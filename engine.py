"""
Provides a training engine for an LSTM model. Training includes a tunable early-stopping mechanism. Intermediate model parameters are stored on each training epoch. In addition key parameters are sent to tensorboard server for diagnostics
"""
 
import copy
import re

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn.utils.rnn
from torch.nn.utils.rnn import PackedSequence

from pycrayon import CrayonClient

import numpy as np

import shutil
import os

import sklearn
from sklearn.metrics import roc_curve

from utils import is_sorted

import time
import datetime

from models import init_weights

from data_handling import JetDatasetSeq

from data_handling import sequence_maker


#class Criterion:
#    def __init__(self,size_average=True):
#        #This could be made more sophisticated -e.g. weights different losses
#        self.crit=nn.BCELoss(size_average=size_average)
#
#    def __call__(self,output,target):
#        #print("output:")
#        #print(output)
#        #print("target:")
#        #print(target)
#        return self.crit(output,target)


class Engine:
    """The training engine 
    
    Performs training and evaluation
    """

    def __init__(self, model,
                 x_train, y_train, seqlengths_train,
                 x_val, y_val, seqlengths_val,
                 x_test, y_test, seqlengths_test,
                 dirpath, data_description, device_id=None, verbose=False):
        self.model = model


        #print("types: {} {} {} {} {} {}".format(x_train.dtype,y_train.dtype,
        #                                        x_val.dtype,y_val.dtype,
        #                                        x_test.dtype,y_test.dtype))
        
        self.model_arch=str(self.model)
        
        #init_weights(self.model)
        
        self.device_id = device_id
        self.verbose = verbose
        self.opt = optim.Adam(self.model.parameters()) # ,eps=1e-3)
        #self.opt = optim.Adamax(self.model.parameters())

        if self.verbose:
            print('this is the list of model parameters')
            print(list(self.model.parameters()))
            
        self.crit = nn.BCELoss()
        self.crit_non_avg = nn.BCELoss(size_average=False)
        ##self.crit = nn.BCEWithLogitsLoss()
        ##self.crit_non_avg = nn.BCEWithLogitsLoss(size_average=False)

        
        #self.crit_per_eg = nn.BCELoss(
        

        self.ds_train=JetDatasetSeq(x_train, seqlengths_train, y_train)
        self.ds_val=JetDatasetSeq(x_val, seqlengths_val, y_val)
        self.ds_test=JetDatasetSeq(x_test, seqlengths_test, y_test)


        self.x_val=x_val
        self.y_val=y_val

        #print('y_val')
        #print(self.y_val[:1024])
        
        self.cc=CrayonClient(hostname="localhost", port=8889)

        if self.device_id is not None:
            model.cuda(device_id)

        self.dirpath=dirpath

        self.data_description=data_description
        
        try:
            os.stat(self.dirpath)
        except:
            print("making a directory for model data: {}".format(self.dirpath))
            os.mkdir(self.dirpath)

        #add the path for the data type to the dirpath
        self.dirpath=self.dirpath+'/'+data_description
        try:
            os.stat(self.dirpath)
        except:
            print("making a directory for model data for data prepared as: {}".format(self.data_description))
            os.makedirs(self.dirpath,exist_ok=True)

        


    def make_training_description(self, batch_size_train, min_epochs, max_epochs, patience, improvement_threshold):

        #description=(str(self.model)+
        description=("LSTM"+
                     "_BatchSize_"+str(batch_size_train)+
                     "_MinEpochs_"+str(min_epochs)+
                     "_MaxEpochs_"+str(max_epochs)+
                     "_Patience_"+str(patience)+
                     "_ImprovementThreshold_"+str(improvement_threshold))
        return description
        
    def train(self,
              min_epochs=50,max_epochs=150,
              batch_size_train=256,
              batch_size_val=1024,
              improvement_threshold=0.0001,
              patience=10,
              improvement_size=10,
              num_workers=32):
        """
        The heavy lifting of training happens here
        
        Keyword arguments:
        min_epochs    -- We will train this long no matter what
        max_epochs    -- Stop training after this many epochs 
                         irrespective of how big the improvement is
        batch_size_train  -- batch size for training pass
        batch_size_val    -- batch size for the validation pass
        improvement_threshold -- extend training if the accuracy increases by this much
        patience      -- set countdown to this if sufficient improvemnt was detected
        improvement_size   -- depreciated
        num_workers    -- munber of threads for the data loaders
        
        """

        launch_time=time.strftime("%Y_%m_%d_%H_%M")

        description=self.make_training_description(batch_size_train,
                                                   min_epochs,
                                                   max_epochs,
                                                   patience,
                                                   improvement_threshold)
        
        exp_name=("exp_"+
                  description+
                  "_Launched_"+launch_time)

        print("\n\n********************************************************************************")
        print("launching experiment %s" % (exp_name))
        
        cc_experiment=self.cc.create_experiment(exp_name)
        
        total_loss=0

        train_dataloader=DataLoader(self.ds_train,batch_size=batch_size_train,
                                    shuffle=True,num_workers=num_workers,
                                    collate_fn=sequence_maker,
                                    pin_memory=True)

        val_dataloader=DataLoader(self.ds_val,batch_size=batch_size_val,
                                  shuffle=False,num_workers=num_workers,
                                  collate_fn=sequence_maker,
                                  pin_memory=True)


        num_batches=len(train_dataloader)

        print("number of batches in taining dataset: ",num_batches)
        
        start_time = time.time()
        prev_time=start_time

        epoch=0

        keep_training=True

        countdown_epochs=min_epochs

        #these are average losses (i.e. per example) on the validation set
        val_current_loss=0.0 #Variable(torch.from_numpy(np.asarray([0.0])))
        val_best_loss=np.inf # Variable(torch.from_numpy(np.asarray([np.inf])))
        val_loss_at_last_patience_increase=np.inf
        prev_train_batch_loss=np.inf
        best_model=None
        
        while epoch<max_epochs and keep_training:

            print("\n\n********************************************************************************")
            print("time %s | epoch: %i begin training loop" % (time.strftime("%H:%M:%S"), epoch))


            #make our model trainable
            self.model.train()

            cumulative_training_average_loss=0.0
            p_threshold=1.0
            
            for batch_idx, (data, target) in enumerate(train_dataloader):

                #print('data')
                #print(data)
                #
                #print('data type')
                #print(type(data))
                
                data=PackedSequence(data[0].cuda(async=True), data[1])
                target=Variable(target.cuda(async=True))
                
                #if batch_idx == 0:
                #    print("first batch cuda status: ", data.is_cuda, target.is_cuda)
                #    print("batch length: ",len(target))
                #data, target = Variable(data), Variable(target)
                #print('data target volatile requires grad {} {} {} {}'.format(data.volatile, data.requires_grad, target.volatile, target.requires_grad))
                self.opt.zero_grad()
                output=self.model(data)

                        
                loss=self.crit(output,target) # this is averaging per example
                if self.verbose and batch_idx == 0:
                    print("first batch length", len(target))
                    print("first batch (average) loss:", loss.data[0])
                    print('output shape {}'.format(output.shape))
                    print('target shape {}'.format(target.shape))
                    
                loss.backward()

                
                if self.verbose and batch_idx == 0:
                    print('loss:')
                    #print(loss.data.cpu().numpy())
                    print(loss)
                    print('output:')
                    print(output)
                    print('target')
                    print(target)
                
                #print('loss gradient')
                #print(loss.grad) # .data.cpu().numpy())

                if self.verbose and batch_idx == 0:
                    print('l5 weight gradient:')
                    print(self.model.l5.weight.grad)
                    print('l5 bias gradient:')
                    print(self.model.l5.bias.grad)

                    print('l5 bias data before step:')
                    print(self.model.l5.bias.data)
                    
                self.opt.step()

                if self.verbose and batch_idx == 0:
                    print('l5 bias data after step:')
                    print(self.model.l5.bias.data)
                
                current_loss=loss.data[0]
                if batch_idx == (num_batches-1):
                    if self.verbose :
                        print("last batch length", len(target))
                        print("last batch (average) loss:", current_loss)
                else:
                    # adding the averages but only for 'non-tail' batches
                    # the .data[0] is to extract the contents of the pytorch variable
                    # we will divide after the batch loop
                    
                    cumulative_training_average_loss+=current_loss

                if current_loss >  2.0*prev_train_batch_loss and epoch > 10:
                    print('CATASTROPHIC LOSS BLOWUP in batch {}'.format(batch_idx))
                    cc_experiment.add_scalar_value("blowup epoch",epoch)
                    cc_experiment.add_scalar_value("blowup batch",batch_idx)
                    
                
                prev_train_batch_loss=current_loss
                
                #calculate the remaining time to finish the epoch
                percent_done=((float(batch_idx)/float(num_batches))*100.0)
                if percent_done>=p_threshold :
                    percent_done_time=time.time()
                    eta_epoch=(percent_done_time-prev_time)*float(100.0-percent_done)/float(percent_done)
                    print("Epoch %i is %.1f%% done; ETA to finish epoch training %.0f seconds at %s" %
                          (epoch,
                           percent_done,
                           eta_epoch,
                           datetime.datetime.fromtimestamp(eta_epoch+percent_done_time).strftime("%H:%M:%S") ))
                    
                    if p_threshold<=1.000001:
                        p_threshold=10.0
                    else:
                        p_threshold+=9.99999999999
                    
            print("time %s | epoch: %i training loop finished" % (time.strftime("%H:%M:%S"), epoch))
            training_average_loss=cumulative_training_average_loss/(num_batches-1)
            cc_experiment.add_scalar_value("training average loss",training_average_loss)
                
            train_done_time=time.time()
            train_duration=train_done_time-prev_time
            print("epoch training done in %.1f seconds" % train_duration)
            cc_experiment.add_scalar_value("epoch train duration", train_duration)
            
            
                    
            print("evaluating on validation set")

            self.model.eval()

            val_len=len(val_dataloader.dataset)
            
            val_correct=0 #this is the count of correctly classified examples in the validation set.

            pred_np_sample=np.empty((0,1))
            target_np_sample=np.empty((0,1))
            
            for batch_idx, (data, target) in enumerate(val_dataloader):
                #data, target in val_dataloader:
                #data=data.cuda(async=True)
                #target=target.cuda(async=True)
                #data, target = Variable(data, volatile=True), Variable(target)

                data=PackedSequence(data[0].cuda(async=True), data[1])
                target=Variable(target.cuda(async=True))
                
                output = self.model(data)
                val_current_loss += self.crit_non_avg(output, target).data[0] # sum up batch loss
                pred_np=output.data.cpu().numpy()
                target_np=target.data.cpu().numpy()

                pred_np_sample=np.append(pred_np_sample,pred_np)
                target_np_sample=np.append(target_np_sample,target_np)
                
                #if self.verbose and batch_idx == 0:
                #    print('pred_np')
                #    print(pred_np)
                #
                #if batch_idx == 0:
                #    print('target_np')
                #    print(target_np)
                
                #print('target shape {}'.format(target_np.shape))
                #fpr, tpr, _ = roc_curve(target_np,pred_np)
                
                #pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                pred_np_thr=np.zeros_like(pred_np)
                pred_np_thr[np.where(pred_np>=0.5)]=1
                #print('prediction shape {}'.format(pred_np_thr.shape))
                val_correct+=np.count_nonzero(pred_np_thr==target_np)

            
            fpr, tpr, thr = roc_curve(np.ravel(target_np_sample), np.ravel(pred_np_sample))
            

            i_E20=np.where(tpr>0.2)[0][0]
            R_E20=1.0/fpr[i_E20]

            i_E50=np.where(tpr>0.5)[0][0]
            R_E50=1.0/fpr[i_E50]

            i_E80=np.where(tpr>0.8)[0][0]
            R_E80=1.0/fpr[i_E80]

            cc_experiment.add_scalar_value("R_E20", R_E20)
            cc_experiment.add_scalar_value("R_E50", R_E50)
            cc_experiment.add_scalar_value("R_E80", R_E80)

            print("R at E of 20 50 80 % {} {} {}".format(R_E20,R_E50,R_E80))
            
            

            #print('is tpr sorted {}'.format(is_sorted(tpr)))
            #print('is fpr sorted {}'.format(is_sorted(fpr)))
            
            
            val_current_loss/=len(val_dataloader.dataset)

            val_accuracy=val_correct/len(val_dataloader.dataset)
            
            val_done_time=time.time()
            val_duration=val_done_time-train_done_time
            print("epoch validation evaluation done in %.1f seconds" % val_duration)
            cc_experiment.add_scalar_value("epoch validation evaluation duration", val_duration)

            cc_experiment.add_scalar_value("validation average loss",val_current_loss)
            cc_experiment.add_scalar_value("validation accuracy",val_accuracy)
            print("Training set loss: %.4f; Validation set loss: %.4f; Validation set accuracy: %.2f" %
                  (training_average_loss, val_current_loss, val_accuracy))

            is_best=False
            if val_current_loss < val_best_loss:
                is_best=True
                best_model=copy.deepcopy(self.model)
                
                print("improvement from "+str(val_best_loss)+
                      " to "+str(val_current_loss)+
                      " saving model")

                val_best_loss=val_current_loss
                
            if val_current_loss<(1.0-improvement_threshold)*val_loss_at_last_patience_increase :
                print("improvement over last jump from %.6f to %.6f" % (val_loss_at_last_patience_increase, val_current_loss))
                if countdown_epochs<=patience: #only extend the training if we are within patience from terminating
                    cc_experiment.add_scalar_value("countdown at patience increase", countdown_epochs)
                    cc_experiment.add_scalar_value("epoch at patience increase", epoch)
                    print("increased countdown from %i to %i" %
                          (countdown_epochs, patience))
                    countdown_epochs=(patience) # NOT: +1 because we always decrement the countdown at the end of the epoch loop
                else:
                    print("will not increase countdown because it is larger than patience")
                val_loss_at_last_patience_increase=val_current_loss

            print("saving checkpoint")
            self.save_checkpoint({
                'description': description,
                'epoch': epoch,
                'arch': self.model_arch,                
                'state_dict': self.model.state_dict(),
                'countdown_epochs': countdown_epochs,
                'training_avg_loss': training_average_loss,
                'val_current_loss': val_current_loss,
                'val_best_loss': val_best_loss,
                'val_loss_at_last_patience_increase': val_loss_at_last_patience_increase,       
                'optimizer' : self.opt.state_dict(),
            }, is_best)
                
            epoch+=1

            if countdown_epochs>0 and epoch<max_epochs:
                keep_training=True
                print("Will keep training countdown: %d " % (countdown_epochs))
            else:
                keep_training=False
                print("no more training")
            
            countdown_epochs-=1
            prev_time=time.time()

        print("Training done at %s; saving model" % (time.strftime("%H:%M:%S")))
        self.model=best_model

        #done training



    def save_checkpoint(self, state, is_best):
        filename='checkpoint_'+state['description']+'_'+str(state['epoch'])+'.pth.tar'
        torch.save(state, self.dirpath+'/'+filename)
        if is_best:
            shutil.copyfile(self.dirpath+'/'+filename, self.dirpath+'/model_best_'+state['description']+'.pth.tar')

              
                
           
        
