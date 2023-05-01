import torch
from torch import nn
import pandas as pd
import pickle
import numpy as np
from timeit import default_timer as timer



class EventLogEntry():
    """
    """
    def __init__(self, iteration, epoch, start_time, end_time, description):
        """
        """
        self.iteration = iteration
        self.epoch = epoch
        self.start_time = start_time
        self.end_time = end_time
        self.elapsed_time = self.end_time - self.start_time
        self.description = description


    def as_dict(self):
        return {'iteration': self.iteration,
                'epoch': self.epoch,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'elapsed_time': self.elapsed_time,
                'description': self.description,
                } 


class NNLogEntry():
    """
    """
    def __init__(self, iteration, epoch, accuracy, cost, dataset, batch=None):
        """
        """
        self.iteration = iteration
        self.epoch = epoch
        self.accuracy = accuracy
        self.cost = cost
        self.dataset = dataset
        self.batch = batch
    

    def as_dict(self):
        return {'iteration': self.iteration,
                'epoch': self.epoch,
                'batch': self.batch,
                'accuracy': self.accuracy,
                'cost': self.cost,
                'dataset': self.dataset,
                } 


class NNLogger():
    """
    """
    
    def __init__(self):
        """
        """
        self._traindata_log = []
        self._traindata_batch_log = []
        self._testdata_log = []
        self._event_log = []
        self.last_traindata = None
        self.last_testdata = None
        self.last_event = None


    def add_entry_traindata_log(self, iteration, epoch, accuracy, cost):
        entry = NNLogEntry(iteration, epoch, accuracy, cost, 'train')
        self._traindata_log.append(entry)
        self.last_traindata = self._traindata_log[-1]


    def add_entry_traindata_batch_log(self, iteration, epoch, accuracy, cost, batch):
        entry = NNLogEntry(iteration, epoch, accuracy, cost, 'train_batch', batch)
        self._traindata_batch_log.append(entry)
        self.last_traindata_batch = self._traindata_batch_log[-1]


    def add_entry_testdata_log(self, iteration, epoch, accuracy, cost):
        entry = NNLogEntry(iteration, epoch, accuracy, cost, 'test')
        self._testdata_log.append(entry)
        self.last_testdata = self._testdata_log[-1]


    def add_entry_event_log(self, iteration, epoch, starttime, endtime, description):
        entry = EventLogEntry(iteration, epoch, starttime, endtime, description)
        self._event_log.append(entry)
        self.last_event = self._event_log[-1]
    

    def perflogdata2pandas(self):
        data = self._traindata_batch_log + self._traindata_log + self._testdata_log
        return pd.DataFrame([x.as_dict() for x in data])
    

    def eventlog2pandas(self):
        data = self._event_log
        return pd.DataFrame([x.as_dict() for x in data])


    def print_last_train_performance(self):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!        TRAIN          !')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('! EPOCH:          %5i' % self.last_traindata.epoch)
        print('! ITERATION:      %5i' % self.last_traindata.iteration)
        print('! COST:          %6.3f' % self.last_traindata.cost)
        print('! ACCURACY:       %4.1f%%' % self.last_traindata.accuracy)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!\n')

    
    def print_last_test_performance(self):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!        TEST           !')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('! EPOCH:          %5i' % self.last_testdata.epoch)
        print('! ITERATION:      %5i' % self.last_testdata.iteration)
        print('! COST:          %6.3f' % self.last_testdata.cost)
        print('! ACCURACY:       %4.1f%%' % self.last_testdata.accuracy)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!\n')
    

    def print_last_performance(self):
        """
        """
        self.print_last_train_performance() 
        self.print_last_test_performance() 


class MyNeuralNetwork():
    """
    """
    
    def __init__(self, model, loss_fn, optimizer, dataloader_train, dataloader_test, device='cpu'):
        """
        """
        
        self.model = model.to(device)
        print(self.model)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.batch_size_train = dataloader_train.batch_size
        self.batch_size_test = dataloader_test.batch_size
        
        self.device = device
        self._logger = NNLogger()
        
        self.epoch = 0
        self.iteration = 0

    
    def _log_traindata(self, iteration, epoch, accuracy, cost):
        self._logger.add_entry_traindata_log(iteration, epoch, accuracy, cost)

    
    def _log_traindata_batch(self, iteration, epoch, accuracy, cost, batch):
        self._logger.add_entry_traindata_batch_log(iteration, epoch, accuracy, cost, batch)


    def _log_testdata(self, iteration, epoch, accuracy, cost):
        self._logger.add_entry_testdata_log(iteration, epoch, accuracy, cost)


    def _log_event(self, iteration, epoch, starttime, endtime, description):
        self._logger.add_entry_event_log(iteration, epoch, starttime, endtime, description)


    def perflog2pandas(self):
        return self._logger.perflogdata2pandas()
    
    
    def eventlog2pandas(self):
        return self._logger.eventlog2pandas()


    def print_epoch_info(self):
        self._logger.print_last_performance()


    def get_total_train_time(self):
        df = self.eventlog2pandas()
        df = df[df.description.str.startswith('Training Epoch ')]
        return np.around(df.elapsed_time.sum(), decimals=1)


    def test(self, dataloader):
        size = len(dataloader.dataset)
        self.model.eval()
        cost, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                cost += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        cost *= dataloader.batch_size/size
        correct /= size
        return 100*correct, cost
    

    def log_model_perf_testdata(self):
        acc, cost = self.model_perf_testdata()
        self._log_testdata(self.iteration, self.epoch, acc, cost)


    def log_model_perf_traindata(self):
        acc, cost = self.model_perf_traindata()
        self._log_traindata(self.iteration, self.epoch, acc, cost)
    

    def model_perf_testdata(self):
        return self.test(self.dataloader_test)
    

    def model_perf_traindata(self):
        return self.test(self.dataloader_train)
    
    
    def _train_batch(self, X, y, batch=None):
        """
        """
        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # Save performance data
        acc = correct * 100 / self.dataloader_train.batch_size
        cost = loss.item()
        self._log_traindata_batch(self.iteration, self.epoch, acc, cost, batch)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return
    
    
    def _get_iterations_when2eval(self, when2eval):
        """
        """
        if when2eval is None:
            test_iterations = (-1,)
        else:
            try:
                tmp_it = iter(when2eval)
                test_iterations = tuple(when2eval)
            except TypeError as te:
                test_iterations = (when2eval,)
        return test_iterations


    def train_epoch(self, when2eval=None):
        """ Do the training for one epoch, i.e., step once through every batch
        """
        # Log initial performance of the model
        if self.epoch == 0 and self.iteration == 0:
            self.log_model_perf_traindata()
            self.log_model_perf_testdata()
            self.print_epoch_info()
        
        start_time = timer()

        self.epoch += 1
        
        test_iterations = self._get_iterations_when2eval(when2eval)

        for ibatch, (X, y) in enumerate(self.dataloader_train):
            self.iteration += 1
            
            X, y = X.to(self.device), y.to(self.device)
            
            self._train_batch(X, y, ibatch)
            
            if self.iteration in test_iterations:
                self.log_model_perf_testdata()
        
        self.log_model_perf_traindata()
        
        # prevent to evaluate the performance twice
        if not self.iteration in test_iterations:
            self.log_model_perf_testdata()
        
        self.print_epoch_info()
        end_time = timer()
        self._log_event(self.iteration, self.epoch, start_time, end_time, 'Training Epoch %i' % self.epoch)

    
    
    def train(self, nepochs=1):
        """
        """
        for iepoch in range(nepochs):
            self.train_epoch()
   

    def to_disk(self, fname):
        with open(fname, 'wb') as fh:
            pickle.dump(self, fh)



def load_nn(fname):
    """
    """
    with open(fname, 'rb') as fh:
        nn = pickle.load(fh)
    return nn



