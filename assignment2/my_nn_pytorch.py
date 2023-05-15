import torch
from torch import nn
import pandas as pd
import pickle
import numpy as np
from timeit import default_timer as timer



class AccuracyCriterion():
    
    def __init__(self):
        self.name = "accuracy"
        self.onehot = OneHotEncoding()

    def __call__(self, logits, target, full=False):
        return [(self.onehot(logits) == target).type(torch.float).cpu().numpy()]


class LogitsToProbabilities():
    
    def __init__(self):
        self.softmax = nn.Softmax(dim=1)
    
    def __call__(self, logits):
        return self.softmax(logits)


class OneHotEncoding():
    
    def __call__(self, y):
        return y.argmax(1)


class MyNeuralNetwork():
    """
    """
    
    def __init__(self, 
                 model, 
                 loss_fn, 
                 optimizer, 
                 dataloader_train, 
                 dataloader_test, 
                 device='cpu', 
                 criterion=AccuracyCriterion(),
                 logits2prob=LogitsToProbabilities(),
                 onehot=OneHotEncoding(),
                 custom_eval=None):
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
        
        self.criterion = criterion
        self.logits2prob = logits2prob
        self.onehot = onehot
        self.device = device
        
        self.log = []
        
        self.epoch = 0
        self.iteration = 0
    
    
    def log2pandas(self):
        return pd.DataFrame(self.log)


    def add_log_entry(self, logtype, **additional_log_data):
        
        entry = {
                'iteration':   self.iteration,
                'epoch':       self.epoch,
                'logtype':     logtype,
                }
        
        for key, value in additional_log_data.items():
            entry[key] = value
        
        self.log.append(entry)
    
    
    
    def get_total_train_time(self):
        df = self.log2pandas()
        df = df[df.logtype == 'timer']
        return np.around(df.elapsed_time.sum(), decimals=1)
    
    
    def evaluate(self, dataloader=None, X=None, y=None):
        
        self.model.eval()
        
        if dataloader == None:
            with torch.no_grad():    
                logits = self.model(X)
                loss_mean = self.loss_fn(logits, y).item()
                crit = self.criterion(logits, y)
        else:
            loss_mean, crit, indices = 0, [], []
            size = len(dataloader.dataset)
            with torch.no_grad():    
                for X, y, index in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    logits = self.model(X)
                    loss_mean += self.loss_fn(logits, y).item() * X.shape[0]
                    crit += self.criterion(logits, y)
                    indices += [index.numpy()]
#            print(crit)
#            print(indices)
#            print(loss_mean)
#            print(size)
            loss_mean /= size
            indices = np.argsort(np.concatenate(indices))
            crit = np.concatenate(crit)
            crit=crit[indices]
        
        crit_mean = crit.mean()
        return loss_mean, crit_mean, crit
    
   
    def evaluate_and_log_perf_traindata(self):
        loss_mean, crit_mean, crit = self.evaluate(self.dataloader_train)
        self.add_log_entry('performance', criterion=crit, criterion_mean=crit_mean,
                           loss=loss_mean, dataset='train')
   

    def evaluate_and_log_perf_testdata(self):
        loss_mean, crit_mean, crit = self.evaluate(self.dataloader_test)
        self.add_log_entry('performance', criterion=crit, criterion_mean=crit_mean,
                           loss=loss_mean, dataset='test')
    
    
    def predict_onehot(self, X):
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
        return self.onehot(logits)
    
    
    def predict_probabilities(self, X):
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
        return self.logits2prob(logits)
    
    
    def _train_batch(self, X, y, batch=None):
        """
        """
        self.model.train()

        # Compute prediction, loss and criterion
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        
        crit = np.array(self.criterion(logits, y))
        
        # Save performance data
        crit_mean = crit.mean()
        loss_val = loss.item()
        self.add_log_entry('performance', criterion_mean=crit_mean,
                           loss=loss_val, dataset='train_batch', batch=batch)
        
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
            self.evaluate_and_log_perf_traindata()
            self.print_performance()
            self.evaluate_and_log_perf_testdata()
            self.print_performance()
        
        start_time = timer()

        self.epoch += 1
        
        test_iterations = self._get_iterations_when2eval(when2eval)

        for ibatch, (X, y, index) in enumerate(self.dataloader_train):
            self.iteration += 1
            
            X, y = X.to(self.device), y.to(self.device)
            
            self._train_batch(X, y, ibatch)
            
            if self.iteration in test_iterations:
                self.evaluate_and_log_perf_testdata()
                self.print_performance()
        
        # Evaluate on all training data
        self.evaluate_and_log_perf_traindata()
        self.print_performance()
        
        # prevent to evaluate the performance twice in case it has already been done for this iteration
        if not self.iteration in test_iterations:
            self.evaluate_and_log_perf_testdata()
            self.print_performance()
        
        # Here we get the timing information and write it to the log
        end_time = timer()
        elapsed_time = end_time - start_time
        self.add_log_entry('timer', start_time=start_time, end_time=end_time,
                           elapsed_time=elapsed_time, dataset='epoch timing')

    
    
    def train(self, nepochs=1):
        """
        """
        # Print the start of the performance output
        self.print_performance_header()

        for iepoch in range(nepochs):
            self.train_epoch()
   

    def to_disk(self, fname):
        with open(fname, 'wb') as fh:
            pickle.dump(self, fh)


    def print_performance_header(self):
        print('%9s %9s %9s %9s %14s' % ('EPOCH', 'ITERATION', 'DATASET', 'COST', self.criterion.name.upper()))


    def print_performance(self, logentry=None, dataset=None, header=False):
        if logentry is None:
            df = self.log2pandas()
            df = df[df.logtype == 'performance']
            if dataset is None:
                df = df.iloc[-1:]
            else:
                if isinstance(dataset, str):
                    df = df[df.dataset == dataset].iloc[-1:]
                else:
                    idx = []
                    for ds in dataset:
                        idx.append(df[df.dataset == ds].index[-1])
                    df = df.loc[idx]
        else:
            df = pd.DataFrame(logentry)

        if header:
            self.print_performance_header()
        for index, row in df.iterrows():
            print('%9i %9i %9s %9.3f %14.4f' % (row.epoch, 
                                                row.iteration, 
                                                row.dataset.upper(), 
                                                row.loss,
                                                row.criterion_mean))
   

    def print_performance_history(self, dataset=['train', 'test'], header=False):
        df = self.log2pandas()
        df = df[df.logtype == 'performance']
        if isinstance(dataset, str):
            df = df[df.dataset == dataset]
        else:
            idx = []
            for ds in dataset:
                idx.append(df[df.dataset == ds].index)
            idx = np.array(idx).reshape(-1)
            df = df.loc[idx]
        df = df.sort_values(by=['iteration'])
        if header:
            self.print_performance_header()
        for index, row in df.iterrows():
            print('%9i %9i %9s %9.3f %14.4f' % (row.epoch, 
                                                row.iteration, 
                                                row.dataset.upper(), 
                                                row.loss,
                                                row.criterion_mean))
        

def load_nn(fname):
    """
    """
    with open(fname, 'rb') as fh:
        nn = pickle.load(fh)
    print(nn.model)
    print(nn.print_performance_history(header=True))
    return nn



