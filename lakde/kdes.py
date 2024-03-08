import torch

from abc import ABC, abstractmethod

from lakde.callbacks import ELBOCallback
from lakde.utils import SaneSummaryWriter

class AbstractKDE(ABC):
    def __init__(self, block_size, verbose, logs):
        self.block_size = block_size
        self.verbose = verbose
        self.iter_steps = 0
        
        if logs:
            if isinstance(logs, str):
                self.logger = SaneSummaryWriter(log_dir=logs)
            else:
                self.logger = SaneSummaryWriter()
        else:
            self.logger = None
    
    @abstractmethod
    def init_parameters_(self, X, sparsity_threshold=None):
        pass
    
    @abstractmethod
    def init_summaries_(self, X, sparsity_threshold=None):
        pass
    
    @abstractmethod
    def partial_step_(self, X, block_idx, sparsity_threshold=None):
        pass
    
    def fit(self, X, iterations=200, callbacks=None, sparsity_threshold=None, dataset_label=None):
        N, D = X.shape
        assert X.is_contiguous(), "Training data must be contiguous, call .contiguous() before fitting"
        
        if sparsity_threshold is None:  # no thresholding
            sparsity_threshold = 0
        
        if callbacks is None:
            callbacks = ELBOCallback(verbose=self.verbose)
        
        num_blocks = (N - 1) // self.block_size + 1
        
        if self.iter_steps < 1:
            if self.verbose:
                print('Initializing parameters...')
            self.init_parameters_(X, sparsity_threshold)
            self.init_summaries_(X, sparsity_threshold)
        
        if self.logger:
            d = self.hparam_state_dict()
            if sparsity_threshold:
                d['threshold'] = str(sparsity_threshold)
            if dataset_label:
                d['dataset'] = str(dataset_label)
            self.logger.add_hparams(d, {'log_likelihood/test': None, 'elbo/elbo': None})
        
        if self.verbose:
            print('Fitting data...')
        
        try:
            for i in range(iterations):
                epoch = self.iter_steps // num_blocks
                for bi in range(num_blocks):
                    self.partial_step_(X, bi, sparsity_threshold=sparsity_threshold)
                    
                    if self.verbose:
                        print('Iteration {} (epoch {}, batch {}/{})'.format(self.iter_steps + 1, epoch + 1,
                                                                            bi + 1, num_blocks), end='')
                    
                    if callbacks is not None:
                        if isinstance(callbacks, (list, tuple)):
                            for cb in callbacks:
                                cb(X, self, self.iter_steps)
                        else:
                            callbacks(X, self, self.iter_steps)
                    
                    if self.verbose:
                        print()
                    
                    self.iter_steps += 1
        
        except (KeyboardInterrupt, StopIteration):
            pass
        return self
    
    @abstractmethod
    def data_log_likelihood_no_rnm(self, X):
        pass
    
    @abstractmethod
    def data_log_likelihood(self, X):
        pass
    
    def data_likelihood(self, X):
        return torch.exp_(self.data_log_likelihood(X))
    
    @abstractmethod
    def compute_elbo(self, X):
        pass
    
    @abstractmethod
    def log_pred_density(self, X, Y):
        pass
    
    def pred_density(self, X, Y):
        return torch.exp_(self.log_pred_density(X, Y))
    
    @abstractmethod
    def sample(self, X, n):
        pass
    
    def load_state_dict(self, state_dict):
        self.block_size = state_dict['block_size']
        self.verbose = state_dict['verbose']
        self.iter_steps = state_dict['iter_steps']
    
    def state_dict(self):
        return {
            'block_size': self.block_size,
            'verbose'   : self.verbose,
            'iter_steps': self.iter_steps
        }
    
    def hparam_state_dict(self):
        return {'block_size': self.block_size}
    
    def save(self, fname, state_dict=None):
        if state_dict is None:
            state_dict = {}
        d = self.state_dict()
        d.update(state_dict)
        torch.save(d, fname)

