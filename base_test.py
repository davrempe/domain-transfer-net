import copy

class BaseTest(object):
    '''
    Abstract class that outlines how a network test case should be defined.
    '''
    
    def __init__(self,use_gpu=True):
        self.use_gpu = use_gpu
        self.log = {}
        self.log['best_model'] = None
        self.log['train_loss'] = []
        self.log['val_loss'] = []
        self.model = None
        self.loss_function = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def create_data_loaders(self):
        '''
        Reads in the desired dataset for the Test, splits into
        a training and validation set, and saves loaders.
        '''
        pass
    
    def visualize_single_batch(self):
        '''
        Plots a minibatch as an example of what the data looks like.
        '''
        pass
    
    def create_model(self):
        '''
        Constructs the model, converts to GPU if necessary. Saves for training.
        '''
        pass
    
    def create_loss_function(self):
        '''
        Constructs the loss function, and saves it to be called during training.
        '''
        pass
    
    def create_optimizer(self):
        '''
        Creates and saves the optimizer to use for training.
        '''
        pass
    
    def train_model(self, num_epochs, **kwargs):
        '''
        Trains the model.
        '''
        pass
    
    def test_model(self):
        '''
        Tests the model and returns the loss.
        '''
        pass
    
    def log_losses(self, train_loss, val_loss):
        '''
        Writes the given training and validation loss to the log.
        '''
        self.log['train_loss'].append(train_loss)
        self.log['val_loss'].append(val_loss)
        
    def log_best_model(self):
        '''
        Writes the "best" model found so far to the log.
        '''
        self.log['best_model'] = copy.deepcopy(self.model)
