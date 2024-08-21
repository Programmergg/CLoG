import torch
from abc import abstractmethod

class BaseLearner(torch.nn.Module):
    def __init__(self):
        super(BaseLearner, self).__init__()
        pass
    
    @abstractmethod
    def train_step(self, x, y):
        pass

    def regularize_with_gradient_before_step(self):
        pass

    def regularize_with_gradient_after_step(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    def load(self):
        pass

    def save(self):
        pass