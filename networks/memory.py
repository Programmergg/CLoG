import os
import torch
import numpy as np

def reservoir(num_seen_examples: int, max_size: int) -> int:
    '''
    Reservoir sampling.
    Return the target index if the current image is sampled, else -1
    '''
    return num_seen_examples if num_seen_examples < max_size else np.random.randint(0, max_size)

class Memory:
    '''
        The replay buffer.
    '''
    def __init__(self, max_size, data_args):
        self.max_size = max_size
        self.num_seen_examples = 0
        self.attributes = ['images', 'labels']
        # FIXME: the shape of 'labels' is simply indices
        self.attributes_shape = [(self.max_size, 3, data_args.image_size, data_args.image_size), (self.max_size)]
        self.attributes_dtype = [torch.float32, torch.int64]
        for attr_str in self.attributes:
            setattr(self, attr_str, torch.zeros(self.attributes_shape[self.attributes.index(attr_str)], dtype=self.attributes_dtype[self.attributes.index(attr_str)]))

    def __len__(self):
        return self.num_seen_examples
    
    def save(self, path):
        obj = {}
        for attr_str in self.attributes:
            obj[attr_str] = getattr(self, attr_str)
        obj['num_seen_examples'] = self.num_seen_examples
        torch.save(obj, os.path.join(path, 'memory.pt'))
    
    def load(self, path):
        obj = torch.load(os.path.join(path, 'memory.pt'), map_location='cpu')
        for attr_str in self.attributes:
            setattr(self, attr_str, obj[attr_str])
        self.num_seen_examples = obj['num_seen_examples']
    
    def empty(self):
        return self.num_seen_examples == 0
    
    @torch.no_grad()
    def add(self, images, labels):
        for i in range(images.size(0)):
            idx = reservoir(self.num_seen_examples, self.max_size)
            self.images[idx] = images[i].to(self.images.device)
            self.labels[idx] = labels[i].to(self.images.device)
            if self.num_seen_examples < self.max_size:
                self.num_seen_examples += 1
    
    @torch.no_grad()
    def sample(self, batch_size):
        if batch_size > min(self.num_seen_examples, self.images.size(0)):
            batch_size = min(self.num_seen_examples, self.images.size(0))
        idx = np.random.choice(min(self.num_seen_examples, self.images.size(0)), batch_size, replace=False)
        return self.images[idx].clone(), self.labels[idx].clone()