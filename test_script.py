import torch
import numpy as np

timesteps = torch.from_numpy(np.arange(0, 10)[::-1].copy().astype(np.int64))
print(timesteps)