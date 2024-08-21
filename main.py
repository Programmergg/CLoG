# import os
# # Set environment variables for HTTP and HTTPS proxies.
# # These variables define the proxy server address (http://127.0.0.1:7897)
# # to route network traffic through a proxy. This is useful when you need to
# # access the internet via a proxy server, such as for network security or bypassing restrictions.
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import random
import numpy as np
from trainer.train import Trainer
from trainer.eval import evaluate
from transformers import HfArgumentParser
from dataloader.data import get_dataloader
from networks.diffusion import ddim, lora_ddim
from utils.configs import ModelArguments, DataArguments, TrainArguments, update_configs

def get_model(model_args, data_args, training_args):
    '''
        model factory function
        A model must implement:
            train_step(x, y, task_id=None) -> loss
            sample(bs, seed) -> image
            load()
            save()
    '''
    if 'ddim' in model_args.model_arch and 'lora' in model_args.method:
        diffusion_model = lora_ddim.Learner(model_args, data_args, training_args)
    elif 'ddim' in model_args.model_arch:
        diffusion_model = ddim.Learner(model_args, data_args, training_args)
    else:
        raise NotImplementedError
    return diffusion_model

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = update_configs(model_args, data_args, training_args)
    print(model_args, data_args, training_args)

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_args.seed)
        torch.cuda.manual_seed_all(training_args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataloaders = get_dataloader(data_args, training_args)
    diffusion_model = get_model(model_args, data_args, training_args)
    trainer = Trainer(diffusion_model, dataloaders, model_args, data_args, training_args)
    trainer.train()
    evaluate(diffusion_model, dataloaders, model_args, data_args, training_args)

if __name__ == '__main__':
    main()