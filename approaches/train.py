import os
import torch
import random
import logging
from tqdm.auto import tqdm
from torch.optim import Adam
from accelerate import Accelerator
from utils.metrics import Validator
from diffusers.optimization import get_cosine_schedule_with_warmup

class Trainer:
    def __init__(self, diffusion_model, dataloaders, model_args, data_args, training_args):
        self.diffusion_model = diffusion_model
        self.dataloaders = dataloaders
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def launch_accelerator(self):
        self.accelerator = Accelerator(
            mixed_precision=self.training_args.mixed_precision,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
            log_with='tensorboard',
            project_dir=os.path.join(self.training_args.logging_dir, 'logs')
        )
    
    def launch_logger(self):
        logging.basicConfig(
            filename=os.path.join(self.training_args.logging_dir, 'log.txt'),
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )

    def train(self):
        self.launch_accelerator()
        self.launch_logger()
        if self.accelerator.is_main_process:
            logging.info("Begin Training...")
            os.makedirs(self.training_args.logging_dir, exist_ok=True)
            self.accelerator.init_trackers("training")
        self.optimizer = Adam([p for p in self.diffusion_model.parameters() if p.requires_grad], lr=self.training_args.learning_rate, weight_decay=self.training_args.weight_decay)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer, 
            num_warmup_steps=self.training_args.lr_warmup_steps, 
            num_training_steps=self.training_args.num_train_epochs * len(self.dataloaders['train_loader'])
        )
        self.diffusion_model.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.diffusion_model.unet, self.optimizer, self.dataloaders['train_loader'], self.lr_scheduler
        )
        self.validator = Validator(self.data_args, self.training_args)
        self.validator.to(self.accelerator.device)
        self.global_step = 0
        self.global_metric = 1e09  # Please ensure: the smaller the better
        for epoch in range(self.training_args.num_train_epochs):
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(self.train_dataloader):
                self.diffusion_model.train()
                loss = self.diffusion_model.train_step(batch['images'], batch['labels'])
                self.accelerator.backward(loss)
                self.diffusion_model.regularize_with_gradient_before_step()
                self.accelerator.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
                self.optimizer.step()
                # regularize model with gradient information (for SI).
                self.diffusion_model.regularize_with_gradient_after_step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                logs = {"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0], 'step': self.global_step}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)
                self.global_step += 1
                if self.global_step % self.training_args.eval_steps == 0:
                    # This will evaluate num_proc times.
                    self.diffusion_model.unet = self.accelerator.unwrap_model(self.diffusion_model.unet)
                    self.diffusion_model.eval()
                    self.validate()
                    self.diffusion_model.unet = self.accelerator.prepare_model(self.diffusion_model.unet)
                    self.diffusion_model.train()
        self.diffusion_model.unet = self.accelerator.unwrap_model(self.diffusion_model.unet)
        self.diffusion_model.eval()
        self.validate()

    def validate(self):
        with torch.no_grad():
            labels = random.choices([self.data_args.sequence.index(x) for x in self.data_args.task_labels], k=self.training_args.per_device_eval_batch_size)
            samples = self.diffusion_model.sample(
                self.training_args.per_device_eval_batch_size, 
                self.training_args.seed,
                labels=torch.tensor(labels, device=self.accelerator.device, dtype=torch.long)
            )
            eval_logs = self.validator.evaluate(samples, labels, self.dataloaders['test_loader'])
            if eval_logs['metric_for_validation'] < self.global_metric:
                self.diffusion_model.save(self.training_args.logging_dir, dataloader=self.train_dataloader)
                self.global_metric = eval_logs['metric_for_validation']
            if self.accelerator.is_main_process:
                logging.info(f"Step {self.global_step}: {str(eval_logs)}")