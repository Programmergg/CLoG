import torch
import random
from copy import deepcopy
import torch.nn.functional as F
from diffusers.utils import make_image_grid
from diffusers import UNet2DModel, DDIMScheduler
from networks.diffusion.base import BaseLearner
from peft import inject_adapter_in_model, LoraConfig
from networks.diffusion.pipeline_ddim import MyDDIMPipeline

class Learner(BaseLearner):
    def __init__(self, model_args, data_args, training_args):
        super(Learner, self).__init__()
        if data_args.image_size == 32:
            block_out_channels = (128, 256, 256, 256)
            down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
            up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
        elif data_args.image_size == 64:
            block_out_channels = (128, 256, 384, 512)
            down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
            up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
        elif data_args.image_size == 128:
            block_out_channels = (128, 128, 256, 384, 512)
            down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D")
            up_block_types = ("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
        self.unet = UNet2DModel(
            sample_size=data_args.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            downsample_type='resnet',
            upsample_type='resnet',
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            num_class_embeds=data_args.tot_class_num,
            dropout=0.1,
        )
        self.scheduler = DDIMScheduler(num_train_timesteps=model_args.diffusion_time_steps)
        self.pipeline = MyDDIMPipeline(unet=self.unet, scheduler=self.scheduler)
        self.inference_steps = model_args.inference_steps
        self.model_args, self.data_args, self.training_args = model_args, data_args, training_args

        self.init_unet = deepcopy(self.unet)
        # For Task0, the full fine-tuning approach is used, while the subsequent tasks are fine-tuned using the LoRA PEFT method.
        if self.data_args.task_id > 0:
            path = f'logs/model_arch={self.model_args.model_arch}/method={self.model_args.method}/dataset_name={self.data_args.dataset_name}/seed={self.train_args.seed}/task_id=0'
            self.load_backbone(path)
            self.lora_config = LoraConfig(
                r=8,
                lora_dropout=0.1,
                lora_alpha=8,
                init_lora_weights=True,
                target_modules=['to_k', 'to_q', 'to_v', 'to_out.0', 'conv1', 'conv2'],
            )
            self.class_embedding_past = [deepcopy(self.unet.class_embedding.weight[i].data) for i in range(self.data_args.class_num * self.data_args.task_id)]
            self.unet = inject_adapter_in_model(self.lora_config, self.unet, "task_{}".format(self.data_args.task_id))
            for n, p in self.unet.named_parameters():
                if 'class_embedding' in n or 'lora' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    def load_backbone(self, path):
        self.unet.load_state_dict(torch.load(f'{path}/model.pth', map_location='cpu'))

    def train_step(self, x, y):
        loss = 0.
        noise = torch.randn(x.shape, device=x.device)
        timesteps = torch.randint(0, self.model_args.diffusion_time_steps, (x.shape[0],), device=x.device, dtype=torch.int64)
        noisy_x = self.scheduler.add_noise(x, noise, timesteps)
        noise_pred = self.unet(noisy_x, timesteps, class_labels=y, return_dict=False)[0]
        loss += F.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, bs, seed, labels):
        # This part is actually problematic, as Continual Generation doesn't actually get the exact task id.
        self.pipeline.to(self.unet.device)
        task_id = (labels[0] // self.data_args.class_num).item()
        self.unet = deepcopy(self.init_unet)
        path = f'logs/model_arch={self.model_args.model_arch}/method={self.model_args.method}/dataset_name={self.data_args.dataset_name}/seed={self.train_args.seed}/task_id=0'
        self.load_backbone(path)
        if task_id > 0:
            self.unet = inject_adapter_in_model(self.lora_config, self.unet, "task_{}".format(task_id))
            self.unet.load_state_dict(torch.load(self.training_args.all_dirs[task_id] + '/lora.pth', map_location='cpu'))
        self.unet.to(self.pipeline.device)
        self.unet.eval()
        self.pipeline.unet = self.unet
        image = self.pipeline(
            batch_size=bs,
            labels=labels,
            num_inference_steps=self.inference_steps,
            generator=torch.manual_seed(seed),
            output_type='pil'
        ).images
        return image

    def save(self, path, dataloader):
        labels = random.choices([self.data_args.sequence.index(x) for x in self.data_args.task_labels], k=self.training_args.per_device_eval_batch_size)
        images = self.sample(
            self.training_args.per_device_eval_batch_size, 
            self.training_args.seed,
            labels=torch.tensor(labels, device=self.unet.device, dtype=torch.long)
        )
        make_image_grid(
            images[:int(self.training_args.per_device_eval_batch_size ** 0.5)*int(self.training_args.per_device_eval_batch_size ** 0.5)], 
            rows=int(self.training_args.per_device_eval_batch_size ** 0.5), 
            cols=int(self.training_args.per_device_eval_batch_size ** 0.5)
        ).save(f'{path}/samples.png')
        torch.save(self.get_backbone_state_dict(), f'{path}/model.pth')
        if self.data_args.task_id > 0:
            torch.save(self.get_lora_state_dict(), f'{path}/lora.pth')

    def get_backbone_state_dict(self):
        state_dict = self.unet.state_dict()
        for i in range(self.data_args.class_num * (self.data_args.task_id + 1)):
            # if we have weight decay, the embedding will gradually decay even if it's not trained...
            state_dict['class_embedding.weight'][i] = self.class_embedding_past[i]
        return {k.replace("base_layer.",  ""): v for k, v in state_dict.items() if 'lora' not in k}

    def get_lora_state_dict(self):
        state_dict = self.unet.state_dict()
        return {k: v for k, v in state_dict.items() if 'lora' in k}