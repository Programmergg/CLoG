import os
import re
import torch
import random
import logging
from accelerate import Accelerator
from utils.metrics import Validator
from diffusers.utils import make_image_grid

def extract_metrics(log_file_path):
    metrics = {}
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(r'Task (\d+):.*?\'metric_for_validation\': ([\d.]+)', line)
            if match:
                task_id = int(match.group(1))
                metric_value = float(match.group(2))
                metrics[task_id] = metric_value
    return metrics

def update_task_id_in_path(path, new_task_id):
    updated_path = re.sub(r'task_id=\d+', f'task_id={new_task_id}', path)
    return updated_path

@torch.no_grad()
def evaluate(diffusion_model, dataloaders, model_args, data_args, training_args):
    accelerator = Accelerator()
    logging.basicConfig(
        filename=os.path.join(training_args.logging_dir, 'log.txt'),  # Updated to append to 'log.txt'
        filemode='a',
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    logging.info("Running CIL Testing")
    diffusion_model.unet = accelerator.prepare(diffusion_model.unet)
    diffusion_model.eval()

    # Initialize variables for metrics
    performance_sum = 0
    curr_task_performance = {}
    for task_id in range(data_args.task_id + 1):
        validator = Validator(data_args, training_args)
        validator.to(accelerator.device)
        all_samples = []
        all_labels = []
        while len(all_samples) < data_args.tot_samples_for_eval:
            labels = random.choices([data_args.sequence.index(x) for x in data_args.all_task_labels[task_id]], k=training_args.per_device_eval_batch_size)
            all_samples.extend(
                diffusion_model.sample(
                    training_args.per_device_eval_batch_size,
                    training_args.seed + len(all_samples),
                    labels=torch.tensor(labels, device=accelerator.device, dtype=torch.long)
                )
            )
            all_labels.extend(labels)
        # Evaluate and log metrics for the current task
        eval_logs = validator.evaluate(all_samples[:data_args.tot_samples_for_eval], all_labels, dataloaders['all_test_loader'][task_id])
        current_performance = eval_logs['metric_for_validation']
        curr_task_performance[task_id] = current_performance
        # Save samples grid image
        rows = cols = 16
        make_image_grid(all_samples[:rows * cols], rows=rows, cols=cols).save(f'{training_args.logging_dir}/{task_id}.png')
        logging.info(f"Task {task_id}: {str(eval_logs)}")
        del validator
        performance_sum += current_performance

    # Update AFQ for the last task
    performance_avg = performance_sum / (data_args.task_id + 1)
    logging.info(f"Average Performance of Task {data_args.task_id}: {performance_avg}")
    task_count = 0
    all_task_performance_sum = 0
    for task_id in range(data_args.task_id + 1):
        task_metrics = extract_metrics(update_task_id_in_path(os.path.join(training_args.logging_dir, 'log.txt'), task_id))
        all_task_performance_sum += sum(task_metrics.values())
        if task_id == 0:
            first_task_performance = sum(task_metrics.values()) / len(task_metrics)
        task_count += len(task_metrics)
    all_task_performance_avg = all_task_performance_sum / task_count
    logging.info(f"Average Incremental Performance of Task {data_args.task_id}: {all_task_performance_avg}")

    # Calculate Forgetting Rate (FR) if not the first task
    if data_args.task_id > 0:
        forgetting = 0.0
        for task_id in range(data_args.task_id):
            init_task_metrics = extract_metrics(update_task_id_in_path(os.path.join(training_args.logging_dir, 'log.txt'), task_id))[task_id]
            forgetting += (curr_task_performance[task_id] - init_task_metrics)
        forgetting /= data_args.task_id
        logging.info(f"Forgetting Rate of Task {data_args.task_id}: {forgetting}")
    else:
        logging.info(f"Forgetting Rate of Task {data_args.task_id}: {0.0}")