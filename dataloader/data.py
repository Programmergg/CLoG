import torch
import itertools
import numpy as np
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk

def get_sequence_map(dataset_name, shuffle_order=False):
    if dataset_name.startswith('CIFAR10-'):
        label_map = list(range(10))
    elif dataset_name.startswith('Mnist-'):
        label_map = list(range(10))
    elif dataset_name.startswith('FashionMnist-'):
        label_map = list(range(10))
    elif dataset_name.startswith('Flowers-'):
        label_map = list(range(102))
    elif dataset_name.startswith('CUB-Birds-'):
        label_map = list(range(200))
    elif dataset_name.startswith('Stanford-Cars-'):
        label_map = list(range(196))
    elif dataset_name.startswith('ImageNet-'):
        label_map = list(range(1000))
    elif dataset_name.startswith('Custom-Objects-'):
        label_map = list(range(150))
    else:
        raise NotImplementedError
    if shuffle_order:
        label_map = np.random.permutation(label_map).tolist()
    return label_map

def divide_list(sequence_list, task_num):
    n = len(sequence_list) // task_num
    remainder = len(sequence_list) % task_num
    sublists = []
    start = 0
    for i in range(task_num):
        if i == 0:
            end = start + n + remainder
        else:
            end = start + n
        sublists.append(sequence_list[start:end])
        start = end
    return sublists, n

def get_dataloader(data_args, training_args):
    data_args.task_num = int(data_args.dataset_name.split('-')[-1].replace('T', ''))
    if data_args.dataset_name.startswith('CIFAR10-'):
        dataset = load_dataset('cifar10')
        dataset = dataset.rename_column('img', 'images')
        dataset = dataset.rename_column('label', 'labels')
        data_args.tot_class_num = 10
    elif data_args.dataset_name.startswith('Mnist-'):
        dataset = load_dataset('mnist')
        dataset = dataset.rename_column('image', 'images')
        dataset = dataset.rename_column('label', 'labels')
        data_args.tot_class_num = 10
    elif data_args.dataset_name.startswith('FashionMnist-'):
        dataset = load_dataset('fashion_mnist')
        dataset = dataset.rename_column('image', 'images')
        dataset = dataset.rename_column('label', 'labels')
        data_args.tot_class_num = 10
    elif data_args.dataset_name.startswith('Flowers-'):
        dataset = load_dataset('../datasets/oxford-flowers')
        dataset = dataset.rename_column('image', 'images')
        dataset = dataset.rename_column('label', 'labels')
        data_args.tot_class_num = 100
    elif data_args.dataset_name.startswith('CUB-Birds-'):
        dataset = load_dataset('cub')
        dataset = dataset.rename_column('image', 'images')
        dataset = dataset.rename_column('label', 'labels')
        data_args.tot_class_num = 200
    elif data_args.dataset_name.startswith('Stanford-Cars-'):
        dataset = load_dataset('stanford_cars')
        dataset = dataset.rename_column('image', 'images')
        dataset = dataset.rename_column('label', 'labels')
        data_args.tot_class_num = 196
    elif data_args.dataset_name.startswith('ImageNet-'):
        dataset = load_from_disk('../datasets/imagenet-1k-64')
        dataset = dataset.rename_column('X_train', 'images')
        dataset = dataset.rename_column('Y_train', 'labels')
        dataset['test'] = dataset['validation']
        data_args.tot_class_num = 1000
    elif data_args.dataset_name.startswith('Custom-Objects-'):
        dataset = load_from_disk('../datasets/custom_objects')
        dataset = dataset.rename_column('img', 'images')
        dataset = dataset.rename_column('lbl', 'labels')
        data_args.tot_class_num = 150
    else:
        raise NotImplementedError
    data_args.sequence = get_sequence_map(data_args.dataset_name, shuffle_order=data_args.shuffle_order)
    data_args.all_task_labels, data_args.class_num = divide_list(data_args.sequence, data_args.task_num)
    data_args.task_labels = data_args.all_task_labels[data_args.task_id]

    def get_preprocess(data_args):
        if 'ImageNet-' in data_args.dataset_name:
            return transforms.Compose(
                [
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize([data_args.image_size, data_args.image_size]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

    def transform(examples):
        preprocess = get_preprocess(data_args)
        images = [preprocess(torch.tensor(image) if 'ImageNet-' in data_args.dataset_name else image.convert('RGB')) for image in examples["images"]]
        return {"images": images, 'labels': examples['labels']}

    def filter(dataset, labels):
        filtered_dataset = deepcopy(dataset)
        train_indices = [index for index, label in enumerate(dataset['train']['labels']) if label in labels]
        filtered_dataset['train'] = dataset['train'].select(train_indices)
        test_indices = [index for index, label in enumerate(dataset['test']['labels']) if label in labels]
        filtered_dataset['test'] = dataset['test'].select(test_indices)
        return filtered_dataset

    # continual learning setting
    if data_args.noncl:
        data_args.task_labels = list(itertools.chain.from_iterable(data_args.all_task_labels))
        task_dataset = filter(dataset, data_args.task_labels)
        task_dataset['train'] = task_dataset['train'].map(lambda x: {'labels': data_args.sequence.index(x)}, input_columns='labels')
        task_dataset['test'] = task_dataset['test'].map(lambda x: {'labels': data_args.sequence.index(x)}, input_columns='labels')
        task_dataset.set_transform(transform)
        dataloader_dict = {
            'train_loader': DataLoader(task_dataset['train'], batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=8),
            'test_loader': DataLoader(task_dataset['test'], batch_size=training_args.per_device_eval_batch_size, shuffle=False, num_workers=8)
        }
    else:
        task_dataset = filter(dataset, data_args.task_labels)
        task_dataset['train'] = task_dataset['train'].map(lambda x: {'labels': data_args.sequence.index(x)}, input_columns='labels')
        task_dataset['test'] = task_dataset['test'].map(lambda x: {'labels': data_args.sequence.index(x)}, input_columns='labels')
        task_dataset.set_transform(transform)
        dataset_seq = [filter(dataset, task_labels) for task_labels in data_args.all_task_labels[:(data_args.task_id + 1)]]
        for task_id in range(data_args.task_id + 1):
            dataset_seq[task_id]['train'] = dataset_seq[task_id]['train'].map(lambda x: {'labels': data_args.sequence.index(x)}, input_columns='labels')
            dataset_seq[task_id]['test'] = dataset_seq[task_id]['test'].map(lambda x: {'labels': data_args.sequence.index(x)}, input_columns='labels')
        for ds in dataset_seq:
            ds.set_transform(transform)
        dataloader_dict = {
            'train_loader': DataLoader(task_dataset['train'], batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=8),
            'test_loader': DataLoader(task_dataset['test'], batch_size=training_args.per_device_eval_batch_size, shuffle=False, num_workers=8),
            'all_test_loader': [DataLoader(dataset_seq[task_id]['test'], batch_size=training_args.per_device_eval_batch_size, shuffle=False, num_workers=8) for task_id in range(data_args.task_id + 1)]
        }
    return dataloader_dict