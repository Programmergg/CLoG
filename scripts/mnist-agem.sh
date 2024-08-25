#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

seed=42
eval_steps=500
weight_decay=0
method='agem'
memory_size=512
replay_batch_size=256
shuffle_order=False
lr_warmup_steps=500
learning_rate="2e-4"
num_train_epochs=200
dataset_name="Mnist-5T"
tot_samples_for_eval=2048
per_device_eval_batch_size=256
per_device_train_batch_size=256
num_tasks=$(echo $dataset_name | grep -o '[0-9]T' | grep -o '[0-9]')

# training and evaluation
for i in $(seq 0 $((num_tasks-1))); do
    python main.py \
        --model_arch ddim \
        --lr_warmup_steps $lr_warmup_steps \
        --weight_decay $weight_decay \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_eval_batch_size \
        --tot_samples_for_eval $tot_samples_for_eval \
        --num_train_epochs $num_train_epochs \
        --eval_steps $eval_steps \
        --learning_rate $learning_rate \
        --shuffle_order $shuffle_order \
        --dataset_name $dataset_name \
        --task_id $i \
        --seed $seed \
        --method $method \
        --memory_size $memory_size \
        --replay_batch_size $replay_batch_size
done