#!/bin/bash

srun --mem=20G -c 10 --gres=gpu:2 -p t4 \
    sh scripts/pretrain_bert_distributed_phase1.sh \
        --partition=t4 \
        --batch_size=32 \
        --gradient_accumulation=2 \
        --nnodes=2 \
        --master_dir=master_ip \
        --master_output=t4_2_nodes_128_seq_len_32_batch_size.ip \
        --ngpu_per_node=2 \
        --node_rank=1