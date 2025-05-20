#!/bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12355
NUM_GPUS=1

python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  run.py \
  --mode train \
  --model_name codereviewer \
  --train_data ./dataset/train.tsv \
  --val_data ../../dataset/val.tsv \
  --test_data ../../dataset/test.tsv \
  --rag_dir ../retrieval/rag_candidate \
  --output_dir ./output/checkpoints \
  --ckp_dir ./output/checkpoints \
  --prompt_mode pair_rag \
  --top_k 30 \
  --batch_size 8 \
  --beam_size 10 \
  --lr 3e-5 \
  --grad_accum 8 \
  --patience 3 \
  --num_epochs 20
