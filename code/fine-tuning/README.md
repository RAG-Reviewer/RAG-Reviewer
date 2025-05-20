# RAG-Reviewer Fine-tuning &Evaluate

This repository provides an end-to-end pipeline for training and evaluating **RAG-Reviewer**, a retrieval-augmented generation framework for code review comment generation.

---

## Supported Pre-trained Models

You can experiment with any of the following 5 pre-trained models, each tailored for code review generation:

- `codereviewer` – Microsoft [CodeReviewer](https://huggingface.co/microsoft/codereviewer) 
- `codet5` – Salesforce [CodeT5](https://huggingface.co/Salesforce/codet5-base)
- `codet5p` – Salesforce [CodeT5+](https://huggingface.co/Salesforce/codet5p-220m)
- `auger` – [Li et al.](https://dl.acm.org/doi/abs/10.1145/3540250.3549099) (FSE 2022)
- `tufanoT5` – [Tufano et al.](https://dl.acm.org/doi/abs/10.1145/3510003.3510621) (ICSE 2022)

Specify the model with `--model_name` in your `run.sh`.

---

## Prompt Modes

Choose one of the following modes using `--prompt_mode`:

- `vanilla` – Only code is used as input (Generation-based method)
- `singleton_rag` – Code + top-K comments
- `pair_rag` – Code + top-K (code, comment) pairs

---

## Directory Conventions

- For **training**, `--output_dir` is the checkpoint save path.
- For **evaluation**, `--output_dir` is where predictions (`predictions.txt`) are written.

---

## Setup Instructions for Special Models

### AUGER

1. Download `best_ppl_pretraining_pytorch.bin` from the [AUGER replication package](https://gitlab.com/ai-for-se-public-data/auger-fse-2022/-/tree/main/AUGER/model?ref_type=heads).
2. Place it in:
   ```
   ./models/AUGER_pre-trained/
   ```

### TufanoT5

1. Download **tokenizer.zip** from the [TufanoT5 replication package](https://zenodo.org/records/5387856#.YTDrPZ4zZyo) and extract to:
   ```
   ./models/TufanoT5_tokenizer/
   ```
   This should contain:
   ```
   TokenizerModel
   TokenizerModel.model
   ```

2. Download **models.zip** from the [TufanoT5 replication package](https://zenodo.org/records/5387856#.YTDrPZ4zZyo) and extract `T5_pre-trained/` to:
   ```
   ./models/TufanoT5_pre-trained/
   ```

3. Convert Tufano's TensorFlow checkpoint to PyTorch by running:
   ```bash
   python tf_2_pytorch_tufanoT5.py
   ```

---

## Training Example (run.sh)

```bash
#!/bin/bash

MASTER_ADDR=localhost                # Address of master node
MASTER_PORT=12355                   # Port for communication
NUM_GPUS=2                          # Number of GPUs for DDP

python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS \     # Number of processes (1 per GPU)
  --master_addr=$MASTER_ADDR \     # Master node address
  --master_port=$MASTER_PORT \     # Master node port
  run.py \                         
  --mode train \                   # Mode: train
  --model_name codereviewer \      # Choose from: codereviewer, codet5, codet5p, auger, tufanoT5
  --train_data ../../dataset/train.tsv \  # Path to training data
  --val_data ../../dataset/val.tsv \      # Path to validation data
  --rag_dir ../retrieval/rag_candidate \  # Path to RAG retrieval data
  --output_dir ./output/checkpoints \     # Where to save checkpoints
  --prompt_mode pair_rag \         # Prompt type: pair_rag | singleton_rag | vanilla
  --top_k 8 \                      # Number of top-K exemplars
  --batch_size 8 \                 # Batch size per device
  --beam_size 10 \                 # Beam search width
  --lr 3e-5 \                      # Learning rate
  --grad_accum 4 \                 # Gradient accumulation steps
  --patience 3 \                   # Early stopping patience
  --num_epochs 20                   # Max number of epochs
```

---

## Evaluation Example (run.sh)

```bash
#!/bin/bash

MASTER_ADDR=localhost                # Address of master node
MASTER_PORT=12355                   # Port for communication
NUM_GPUS=2                          # Number of GPUs for DDP

python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS \     # Number of processes (1 per GPU)
  --master_addr=$MASTER_ADDR \     # Master node address
  --master_port=$MASTER_PORT \     # Master node port
  run.py \
  --mode evaluate \                # Mode: evaluate
  --model_name codereviewer \      # Choose from: codereviewer, codet5, codet5p, auger, tufanoT5
  --test_data ../../dataset/test.tsv \    # Path to test data
  --rag_dir ../retrieval/rag_candidate \  # Path to RAG retrieval data
  --output_dir ./output/inference \       # Where predictions will be written
  --ckp_dir ./output/checkpoints/best_model \ # Directory of trained checkpoint
  --prompt_mode pair_rag \         # Prompt type: pair_rag | singleton_rag | vanilla
  --top_k 8 \                      # Number of top-K exemplars
  --batch_size 8 \                 # Batch size per device
  --beam_size 10                    # Beam search width
```

