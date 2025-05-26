# RAG-Reviewer Fine-tuning & Evaluation

This directory contains the training and evaluation scripts for **RAG-Reviewer**, a retrieval-augmented generation framework for code review comment generation.

---

## Supported Pre-trained Models

You can fine-tune or evaluate RAG-Reviewer with any of the following pre-trained models:

- `codereviewer` – [CodeReviewer (FSE 2022)](https://arxiv.org/abs/2203.09095)
- `codet5` – [CodeT5 (EMNLP 2021)](https://arxiv.org/abs/2109.00859)
- `codet5p` – [CodeT5+ (arXiv:2305.07922, 2023)](https://arxiv.org/abs/2305.07922)
- `auger` – [AUGER (FSE 2022)](https://dl.acm.org/doi/abs/10.1145/3540250.3549099)
- `tufanoT5` – [Tufano et al. (ICSE 2022)](https://dl.acm.org/doi/abs/10.1145/3510003.3510621)

---

## Setup for Manually Download Checkpoints

### AUGER

1. Download `best_ppl_pretraining_pytorch.bin` from the [AUGER replication package](https://gitlab.com/ai-for-se-public-data/auger-fse-2022/-/tree/main/AUGER/model?ref_type=heads).
2. Place it in:
   ```
   ./pre-trained_checkpoints/AUGER_pre-trained/
   ```

### TufanoT5

1. Download `tokenizer.zip` from [Tufano's replication](https://zenodo.org/records/5387856#.YTDrPZ4zZyo) and extract to:
   ```
   ./pre-trained_checkpoints/TufanoT5_tokenizer/
   ```

2. Download `models.zip` and extract `T5_pre-trained/` into:
   ```
   ./pre-trained_checkpoints/TufanoT5_pre-trained/tf_version/
   ```

3. Convert TensorFlow checkpoint to PyTorch format:
   ```bash
   python tf_2_pytorch_tufanoT5.py
   ```

---

## Fine-tuning & Evaluation Instructions

### Fine-tuning

Refer to the notebook: `finetuning.ipynb`

> **Note**: For `auger` and `tufanoT5`, you must prepare the pre-trained checkpoints manually before running fine-tuning (see previous section).

### Evaluation

Refer to the notebook: `evaluate.ipynb`

The checkpoints used in our paper’s experiments are publicly available on [here](https://figshare.com/articles/dataset/Replication_package_for_RAG-Reviewer_A_Retrieval-Augmented_Generation_Framework_for_Automated_Code_Review_Comment_Generation_/29147681).

Download `fine_tuned_checkpoints.zip` from extract it into:
>
> ```
> ./output/fine_tuned_checkpoints/
> ```

Each checkpoint folder follows the naming convention:

```
modelName_ragStrategy_finetuned_best_ckp_epoch
```

Where:
- `modelName` specifies the backbone model (e.g., CodeT5, CodeReviewer, etc.).
- `ragStrategy` indicates the prompting strategy: `rag_pair`, `rag_singleton`, or `vanilla`.
- `epoch` refers to the best-performing epoch (based on validation performance).
---
