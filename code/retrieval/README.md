# Dataset Preprocessing and Retrieval Candidate Generation

This directory contains the code for building the **retrieval database** used in the **RAG-Reviewer** pipeline. It includes code for encoding source code and generating retrieval-based exemplars, which are essential for later fine-tuning or inference using **Retrieval-Augmented Generation (RAG)**.

---

## Overview

The process consists of two main stages:

---

### 1. Code Embedding with `code_encoder_module.py`

This Python script generates vector embeddings for source code in the Tufano et al. dataset (train/val/test splits).

- **Model**: [UniXcoder-base-nine](https://huggingface.co/microsoft/unixcoder-base-nine)
- **Function**: Encodes each code snippet into a dense vector representation using `AutoTokenizer` and `AutoModel` from Hugging Face Transformers.
- **Execution**: Run the script with:

```bash
python code_encoder_module.py
```

- **Output Format**: Embeddings are stored in `.pkl` files for downstream retrieval.

#### Outputs:
- `train_code_embeddings.pkl`
- `val_code_embeddings.pkl`
- `test_code_embeddings.pkl`

These files are saved under `./code_embeddings/`.

You can download `code_embeddings.zip` from [here](https://figshare.com/articles/dataset/Replication_package_for_RAG-Reviewer_A_Retrieval-Augmented_Generation_Framework_for_Automated_Code_Review_Comment_Generation_/29147681).

---

### 2. Top-K Retrieval with `generate_rag_candidates.py`

This Python code retrieves the top-K most similar code snippets from the **training set** for each item in the **train**, **val**, and **test** datasets.

- **Similarity Metric**: Cosine similarity computed over code embeddings.

For training samples, self-retrieval (i.e., retrieving the ground-truth) is excluded to prevent data leakage in the RAG setup.

#### Outputs:
- `rag_candidate/train_to_train_retrieval_top30_code.txt`
- `rag_candidate/train_to_train_retrieval_top30_comment.txt`
- `rag_candidate/val_to_train_retrieval_top30_code.txt`
- `rag_candidate/val_to_train_retrieval_top30_comment.txt`
- `rag_candidate/test_to_train_retrieval_top30_code.txt`
- `rag_candidate/test_to_train_retrieval_top30_comment.txt`

You can download `rag_candidate.zip` from [here](https://figshare.com/articles/dataset/Replication_package_for_RAG-Reviewer_A_Retrieval-Augmented_Generation_Framework_for_Automated_Code_Review_Comment_Generation_/29147681).