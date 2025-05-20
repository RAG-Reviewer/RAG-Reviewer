# RAG-Reviewer: Retrieval-Augmented Code Review Comment Generation

RAG-Reviewer is a framework that improves automated code review comment generation by leveraging **retrieval-augmented generation (RAG)**. It combines retrieved exemplars with pre-trained language models to generate high-quality review feedback for source code.

---

## Instructions

Follow the steps below to reproduce our results:

---

### 1. Install Dependencies

Ensure Python ≥ 3.10.6 is installed. Then install required packages via:

```bash
pip install -r requirements.txt
```

---

### 2. Download Dataset

Download the [Tufano et al. code-to-comment dataset](https://zenodo.org/records/5387856#.YTDrPZ4zZyo).  
From the downloaded `dataset.zip`, extract the following files from the `fine-tuning/new_large/code-to-comment/` directory:

- `train.tsv`
- `val.tsv`
- `test.tsv`

Place them in the `./dataset/` directory:

```
./dataset/train.tsv
./dataset/val.tsv
./dataset/test.tsv
```

---

### 3.Run RAG-Reviewer

The process is divided into two major steps:

#### (1) Build Retrieval Database

Construct the retrieval database (based on UniXCoder embeddings) that will be used for RAG input.  
Refer to the instructions in [`./code/retrieval/README.md`](./code/retrieval/README.md)

#### (2) Fine-Tune or Evaluate RAG-Reviewer

Run training or evaluation using the pre-built retrieval database.  
See the guide in [`./code/fine-tuning/README.md`](./code/fine-tuning/README.md)

---

## 📁 Repository Structure

```
RAG-Reviewer/
├─ code/
│  ├─ fine-tuning/
│  │  ├─ models/
│  │  │  ├─ AUGER_pre-trained/
│  │  │  ├─ TufanoT5_pre-trained/
│  │  │  │  ├─ pytorch_version/
│  │  │  │  └─ tf_version/
│  │  │  └─ TufanoT5_tokenizer/
│  │  ├─ output/
│  │  │  ├─ checkpoints/     # Saved fine-tuned models
│  │  │  └─ inference/       # Output predictions
│  │  └─ utils/              # Modular training utilities
│  └─ retrieval/
│     ├─ code_embeddings/    # Vector database (UniXCoder)
│     └─ rag_candidate/      # Retrieved top-K exemplars
├─ dataset/                  # Contains train/val/test .tsv
├─ manual analysis.xlsx      # Manual evaluation result
├─ requirements.txt          # Python dependencies
```

