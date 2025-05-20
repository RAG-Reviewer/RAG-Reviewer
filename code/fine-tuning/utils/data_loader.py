import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer

class ReviewDataset(Dataset):
    def __init__(self, codes, comments, rag_codes, rag_comments, tokenizer: PreTrainedTokenizer, prompt_mode, top_k):
        self.codes = codes
        self.comments = comments
        self.rag_codes = rag_codes
        self.rag_comments = rag_comments
        self.tokenizer = tokenizer
        self.prompt_mode = prompt_mode
        self.top_k = top_k

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        prompt = self.codes[idx]
        if self.prompt_mode == "singleton_rag":
            for c in self.rag_comments[idx][:self.top_k]:
                prompt += " [SEP] " + c
        elif self.prompt_mode == "pair_rag":
            for j in range(self.top_k):
                prompt += f" [SEP] {self.rag_comments[idx][j]} [CSEP] {self.rag_codes[idx][j]}"

        enc = self.tokenizer(prompt, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        tgt = self.tokenizer(self.comments[idx], truncation=True, max_length=128, padding="max_length", return_tensors="pt")

        input_ids = enc.input_ids.squeeze(0)
        attn = enc.attention_mask.squeeze(0)
        labels = tgt.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def read_tsv(path):
    codes, comments = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            code, comment = line.strip().split("\t", 1)
            codes.append(code)
            comments.append(comment)
    return codes, comments

def get_retrievals(rag_dir, split, total_topk=30):
    code_file = os.path.join(rag_dir, f"{split}_to_train_retrieval_top{total_topk}_code.txt")
    comment_file = os.path.join(rag_dir, f"{split}_to_train_retrieval_top{total_topk}_comment.txt")
    codes = open(code_file, encoding="utf-8").read().splitlines()
    comments = open(comment_file, encoding="utf-8").read().splitlines()
    n = len(codes) // total_topk
    return [codes[i*total_topk:(i+1)*total_topk] for i in range(n)], [comments[i*total_topk:(i+1)*total_topk] for i in range(n)]

def get_dataloaders(args, mode="train"):
    if mode == "train":
        train_codes, train_comments = read_tsv(args.train_data)
        val_codes, val_comments = read_tsv(args.val_data)

        if args.prompt_mode != "vanilla":
            rag_train_c, rag_train_q = get_retrievals(args.rag_dir, "train")
            rag_val_c, rag_val_q = get_retrievals(args.rag_dir, "val")
        else:
            rag_train_c = rag_train_q = rag_val_c = rag_val_q = None

        train_ds = ReviewDataset(train_codes, train_comments, rag_train_c, rag_train_q, args.tokenizer, args.prompt_mode, args.top_k)
        val_ds = ReviewDataset(val_codes, val_comments, rag_val_c, rag_val_q, args.tokenizer, args.prompt_mode, args.top_k)

        train_sampler = DistributedSampler(train_ds) if torch.distributed.is_initialized() else None
        val_sampler = DistributedSampler(val_ds) if torch.distributed.is_initialized() else None

        return (
            DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None)),
            DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler)
        )
    else:
        test_codes, test_comments = read_tsv(args.test_data)
        rag_test_c, rag_test_q = get_retrievals(args.rag_dir, "test") if args.prompt_mode != "vanilla" else (None, None)
        test_ds = ReviewDataset(test_codes, test_comments, rag_test_c, rag_test_q, args.tokenizer, args.prompt_mode, args.top_k)
        return DataLoader(test_ds, batch_size=args.batch_size)
