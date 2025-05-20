import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.evaluator import evaluate_model

def setup_ddp(args):
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        print(f"[INFO] Initialized DDP on rank {args.local_rank} / {dist.get_world_size()}")
    else:
        args.local_rank = 0
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[INFO] Running in non-DDP mode")

def train_model(args, train_loader, val_loader):
    model = args.model.to(args.device)
    tokenizer = args.tokenizer

    if dist.is_initialized():
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    name = args.model_name.lower()
    if "tufano" in name or "auger" in name:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr or 3e-4)
        scheduler = None
        grad_accum = args.grad_accum
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr or 3e-5, weight_decay=0.01)
        total_steps = (len(train_loader) // args.grad_accum) * args.num_epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        grad_accum = args.grad_accum

    scaler = GradScaler()
    best_pp = 0.0
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / grad_accum

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        if args.local_rank == 0 or not dist.is_initialized():
            val_bleu, val_pp = evaluate_model(args, val_loader)
            print(f"[Validation] BLEU: {val_bleu:.2f}, PP: {val_pp*100:.2f}%")

            if val_pp > best_pp:
                best_pp = val_pp
                patience_counter = 0
                ckpt_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(ckpt_path, exist_ok=True)
                model.save_pretrained(ckpt_path) if dist.is_initialized() else model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print("New best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"No improvement for {args.patience} epochs. Early stopping.")
                    break
