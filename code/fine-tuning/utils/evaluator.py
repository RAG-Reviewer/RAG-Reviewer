from tqdm import tqdm
import torch
import sacrebleu
import os

def evaluate_model(args, loader):
    model = args.model
    tokenizer = args.tokenizer
    device = args.device

    model.eval()
    all_refs, all_preds = []

    print("[INFO] Starting evaluation...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=args.beam_size
            )

            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_preds.extend([p.strip() for p in preds])

            for lbl in labels:
                ids = [i for i in lbl.tolist() if i != -100]
                ref = tokenizer.decode(ids, skip_special_tokens=True).strip()
                all_refs.append(ref)

    # Compute BLEU and PP
    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs]).score
    perfect_matches = sum(1 for p, r in zip(all_preds, all_refs) if p == r)
    pp = perfect_matches / len(all_refs)

    print(f"[RESULT] Perfect Prediction (PP): {pp * 100:.2f}%")
    print(f"[RESULT] BLEU-4 Score: {bleu:.2f}")

    # Write predictions to file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "predictions.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in all_preds:
            f.write(pred + "\n")

    print(f"[INFO] Predictions written to {output_file}")

    return bleu, pp
