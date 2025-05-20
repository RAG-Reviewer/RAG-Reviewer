import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="RAG-Reviewer Unified Pipeline")

    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=False)
    parser.add_argument("--val_data", type=str, required=False)
    parser.add_argument("--test_data", type=str, required=False)
    parser.add_argument("--rag_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckp_dir", type=str, required=False)

    parser.add_argument("--prompt_mode", choices=["vanilla", "singleton_rag", "pair_rag"], required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=20)

    return parser.parse_args()
