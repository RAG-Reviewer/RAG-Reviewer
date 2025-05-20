from utils.args import parse_args
from utils.model_loader import load_model_and_tokenizer
from utils.data_loader import get_dataloaders
from utils.trainer import setup_ddp, train_model
from utils.evaluator import evaluate_model

def main():
    args = parse_args()
    setup_ddp(args)
    model, tokenizer, device = load_model_and_tokenizer(args)
    args.model, args.tokenizer, args.device = model, tokenizer, device

    if args.mode == "train":
        train_loader, val_loader = get_dataloaders(args, mode="train")
        train_model(args, train_loader, val_loader)
    else:
        test_loader = get_dataloaders(args, mode="evaluate")
        evaluate_model(args, test_loader)

if __name__ == "__main__":
    main()
