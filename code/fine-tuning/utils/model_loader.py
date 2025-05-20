import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration, T5Tokenizer,
    RobertaTokenizer
)

def load_model_and_tokenizer(args):
    name = args.model_name.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode =="trian":
        if "codereviewer" == name:
            model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/codereviewer")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")

        elif "codet5" in name:
            if "codet5p" in name:
                model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m")
                tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")
            else:
                model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
                tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

        elif "auger" == name:
            base = "SEBS/code_trans_t5_base_code_documentation_generation_java_multitask"
            ckpt = "./models/AUGER_pre-trained/best_ppl_pretraining_pytorch.bin"
            model = T5ForConditionalGeneration.from_pretrained(base)
            model.resize_token_embeddings(32101)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state, strict=False)
            tokenizer = T5Tokenizer.from_pretrained(base)

        elif "tufanoT5" == name:
            model = T5ForConditionalGeneration.from_pretrained("./models/TufanoT5_pre-trained/pytorch_version")
            tokenizer = T5Tokenizer.from_pretrained("./models/TufanoT5_tokenizer/tokenizer/TokenizerModel.model")

        else:
            raise ValueError(f"Unsupported model name: {args.model_name}")
    else:
        if "codereviewer" == name:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.ckp_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.ckp_dir)

        elif "codet5" in name:
            if "codet5p" in name:
                model = T5ForConditionalGeneration.from_pretrained(args.ckp_dir)
                tokenizer = AutoTokenizer.from_pretrained(args.ckp_dir)
            else:
                model = T5ForConditionalGeneration.from_pretrained(args.ckp_dir)
                tokenizer = RobertaTokenizer.from_pretrained(args.ckp_dir)

        elif "AUGER" == name:
            model = T5ForConditionalGeneration.from_pretrained(args.ckp_dir)
            tokenizer = T5Tokenizer.from_pretrained(args.ckp_dir)

        elif "tufanoT5" == name:
            model = T5ForConditionalGeneration.from_pretrained(args.ckp_dir)
            tokenizer = T5Tokenizer.from_pretrained(args.ckp_dir)

        else:
            raise ValueError(f"Unsupported model name: {args.model_name}")

    model.to(device)
    print(f"[INFO] Loaded model on {device}")
    return model, tokenizer, device
