from transformers import T5Config, T5ForConditionalGeneration


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path: str, config_file: str, pytorch_dump_path: str) -> None:
    config = T5Config.from_pretrained(config_file)
    print(f"Building PyTorch model from configuration: {config}")

    model = T5ForConditionalGeneration.from_pretrained(
        tf_checkpoint_path, from_tf=True, config=config
    )

    # Save PyTorch model
    print(f"Saving PyTorch model to: {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)


def main():
    convert_tf_checkpoint_to_pytorch(
        tf_checkpoint_path="./models/TufanoT5_pre-trained/tf_version",
        config_file="./models/TufanoT5_pre-trained/config.json",
        pytorch_dump_path="./models/TufanoT5_pre-trained/pytorch_version",
    )


if __name__ == "__main__":
    main()
