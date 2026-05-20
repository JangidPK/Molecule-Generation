import platform
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_config_summary(config, tokenizer):
    """
    Prints a summary of the model configuration, 
    training settings, and system information.
    """

    arch = config["model"]["architecture"]

    vocab_size = len(tokenizer.token_to_idx) # this will depend on the tokenizer

    common_cfg = config["model"]["common"]

    model_cfg = config["model"][arch]


    print("\n" + "=" * 60)

    print("MOLECULAR GENERATOR CONFIGURATION SUMMARY")
    print("Author: Pankaj")

    print("=" * 60)


    print("\nMODEL CONFIGURATION")
    print(f"Architecture      : {arch.upper()}")
    print(f"Vocabulary size   : {vocab_size}")
    print(f"Block size        : {common_cfg['block_size']}")


    if arch == "lstm":
        print(f"Embedding dim     : {model_cfg['embedding_dim']}")
        print(f"Hidden dim        : {model_cfg['hidden_dim']}")
        print(f"Layers            : {model_cfg['num_layers']}")
        print(f"Dropout           : {model_cfg['dropout']}")


    elif arch == "transformer":
        print(f"Embedding dim     : {model_cfg['embedding_dim']}")
        print(f"Heads             : {model_cfg['n_head']}")
        print(f"Layers            : {model_cfg['n_layer']}")
        print(f"Dropout           : {model_cfg['dropout']}")


    print("\nTRAINING CONFIGURATION")
    print(f"Batch size        : {config['training']['batch_size']}")
    print(f"Learning rate     : {config['training']['learning_rate']}")
    print(f"Epochs            : {config['training']['epochs']}")
    print(f"Train size        : {config['training']['train_size']}")
    print(f"Dataset size      : {config['training']['dataset_size']}")


    print("\nSYSTEM CONFIGURATION")
    print(f"Device            : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version   : {torch.__version__}")
    print(f"Python version    : {platform.python_version()}")


    print("\nTOKENIZER SETTINGS")
    print("Special tokens    : <PAD>, <START>, <END>, <UNK>")

    print("=" * 60 + "\n")
