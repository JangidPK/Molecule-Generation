import argparse
import pandas as pd
import torch
import yaml
from molecule_generator.reinforcement_learning.optimize_properties import rl_optimize
from molecule_generator.preprocessing.scrape_data import PubChemSmilesScraper
from molecule_generator.preprocessing.tokenizer import SmilesTokenizer
from molecule_generator.generate.sampling import Sampler
from molecule_generator.training.trainer import Trainer
from molecule_generator.utils.paths import RAW_SMILES, ARTF_DIR, CONFIG_FILE


def scrape():

    pcs = PubChemSmilesScraper(
        chunk_size=500,
        num_chunks=10
    )

    pcs.write_data(output_file=RAW_SMILES)


def preprocess():

    smiles_data = pd.read_csv(
        RAW_SMILES
    ).values.squeeze()

    tokenizer = SmilesTokenizer()

    filtered = tokenizer.filter_smiles(smiles_data)

    tokenizer.fit(filtered)

    tokenizer.save(
        ARTF_DIR / "vocab.pkl"
    )

    encoded = tokenizer.encode_batch(filtered)

    torch.save(
        encoded,
        ARTF_DIR / "encoded_dataset.pt"
    )

    print("Preprocessing complete.")


def train():

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    arch = config["model"]["architecture"]
    print(f"Training {arch} model...")


    tokenizer = SmilesTokenizer()

    tokenizer.load(
        ARTF_DIR / "vocab.pkl"
    )

    encoded_sequences = torch.load(
        ARTF_DIR / "encoded_dataset.pt"
    )

    trainer = Trainer(
        encoded_sequences,
        tokenizer,
        config
    )

    trainer.train(
        epochs=config["training"]["epochs"]
    )


    model_path = ARTF_DIR / f"{arch}_model.pt"
    trainer.save(model_path)
    print(f"Model saved to {model_path}")

def optimize():

    with open(CONFIG_FILE) as f:

        config = yaml.safe_load(f)
    arch = config["model"]["architecture"]
    print(f"Optimizing {arch} model with reinforcement learning...")


    tokenizer = SmilesTokenizer()

    tokenizer.load(
        ARTF_DIR / "vocab.pkl"
    )


    encoded_sequences = torch.load(
        ARTF_DIR / "encoded_dataset.pt"
    )


    trainer = Trainer(
        encoded_sequences,
        tokenizer,
        config
    )

    model_path = ARTF_DIR / f"{arch}_model.pt"
    trainer.model.load_state_dict(
        torch.load(model_path)
    )


    rl_optimize(
        trainer.model,
        trainer.optimizer,
        tokenizer,
        steps=config["rl"]["steps"]
    )

    model_path = ARTF_DIR / f"{arch}_model.pt"
    trainer.save(model_path)
    print(f"Model saved to {model_path}")


def sample():

    sampler = Sampler(CONFIG_FILE)

    molecules = sampler.sample_valid(20)

    img = sampler.visualize(molecules)

    img.show()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "stage",
        choices=[
            "scrape",
            "preprocess",
            "train",
            "optimize",
            "sample"],
    )

    args = parser.parse_args()


    if args.stage == "scrape":

        scrape()

    elif args.stage == "preprocess":

        preprocess()

    elif args.stage == "train":

        train()

    elif args.stage == "optimize":

        optimize()

    elif args.stage == "sample":

        sample()
