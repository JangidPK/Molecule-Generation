import torch
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from molecule_generator.preprocessing.tokenizer import SmilesTokenizer
from molecule_generator.models.lstm_model import LSTM
from molecule_generator.models.transformer_model import Transformer
from molecule_generator.utils.paths import ARTF_DIR
from molecule_generator.utils.config_summary import print_config_summary

RDLogger.DisableLog("rdApp.*")


class Sampler:

    def __init__(self, config_path):

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.temperature = self.config["sampling"]["temperature"]
        self.tokenizer = SmilesTokenizer()

        self.tokenizer.load(
            ARTF_DIR /"vocab.pkl"
        )
        vocab_size = len(self.tokenizer.token_to_idx)
        self.config["model"]["vocab_size"] = vocab_size

        print("\nUsing device:", self.device)
        print_config_summary(self.config, self.tokenizer)

        arch = self.config["model"]["architecture"]
        
        if arch == "lstm":
            self.model = LSTM(self.config)
            self.model.load_state_dict(
            torch.load(
                ARTF_DIR /"lstm_model.pt",
                map_location=self.device
            )
        )
        elif arch == "transformer":
            self.model = Transformer(self.config)
            self.model.load_state_dict(
            torch.load(
                ARTF_DIR /"transformer_model.pt",
                map_location=self.device
            )
        )
            
        


        self.model.to(self.device)
        self.model.eval()


        self.start_token = self.tokenizer.token_to_idx["<START>"]
        self.end_token = self.tokenizer.token_to_idx["<END>"]


    def sample(self, max_length=128):

        sequence = torch.tensor(
            [[self.start_token]],
            device=self.device
        )


        with torch.no_grad():

            for _ in range(max_length):

                logits = self.model(sequence)

                probs = torch.softmax(
                    logits[:, -1, :]/self.temperature,
                    dim=-1
                )


                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )


                sequence = torch.cat(
                    (sequence, next_token),
                    dim=1
                )


                if next_token.item() == self.end_token:
                    break


        tokens = self.tokenizer.decode(
            sequence.squeeze().tolist()
        )


        smiles = self.tokenizer.detokenize(tokens)


        return smiles


    def sample_valid(self, n_samples=20):

        valid_mols = []


        for _ in range(n_samples):

            smiles = self.sample()


            mol = Chem.MolFromSmiles(smiles)


            if mol:

                valid_mols.append(mol)


        return valid_mols


    def visualize(self, mols):

        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(200, 200)
        )


        return img
