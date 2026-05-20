# trainer.py

import torch
from torch.utils.data import DataLoader, random_split
from molecule_generator.models.lstm_model import LSTM
from molecule_generator.models.transformer_model import Transformer
from molecule_generator.preprocessing.dataset_builder import SmilesDataset
from molecule_generator.utils.config_summary import print_config_summary, count_parameters


class Trainer:

    def __init__(self, sequences, tokenizer, config):
        '''
        trains both LSTM and Transformer models based on the config settings
        '''

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        vocab_size = len(tokenizer.token_to_idx)
        print("\nUsing device:", self.device)

        arch = config["model"]["architecture"]
        model_cfg = config["model"][arch]
        common_cfg = config["model"]["common"]
        config["model"]["vocab_size"] = vocab_size
        config["model"]["block_size"] = common_cfg["block_size"]


        if arch == "lstm":
            self.model = LSTM(config)

        elif arch == "transformer":
            self.model = Transformer(config)

        else:
            raise ValueError(f"Unknown architecture: {arch}")

        self.model = self.model.to(self.device)
        print(f"Trainable params  : {count_parameters(self.model):,}")

        dataset = SmilesDataset(sequences)

        config["training"]["dataset_size"] = len(dataset)
        train_size = int(config["training"]["train_size"] * len(dataset))
        val_size = len(dataset) - train_size

        self.config = config
        self.tokenizer = tokenizer

        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size]
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=True
        )


        self.val_loader = DataLoader(
            val_ds,
            batch_size=config["training"]["batch_size"]
        )


        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=0
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"]
        )


        self.train_losses = []
        self.val_losses = []




    def train(self, epochs):

        print("\nTraining model...")

        print_config_summary(self.config, self.tokenizer)

        for epoch in range(epochs):

            self.model.train()

            train_loss = 0

            for X, Y in self.train_loader:

                X = X.to(self.device)
                Y = Y.to(self.device)


                output = self.model(X)


                loss = self.criterion(
                    output.reshape(-1, output.shape[-1]),
                    Y.reshape(-1)
                )


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                train_loss += loss.item()


            avg_train_loss = train_loss / len(self.train_loader)


            val_loss = self.evaluate()


            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)


            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )


    def evaluate(self):

        self.model.eval()

        total_loss = 0

        with torch.no_grad():

            for X, Y in self.val_loader:

                X = X.to(self.device)
                Y = Y.to(self.device)


                output = self.model(X)


                loss = self.criterion(
                    output.reshape(-1, output.shape[-1]),
                    Y.reshape(-1)
                )


                total_loss += loss.item()


        return total_loss / len(self.val_loader)


    def save(self, path):

        torch.save(self.model.state_dict(), path)
