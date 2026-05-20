import torch
import torch.nn as nn


class LSTM(torch.nn.Module):

    def __init__(self, config):

        super().__init__()

        model_cfg = config["model"]["lstm"]
        common_cfg = config["model"]["common"]
        self.vocab_size = config["model"]["vocab_size"]
        self.embedding_dim = model_cfg["embedding_dim"]
        self.hidden_dim = model_cfg["hidden_dim"]
        self.num_layers = model_cfg["num_layers"]
        self.dropout = model_cfg["dropout"]
        self.block_size = common_cfg["block_size"]
        
    
        self.block_size = common_cfg["block_size"]

        self.temperature = config["sampling"]["temperature"]

        self.embedding = torch.nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=0
        )


        self.lstm = torch.nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            self.num_layers,
            batch_first=True
        )


        self.dropout_layer = torch.nn.Dropout(
            self.dropout
        )


        self.fc = torch.nn.Linear(
            self.hidden_dim,
            self.vocab_size
        )


    def forward(self, x):

        x = self.embedding(x)

        x, _ = self.lstm(x)

        x = self.dropout_layer(x)

        x = self.fc(x)

        return x


    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)

            logits = logits[:, -1, :]/self.temperature

            probs = torch.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def compute_log_probs(self, sequence):

        logits = self(sequence[:, :-1])

        targets = sequence[:, 1:]

        log_probs = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none"
        )

        return -log_probs
    