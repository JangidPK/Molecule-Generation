import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, head_size, embedding_dim, block_size, dropout):

        super().__init__()

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_head,
        embedding_dim,
        block_size,
        dropout
    ):

        super().__init__()

        head_size = embedding_dim // n_head

        self.heads = nn.ModuleList(

            [
                Head(
                    head_size,
                    embedding_dim,
                    block_size,
                    dropout
                )
                for _ in range(n_head)
            ]
        )

        self.proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        x = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        x = self.proj(x)

        return self.dropout(x)


class FeedForward(nn.Module):

    def __init__(self, embedding_dim, dropout):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                embedding_dim,
                4 * embedding_dim
            ),

            nn.ReLU(),

            nn.Linear(
                4 * embedding_dim,
                embedding_dim
            ),

            nn.Dropout(dropout)
        )


    def forward(self, x):

        return self.net(x)


class Block(nn.Module):

    def __init__(
        self,
        embedding_dim,
        n_head,
        block_size,
        dropout
    ):

        super().__init__()

        self.sa = MultiHeadAttention(
            n_head,
            embedding_dim,
            block_size,
            dropout
        )

        self.ffwd = FeedForward(
            embedding_dim,
            dropout
        )

        self.ln1 = nn.LayerNorm(
            embedding_dim
        )

        self.ln2 = nn.LayerNorm(
            embedding_dim
        )


    def forward(self, x):

        x = x + self.sa(self.ln1(x))

        x = x + self.ffwd(self.ln2(x))

        return x


class Transformer(nn.Module):

    def __init__(self, config):

        super().__init__()

        model_cfg = config["model"]["transformer"]

        common_cfg = config["model"]["common"]


        self.block_size = common_cfg["block_size"]

        self.embedding_dim = model_cfg["embedding_dim"]

        self.vocab_size = config["model"]["vocab_size"]

        self.n_head = model_cfg["n_head"]

        self.n_layer = model_cfg["n_layer"]

        dropout = model_cfg["dropout"]


        self.token_embedding_table = nn.Embedding(
            self.vocab_size,
            self.embedding_dim
        )


        self.position_embedding_table = nn.Embedding(
            self.block_size,
            self.embedding_dim
        )


        self.blocks = nn.Sequential(

            *[
                Block(
                    self.embedding_dim,
                    self.n_head,
                    self.block_size,
                    dropout
                )

                for _ in range(self.n_layer)
            ]
        )


        self.ln_f = nn.LayerNorm(
            self.embedding_dim
        )


        self.lm_head = nn.Linear(
            self.embedding_dim,
            self.vocab_size
        )


    def forward(self, idx, targets=None):

        B, T = idx.shape


        tok_emb = self.token_embedding_table(idx)


        pos_emb = self.position_embedding_table(
            torch.arange(
                T,
                device=idx.device
            )
        )


        x = tok_emb + pos_emb


        x = self.blocks(x)


        x = self.ln_f(x)


        logits = self.lm_head(x)


        loss = None


        if targets is not None:

            logits = logits.view(
                B * T,
                self.vocab_size
            )

        return logits


    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

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
    