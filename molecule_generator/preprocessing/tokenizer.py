import re
import pickle
from collections import Counter

# all allowed elements 
# will be updated if necessary
ALLOWED_ELEMENTS = {
            "H", "C", "N", "O", "P", "S",
            "F", "Cl", "Br", "I",
            "B", "Si"
        }

class SmilesTokenizer:

    def __init__(self, max_length=128):

        self.max_length = max_length
        self.token_to_idx = None
        self.idx_to_token = None

        self.pattern = r"Cl|Br|\[[^\]]+\]|\.|\=|\#|\-|\+|\\\\|\/|\(|\)|[A-Za-z]|[0-9]"


    def contains_only_allowed_atoms(self, smiles):

        bracket_atoms = re.findall(r"\[([A-Z][a-z]?)", smiles)
        simple_atoms = re.findall(r"Cl|Br|[A-Z][a-z]?", smiles)

        atoms = set(bracket_atoms + simple_atoms)

        return atoms.issubset(ALLOWED_ELEMENTS)


    def contains_disallowed_bracket_atoms(self, smiles) -> bool :
        '''
        Checks whether the smiles string contains disallowed characters

        Returns True or False.
        '''

        bracket_tokens = re.findall(r"\[[^\]]+\]", smiles)

        for token in bracket_tokens:
            if token not in ALLOWED_ELEMENTS:
                return True

        return False
    

    def contains_charge(self, smiles):

        charged_atoms = re.findall(r"\[[^\]]*[\+\-][^\]]*\]", smiles)

        return len(charged_atoms) > 0


    def filter_smiles(self, smiles_list):
        
        '''
        Filters the given SMILES based on some conditions
        
        More conditions can be added or removed 
        '''

        filtered = [
            smi for smi in smiles_list
            if self.contains_only_allowed_atoms(smi)
            and not self.contains_charge(smi)
            and not self.contains_disallowed_bracket_atoms(smi)
            and "." not in smi
        ]

        return filtered
    



    def tokenize(self, smiles):

        tokens = re.findall(self.pattern, smiles)

        return ["<START>"] + tokens + ["<END>"]


    def fit(self, smiles_list):
        '''
        Createa a vocab and inverse vocab 
        '''

        smiles_list = self.filter_smiles(smiles_list)

        tokenized = [self.tokenize(s) for s in smiles_list]

        vocab = Counter()

        for seq in tokenized:
            vocab.update(seq)

        vocab_list = ["<PAD>", "<UNK>"] + sorted(vocab.keys())

        self.token_to_idx = {
            token: idx for idx, token in enumerate(vocab_list)
        }

        self.idx_to_token = {
            idx: token for token, idx in self.token_to_idx.items()
        }


    def encode(self, smiles):

        tokens = self.tokenize(smiles)

        encoded = [
            self.token_to_idx.get(token, self.token_to_idx["<UNK>"])
            for token in tokens
            ]
        
        # here i am slicing the sequence to max_length
        # however sequences with excedding lengths can be excluded

        encoded = encoded[:self.max_length]

        encoded += [
            self.token_to_idx["<PAD>"]
            ] * (self.max_length - len(encoded))

        return encoded

    def encode_batch(self, smiles_list):
        return [
            self.encode(smi)
            for smi in smiles_list
        ]


    def decode(self, indices):

        tokens = []

        for i in indices:

            token = self.idx_to_token.get(i, "<UNK>")

            if token == "<END>":
                break

            if token != "<PAD>":
                tokens.append(token)

        return tokens

    def decode_batch(self, indices_list):
        return [
            self.decode(indices)
            for indices in indices_list
        ]

    def detokenize(self, tokens):
        '''
        Remove the token characters
        '''

        return "".join(
            t for t in tokens
            if t not in {"<START>", "<END>"}
        )


    def save(self, filepath):

        with open(filepath, "wb") as f:

            pickle.dump({
                "token_to_idx": self.token_to_idx,
                "idx_to_token": self.idx_to_token
            }, f)


    def load(self, filepath):

        with open(filepath, "rb") as f:

            data = pickle.load(f)

        self.token_to_idx = data["token_to_idx"]
        self.idx_to_token = data["idx_to_token"]