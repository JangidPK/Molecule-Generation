from rdkit import Chem
from rdkit.Chem import QED, Descriptors


def compute_reward(smiles):

    '''
    Criteria for reward:
    1. Validity: Check if the generated SMILES string corresponds to a valid molecule.
    2. QED Score: Calculate the Quantitative Estimate of Drug-likeness (QED) for the molecule. 
        Higher QED scores indicate better drug-like properties.
    3. LogP: Calculate the logarithm of the partition coefficient (LogP) to assess the molecule's hydrophobicity. 
    A reward can be designed to favor molecules with LogP values within a certain
    '''

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return -1.0

    qed = QED.qed(mol)

    logp = Descriptors.MolLogP(mol)

    reward = qed - 0.1 * abs(logp - 2)
    

    return reward