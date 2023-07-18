import numpy as np
from rdkit.Chem import DataStructs, AllChem
import pandas as pd
import torch
from torch.utils.data import Dataset

def fingerprint_array_df(fp_list: list):
    finp = []
    # convert it to a list of bits
    for m in fp_list:
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(m, arr)
        finp.append(arr)


    all_fin = pd.DataFrame(finp)
    return all_fin

def mse_loss_with_nans(input, target, ignore_value=None):

    # Missing data are nan's
    if ignore_value:
        # mask the value
        mask = target == ignore_value
    else:
        mask = torch.isnan(target)

    out = (input[~mask]-target[~mask])**2
    loss = out.mean()

    return loss

def generate_fp_array(mol):
    # generate the fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
    arr1 = np.zeros((len(fp),))
    DataStructs.ConvertToNumpyArray(fp, arr1)
    arr1 = arr1.astype(int)
    fingerprint = arr1

    return fingerprint

class MorganDataset(Dataset):
    def __init__(self, fps):
        self.fps = fps

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        input = self.fps[idx]

        return input
