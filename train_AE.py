import pandas as pd
import numpy as np
from AE_utils import generate_fp_array, MorganDataset, fingerprint_array_df
from AE_models import LBAE, AE
import torch
import joblib
import rdkit.Chem as Chem
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

    
def jaccard_distance(x, y):
    """Exact same thing as tanimoto similarity.
    Args:
        x (List): A bit vector in list format.
        y (List): A bit vector in list format to be compared against x.

    Returns:
        _type_: a numpy float in the domain of (0,1) that represents the tanimoto similarity.
    """
    x = np.asarray(x, bool) # Not necessary, if you keep your data
    y = np.asarray(y, bool) # in a boolean array already!

    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())




if __name__ == "__main__":

    # the pre-calculated 1024 morgan fingerprints of all structures found in BindingDB. Around 2.4 million structures.
    train_filename = '/Users/docmartin/Downloads/BindingDB_SMILES_clean_morgan_fps.joblib'
    encoded_fps = joblib.load(train_filename)
    save_model_name = '/Users/docmartin/Downloads/LBAE_model_encode_128.joblib'

    # the acetylcholinesterase dataset to be used as a test set for the LBAE
    test_filename = '/Users/docmartin/Downloads/Acetylcholinesterase_human_IC50_ChEMBLBindingDB_spliton6_binary_1micromolar.csv'
    df = pd.read_csv(test_filename)
    encoded_fps_test = np.array([generate_fp_array(Chem.MolFromSmiles(x)) for x in df['SMILES']])

    # parameters used for the model.
    batch_size = 16
    epochs = 5
    learn_r = 1e-4

    # create the dataloaders to be used by the model
    dataset = MorganDataset(torch.FloatTensor(encoded_fps))
    test_dataset = MorganDataset(torch.FloatTensor(encoded_fps_test))

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    
    # set up the model
    model = LBAE()

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_r)

    outputs = []
    losses = []

    # for saving and inspecting
    real_fps = []
    latent_fps = []
    reconstructed_fps = []
    rounded_latent_fps = []
    rounded_reconstructed_fps = []

    # training loop
    for i, epoch in enumerate(range(epochs)):
        for fingerprint in tqdm(dataloader):

            model.train()
            # Output of Autoencoder
            reconstructed_fingerprint = model(fingerprint)
            
            # Calculating the loss
            loss = loss_function(reconstructed_fingerprint, fingerprint)
            
            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Storing the losses in a list for plotting
            losses.append(loss.cpu().detach())
            outputs.append((epochs, fingerprint, reconstructed_fingerprint))

            # save the model regularly in case it fails during some training epoch. 
            joblib.dump(model, save_model_name)

        # at the end of each epoch, test
        for test_fp in tqdm(test_dataloader):
            with torch.no_grad():
                model.eval()
                reconstructed_test = model(test_fp)
        print(f" fingerprint: {test_fp[0]} \t reconstruction: {reconstructed_test[0]}")
        print(f"tanimoto example 1: {jaccard_distance(torch.round(test_fp[0]).detach().numpy(), torch.round(reconstructed_test[0]).detach().numpy())}")
        print(f"tanimoto example 2: {jaccard_distance(torch.round(test_fp[1]).detach().numpy(), torch.round(reconstructed_test[1]).detach().numpy())}")
        print(f"tanimoto example 3: {jaccard_distance(torch.round(test_fp[2]).detach().numpy(), torch.round(reconstructed_test[2]).detach().numpy())}")
        print(f"Current loss: {loss.detach()}")
