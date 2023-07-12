
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

class MorganDataset(Dataset):
    def __init__(self, fps):
        self.fps = fps

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        input = self.fps[idx]

        return input
    
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
if __name__ == "__main__":

    # dataset to read in
    filename = ''
    df = pd.read_csv(filename)

    batch_size = 64

    # encode the X
    encoded_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x_i), 3, nBits=1024) for x_i in df['SMILES']]
    dataset = MorganDataset(encoded_fps)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    # set up the model
    model = AE()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)

    epochs = 2
    outputs = []
    losses = []


    for epoch in range(epochs):
        for fingerprint in tqdm(dataloader):
           
            # Output of Autoencoder
            reconstructed = model(fingerprint)
            
            # Calculating the loss function
            loss = loss_function(reconstructed, fingerprint)
            
            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Storing the losses in a list for plotting
            losses.append(loss.cpu().detach())
            outputs.append((epochs, fingerprint, reconstructed))
        
    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    # Plotting the last 100 values
    plt.plot(losses[-100:])