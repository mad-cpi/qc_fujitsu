
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm
import joblib
from scipy.spatial import distance
class MorganDataset(Dataset):
    def __init__(self, fps):
        self.fps = fps

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        input = self.fps[idx]

        return input

class QuantizerFunc(torch.autograd.Function):
    @staticmethod
    def forward(self, input, npoints=4, dropout=0):
        # self.save_for_backward(input)
        # self.constant = npoints
        if npoints < 0:
            x = torch.sign(input)
            x[x==0] = 1
            return x

        scale = 10**npoints
        input = input * scale
        input = torch.round(input)
        input = input / scale
        return input

    @staticmethod
    def backward(self, grad_output):
        # input, = self.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        # grad_input[:] = 1
        return grad_input, None
    
class LBAE_encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tan = nn.Tanh()
        self.quant = QuantizerFunc.apply
        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.LeakyReLU(0.02)
        self.enc0a = torch.nn.Linear(1024, 1024)
        self.enc0b = torch.nn.Linear(1024, 512)

        self.enc1a = torch.nn.Linear(512, 512)
        self.enc1b = torch.nn.Linear(512, 256)

        self.enc2a = torch.nn.Linear(256, 256)
        self.enc2b = torch.nn.Linear(256, 128)

    def forward(self, x):
        x = self.enc0a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.enc0b(x)

        x = self.enc1a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.enc1b(x)

        x = self.enc2a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.enc2b(x)
        x = torch.tanh(x)
        x = self.quant(x)

        return x
class LBAE_decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.LeakyReLU(0.02)
        self.dec0a = torch.nn.Linear(128, 128)
        self.dec0b = torch.nn.Linear(128, 256)

        self.dec1a = torch.nn.Linear(256, 256)
        self.dec1b = torch.nn.Linear(256, 512)

        self.dec2a = torch.nn.Linear(512, 512)
        self.dec2b = torch.nn.Linear(512, 1024)

    def forward(self, x):
        x = self.dec0a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dec0b(x)

        x = self.dec1a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dec1b(x)

        x = self.dec2a(x) + x
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dec2b(x)
        x = torch.sigmoid(x)

        return x

class LBAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LBAE_encoder()
        self.decoder = LBAE_decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.tan = nn.Tanh()
        self.quant = QuantizerFunc.apply
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)

        # now for the quantized part
        encoded = torch.tanh(encoded)
        encoded = self.quant(encoded)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode_only(self, x):
        encoded = self.encoder(x)
        encoded = torch.tanh(encoded)
        encoded = self.quant(encoded)
        return encoded
    

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


if __name__ == "__main__":

    # set a flag for the giant dataset or the toy one to play around with the model
    toy = True
    #if toy == True:
        # read in the toy dataset, create the FPs
    filename = '/Users/docmartin/Downloads/Acetylcholinesterase_human_IC50_ChEMBLBindingDB_spliton6_binary_1micromolar.csv'
    df = pd.read_csv(filename)
    encoded_fps_test = np.array([generate_fp_array(Chem.MolFromSmiles(x)) for x in df['SMILES']])
    #else:
    # dataset to read in
    filename = '/Users/docmartin/Downloads/BindingDB_SMILES_clean_morgan_fps.joblib'
    #df = pd.read_csv(filename)
    encoded_fps = joblib.load(filename)

    batch_size = 1
    epochs = 1
    # encode the X
    #encoded_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x_i), 3, nBits=1024) for x_i in df['SMILES']]
    dataset = MorganDataset(torch.FloatTensor(encoded_fps))
    test_dataset = MorganDataset(torch.FloatTensor(encoded_fps_test))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(dataset, shuffle=True, batch_size=64)
    # set up the model
    #model = AE()
    model = LBAE()
    #loss_function = torch.nn.MSELoss()
    loss_function = torch.nn.BCEWithLogitsLoss()
    #loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)


    outputs = []
    losses = []


    for epoch in range(epochs):
        for i, fingerprint in enumerate(tqdm(dataloader)):
            model.train()
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
        
            if i % 100 == 0:
                for test_fp in tqdm(test_dataloader):
                    with torch.no_grad():
                        model.eval()
                        reconstructed = model(test_fp)
                        print(f"Loss: {loss}")
                        print(f" fingerprint: {test_fp} \t reconstruction: {reconstructed}")
                        print(f"tanimoto: {distance.jaccard(torch.round(test_fp[0]).detach().numpy(), torch.round(reconstructed[0]).detach().numpy())}")
                        print(f"tanimoto: {distance.jaccard(test_fp[1].detach().numpy(), reconstructed[1].detach().numpy())}")
                        print(f"tanimoto: {distance.jaccard(test_fp[2].detach().numpy(), reconstructed[2].detach().numpy())}") 

    # next, we want to encode all of the SMILES into a bit vector
    test_outputs = []
    test_latent = []
    actual_fps = []
    for test_fp in tqdm(test_dataloader):
        with torch.no_grad():
            model.eval()

            #test_outputs.extend(model(test_fp))
            #test_latent.extend(model.encode_only(test_fp))
            #actual_fps.extend(test_fp)

    #final_df = pd.DataFrame(list(zip([test_outputs, test_latent, actual_fps])))
    #print(final_df)
    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    # Plotting the last 100 values
    plt.plot(losses[-100:])