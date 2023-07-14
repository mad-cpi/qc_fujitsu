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
from scipy.spatial import distance
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
    #encoded_fps = joblib.load(filename)
    encoded_fps = encoded_fps_test
    batch_size = 1
    epochs = 10
    # encode the X
    #encoded_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x_i), 3, nBits=1024) for x_i in df['SMILES']]
    dataset = MorganDataset(torch.FloatTensor(encoded_fps))
    test_dataset = MorganDataset(torch.FloatTensor(encoded_fps_test))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(dataset, shuffle=True, batch_size=64)
    # set up the model
    # model = AE()
    model = LBAE()
    #loss_function = torch.nn.MSELoss()
    #loss_function = torch.nn.BCEWithLogitsLoss()
    #loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)


    outputs = []
    losses = []

    # for saving and inspecting
    real_fps = []
    latent_fps = []
    reconstructed_fps = []
    rounded_latent_fps = []
    rounded_reconstructed_fps = []
    for i, epoch in enumerate(tqdm(range(epochs))):
        for fingerprint in dataloader:
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
        
        # at the end of each epoch, test
        for test_fp in tqdm(test_dataloader):
            with torch.no_grad():
                model.eval()
                reconstructed_test = model(test_fp)
                latent_pred = model.encoder(test_fp)
        print(f" fingerprint: {test_fp} \t reconstruction: {reconstructed_test}")
        print(f"tanimoto: {distance.jaccard(torch.round(test_fp[0]).detach().numpy(), torch.round(reconstructed_test[0]).detach().numpy())}")
        print(f"tanimoto: {distance.jaccard(torch.round(test_fp[1]).detach().numpy(), torch.round(reconstructed_test[1]).detach().numpy())}")
        print(f"tanimoto: {distance.jaccard(torch.round(test_fp[2]).detach().numpy(), torch.round(reconstructed_test[2]).detach().numpy())}")
        print(f"Current loss: {loss.detach()}")
        real_fps.append(test_fp[0].detach().numpy())
        latent_fps.append(latent_pred[0].detach().numpy())
        reconstructed_fps.append(reconstructed_test[0].detach().numpy())
        rounded_latent_fps.append(torch.round(latent_pred[0]).detach().numpy())
        rounded_reconstructed_fps.append(torch.round(reconstructed_test[0]).detach().numpy())
                # save a few fps for checking out
        ##if i % 10 == 0:

            # next, we want to encode all of the SMILES into a bit vector
    real_fps_df = pd.DataFrame(real_fps)
    real_fps_df.to_csv(f'/Users/docmartin/Downloads/LBAE/acetyl_LBAE_real_fps.csv')
    latent_fps_df = pd.DataFrame(latent_fps)
    latent_fps_df.to_csv(f'/Users/docmartin/Downloads/LBAE/acetyl_LBAE_latent_fps.csv')
    reconstructed_fps_df = pd.DataFrame(reconstructed_fps)
    reconstructed_fps_df.to_csv(f'/Users/docmartin/Downloads/LBAE/acetyl_LBAE_reconstructed_fps.csv')
    rounded_latent_fps_df = pd.DataFrame(rounded_latent_fps)
    rounded_latent_fps_df.to_csv(f'/Users/docmartin/Downloads/LBAE/acetyl_LBAE_latent_fps_rounded.csv')
    rounded_reconstructed_fps_df = pd.DataFrame(rounded_reconstructed_fps)
    rounded_reconstructed_fps_df.to_csv(f'/Users/docmartin/Downloads/LBAE/acetyl_LBAE_reconstructed_fps_rounded.csv')



    #test_outputs = []
    #test_latent = []
    #actual_fps = []
    #for test_fp in tqdm(test_dataloader):
        #with torch.no_grad():
            #model.eval()

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