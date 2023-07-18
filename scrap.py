import pandas as pd
from rdkit.Chem import AllChem
import rdkit.Chem as Chem
import joblib

df = pd.read_parquet('/Users/docmartin/Downloads/BindingDB_All_2D_clean.parquet')
print(df)

morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(x_i), 3, nBits=1024) for x_i in df['SMILES']]

joblib.dump(morgan_fps, '/Users/docmartin/Downloads/BindingDB_SMILES_clean_morgan_fps.joblib')