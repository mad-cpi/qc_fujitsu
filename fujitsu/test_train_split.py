
## Matthew Dorsey 
## 2023.08.07
## Program for splitting training datasets into training and validation sets

import sys, os, math
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit

## PARAMETERS
# fraction of dataset that should be assigned the testing catagory
test_set_size = 0.8


## ARGUMENTS
# path to dataset that should be split
path = sys.argv[1]
# title of SMILE col
smile_col = sys.argv[2]
# title of class col
class_col = sys.argv[3]


## SCRIPT 
# load the data
df = pd.read_csv(path)
smiles = df[smile_col].tolist()
classifications = df[class_col].tolist()


# split data set into testing and training sets
rs = ShuffleSplit(n_splits = 5, test_size = test_set_size, random_state = 42)

# save the data sets 