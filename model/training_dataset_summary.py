import os
from pathlib import Path
import numpy as np 
import pandas as pd


GLOBAL_PATH = os.getcwd()
TRAINING_DATASET_PATH = Path(GLOBAL_PATH,"datasets/training")
files = os.listdir(TRAINING_DATASET_PATH)

for file in files:
    filepath = Path(TRAINING_DATASET_PATH,file)
    data = np.load(filepath)['facies']
    
    print(np.shape(data))