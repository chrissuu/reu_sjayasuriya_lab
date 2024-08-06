import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torcheval.metrics.functional import binary_auprc
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from Datacompiler import generate_compiler
from utils import *
from train_test import test
from train_test import train
from QDataset import QuantumDataset

from SQNN import SQNN
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CLOUD = False
LINUX = False
HARDSTOP = 500 # how many imgs to use. 2 * HARDSTOP, balanced
HARDSTOP_TST = 120
BATCH_SIZE = 20 # MAKE SURE BATCH_SIZE ARE FACTORS OF HARDSTOP_TST AND HARDSTO
QUBIT = "lightning.qubit" 
WIRES = 4
# if not LINUX else "lightning.gpu"
path_trn = ""
path_tst = ""
path_hdf = ""

path_vhn_sv = ""
path_vhn_sv_wghts = ""
path_reg_sv = ""

SAVE_PATH = "/Users/chrissu/Desktop/research_data/qnn_data/"  # Data saving folder

if CLOUD:
    path_trn = '/data/sjayasur/greg_data/train/'
    path_tst = '/data/sjayasur/greg_data/test/'
    path_hdf = 'DL_info/chip_info/cube_raw'
    path_vhn_sv_wghts = '/home/chrissu/saves/vhn_weights.txt'
    path_vhn_sv = '/home/chrissu/saves/res_vhn.txt'
    path_reg_sv = '/home/chrissu/saves/res_reg.txt'
elif LINUX:
    path_trn = '/home/imaginglyceum/Desktop/research_data/sas_nov_data/'
    path_tst = '/home/imaginglyceum/Desktop/research_data/sas_june_data/'
    path_hdf = 'DL_info/chip_info/cube_raw'
    path_vhn_sv_wghts = '/home/imaginglyceum/Desktop/reu_suren_lab2024/research/scripts/pipeline_test/saves/vhn_wghts.npy'
    path_vhn_sv = '/home/imaginglyceum/Desktop/reu_suren_lab2024/research/scripts/pipeline_test/saves/res_vhn.txt'
    path_reg_sv = '/home/imaginglyceum/Desktop/reu_suren_lab2024/research/scripts/pipeline_test/saves/res_reg.txt'
else:
    path_trn = '../../../../research_data/sas_nov_data/'
    path_tst = '../../../../research_data/sas_june_data/'
    path_hdf = 'DL_info/chip_info/cube_raw'
    path_vhn_sv_wghts ='/Users/chrissu/Desktop/reu_suren_lab2024/research/scripts/pipeline/saves/vhn_wghts.npy'
    path_vhn_sv = './saves/res_vhn.txt'
    path_reg_sv = './saves/res_reg.txt'

wghts = np.load(path_vhn_sv_wghts)

n_layers = 1   # Number of random layers


np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_seed(0)       # Seed for TensorFlow random number generator

q_train_images = np.array([])
q_test_images = np.array([])
np.save(SAVE_PATH + "/train", q_train_images)
np.save(SAVE_PATH + "/test", q_test_images)

skip = 1 #skips ~ num filters. #kerneled imgs = skip * 8
iters = 1 #only change iters if you want to test scaling laws
n_layers = 1
train_labels = np.load(SAVE_PATH + "q_train_labels.npy")
test_labels = np.load(SAVE_PATH + "q_test_labels.npy")
dataset_trn_np = np.load(SAVE_PATH + "q_train_dataset.npy")
dataset_tst_np = np.load(SAVE_PATH + "q_test_dataset.npy")

print(f"Type of dataset: {type(dataset_trn_np)}")
print(f"Shape of TRAIN dataset: {dataset_trn_np.shape}")
print(f"Shape of TEST dataset: {dataset_tst_np.shape}")

dataset_trn = QuantumDataset(dataset_trn_np, np.array(train_labels, dtype = 'float'), HARDSTOP)
dataset_tst = QuantumDataset(dataset_tst_np, np.array(test_labels, dtype = 'float'), HARDSTOP_TST)

dldr_trn = DataLoader(dataset_trn, batch_size = BATCH_SIZE, shuffle = True)
dldr_tst = DataLoader(dataset_tst, batch_size = BATCH_SIZE, shuffle  = True)

netsq = SQNN(bz = BATCH_SIZE, rand_params=None, q_dev = None, w=WIRES) # initializes REG convnet; nc = input should have 1 channel
criterion1 = nn.BCELoss()
criterion2 = None
optimizer = optim.Adam(netsq.parameters(), lr=0.0004)

START_EPOCHS = 12
NUM_EPOCHS = 14
arr_epoch = [i for i in range(START_EPOCHS, NUM_EPOCHS, 2)]
atrp_aucpr_tst = []
sqnn_aucpr_tst = []
atrp_aucpr_trn = []
sqnn_aucpr_trn = []


from VHN_ATR import ATR

net_vhn = ATR(nc = 1) # initializes VHN convnet; nc = input should have 1 channel
criterion1 = nn.BCELoss()
criterion2 = nn.MSELoss()
optimizer = optim.Adam(net_vhn.parameters(), lr=0.0004)

vhn_aucpr_tst = []
vhn_aucpr_trn = []

for i in range(START_EPOCHS, NUM_EPOCHS, 2):
    train(criterion1, criterion2, optimizer, net_vhn, num_epochs = i, dldr_trn = dldr_trn)
    net_vhn = ATR(nc = 1) # initializes VHN convnet; nc = input should have 1 channel
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(net_vhn.parameters(), lr=0.0004)

    accuracy, aucpr, str_accuracy, str_aucpr = test(netsq, dldr_tst=dldr_tst)

    vhn_aucpr_tst.append(aucpr)

    accuracy, aucpr, str_accuracy, str_aucpr = test(netsq, dldr_tst=dldr_trn)

    vhn_aucpr_trn.append(aucpr)





import matplotlib.pyplot as plt

plt.plot(arr_epoch, vhn_aucpr_tst, label='atrp_tst', linestyle='--', marker='s', color='y')
plt.plot(arr_epoch, vhn_aucpr_trn, label='sqnn_tst', linestyle='--', marker='s', color='g')

plt.xlabel('Num epochs')
plt.ylabel('ATRP (Blue), SQNN (Red)')
plt.title('Plot of AUCPR of atrp and sqnn with respect to num of epochs')

plt.legend()

# Add grid
plt.grid(True)

# Show the plot
plt.show()

