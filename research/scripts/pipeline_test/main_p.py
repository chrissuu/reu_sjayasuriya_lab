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


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CLOUD = False
LINUX = False
HARDSTOP = 2

path_trn = ""
path_tst = ""
path_hdf = ""

path_vhn_sv = ""
path_vhn_sv_wghts = ""
path_reg_sv = ""

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

# dldr_trn_reg, dldr_tst_reg = copy.deepcopy(dldr_trn), copy.deepcopy(dldr_tst)

n_layers = 1   # Number of random layers

SAVE_PATH = "../../../research_data/qnn_data/"  # Data saving folder
PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_seed(0)       # Seed for TensorFlow random number generator

print("finished loading dataset trn + tst\n\n")

#creates iters datasets with skip filters
def generate_datum(train_images, n_layers, iters, skip):
    
    train_datasets = [[[] for img in train_images] for i in range(iters)]
    stores = [[] for i in range(len(train_images))]
    
        
    dev = qml.device("lightning.qubit", wires=8)
    # Random circuit parameters
    

    @qml.qnode(dev)
    def circuit(phi, rand_params):
        # Encoding of 8 classical input values
        for j in range(8):
            qml.RY(np.pi * phi[j], wires=j)
        # print("qml.RY(np.pi * phi[j], wires = j") 
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(8)))
        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(8)]

    #produces 4 kerneled images of size 14,14 per image
    def quanv(image, rand_params):
        """Convolves the input image with many applications of the same quantum circuit."""
        tot = np.zeros((32, 32, 50, 8))
        
        # Loop over the coordinates of the top-left pixel of 2X2 squares
        for j in range(0, 64, 2):
            for k in range(0, 64, 2):
                for m in range(0, 100, 2):
                    # Process a squared 2x2 region of the image with a quantum circuit
                    q_results = circuit(
                        [
                            image[j, k, m],
                            image[j, k + 1, m],
                            image[j + 1, k, m],
                            image[j + 1, k + 1, m],
                            image[j, k, m+1],
                            image[j, k + 1, m+1],
                            image[j + 1, k, m+1],
                            image[j + 1, k + 1, m+1],
                        ], 
                        rand_params
                    )
                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    for c in range(8):
                        tot[j // 2, k // 2, m //2, c] = q_results[c]
        return tot
    
    start = time.time()
    for i in range(skip * iters): 
        # for skip*iters iterations, create a random circuit 
        # and use that circuit to generate 4 kerneled images
        # when this loop exits, there would be skip * iters * 4 kerneled images,
        # imgs, in stores[imgs]
        print(f"generating {i+1}th iteration of filtered images\n")
        rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 8))
        for idx, img in enumerate(train_images):
            start_temp = time.time()
            stores[idx].append(quanv(img, rand_params))
            end_temp = time.time()
            print(f"img {idx} took {end_temp - start_temp} seconds to process")
    end = time.time()
    print(f"finished creating filtered images. this task took {end - start} seconds. \n\nentering dataset generation\n\n")
   
    # from the previous loop, we now want to put these imgs in datasets
    # i loops over the number of datasets (iters)

    
    for i in range(0, iters):
        
        # for each img array in stores, 
        # enumerating by idx, we take a subarray of size i * skip, copy, then reshape to desired size
        # appropriate data with appropriate idx is added
        for idx, img_array in enumerate(stores):
            temp = np.array(img_array[0:(i+1) * skip]).copy().reshape((32, 32, 50, 8 * (i+1) * skip))
            train_datasets[i][idx].append(temp)
    
    print("finished dataset generation\n\n")
    return train_datasets

def generate_data(images, num_filters, n_layers):
    return generate_datum(train_images=images, iters = 1, skip = num_filters, n_layers = n_layers)

def create_lists(path, path_hdf, BZ, IR, HARDSTOP):
    dldr = generate_compiler(data_root = path, \
                hdf_data_path = path_hdf, \
                BZ = BZ, IR = IR)
    images = []
    labels = []
    for i, data in enumerate(dldr,0):
        inputs, label = data
        
        # print(f"type of image: {type(inputs)}\n\n") 
        if i < HARDSTOP:
            if i % 50 == 0 and HARDSTOP >= 50:
                print(f"loaded {i} elts") 
            else:
                print(f"loaded {i+1} elts")

            inputs = torch.log10(torch.tensor(inputs) + 1)
            # print(inputs.shape)
            label = torch.tensor(label)
            images.append(inputs)
            labels.append(label)
        
    return images, labels

train_images, train_labels = create_lists(path = path_trn, path_hdf = path_hdf, BZ = 1, IR = 1, HARDSTOP = HARDSTOP)

test_images, test_labels = create_lists(path = path_tst, path_hdf = path_hdf, BZ = 1, IR = 1, HARDSTOP = HARDSTOP)


print(f"len of train dataset: {len(train_images)}")
print(f"len of test dataset: {len(test_images)}")
print(f"shape of data: {test_images[0].shape}")
# shape shold be 20, 101, 64, 64 before CNN input.#
    
skip = 1 #skips ~ num filters. #kerneled imgs = skip * 8
iters = 1 #only change iters if you want to test scaling laws
n_layers = 1
dataset_trn_np = generate_data(images=train_images, n_layers = n_layers, num_filters=skip)
dataset_tst_np = generate_data(images=test_images, n_layers = n_layers, num_filters = skip)
print(type(dataset_trn_np))
# for i, dataset in enumerate(datasets):
#     datasets[i] = np.array(np.array(dataset).reshape(min(HARDSTOP, len(train_images)), 32, 32, 50, skip * (i+1) * 8))

# for dataset in datasets:

#     print((dataset).shape)

dataset_trn = QuantumDataset(dataset_trn_np, np.array(train_labels))
dataset_tst = QuantumDataset(dataset_tst_np, np.array(test_labels))

dldr_trn = DataLoader(dataset_trn, batch_size = 4, shuffle = True)
dldr_tst = DataLoader(dataset_tst, batch_size = 4, shuffle  = True)

class ATRP(nn.Module):
    def __init__(self, nc, device = None, N_filters=4, N_output = 1, ker = 4, s = 2, pad =1):
        super(ATRP, self).__init__()
        self.ker = ker
        self.s = s
        self.pad = pad
        self.nc = nc
        # self.device = device
        self.N_filters = N_filters
        self.N_output = N_output
        self.conv1 = nn.Conv3d(nc, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv2 = nn.Conv3d(N_filters, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv3 = nn.Conv3d(N_filters, N_filters, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.conv4 = nn.Conv3d(N_filters, 1, kernel_size = (3, 3, 3), stride=(1, 2, 2), padding= (0, 1, 1))
        self.avgpool = nn.AvgPool3d(kernel_size = (6, 2, 2), stride= (1, 1, 1), padding= (0, 0, 0))
    

        # "columns input x and output columnes"

        self.f2 = nn.Linear(1248,1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x0):
        "image vectorization"
        # print(x0.shape)
        x = self.conv1(x0.unsqueeze(1))
        x = self.avgpool(F.relu(x))
        x = self.conv2(x)
        x = self.avgpool(F.relu(x))
        x = self.conv3(x)
        x = self.avgpool(F.relu(x))
        x = self.conv4(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = self.f2(x)

        y = self.sigmoid(x)

        return y
    

netp = ATRP(nc = 1) # initializes REG convnet; nc = input should have 1 channel
criterion1 = nn.BCELoss()
criterion2 = None
optimizer = optim.Adam(netp.parameters(), lr=0.001)

train(criterion1, criterion2, optimizer, netp, num_epochs=5, dldr_trn = dldr_trn)

print("finished training REG ATR\n")

res = test(netp, dldr_tst=dldr_tst)

print("finished testing REG ATR\n")

torch.save(netp, './saves/reg.pt')

res_reg_txt = open(path_reg_sv, 'w')
res_reg_txt.write(str(res))
res_reg_txt.close()