import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import copy
import numpy as np

from Dataloader import generate_generators
from utils import *

wghts = np.load('/home/imaginglyceum/Desktop/reu_suren_lab2024/research/scripts/pipeline/saves/vhn_wghts.npy')
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CLOUD = False
LINUX = True

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
    path_vhn_sv_wghts = './saves/vhn_wghts.npy'
    path_vhn_sv = './saves/res_vhn.txt'
    path_reg_sv = './saves/res_reg.txt'


dldr_trn = generate_generators(data_root = path_trn, \
                hdf_data_path = path_hdf, \
                BZ = 20, IR = 1)

# dldr_tst = generate_generators(data_root=path_tst, \
#                 hdf_data_path = path_hdf, \
#                 BZ = 20, IR = 1)

# dldr_trn_reg, dldr_tst_reg = copy.deepcopy(dldr_trn), copy.deepcopy(dldr_tst)

n_epochs = 10   # Number of optimization epochs
n_layers = 1   # Number of random layers
n_train = 3000  # Size of the train dataset
n_test = 2000  # Size of the test dataset
max_filters = 80

q_histories = []
# c_histories = []
q_img_history_trn = []
q_img_history_tst = []

train_images = []
train_labels = []

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
        tot = np.zeros((64, 64, 101, 8))
        
        # Loop over the coordinates of the top-left pixel of 2X2 squares
        for j in range(0, 64, 2):
            for k in range(0, 64, 2):
                for m in range(0, 100, 2):
                    # Process a squared 2x2 region of the image with a quantum circuit
                    q_results = circuit(
                        [
                            image[j, k, m, 0],
                            image[j, k + 1, m, 0],
                            image[j + 1, k, m, 0],
                            image[j + 1, k + 1, m, 0],
                            image[j, k, m+1, 0],
                            image[j, k + 1, m+1, 0],
                            image[j + 1, k, m+1, 0],
                            image[j + 1, k + 1, m+1, 0],
                        ], 
                        rand_params
                    )
                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    for c in range(8):
                        tot[j // 2, k // 2, m //2, c] = q_results[c]
        return tot
    
    
    for i in range(skip * iters): 
        # for skip*iters iterations, create a random circuit 
        # and use that circuit to generate 4 kerneled images
        # when this loop exits, there would be skip * iters * 4 kerneled images,
        # imgs, in stores[imgs]
        print(f"generating {i}th iteration of filtered images\n".format(i = i))
        rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 8))
        for idx, img in enumerate(train_images):
            stores[idx].append(quanv(img, rand_params))

    print("finished creating filtered images\n\nentering dataset generation\n\n")
   
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

def generate_data(train_images, num_filters, n_layers):
    return generate_datum(train_images=train_images, iters = 1, skip = num_filters, n_layers = n_layers)

for i, data in enumerate(dldr_trn,0):
    inputs, label = data
    if i % 50 == 0:
        print(f"loaded {i} elts") 
    # print(f"type of image: {type(inputs)}\n\n") 
    inputs = torch.log10(torch.tensor(inputs) + 1)
    # print(inputs.shape)
    label = torch.tensor(label)
    train_images.append(inputs)
    train_labels.append(label)
    

print(f"len of dataset: {len(train_images)}")
print(f"shape of data: {train_images[0].shape}")
# shape shold be 20, 101, 64, 64 before CNN input.#
    
skip = 1
iters = 2
n_layers = 1
datasets = generate_data(train_images=train_images, n_layers = n_layers, num_filters=skip)

for i, dataset in enumerate(datasets):
    datasets[i] = np.array(np.array(dataset).reshape((len(train_images), 32, 32, 50, skip * (i+1) * 4)))

for dataset in datasets:

    print((dataset).shape)