import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics.functional import binary_auprc
import copy
from VHN_conv_net import ATR
from REG_conv_net import ATRP
from Dataloader import generate_generators
from utils import *

CLOUD = False

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
else:
    path_trn = '../../../../research_data/sas_nov_data/'
    path_tst = '../../../../research_data/sas_june_data/'
    path_hdf = 'DL_info/chip_info/cube_raw'
    path_vhn_sv_wghts = './saves/vhn_wghts.npy'
    path_vhn_sv = './saves/res_vhn.txt'
    path_reg_sv = './saves/res_reg.txt'


dldr_trn, dldr_tst = generate_generators(data_root = path_trn, \
                                         hdf_data_path = path_hdf, \
                                         BZ = 20, IR = 1), \
                     generate_generators(data_root=path_tst, \
                                         hdf_data_path = path_hdf, \
                                         BZ = 20, IR = 1)

dldr_trn_reg, dldr_tst_reg = copy.deepcopy(dldr_trn), copy.deepcopy(dldr_tst)
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics.functional import binary_auprc
from utils import *

# functions for training and testing the network

def train(criterion1, criterion2, optimizer, net, num_epochs, dldr_trn):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dldr_trn, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs, labels = data
            # print(f"Inputs shape: {inputs.shape}")
            
            temp_inputs = (torch.log10(torch.tensor(inputs) + 1)).float().squeeze(0)
            # print(f"Inputs shape post: {inputs.shape}")

            # print(inputs.shape)
            temp_labels = labels
            # zero the parameter gradients
            optimizer.zero_grad()

            # print(inputs)
            # forward + backward + optimize
            # print(f"TEMP INPUTS TYPE: {type(temp_inputs.item())}")
            outputs = net(temp_inputs)
            loss = criterion1(outputs, torch.tensor(temp_labels.reshape(dldr_trn.batch_size,1)).type(torch.float32))
            if criterion2: 
                # print("SHAPES")
                # print(curly_Nprime(net.vhn.weights).shape)
                # print(torch.sum(temp_inputs, dim = 0).shape)
                # print(temp_inputs.shape)
                _temp = temp_inputs.reshape((dldr_trn.batch_size, 101, 64, 64))
                x_bar = np.zeros((dldr_trn.batch_size, 101, 64, 64), dtype='float')
                target_cnt = 0
                for i in range(dldr_trn.batch_size):
                    if int(temp_labels[i]) == 1:
                        x_bar = np.add(x_bar, _temp[i])
                        target_cnt += 1
                x_bar /= target_cnt
                loss += criterion2(curly_Nprime(net.vhn.weights), curly_N(x_bar.float()))
                loss = loss.float()
                
            # print("netvhn", net.vhn.weights.shape)
            # print(curly_N(torch.sum(inputs, dim = 0) / dldr.batch_size).shape)
       
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            # if i % 5 == 4:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
              
            #     running_loss = 0.0

    return

def test(net, dldr_tst):
    preds = []
    labels = []
    # imax = []
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dldr_tst,0):
            inputs, label = data
            
            temp_inputs = torch.log10(torch.tensor(inputs) + 1).float().squeeze(0)
            # print(inputs.shape)
            temp_label = label
            # calculate outputs by running images through the 
            output = net(temp_inputs)

            preds.append(output.tolist())
            labels.append(temp_label.tolist())
            # imax.append(i)
            # print(inputs.shape)
    preds_parsed = []
    labels_parsed = []
    # print(f"num_test {i}".format(i = max(imax)))
    for i, pred_list in enumerate(preds):
        total += len(pred_list)
        for k, pred in enumerate(pred_list):
            preds_parsed.append(pred)
            # if pred[0] >= 0.5:
            #     preds_parsed.append([1])
            # else:
            #     preds_parsed.append([0])
            labels_parsed.append(labels[i][k])
            if pred[0] >= 0.5: 
                if labels[i][k] == 1:
                    correct += 1
            else:
                if labels[i][k] == 0:
                    correct += 1

    
    print(len(preds_parsed))
    print(len(labels_parsed))
    
    return correct/total, binary_auprc(torch.tensor(preds_parsed).squeeze(1), 
           torch.tensor(labels_parsed), num_tasks=1).mean(), f"ACCURACY {correct / total}", f"PRAUC {binary_auprc(torch.tensor(preds_parsed).squeeze(1), torch.tensor(labels_parsed), num_tasks=1).mean()}"

START_EPOCHS = 4
NUM_EPOCHS = 6
netp = ATR(nc = 1)
total_paramsp = sum(
    param.numel() for param in netp.parameters()
)
print(f"num_params {total_paramsp}")
arr_epoch = [i for i in range(START_EPOCHS,NUM_EPOCHS,2)]
vhn_aucpr_tst = []
vhn_aucpr_trn = []
for i in range(START_EPOCHS, NUM_EPOCHS, 2):
    net_vhn = ATR(nc = 1) # initializes VHN convnet; nc = input should have 1 channel
    
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(net_vhn.parameters(), lr=0.001)

    train(criterion1, criterion2, optimizer, net_vhn, num_epochs = i, dldr_trn = dldr_trn)

    accuracy, aucpr, str_accuracy, str_aucpr = test(net_vhn, dldr_tst=dldr_tst)

    vhn_aucpr_tst.append(aucpr)

    accuracy, aucpr, str_accuracy, str_aucpr = test(net_vhn, dldr_tst=dldr_trn)

    vhn_aucpr_trn.append(aucpr)

import matplotlib.pyplot as plt

# plt.plot(arr_epoch, atrp_aucpr_trn, label='atrp_trn', linestyle='--', marker='s', color='b')
# plt.plot(arr_epoch, sqnn_aucpr_trn, label='sqnn_trn', linestyle='--', marker='s', color='r')
# plt.plot(arr_epoch, atrp_aucpr_tst, label='atrp_tst', linestyle='--', marker='s', color='y')
# plt.plot(arr_epoch, sqnn_aucpr_tst, label='sqnn_tst', linestyle='--', marker='s', color='g')

plt.plot(arr_epoch, vhn_aucpr_tst, label='atrp_tst', linestyle='--', marker='s', color='y')
plt.plot(arr_epoch, vhn_aucpr_trn, label='sqnn_tst', linestyle='--', marker='s', color='g')
# Add labels and title
plt.xlabel('Num epochs')
plt.ylabel('ATRP (Blue), SQNN (Red)')
plt.title('Plot of AUCPR of atrp and sqnn with respect to num of epochs')

# Add a legend
plt.legend()

# Add grid
plt.grid(True)

# Show the plot
plt.show()
