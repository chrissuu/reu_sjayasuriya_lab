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

def train(criterion1, criterion2, optimizer, net, num_epochs, dldr_trn):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("epoch {epoch}".format(epoch=epoch))
        running_loss = 0.0
        for i, data in enumerate(dldr_trn, 0):
            # get the inputs; data is a list of [inputs, labels]
        
            inputs, labels = data
            
            inputs = torch.log10(torch.tensor(inputs).transpose(1,3) + 1)
            # print(inputs.shape)
            labels = torch.tensor(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # print(inputs)
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion1(outputs, labels.reshape(20,1).type(torch.float32))
            if criterion2: 
                loss += criterion2(curly_Nprime(net.vhn.weights), \
                                curly_N(torch.sum(inputs, dim = 0) \
                                        / dldr_trn.batch_size))
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


    with torch.no_grad():
        for i, data in enumerate(dldr_tst,0):
            inputs, label = data
            
            inputs = torch.log10(torch.tensor(inputs).transpose(1,3) + 1)
            # print(inputs.shape)
            label = torch.tensor(label)
            # calculate outputs by running images through the 
            output = net(inputs)

            preds.append(output.tolist())
            labels.append(label.tolist())
            # imax.append(i)
            # print(inputs.shape)

    # print(f"num_test {i}".format(i = max(imax)))

    return "PRAUC" + str(binary_auprc(torch.tensor(preds).squeeze(0).squeeze(2), \
           torch.tensor(labels), num_tasks=len(preds)).mean())

for i in range(1, 15):
    net_vhn = ATR(nc = 1) # initializes VHN convnet; nc = input should have 1 channel
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(net_vhn.parameters(), lr=0.001)

    train(criterion1, criterion2, optimizer, net_vhn, num_epochs = i, dldr_trn = dldr_trn)

    print("finished training VHN ATR\n")

    res = test(net_vhn, dldr_tst=dldr_tst)

    print("finished testing VHN ATR\n")

    torch.save(net_vhn, './saves/vhn.pt')

    vhn_weight = net_vhn.vhn.weights

    np.save(path_vhn_sv_wghts, vhn_weight.detach().numpy())

    res_vhn_txt = open(f"test_iter{i}.txt", 'w')
    res_vhn_txt.write(str(res))
    res_vhn_txt.close()

# netp = ATRP(nc = 1) # initializes REG convnet; nc = input should have 1 channel
# criterion1 = nn.BCELoss()
# criterion2 = None
# optimizer = optim.Adam(netp.parameters(), lr=0.001)

# train(criterion1, criterion2, optimizer, netp, num_epochs=5, dldr_trn = dldr_trn_reg)

# print("finished training REG ATR\n")

# res = test(netp, dldr_tst=dldr_tst_reg)

# print("finished testing REG ATR\n")

# torch.save(netp, './saves/reg.pt')

# res_reg_txt = open(path_reg_sv, 'w')
# res_reg_txt.write(str(res))
# res_reg_txt.close()
