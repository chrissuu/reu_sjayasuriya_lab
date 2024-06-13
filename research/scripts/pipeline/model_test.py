import torch
from utils import *
from main import test
import copy
from Dataloader import generate_generators

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

model_with_vhn = torch.load('./saves/vhn.pt')

model_with_reg = torch.load('./saves/reg.pt')

model_with_vhn.eval()

model_with_reg.eval()

res = test(model_with_vhn, dldr_tst)

print(f"vhn res \n {res}".format(res = res))

res_reg = test(model_with_reg, dldr_tst_reg)

print(f"reg res \n {res_reg}".format(res_reg = res_reg))
