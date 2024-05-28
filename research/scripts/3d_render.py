import numpy as np
import matplotlib.pyplot as plt
import h5py
import ipyvolume as ipv

path = '../data/Deep_194356/SERDP_LF_down_0.hdf5'

f = h5py.File(path, 'r')

real_data = f['Data']['Real'][:]
imag_data = f['Data']['Imag'][:]
complex_data = real_data + 1j * imag_data
print(complex_data.shape)
magnitude_data = np.abs(complex_data)

nan_mask = np.isnan(magnitude_data)
inf_mask = np.isinf(magnitude_data)

cleaned_data = np.where(nan_mask | inf_mask, 0, magnitude_data)
log_comp_data = np.log(cleaned_data+1)
ipv.quickvolshow(log_comp_data, level = [0.1,0.5,0.8], opacity = 0.1)
ipv.show()
