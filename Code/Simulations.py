from itertools import product
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentSphericalMeanModel
from dmipy.signal_models import cylinder_models, gaussian_models, sphere_models
from scipy.io import savemat
from scipy.special import erf
from tqdm import tqdm

np.random.seed(100)

def sim_sig_np(bf,be,tm,adc,sigma,axr):
    be = np.expand_dims(be, axis=0)
    bf = np.expand_dims(bf, axis=0)
    tm = np.expand_dims(tm, axis=0)
    
    if adc.size != 1:
        adc = np.expand_dims(adc, axis=1)
        sigma = np.expand_dims(sigma, axis=1)
        axr = np.expand_dims(axr, axis=1)                                   

    tm[(tm == np.min(tm)) & (bf == 0)] = np.inf

    adc_prime = adc * (1 - sigma* np.exp(-tm*axr))
    normalised_signal = np.exp(-adc_prime * be)

    return normalised_signal, adc_prime

nvox = 10000 # number of voxels to simulate
noise = 0 # 1 for noise, 0 for no noise

bf = np.array([0, 0, 250, 250, 250, 250, 250, 250]) * 1e-3   # filter b-values [ms/um2]
be = np.array([0, 250, 0, 250, 0, 250, 0, 250]) * 1e-3       # encoding b-values [ms/um2]
tm = np.array([20, 20, 20, 20, 200, 200, 400, 400], dtype=np.float32) * 1e-3 # mixing time [s]


adc_lb = 0.1        #[um2/ms]
adc_ub = 3.5        #[um2/ms]
adc_ub = 6.15          #[um2/ms] based off intuition, look at max possible value
sig_lb = 0          #[a.u.]
sig_ub = 1.0          #[a.u.]
axr_lb = 0.1        #[s-1]
#axr_lb = 1          #[s-1]
axr_ub = 20         #[s-1]

#consider doing in si units

limits = np.array([[adc_lb, adc_ub], [sig_lb, sig_ub] , [axr_lb, axr_ub]])

adc_init = (adc_lb + adc_ub) / 2 #[um2/ms]
sig_init = (sig_lb + sig_ub) / 2 #[a.u.]
axr_init = (axr_lb + axr_ub) / 2 #[ms-1]

num_inits = 5

# Create equally spaced arrays for each parameter
# remove first and last values which are on the "face of the cube"
adc_inits = np.linspace(adc_lb, adc_ub, num_inits)[1:-1]
sig_inits = np.linspace(sig_lb, sig_ub, num_inits)[1:-1]
axr_inits = np.linspace(axr_lb, axr_ub, num_inits)[1:-1]

# Generate all permutations of combinations
all_inits = list(product(adc_inits, sig_inits, axr_inits))

# Convert the list of tuples to a NumPy array
all_inits = np.array(all_inits)

#new method:
fieq = np.random.uniform(0,0.1,nvox)                            # fieq, simulated [a.u] #change
feeq = 1 - fieq                                                 # feeq, simulated [a.u]
# ranges for De & Di from lizzies paper 
De = np.random.uniform(0.1, 3.5, nvox)                          # De, simulated [um2/ms]
Di = np.random.uniform(3, 30, nvox)                             # Di, simulated [um2/ms]

# Check if De is smaller than Di and regenerate values if necessary
while np.any(De > Di):
    indices = np.where(De > Di)
    De[indices] = np.random.uniform(0.1, 3.5, len(indices[0]))
    Di[indices] = np.random.uniform(3, 30, len(indices[0]))                              


De = np.expand_dims(De,axis=1)
Di = np.expand_dims(Di,axis=1)
fieq = np.expand_dims(fieq,axis=1)
feeq = np.expand_dims(feeq,axis=1)

sim_adc = feeq * De + fieq * Di                                 # ADC, simulated [um2/ms] (I think units are the same because it is a weighted sum)

#s0 = 1 at this point, so don't need to divide by anything
sbf_s0 = ((1-fieq)*np.exp(-bf*De)+fieq*np.exp(-bf*Di))          # s(bf), simulated [a.u]
fe0 = (feeq*np.exp(-bf*De))/sbf_s0

sim_sigma = ((De-Di)*(feeq - fe0))/sim_adc      
# we take 3rd column because it is one of the columns where bf=250
sim_sigma = sim_sigma[:,2]                                      # sigma, simulated [a.u.]
sim_adc = np.squeeze(sim_adc)

sim_axr = np.random.uniform(axr_lb,axr_ub,nvox)                 # AXR, simulated [s-1]


sim_E_vox, sim_adc_prime = sim_sig_np(bf,be,tm,sim_adc,sim_sigma,sim_axr)

if noise == 1:
    # Adding rician noise to the simulated signal
    Sim_E_vox_real = sim_E_vox + np.random.normal(scale=1/50, size=np.shape(sim_E_vox)) # adding rician noise, snr = 50
    Sim_E_vox_imag = np.random.normal(scale=1/50, size=np.shape(sim_E_vox))
    E_vox = np.sqrt(Sim_E_vox_real**2 + Sim_E_vox_imag**2)
else: 
    E_vox = sim_E_vox