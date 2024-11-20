# .\env\Scripts\activate to activate virtual environment
# pip freeze > requirements.txt You can export a list of all installed packages
# pip install -r requirements.txt to install from a requirements file
# pip list to list all packages

import numpy as np
import matplotlib.pyplot as plt
from func import *

dim = np.array([6, 6])  # kernel dimensions (in pixels)
ppa = np.array([8, 8])  # pixels per arc (scaling factor)
ang = np.ceil(dim / ppa)  # angular size based on pixels per arc
ctr = (1/3) * dim[0]  # center size as a fraction of kernel size
sur = (2/3) * dim[0]  # surround size as a fraction of kernel size

kernelON = dogKernel(dim = dim, ang = ang, ppa = ppa, ctr = ctr, sur = sur)
kernelOFF = dogKernel(dim = dim, ang = ang, ppa = ppa, ctr = ctr, sur = sur)

kernelON.setFilterCoefficients(ONOFF="ON")
kernelOFF.setFilterCoefficients(ONOFF="OFF")

kernelON.displayKernel(show_plt=True, show_hist=True)

# Next, read the SNNTorch documentation to determine how to derive spikes from images, and input them into the networks
