#------------------
# Ici, on crée un énorme dictionnaire qui contient toutes les données des fichiers du dossier où on est
#------------------

import numpy as np
import numpy.fft as fft
import scipy as sp
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d
from scipy import constants
from scipy.signal import savgol_filter
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import yaml
import re
from yaml import load, dump
import pandas as pd
import plot_functions as myplt
import get_functions as gt
import matplotlib.image as mpimg
import os
from scipy.optimize import curve_fit
import glob
import seaborn as sns

sns.set_style("ticks", {"xtick.direction" : u"in", "ytick.direction" : u"in"})

os.listdir('.')


# Read the configuration file.
yaml_files = glob.glob('*.yml')
try:
    assert(len(yaml_files) == 1)
except AssertionError:
    raise RuntimeError('Only one yaml file must be present in the current directory.')

variables = yaml.load(open(yaml_files[0], 'r'))


bd = gt.big_dictionnary()


name = 'Lam3_V12_001'

plt.plot(bd[name]['time'], bd[name]['force'], label='force')

area = gt.get_penetrated_area(bd[name]['position'], bd[name]['boundaries'], variables)
area /= 700

plt.plot(bd[name]['time'], area, label='area')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Force et aire (sans dimension)')
plt.savefig('truc.png')


