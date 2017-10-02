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
import matplotlib.font_manager
import yaml
import re
from yaml import load, dump
import pandas as pd
import plot_functions as myplt
import get_functions as gt
import matplotlib.image as mpimg
import os
from scipy.optimize import curve_fit
import scipy.optimize as spopt
import glob
import seaborn as sns
import lineregress as myregress
import pylab as pl
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# print(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))


sns.set(context='paper', style ='ticks', font_scale=2.5, font = 'Symbola', rc = {"xtick.direction" : u"in", "ytick.direction" : u"in"})

sns.set_palette(['forestgreen', 'LightSeaGreen',  'Aquamarine', 'MediumSlateBlue', 'MidnightBlue', 'RosyBrown', 'Brown', 'Chocolate', 'goldenrod', 'olive', 'yellowgreen', 'gold', 'darkorange', 'orangered', 'crimson'])
os.listdir('.')

#current_palette = sns.color_palette()


# Read the configuration file.
yaml_files = glob.glob('*.yml')
try:
    assert(len(yaml_files) == 1)
except AssertionError:
    raise RuntimeError('Only one yaml file must be present in the current directory.')

variables = yaml.load(open(yaml_files[0], 'r'))

bd = gt.big_dictionnary()

gt.full_stress_processing_relaxation(bd)






"""
----------------
Pour faire les figures du rapport sur la partie extraction
----------------



name = 'Lam2_V12_001'
bound1 = bd[name]['boundaries']['relaxation_to_elastic']
bound2 = bd[name]['boundaries']['elastic_to_fluidized']
bound3 = bd[name]['boundaries']['fluidized_to_meniscus']
bound4 = bd[name]['boundaries']['meniscus_to_breakage']
def take_time(bound, bound2):
	bound1 = bd[name]['boundaries']['relaxation_to_elastic']
	return bd[name]['time'][bound:bound2] - bd[name]['time'][bound1]


plt.plot(take_time(bound1,bound2), bd[name]['force'][bound1:bound2], linestyle='', marker='o', ms = 4, color='mediumaquamarine')
plt.plot(take_time(bound2,bound3), bd[name]['force'][bound2:bound3], linestyle='', marker='o', ms = 4, color='cornflowerblue')
plt.plot(take_time(bound3,bound4), bd[name]['force'][bound3:bound4], linestyle='', marker='o', ms = 4, color='mediumslateblue')
plt.xlabel(r'$t $(s)')
plt.ylabel(r'$ F $ (mN)')
plt.savefig('Retire.svg')
plt.close()



plt.plot(take_time(bound1,bound2), bd[name]['stress'][bound1:bound2], linestyle='', marker='o', ms = 4, color='mediumaquamarine')
plt.plot(take_time(bound2,bound3), bd[name]['stress'][bound2:bound3], linestyle='', marker='o', ms = 4, color='cornflowerblue')
plt.plot(take_time(bound3,bound4), bd[name]['stress'][bound3:bound4], linestyle='', marker='o', ms = 4, color='cornflowerblue')
plt.xlabel(r'$t $(s)')
plt.ylabel(r'$ \tau_p $ (Pa)')
plt.savefig('stress_retire.svg')
plt.close()
"""
