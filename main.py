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
import pylab as pl
from matplotlib import rc
import decimal

# Annoyingly long part for graphics settings.

plt.rc('text', usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rcParams["figure.figsize"] = [12.8,8.8]
sns.set(context='poster', style ='ticks', font_scale=2.5, font = 'Symbola', rc = {"xtick.direction" : u"in", "ytick.direction" : u"in"})
sns.set_palette(['forestgreen', 'LightSeaGreen',  'Aquamarine', 'MediumSlateBlue', 'MidnightBlue', 'RosyBrown', 'Brown', 'Chocolate', 'goldenrod', 'olive', 'yellowgreen', 'gold', 'darkorange', 'orangered', 'crimson'])

# ------------------- Read the configuration file

os.listdir('.')
yaml_file = glob.glob('*.yml')
try:
    assert(len(yaml_file) == 1)
except AssertionError:
    raise RuntimeError('Only one yaml file must be present in the current directory.')
variables = yaml.load(open(yaml_file[0], 'r'))

# ------------------- Real work

bd = gt.big_dictionnary()
