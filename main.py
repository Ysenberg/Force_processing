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


names_by_plate = gt.get_names_files_same_plate(bd)



myplt.plot_relaxation_s_vs_t(bd)








"""

for lame, _ in sorted(names_by_plate.items()):
	for name in sorted(names_by_plate[lame]):
		bound1 = bd[name]['boundaries']['arg_force_max']
		bound2 = bd[name]['boundaries']['relaxation_to_elastic']
		time = bd[name]['time'][bound1:bound2] - bd[name]['time'][bound1]

		normalisee_force = bd[name]['force'][bound1:bound2]
		normalisee_force /= bd[name]['force'][bound1+1]

		plt.plot(time, normalisee_force, label=bd[name]['file_splitted_name'][1], linestyle='', marker='o', ms = 3)
	plt.legend(title='Vitesse (mm/s) :', prop={'fontsize':6})
	plt.xscale("log")
	plt.yscale("log")
	plt.xlabel('T (s)')
	plt.ylabel('F/Fo')
	plt.savefig( 'Force_vs_time_during_relaxation_' + bd[name]['file_splitted_name'][0] + '.pdf')
	plt.close()



print(sorted(names_by_plate))

"""

