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
import plot_functions as myplt
import get_functions as gt
import matplotlib.image as mpimg
import os
from scipy.optimize import curve_fit
import glob

os.listdir('.')


# Read the configuration file.
yaml_files = glob.glob('*.yml')
try:
    assert(len(yaml_files) == 1)
except AssertionError:
    raise RuntimeError('Only one yaml file must be present in the current directory.')

variables = yaml.load(open(yaml_files[0], 'r'))
# bd = {}

# # remplit le dictionnaire bd avec toutes les infos qu'on peut trouver sur les fichiers présents dans le même dossier que ce .py

# for filename in glob.glob('*.txt'):

# 	file_splitted_name = os.path.splitext(filename)[0].split('_') # Les fichiers seront nommés Lam*_V*_00*.txt
# 	Lam = file_splitted_name[0]
# 	V = file_splitted_name[1]
# 	Nb = file_splitted_name[2]
# 	print(filename) #juste pour voir si tout va bien. A retirer ensuite
# 	data = np.loadtxt(filename, unpack=True) # charge les données du fichier de la boucle en cours
# 	time = data[0]
# 	position = data[1]
# 	force = data[2]



# 	#Trouver la vitesse associée au fichier
# 	speed = int(re.findall(r'\d+', V)[0])
# 	#Trouver la lame associée au fichier
# 	plate = int(re.findall(r'\d+', Lam)[0])


# 	boundaries = gt.get_boundaries(time, position, force, variables)

# 	# Mettre la force à zéro au début
# 	dist_init = variables['dist_plate_to_foam']
# 	number_first_points = (int(float(((len(np.nonzero((position - dist_init)<0)[0]))/2))))
# 	zero_load = force[:number_first_points].mean()
# 	force -= zero_load

# 	# Calcul du stress
# 	stress = gt.get_stress(boundaries, force, position, variables)




# 	# Le fit de la partie pénétration !
# 	x = time[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']]
# 	y = position[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']]
# 	fit_penetration = np.polyfit(x,y,1) # cette fonction met les paramètres du fit par ordre d'exposant décroissant (la pente est dont fit[0])

# 	bd.update({ '_'.join([Lam, V, Nb]) :
# 	{'time' : time, # ----------------------------------------- array
# 	'position': position, # ----------------------------------- array
# 	'force' : force, # ---------------------------------------- array
# 	'boundaries' : boundaries, # ------------------------------ dictionnary
# 	'stress' : stress, # -------------------------------------- array
# 	'file_splitted_name' : file_splitted_name, # -------------- list of strings
# 	'fit_penetration' : fit_penetration, # -------------------- list of floats
# 	'speed' : speed, # ---------------------------------------- int
# 	'plate' : plate # ----------------------------------------- int
# 	}
# 	}
# 	)


# smoothe le signal de force en utilisant Savitzty-Golay filter. 
#Attention, ce filtre semble supposer que les points sont espacés du même temps. Si on a des choses abberrantes c'est certainement l'erreur. 

# for name in bd.keys():

# 	sf = savgol_filter(bd[name]['force'], 9, 5, mode='mirror')
# 	sf2 = savgol_filter(bd[name]['force'], 9, 5, deriv=1, mode='mirror')
# 	sf3 = savgol_filter(bd[name]['force'], 9, 5, deriv=2, mode='mirror')
# 	#plt.plot(bd[name]['time'], bd[name]['force'], label='data')
# 	#plt.plot(bd[name]['time'], sf, label = 'smoothed data')
# 	plt.plot(bd[name]['time'], sf2, label = 'derivative of smoothed data')
# 	plt.plot(bd[name]['time'], sf3, label = 'second derivative of smoothed data')
# 	plt.legend()
# 	plt.xlabel('Time (ms)')
# 	plt.ylabel('Force or its derivative')
# 	plt.title(name)
# 	plt.savefig(name + '_force_derivative.png')
# 	plt.close()


# for filename in glob.glob('*.txt'):

#         file_splitted_name = os.path.splitext(filename)[0].split('_') # Les fichiers seront nommés Lam*_V*_00*.txt
#         Lam = file_splitted_name[0]
#         V = file_splitted_name[1]
#         Nb = file_splitted_name[2]
#         data = np.loadtxt(filename, unpack=True) # charge les données du fichier de la boucle en cours
#         time = data[0]
#         time /= 1000
#         position = data[1]
#         force = data[2]

#         plt.plot(time, force, marker='', linestyle='-', ms=0.1, lw=0.2, label=V)
#         plt.legend()
#         plt.xlabel('Time(s)')
#         plt.ylabel('Force(mN)')
#         plt.savefig(Lam + V + Nb + '.png')

#         plt.close()

bd = gt.big_dictionnary()

myplt.plot_mean_stress_vs_speed(bd)




