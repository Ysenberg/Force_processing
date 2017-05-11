#------------------
# Ici, on crée un énorme dictionnaire qui contient toutes les données des fichiers du dossier où on est
#------------------

import numpy as np
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d
from scipy import constants
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
bd = {}

# remplit le dictionnaire bd avec toutes les infos qu'on peut trouver sur les fichiers présents dans le même dossier que ce .py

for filename in glob.glob('*.txt'):

	file_splitted_name = os.path.splitext(filename)[0].split('_') # Les fichiers seront nommés Lam*_V*_00*.txt
	Lam = file_splitted_name[0]
	V = file_splitted_name[1]
	Nb = file_splitted_name[2]
	print(filename) #juste pour voir si tout va bien. A retirer ensuite
	data = np.loadtxt(filename, unpack=True) # charge les données du fichier de la boucle en cours
	time = data[0]
	position = data[1]
	force = data[2]



	#Trouver la vitesse associée au fichier
	speed = int(re.findall(r'\d+', V)[0])
	#Trouver la lame associée au fichier
	plate = int(re.findall(r'\d+', Lam)[0])


	boundaries = gt.get_boundaries(time, position, force, variables)

	# Mettre la force à zéro au début
	dist_init = variables['dist_plate_to_foam']
	number_first_points = (int(float(((len(np.nonzero((position - dist_init)<0)[0]))/2))))
	zero_load = force[:number_first_points].mean()
	force -= zero_load

	# Calcul du stress
	stress = gt.get_stress(boundaries, force, position, variables)




	# Le fit de la partie pénétration !
	x = time[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']]
	y = position[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']]
	fit_penetration = np.polyfit(x,y,1) # cette fonction met les paramètres du fit par ordre d'exposant décroissant (la pente est dont fit[0])

	bd.update({ '_'.join([Lam, V, Nb]) :
	{'time' : time, # ----------------------------------------- array
	'position': position, # ----------------------------------- array
	'force' : force, # ---------------------------------------- array
	'boundaries' : boundaries, # ------------------------------ dictionnary
	'stress' : stress, # -------------------------------------- array
	'file_splitted_name' : file_splitted_name, # -------------- list of strings
	'fit_penetration' : fit_penetration, # -------------------- list of floats
	'speed' : speed, # ---------------------------------------- int
	'plate' : plate # ----------------------------------------- int
	}
	}
	)



# def func(x, a, b, c, d, f):
#     return a*np.exp(-b*x) + c*np.exp(-d*x) + f

# x = bd['Lam1_V10_001']['force'][bd['Lam1_V10_001']['boundaries']['penetration_to_relaxation']:bd['Lam1_V10_001']['boundaries']['relaxation_to_elastic']]
# y = bd['Lam1_V10_001']['time'][bd['Lam1_V10_001']['boundaries']['penetration_to_relaxation']:
# bd['Lam1_V10_001']['boundaries']['relaxation_to_elastic']] - bd['Lam1_V10_001']['time'][bd['Lam1_V10_001']['boundaries']['penetration_to_relaxation']]

# popt, pcov = curve_fit(func, x, y)
# print(popt)

# plt.plot(x,y, marker='o', linestyle='')
# plt.plot(x, func(x, *popt), '-', label = 'fit_dbl_exp')
# plt.legend()
# plt.savefig('fit_double_exponentielle.png')


# def func(x, a, b, c):
#     return a*np.exp(-b*x) + c

# x0 = np.array([50000 , 3, 300])

# x = np.asarray(bd['Lam1_V10_001']['force'][bd['Lam1_V10_001']['boundaries']['penetration_to_relaxation']:bd['Lam1_V10_001']['boundaries']['relaxation_to_elastic']-11])
# y = np.asarray(bd['Lam1_V10_001']['time'][bd['Lam1_V10_001']['boundaries']['penetration_to_relaxation']:
# bd['Lam1_V10_001']['boundaries']['relaxation_to_elastic']-11] - bd['Lam1_V10_001']['time'][bd['Lam1_V10_001']['boundaries']['penetration_to_relaxation']])

# popt, pcov = curve_fit(func, x, y, x0)
# print(popt)

# plt.plot(x,y, marker='o', linestyle='')
# plt.plot(x, func(x, *popt), '-', label = 'fit_dbl_exp')
# plt.legend()
# plt.savefig('fit_double_exponentielle_debut.png')

# x0 = np.array([1, 1, 1 ])


# x = np.asarray(bd['Lam1_V10_001']['force'][(bd['Lam1_V10_001']['boundaries']['relaxation_to_elastic']-11):bd['Lam1_V10_001']['boundaries']['relaxation_to_elastic']])
# y = np.asarray(bd['Lam1_V10_001']['time'][(bd['Lam1_V10_001']['boundaries']['relaxation_to_elastic']-11):
# bd['Lam1_V10_001']['boundaries']['relaxation_to_elastic']] - bd['Lam1_V10_001']['time'](bd['Lam1_V10_001']['boundaries']['relaxation_to_elastic']-11)


# plt.plot(x,y, marker='o', linestyle='')
# plt.show()

# popt, pcov = curve_fit(func, x, y, x0)
# print(popt)

# plt.plot(x,y, marker='o', linestyle='')
# plt.plot(x, func(x, *popt), '-', label = 'fit_dbl_exp')
# plt.legend()
# plt.savefig('fit_double_exponentielle_fin.png')




myplt.plot_for_each_file(bd)




