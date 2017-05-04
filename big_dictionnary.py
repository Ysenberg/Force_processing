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
import glob

os.listdir('.')


# D'abord le fichier yml qui passe dans le coin. Attention, ici si il y en a deux on est mal
fichier_yml = open(glob.glob('*.yml')[0], 'r')
variables = yaml.load(fichier_yml)
bd = {}
	

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

#print(len(bd)) # donne la longueur du dictionnaire (et donc le nombre de fichiers)
#print(bd.keys()) # donne les clés du dictionnaire (et donc les noms de fichiers sans le .txt)


def plot_for_each_file(bd):

	"""
    Save plots involving only one file each
    
    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.
    
    Returns
    -------
    Nothing, but saves png
    
    """

	for name in bd.keys():
		# vérification
		myplt.plot_boundaries(bd[name]['time'],bd[name]['force'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
		# la force en fonction de la position 
		myplt.plot_diff_regions_f_vs_p(bd[name]['position'],bd[name]['force'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
		# trace les subplots de force 
		myplt.plot_f_vs_t_subplots(bd[name]['time'],bd[name]['force'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
		# trace la contrainte en fonction du temps 
		myplt.plot_s_vs_t(bd[name]['time'],bd[name]['stress'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
		# trace la contrainte en fonction de la position 
		myplt.plot_s_vs_p(bd[name]['position'],bd[name]['stress'],bd[name]['boundaries'],bd[name]['file_splitted_name'])


def plot_all_velocities(bd):
	"""
    Save plots involving all files in *txt in the folder
    
    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.
    
    Returns
    -------
    Nothing, but saves png
    
    """

	for name in bd.keys():
		plt.plot(bd[name]['time'],bd[name]['force'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
		plt.legend()

	plt.xlabel('Time (ms)')
	plt.ylabel('Force (mN)')
	plt.minorticks_on()
	plt.title('Force_vs_Time' + bd[name]['file_splitted_name'][0])
	name_file = ''.join(['Force_vs_time_varV_', bd[name]['file_splitted_name'][0], '.png'])
	plt.savefig(name_file, bbox_inches='tight')
	plt.close()

	for name in bd.keys():
		plt.plot(bd[name]['position'],bd[name]['force'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
		plt.legend()

	plt.xlabel('Position (mm)')
	plt.ylabel('Force (mN)')
	plt.minorticks_on()
	plt.title('Force_vs_Position' + bd[name]['file_splitted_name'][0])
	name_file = ''.join(['Position_vs_force_varV_', bd[name]['file_splitted_name'][0], '.png'])
	plt.savefig(name_file, bbox_inches='tight')
	plt.close()

	for name in bd.keys():
		plt.plot(bd[name]['time'],bd[name]['stress'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
		plt.legend()

	plt.xlabel('Time (ms)')
	plt.ylabel('Stress (mN/mm²)')
	plt.minorticks_on()
	plt.title('Stress_vs_time' + bd[name]['file_splitted_name'][0])
	name_file = ''.join(['Stress_vs_time_varV', bd[name]['file_splitted_name'][0], '.png'])
	plt.savefig(name_file, bbox_inches='tight')
	plt.close()


	for name in bd.keys():
		plt.plot(bd[name]['position'],bd[name]['stress'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
		plt.legend()

	plt.xlabel('Position (mm)')
	plt.ylabel('Stress (mN/mm²)')
	plt.minorticks_on()
	plt.title('Stress_vs_position' + bd[name]['file_splitted_name'][0])
	name_file = ''.join(['Stress_vs_position_varV', bd[name]['file_splitted_name'][0], '.png'])
	plt.savefig(name_file, bbox_inches='tight')
	plt.close()

#penetration_slope = []
#speed_in_files = []

#for name in bd.keys():
#	penetration_slope += [bd[name]['fit_penetration'][0]]
#	speed_in_files += [bd[name]['speed']]

#	plt.plot(speed_in_files, penetration_slope, marker='o', linestyle='', ms ='4', label=bd[name]['file_splitted_name'][0])
#	plt.legend()
#plt.xlabel('Speed (mm/ms)')
#plt.ylabel('Slope_penetration (mN/ms)')
#plt.title('Slope_penetration_vs_speed' + Lam + '.png')
#plt.savefig('Slope_penetration_vs_speed_' + Lam + '.png')
#plt.close()

### Tracer le stress moyen pendant la pénétration pour toutes les vitesses d'une même lame 

names_by_plate = gt.get_names_files_same_plate(bd)

for lame in names_by_plate.keys():
	print(lame)
	speed = []
	mean_stress = []
	std_stress = []
	'stress_' + lame 
	for name in names_by_plate[lame]:
		speed += [bd[name]['speed']]
		mean_stress += [np.asarray(bd[name]['stress'][bd[name]['boundaries']['contact_to_penetration']+1:bd[name]['boundaries']['penetration_to_relaxation']]).mean()]
		std_stress += [np.std(bd[name]['stress'][bd[name]['boundaries']['contact_to_penetration']+1:bd[name]['boundaries']['penetration_to_relaxation']])]

	plt.plot(speed,mean_stress,label=bd[name]['file_splitted_name'][0], linestyle='', marker='o')
	plt.legend()

plt.title('Mean_stress_while_penetration_vs_speed_' )
plt.savefig('Mean_stress_while_penetration_vs_speed_' + '.png')
plt.close()


		


	




