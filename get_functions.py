import numpy as np
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d
from scipy import constants
import scipy as sp
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from itertools import compress
import glob
import yaml
import re
import os
from scipy.signal import savgol_filter
from yaml import load, dump
import pandas as pd
import plot_functions as myplt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
import scipy.optimize as spopt
import glob
import seaborn as sns
import lineregress as myregress
import pylab as pl
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

def get_boundaries(time, position, force, variables):
    """
    Return the boundaries of each regime.

    Parameters
    ----------
    time : array
        Time.
    position : array
        Positions.
    force : array
        Force.
    variables : dictionnary
    	Contains the yaml informations

    Returns
    -------
    dictionnary

    """
    dist_init = variables['dist_plate_to_foam']
    number_first_points = (int(float(((len(np.nonzero((position - dist_init)<0)[0]))/2))))
    number_last_points = int(float(number_first_points/2))

    # Approach to contact
    # Calculate the zero load based on the first points
    # and quantify how much we deviate from that.
    threshold_force_in = np.std(force[:number_last_points])
    zero_load = force[:number_first_points].mean()
    far_from_threshold = np.abs((force - zero_load) / force) > threshold_force_in
    arg_approach_to_contact = far_from_threshold.nonzero()[0][0]

    # Max of force
    arg_force_max = force.argmax()


    # # Contact to penetration
    # far_from_threshold = (force - zero_load) / force > threshold_force_in
    # arg_contact_to_penetration = far_from_threshold.nonzero()[0][0]
    #
    test_force = force[arg_approach_to_contact:] - zero_load
    test_force[test_force < 0] = 0
    arg_contact_to_penetration = test_force.nonzero()[0][0] + arg_approach_to_contact
    assert(arg_approach_to_contact <= arg_contact_to_penetration)



    # Penetration to relaxation
    # End of motion
    args_position_plateau = np.nonzero(position - position.max() >= 0)
    arg_penetration_to_relaxation = args_position_plateau[0][0]

    # Relaxation to elastic regime
    arg_relaxation_to_elastic = args_position_plateau[0][-1]

    # Elastic to fluidized regime
    arg_elastic_to_fluidized = force.argmin()

    # Fluidized to meniscus
    pos_contact_with_foam = position[arg_approach_to_contact]
    length_in_foam = (position[arg_approach_to_contact:] - pos_contact_with_foam)
    arg_fluidized_to_meniscus = np.nonzero(length_in_foam < 0)[0][0] + arg_approach_to_contact

    # Meniscus to breakage
    inv_force = force[::-1]
    threshold_force_out = np.std(inv_force[:number_last_points])
    end_load = inv_force[:number_last_points].mean()
    far_from_threshold = np.abs((inv_force - end_load) / inv_force) > threshold_force_out
    arg_meniscus_to_breakage = force.shape[0] - far_from_threshold.nonzero()[0][0] - 1

    return {'approach_to_contact': arg_approach_to_contact,
            'contact_to_penetration' : arg_contact_to_penetration,
            'penetration_to_relaxation' : arg_penetration_to_relaxation,
            'relaxation_to_elastic' : arg_relaxation_to_elastic,
            'elastic_to_fluidized' :  arg_elastic_to_fluidized,
            'fluidized_to_meniscus' : arg_fluidized_to_meniscus,
            'meniscus_to_breakage' : arg_meniscus_to_breakage,
            'arg_force_max' : arg_force_max
           }

def get_penetrated_area(position, boundaries, variables):
    """
    Return the penetrated area.

    The value is set to zero if the plate is above the foam
    interface.

    Parameters
    ----------
    position : array
        Array of plate positions.
    boundaries : dictionnary
        Dictionnary containing limits btw regimes
    varaibles : dictionnary
        Informations from the yaml file.

    Returns
    -------
    area : array
    """
    area = (position - position[boundaries['contact_to_penetration']]) * variables['plate_width']
    area[area < 0] = 0
    area = area *2
    return area

def get_stress(boundaries, force, position, variables):
    """
    Calculate stress by dividing force by the penetrated area

    Parameters
    ----------
    boundaries : dictionnary
        Limits btw regions
    force : array
        Force.
    position : array
        Position.
    variables : dictionnary
        Contains informations about the experiment.

    Returns
    -------
    array

    """
    penetrated_area = get_penetrated_area(position, boundaries, variables)
    arg_first_nonzero_area = penetrated_area.nonzero()[0][0]
    inv_area = penetrated_area[::-1]
    arg_last_nonzero_area = len(penetrated_area) - inv_area.nonzero()[0][0] - 1



    stress = [0]*(arg_first_nonzero_area)
    stress.extend(force[arg_first_nonzero_area:arg_last_nonzero_area])
    stress.extend([0]*(len(position) - arg_last_nonzero_area))

    stress[arg_first_nonzero_area:arg_last_nonzero_area] /= penetrated_area[arg_first_nonzero_area:arg_last_nonzero_area]

    return stress

def get_names_files_same_plate(bd):
    """
    Returns one dictionnary containing lists of names of the files for a same plate

    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.

    Returns
    -------
    dict_names_plates : dictionnary
        Dictionnary containing all names of files ordered by plate number

    """

    # ressort un dictionnaire contenant des listes de noms, les clés sont 'names_Lam1', par exemple. Permet d'accéder à toutes les données pour une même lame.
    list_of_plates = []
    dict_names_plates = {}
    for name in bd.keys():
        if bd[name]['file_splitted_name'][0] in list_of_plates :
            dict_names_plates['names_' + bd[name]['file_splitted_name'][0]] += [name]
        else :
            list_of_plates += [bd[name]['file_splitted_name'][0]]
            list_name = [name]
            dict_names_plates.update({
            'names_' + bd[name]['file_splitted_name'][0] : list_name
            })

    return dict_names_plates

def get_names_files_same_speed(bd):
    """
    Returns one dictionnary containing lists of names of the files for a same speed

    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.

    Returns
    -------
    dict_names_plates : dictionnary
        Dictionnary containing all names of files ordered by speed

    """

    # ressort un dictionnaire contenant des listes de noms, les clés sont 'names_V30', par exemple. Permet d'accéder à toutes les données pour une même vitesse.
    list_of_speed = []
    dict_names_speed = {}
    for name in bd.keys():
        if bd[name]['file_splitted_name'][1] in list_of_plates :
            print('Im here')
            dict_names_speed['names_' + bd[name]['file_splitted_name'][1]] += [name]
        else :
            list_of_speed += [bd[name]['file_splitted_name'][1]]
            list_name = [name]
            dict_names_speed.update({
            'names_' + bd[name]['file_splitted_name'][1] : list_name
            })

    return dict_names_speed

def big_dictionnary():
    """
    Returns one dictionnary containing informations about files in the current folder

    Parameters
    ----------

    Returns
    -------
    bd : dictionnary
        Dictionnary containing :
            - time : array
                Contains the first column of data files in seconds
            - position : array
                Contains the second column of data files in mm
            - force : array
                Contains the third column of data files in mN
            - boundaries : array
                Values of the delimitaions between regions
            - stress : array
                Contains the force divided by the immerged area of the plate in pascals
            - file_splitted_name : list of strings
                Contains strings from the file name
            - fit penetration :
                Contains parameters (decreasing exponent) of a linear fit of the penetration regime
            - speed : float
                Speed of the plate, extracted from the file name
            - plate : int
                Contains the int corresponding to the studied roughness, extracted from the file name
            - delta : float
                Distance correspondant au régime de réponse élastique

    """

    bd = {}
    yaml_files = glob.glob('*.yml')
    try:
        assert(len(yaml_files) == 1)
    except AssertionError:
        raise RuntimeError('Only one yaml file must be present in the current directory.')

    variables = yaml.load(open(yaml_files[0], 'r'))

    for filename in glob.glob('*.txt'):

        file_splitted_name = os.path.splitext(filename)[0].split('_') # Les fichiers seront nommés Lam*_V*_00*.txt
        Lam = file_splitted_name[0]
        V = file_splitted_name[1]
        Nb = file_splitted_name[2]
        #print(filename) #juste pour voir si tout va bien. A retirer ensuite
        data = np.loadtxt(filename, unpack=True) # charge les données du fichier de la boucle en cours
        time = data[0]
        time /= 1000
        position = data[1]
        force = data[2]



        #Trouver la vitesse associée au fichier
        speed = int(re.findall(r'\d+', V)[0])
        #Trouver la lame associée au fichier
        plate = int(re.findall(r'\d+', Lam)[0])


        boundaries = get_boundaries(time, position, force, variables)

        # Mettre la force à zéro au début
        dist_init = variables['dist_plate_to_foam']
        number_first_points = (int(float(((len(np.nonzero((position - dist_init)<0)[0]))/2))))
        zero_load = force[:number_first_points].mean()
        force -= zero_load

        # Calcul du stress
        stress = get_stress(boundaries, force, position, variables)
        stress = np.asarray(stress) *1000 # pour passer en pascals


        # Le fit de la partie pénétration !
        x = time[boundaries['contact_to_penetration']:boundaries['arg_force_max']] # temps en secondes
        y = position[boundaries['contact_to_penetration']:boundaries['arg_force_max']] # position en mm
        fit_penetration = sp.stats.linregress(x, y) # cette fonction met les paramètres du fit par ordre d'exposant décroissant (la pente est donc fit[0])

        # Le fit de la partie fluidized !
        z = time[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']] # temps en secondes
        w = position[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']] # position en mm
        fit_meniscus = sp.stats.linregress(z, w) # cette fonction met les paramètres du fit par ordre d'exposant décroissant (la pente est donc fit[0])
        """ Renvoie un tuple avec 5 valeurs :
         - la pente
         - l'ordonnée à l'origine
         - le coefficient de corrélation
         - la p-value
         - l'erreur standard de l'estimation
        """

        # Retourne delta, la distance parcourue dans la mousse entre la fin de la relaxation et le min de force.
        delta = abs(position[boundaries['elastic_to_fluidized']] - position[boundaries['relaxation_to_elastic']])




        bd.update({ '_'.join([Lam, V, Nb]) :
        {'time' : time, # ----------------------------------------- array
        'position': position, # ----------------------------------- array
        'force' : force, # ---------------------------------------- array
        'boundaries' : boundaries, # ------------------------------ dictionnary
        'stress' : stress, # -------------------------------------- array
        'file_splitted_name' : file_splitted_name, # -------------- list of strings
        'fit_penetration' : fit_penetration, # -------------------- list of floats
        'fit_meniscus' : fit_meniscus, # -------------------------- list of floats
        'speed' : speed, # ---------------------------------------- int
        'plate' : plate, # ---------------------------------------- int
        'delta' : delta # ----------------------------------------- float

        }
        }
        )

    return bd

def get_mean_stress_penetration(bd):

    """
    Returns one dictionnary containing informations about mean stress during the penetration regime

    Parameters
    ----------
    bd : dictionnary

    Returns
    -------
    mean_stress : dictionnary
        Contains :
         - speed : array
            Speed at which the plate is moving
         - mean_stress_penetration : array
            Mean of stress during penetration regime
         - std_stress_penetration : array
            Standard deviation regarding the mean stress calculated before

    """

    names_by_plate = get_names_files_same_plate(bd)

    mean_stress = {}

    for lame, _ in sorted(names_by_plate.items()):
        speed = []
        mean_stress_penetration = []
        std_stress_penetration = []
        for name in sorted(names_by_plate[lame]):
            speed += [bd[name]['speed']]
            mean_stress_penetration += [np.asarray(bd[name]['stress'][bd[name]['boundaries']['contact_to_penetration']+10:bd[name]['boundaries']['arg_force_max']]).mean()]
            std_stress_penetration += [np.std(bd[name]['stress'][bd[name]['boundaries']['contact_to_penetration']+10:bd[name]['boundaries']['arg_force_max']])]

        mean_stress.update({lame :
            {'speed' : speed ,
            'mean_stress_penetration' : mean_stress_penetration,
            'std_stress_penetration' : std_stress_penetration
            }})

    return mean_stress

def get_mean_stress_relaxation(bd):

    """
    Returns one dictionnary containing informations about mean stress during the penetration regime

    Parameters
    ----------
    bd : dictionnary

    Returns
    -------
    mean_stress : dictionnary
        Contains :
         - speed : array
            Speed at which the plate is moving
         - mean_stress_penetration : array
            Mean of stress during penetration regime
         - std_stress_penetration : array
            Standard deviation regarding the mean stress calculated before

    """

    names_by_plate = get_names_files_same_plate(bd)

    mean_stress_rel = {}


    for lame, _ in sorted(names_by_plate.items()):
        speed = []
        mean_stress = []
        std_stress = []
        for name in sorted(names_by_plate[lame]):
            bound1 = bd[name]['boundaries']['arg_force_max'] + 100
            bound2 = bd[name]['boundaries']['relaxation_to_elastic']
            speed += [int(bd[name]['speed'])]
            mean_stress += [np.asarray(bd[name]['stress'][bound1:bound2]).mean()]
            std_stress += [np.std(bd[name]['stress'][bound1:bound2])]

        mean_stress_rel.update({lame :
            {'speed' : speed ,
            'mean_stress' : mean_stress,
            'std_stress' : std_stress
            }})

    return mean_stress_rel

def get_dict_plates():
    """
    Returns one dictionnary containing informations about roughness of the plates (beads radius in mm)

    Parameters
    ----------

    Returns
    -------
    dict_plates : dictionnary

    """

    dict_plates = {'0' : 0 , '1' : 0.03, '2' : 0.04, '3' : 0.08, '4' : 0.13, '5' : 0.22}

    return dict_plates

def full_stress_processing_penetration(bd):
    """
    Plots :
     - all mean_stress vs speed files with fitting line and error bars
     - intersection values of the preceding plots
     - gradient values of the plots

    Parameters
    ----------

    Returns
    -------

    """

    mean_stress = get_mean_stress_penetration(bd)

    dict_plates = get_dict_plates()

    names_by_plate = get_names_files_same_plate(bd)
    lames = []
    list_pentes = []
    list_ord_origine = []
    err_pente = []
    err_origine = []

    for lame, _ in sorted(names_by_plate.items()):


        x = np.asarray(mean_stress[lame]['speed'])
        y = np.asarray(mean_stress[lame]['mean_stress_penetration'])
        y_err = np.asarray(mean_stress[lame]['std_stress_penetration'])
        def fit_funct(x, pente, ord_origine):
            return pente*x + ord_origine
        params = curve_fit(fit_funct, x, y)
        [pente, ord_origine] = params[0]
        [std_pente, std_origine] = np.sqrt(np.diag(params[1]))
        x_fitted = [0] + mean_stress[lame]['speed']
        truc = list(pente * x + ord_origine)
        y_fitted = [ord_origine] + truc

        list_pentes += [pente]
        list_ord_origine += [ord_origine]
        err_pente += [std_pente]
        lame_number = int(re.findall(r'\d+', lame)[0])
        lames += [dict_plates[str(lame_number)]]
        err_origine += [std_origine]



        plt.plot(x, y, linestyle='', marker='o', ms = 4) # maintenant x est Ca !!
        plt.errorbar(x, y, yerr=y_err, fmt='', marker='', linestyle='none', elinewidth=0.5, capthick=0.5, capsize=1, color='gray')
        plt.plot(x_fitted, y_fitted, linestyle='-', marker='', linewidth=0.5, color = 'black')
        #plt.plot(x, funct(x), linestyle='-', marker='', ms = 4, color='black')
        #plt.xlabel(r'Ca  ')
        plt.xlabel(r'$V$ (mm/s)')
        plt.ylabel(r'$ \bar{ \tau_p } $ (Pa)')
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        #plt.xscale("log")
        #plt.yscale("log")
        plt.tight_layout()
        plt.savefig('Mean_stress_while_penetration_vs_speed_' + lame + '.svg')
        plt.close()

    plt.plot(lames, list_pentes, linestyle='', marker='o', ms = 4)
    plt.errorbar(lames, list_pentes, yerr=err_pente, fmt='', marker='', linestyle='none', elinewidth=0.5, capthick=0.5, capsize=1, color='gray')
    plt.xlabel(r'$a$ (mm)')
    plt.ylabel(r'$\kappa$ (Pa.s/mm)')
    plt.ticklabel_format(style='sci', axis='both')
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig('Pente_contrainte_moyenne' + '.svg')
    plt.close()

    plt.plot(lames, list_ord_origine, linestyle='', marker='o', ms = 4)
    plt.errorbar(lames, list_ord_origine, yerr=err_origine, fmt='', marker='', linestyle='none', elinewidth=0.5, capthick=0.5, capsize=1, color='gray')
    plt.ylim(ymin=0, ymax=2.5)
    plt.xlabel(r'$a$ (mm)')
    plt.ylabel(r'$\tau_0 $ (Pa)')
    plt.tight_layout()
    plt.savefig('Origine_contrainte_moyenne' + '.svg')
    plt.close()


def full_stress_processing_relaxation(bd):
    """
    Plots :
     - all mean_stress vs speed files with fitting line and error bars
     - intersection values of the preceding plots
     - gradient values of the plots

    Parameters
    ----------

    Returns
    -------

    """

    mean_stress = get_mean_stress_relaxation(bd)

    dict_plates = get_dict_plates()

    names_by_plate = get_names_files_same_plate(bd)
    lames = []
    list_pentes = []
    list_ord_origine = []
    err_pente = []
    err_origine = []
    err_tau_m = []
    tau_m = []

    for lame, _ in sorted(names_by_plate.items()):


        x = np.asarray(mean_stress[lame]['speed'])
        y = np.asarray(mean_stress[lame]['mean_stress'])
        y_err = np.asarray(mean_stress[lame]['std_stress'])
        """
        def fit_funct(x, pente, ord_origine):
            return pente*x + ord_origine
        params = curve_fit(fit_funct, x, y)
        [pente, ord_origine] = params[0]
        [std_pente, std_origine] = np.sqrt(np.diag(params[1]))
        x_fitted = [0] + mean_stress[lame]['speed']
        y_fitted = [ord_origine] + list(pente * x + ord_origine)
        list_pentes += [pente]
        list_ord_origine += [ord_origine]
        err_pente += [std_pente]
        err_origine += [std_origine]
        """
        lame_number = int(re.findall(r'\d+', lame)[0])
        lames += [dict_plates[str(lame_number)]]
        tau_m += [np.mean(y)]
        tau_m_now = np.mean(y)
        err_tau_m += [np.std(y)]
        plt.plot(x, y, linestyle='', marker='o', ms = 4)
        plt.errorbar(x, y, yerr=y_err, fmt='', marker='', linestyle='none', elinewidth=0.5, capthick=0.5, capsize=1, color='gray')
        plt.plot([0,32], [tau_m_now, tau_m_now], linestyle='--', marker='', lw=1.5, color='black')
        plt.xlim(xmin=0)
        plt.ylim(ymin=0, ymax=1.2)
        plt.xlabel(r'$V$ (mm/s)')
        plt.ylabel(r'$ \bar{\tau_p} $ (Pa)')
        plt.tight_layout()
        plt.savefig('Mean_stress_while_relaxation_vs_speed_' + lame + '.svg')
        plt.close()
"""
    plt.plot(lames, list_pentes, linestyle='', marker='o', ms = 4)
    plt.errorbar(lames, list_pentes, yerr=err_pente, fmt='', marker='', linestyle='none', elinewidth=0.5, capthick=0.5, capsize=1, color='gray')
    plt.xlabel(r'$a$ (mm)')
    plt.ylabel(r'$\kappa$ (Pa.s/mm)')
    plt.tight_layout()
    plt.savefig('Pente_contrainte_moyenne_rel' + '.svg')
    plt.close()

    err_origine = np.asarray(err_origine) / 4

    plt.plot(lames, list_ord_origine, linestyle='', marker='o', ms = 4)
    #plt.errorbar(lames, list_ord_origine, yerr=err_origine, fmt='', marker='', linestyle='none', elinewidth=0.5, capthick=0.5, capsize=1, color='gray')
    plt.xlabel(r'$a$ (mm)')
    plt.ylabel(r'$\tau_0 $ (Pa)')
    plt.tight_layout()
    plt.savefig('Origine_contrainte_moyenne_rel' + '.svg')
    plt.close()

    plt.plot(lames, tau_m, linestyle='', marker='o', ms = 4)
    plt.errorbar(lames, tau_m, yerr=err_tau_m, fmt='', marker='', linestyle='none', elinewidth=0.5, capthick=0.5, capsize=1, color='gray')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=3.2)
    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel(r'$a$ (mm)')
    plt.ylabel(r'$\tau_m $ (Pa)')
    plt.tight_layout()
    plt.savefig('tau_m_relaxation' + '.svg')
    plt.close()

    return lames, tau_m, err_tau_m

    """

def get_integrale_extraction(bd):
    """
    Parameters
    bd : big dictionnary containing all informations about the data files
    -------

    Returns
    -------

    """
