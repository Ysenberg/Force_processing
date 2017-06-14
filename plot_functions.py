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
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})


""" 
------------------------------
Fonctions définies dans ce fichier : 
------------------------------
  
  - plot_diff_regions_f_vs_t   
  - plot_diff_regions_f_vs_p           
  - plot_f_vs_t_subplots                        
  - plot_s_vs_t 
  - plot_boundaries
  - plot_s_vs_p
  - plot_f_vs_t
  - plot_mean_stress_vs_speed_all_files
  - plot_all_velocities
  - plot_for_each_file
  - plot_force_derivative


"""

def plot_diff_regions_f_vs_t(time,force,boundaries,file_splitted_name):
    """
    Returns a plot of the different regimes
    Force versus time.
    
    Parameters
    ----------
    time : array
        Time.
    force : array
        Force.
    file_splitted_names : list of strings
        Name of the filed used now splitted in strings.
    
    Returns
    -------
    png file
    
    """
    plt.plot(time[:boundaries['approach_to_contact']],
             force[:boundaries['approach_to_contact']], 
             'sandybrown', marker='o', linestyle='', ms='3')

    plt.plot(time[boundaries['approach_to_contact']:boundaries['contact_to_penetration']],
             force[boundaries['approach_to_contact']:boundaries['contact_to_penetration']], 
             'salmon', marker='o', linestyle='', ms='3')

    plt.plot(time[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']],
             force[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']], 
             'darkseagreen', marker='o', linestyle='', ms='3')
 
    plt.plot(time[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']],
             force[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
             'palegreen', marker='o', linestyle='', ms='3')

    plt.plot(time[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']],
             force[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']], 
             'mediumaquamarine', marker='o', linestyle='', ms='3')

    plt.plot(time[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']],
             force[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']], 
             'cornflowerblue', marker='o', linestyle='', ms='3')

    plt.plot(time[boundaries['fluidized_to_meniscus']:boundaries['meniscus_to_breakage']],
             force[boundaries['fluidized_to_meniscus']:boundaries['meniscus_to_breakage']], 
             'mediumslateblue', marker='o', linestyle='', ms='3')

    plt.plot(time[boundaries['meniscus_to_breakage']:],
             force[boundaries['meniscus_to_breakage']:], 
             'purple', marker='o', linestyle='', ms='3')
    plt.xlabel('t (s)')
    plt.ylabel('F (mN)')
    name = ''.join(['Force_vs_tps_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.pdf'])
    plt.savefig(name, bbox_inches='tight', dpi=200)
    plt.close()


def plot_diff_regions_f_vs_p(position,force,boundaries,file_splitted_name):
    """
    Returns a plot of the different regimes
    Force versus position.
    
    Parameters
    ----------
    position : array
        Position.
    force : array
        Force.
    file_splitted_names : list of strings
        Name of the filed used now splitted in strings.
    
    Returns
    -------
    png file
    
    """
    plt.plot(position[:boundaries['approach_to_contact']],
             force[:boundaries['approach_to_contact']], 
             'sandybrown', marker='o', linestyle='', ms='3')

    plt.plot(position[boundaries['approach_to_contact']:boundaries['contact_to_penetration']],
             force[boundaries['approach_to_contact']:boundaries['contact_to_penetration']], 
             'salmon', marker='o', linestyle='', ms='3')

    plt.plot(position[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']],
             force[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']], 
             'darkseagreen', marker='o', linestyle='', ms='3')
     
    plt.plot(position[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']],
             force[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
             'palegreen', marker='o', linestyle='', ms='3')

    plt.plot(position[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']],
             force[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']], 
             'mediumaquamarine', marker='o', linestyle='', ms='3')

    plt.plot(position[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']],
             force[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']], 
             'cornflowerblue', marker='o', linestyle='', ms='3')

    plt.plot(position[boundaries['fluidized_to_meniscus']:boundaries['meniscus_to_breakage']],
             force[boundaries['fluidized_to_meniscus']:boundaries['meniscus_to_breakage']], 
             'mediumslateblue', marker='o', linestyle='', ms='3')

    plt.plot(position[boundaries['meniscus_to_breakage']:],
             force[boundaries['meniscus_to_breakage']:], 
             'purple', marker='o', linestyle='', ms='3')
    plt.xlabel('p (mm)')
    plt.ylabel('F (mN)')
    name = ''.join(['Force_vs_position_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.pdf'])
    plt.savefig(name, bbox_inches='tight', dpi=200)
    plt.close()


def plot_f_vs_t_subplots(time,force,boundaries,file_splitted_name):
    """
    Returns a plot of the different regimes, each being in a different subplot
    Force vs time.


    Parameters
    ----------
    time : array
        Time.
    force : array
        Force.
    boundaries : dictionnary
        Limits of the regions
    file_splitted_names : list of strings
        Name of the filed used now splitted in strings.
    
    Returns
    -------
    png file
    
    """
    fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(15, 20))
    ax = axes.ravel()
    ax[0].plot(time[:boundaries['approach_to_contact']], 
               force[:boundaries['approach_to_contact']], 'sandybrown', marker='o', linestyle='')
    ax[0].set_title('Approach')

    ax[1].plot(time[boundaries['approach_to_contact']:boundaries['contact_to_penetration']], 
               force[boundaries['approach_to_contact']:boundaries['contact_to_penetration']], 'salmon', marker='o', linestyle='')
    ax[1].set_title('Contact')

    ax[2].plot(time[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']], 
               force[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']],'darkseagreen', marker='o', linestyle='')
    ax[2].set_title('Penetration')

    ax[3].plot(time[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
               force[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 'palegreen', marker='o', linestyle='')
    ax[3].set_title('Relaxation')

    ax[4].plot(time[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']], 
               force[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']], 'mediumaquamarine', marker='o', linestyle='')
    ax[4].set_title('Elastic')

    ax[5].plot(time[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']], 
               force[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']], 'cornflowerblue', marker='o', linestyle='')
    ax[5].set_title('Fluidized')

    ax[6].plot(time[boundaries['fluidized_to_meniscus']:boundaries['meniscus_to_breakage']], 
               force[boundaries['fluidized_to_meniscus']:boundaries['meniscus_to_breakage']], 'mediumslateblue', marker='o', linestyle='')
    ax[6].set_title('Meniscus')

    ax[7].plot(time[boundaries['meniscus_to_breakage']:], 
               force[boundaries['meniscus_to_breakage']:], 'purple', marker='o', linestyle='')
    ax[7].set_title('End!')
    name = ''.join(['Force_vs_time_subplots_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.pdf'])
    plt.savefig(name, bbox_inches='tight', dpi=200)
    plt.close()


def plot_s_vs_t(time,stress,boundaries,file_splitted_name):
    """
    Returns a plot of the different regimes from contact to fluidized regime (both comprised)
    The two next plots corresponds  respectively to the regimes from contact to relaxation and
    to the ones from elastic regime to fluidized regime. 
    Stress vs time.
    
    Parameters
    ----------
    time : array
        Time.
    stress : array
        Stress.
    boundaries : dictionnary
        Limits of the regions
    file_splitted_names : list of strings
        Name of the filed used now splitted in strings.
    
    Returns
    -------
    png file
    
    """
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(10,15))
    ax = axes.ravel()
    #ax[0].plot(time[boundaries['approach_to_contact']:boundaries['contact_to_penetration']],
    #         stress[boundaries['approach_to_contact']:boundaries['contact_to_penetration']], 
    #         'salmon', marker='o', linestyle='')

    ax[0].plot(time[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']],
             stress[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']], 
             'darkseagreen', marker='o', linestyle='')

    ax[0].plot(time[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']],
             stress[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
             color='palegreen', marker='o', linestyle='')
     
    ax[0].plot(time[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']],
             stress[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
             'palegreen', marker='o', linestyle='')

    ax[0].plot(time[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']],
             stress[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']], 
             'mediumaquamarine', marker='o', linestyle='')

    ax[0].plot(time[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']],
             stress[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']], 
             'cornflowerblue', marker='o', linestyle='')

    ax[0].set_xlabel('t (s)')
    ax[0].set_ylabel(r'$\tau (Pa)$')
    ax[0].set_title(''.join(['Stress_vs_Time_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))

    #ax[1].plot(time[boundaries['approach_to_contact']:boundaries['contact_to_penetration']],
    #         stress[boundaries['approach_to_contact']:boundaries['contact_to_penetration']], 
    #         'salmon', marker='o', linestyle='')

    ax[1].plot(time[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']],
             stress[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']], 
             'darkseagreen', marker='o', linestyle='')

    ax[1].plot(time[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']],
             stress[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
             color='palegreen', marker='o', linestyle='')
     
    ax[1].plot(time[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']],
             stress[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
             'palegreen', marker='o', linestyle='')

    ax[1].set_xlabel('t (s)')
    ax[1].set_ylabel(r'$\tau (Pa)$')
    ax[1].set_title(''.join(['Stress_vs_Time_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))


    ax[2].plot(time[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']],
             stress[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']], 
             'mediumaquamarine', marker='o', linestyle='')

    ax[2].plot(time[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']],
             stress[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']], 
             'cornflowerblue', marker='o', linestyle='')

    ax[2].set_xlabel('t (s)')
    ax[2].set_ylabel(r'$\tau (Pa)$')
    ax[2].set_title(''.join(['Stress_vs_Time_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))
    name = ''.join(['Stress_vs_Time_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.pdf'])
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def plot_boundaries(time,force,boundaries,file_splitted_name):
    """
    Returns a plot of the force versus time, enlighting the boundaries
    
    Parameters
    ----------
    time : array
        Time.
    force : array
        Force.
    boundaries : dictionnary
        Limits of the regions
    file_splitted_names : list of strings
        Name of the filed used now splitted in strings.
    
    Returns
    -------
    png file
    
    """
    plt.plot(time, force, 'o', ms ='2')
    plt.plot(time[boundaries['approach_to_contact']], force[boundaries['approach_to_contact']], 'ro')
    plt.plot(time[boundaries['contact_to_penetration']], force[boundaries['contact_to_penetration']], 'ro')
    plt.plot(time[boundaries['penetration_to_relaxation']], force[boundaries['penetration_to_relaxation']], 'ro')
    plt.plot(time[boundaries['relaxation_to_elastic']], force[boundaries['relaxation_to_elastic']], 'ro')
    plt.plot(time[boundaries['elastic_to_fluidized']], force[boundaries['elastic_to_fluidized']], 'ro')
    plt.plot(time[boundaries['fluidized_to_meniscus']], force[boundaries['fluidized_to_meniscus']], 'ro')
    plt.plot(time[boundaries['meniscus_to_breakage']], force[boundaries['meniscus_to_breakage']], 'ro')
    plt.xlabel('t (s)')
    plt.ylabel('F (mN)')
    plt.minorticks_on()
    name = ''.join(['Force_vs_tps_boundaries_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.pdf'])
    plt.savefig(name, bbox_inches='tight', dpi=200)
    plt.close()


def plot_s_vs_p(position,stress,boundaries,file_splitted_name):
    """
    Returns a plot of the different regimes of the stress versus position 
    
    Parameters
    ----------
    position : array
        Position.
    stress : array
        Stress.
    boundaries : dictionnary
        Limits of the regions
    file_splitted_names : list of strings
        Name of the filed used now splitted in strings.
    
    Returns
    -------
    png file
    
    """
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(10,15))
    ax = axes.ravel()
    #ax[0].plot(position[boundaries['approach_to_contact']:boundaries['contact_to_penetration']],
    #         stress[boundaries['approach_to_contact']:boundaries['contact_to_penetration']], 
    #        'salmon', marker='o', linestyle='')

    ax[0].plot(position[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']],
             stress[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']], 
             'darkseagreen', marker='o', linestyle='')

    ax[0].plot(position[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']],
             stress[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
             color='palegreen', marker='o', linestyle='')
     
    ax[0].plot(position[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']],
             stress[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
             'palegreen', marker='o', linestyle='')

    ax[0].plot(position[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']],
             stress[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']], 
             'mediumaquamarine', marker='o', linestyle='')

    ax[0].plot(position[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']],
             stress[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']], 
             'cornflowerblue', marker='o', linestyle='')

    ax[0].set_xlabel('p (mm)')
    ax[0].set_ylabel(r'$\tau (Pa)$')
    ax[0].set_title(''.join(['Stress_vs_Position_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))

    #ax[1].plot(position[boundaries['approach_to_contact']:boundaries['contact_to_penetration']],
    #         stress[boundaries['approach_to_contact']:boundaries['contact_to_penetration']], 
    #         'salmon', marker='o', linestyle='')

    ax[1].plot(position[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']],
             stress[boundaries['contact_to_penetration']:boundaries['penetration_to_relaxation']], 
             'darkseagreen', marker='o', linestyle='')

    ax[1].plot(position[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']],
             stress[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
             color='palegreen', marker='o', linestyle='')
     
    ax[1].plot(position[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']],
             stress[boundaries['penetration_to_relaxation']:boundaries['relaxation_to_elastic']], 
             'palegreen', marker='o', linestyle='')

    ax[1].set_xlabel('p (mm)')
    ax[1].set_ylabel(r'$\tau (Pa)$')
    ax[1].set_title(''.join(['Stress_vs_Position_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))


    ax[2].plot(position[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']],
             stress[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']], 
             'mediumaquamarine', marker='o', linestyle='')

    ax[2].plot(position[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']],
             stress[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']], 
             'cornflowerblue', marker='o', linestyle='')

    ax[2].set_xlabel('p (mm)')
    ax[2].set_ylabel(r'$\tau (Pa)$')
    ax[2].set_title(''.join(['Stress_vs_Time_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))
    plt.minorticks_on()
    name = ''.join(['Stress_vs_Position_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.pdf'])
    plt.savefig(name, bbox_inches='tight', dpi=200)
    plt.close()


def plot_f_vs_t(time,force,color,file_splitted_name):
    """
    Plots the force versus time for the current file. 
    Returns nothing.
    
    Parameters
    ----------
    time : array
        Time.
    force : array
        Force.
    color : string
        One of the named colors of matplotlib
    file_splitted_names : list of strings
        Name of the filed used now splitted in strings.
    
    Returns
    -------
    nothing

    """
    plt.plot(time, force, color, marker='o', linestyle='', ms ='2', label=file_splitted_name[1])
    plt.legend(prop={'size':11})


def plot_mean_stress_vs_speed_all_files(bd):
    """
    Saves a plot of all mean stress during penetration period for all velocities
    
    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.
    
    Returns
    -------
    Nothing, but saves a .pdf
    
    """

    names_by_plate = gt.get_names_files_same_plate(bd)

    for lame, _ in sorted(names_by_plate.items()):
        speed = []
        mean_stress = []
        std_stress = []
        for name in sorted(names_by_plate[lame]):
            speed += [bd[name]['speed']]
            mean_stress += [np.asarray(bd[name]['stress'][bd[name]['boundaries']['contact_to_penetration']+1:bd[name]['boundaries']['arg_force_max']]).mean()]
            std_stress += [np.std(bd[name]['stress'][bd[name]['boundaries']['contact_to_penetration']+1:bd[name]['boundaries']['arg_force_max']])]

        plt.plot(speed, mean_stress,label=bd[name]['file_splitted_name'][0], linestyle='', marker='o', ms = 4)
        plt.errorbar(speed, mean_stress, yerr=std_stress, fmt='', marker='', elinewidth=0.5, linestyle='none', color='gray', capthick=0.5, capsize=1)
    plt.xlabel('V (mm/s)')
    plt.ylabel(r'$ < \tau > (Pa)$')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.savefig('Mean_stress_while_penetration_vs_speed_' + '.pdf')
    plt.close()


def plot_mean_stress_vs_speed_by_plate(bd):
    """
    Saves a plot of all mean stress during penetration period for all velocities
    
    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.
    
    Returns
    -------
    Nothing, but saves a .pdf per plate
    
    """

    names_by_plate = gt.get_names_files_same_plate(bd)

    for lame, _ in sorted(names_by_plate.items()):
        speed = []
        mean_stress_penetration = []
        std_stress_penetration = []
        mean_stress_fluidized = []
        std_stress_fluidized = []
        for name in sorted(names_by_plate[lame]):
            speed += [bd[name]['speed']]
            mean_stress_penetration += [np.asarray(bd[name]['stress'][bd[name]['boundaries']['contact_to_penetration']+1:bd[name]['boundaries']['arg_force_max']]).mean()]
            std_stress_penetration += [np.std(bd[name]['stress'][bd[name]['boundaries']['contact_to_penetration']+1:bd[name]['boundaries']['arg_force_max']])]
        plt.plot(speed, mean_stress_penetration,label=bd[name]['file_splitted_name'][0], linestyle='', marker='o', ms = 4)
        plt.errorbar(speed, mean_stress_penetration, yerr=std_stress_penetration, fmt='', marker='', linestyle='none', elinewidth=0.5, capthick=0.5, capsize=1, color='gray')
        plt.xlabel('V (mm/s)')
        plt.ylabel(r'$ < \tau  > (Pa)$')
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.savefig('Mean_stress_while_penetration_vs_speed_' + bd[name]['file_splitted_name'][0] + '.pdf')
        plt.close()


    


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
    names_by_plate = gt.get_names_files_same_plate(bd)

    for lame, _ in sorted(names_by_plate.items()):

        for name in sorted(names_by_plate[lame]):
            plt.plot(bd[name]['time'],bd[name]['force'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
            plt.legend(title='Vitesse (mm/s) :', prop={'size':11})
        plt.xlabel('t (s)')
        plt.ylabel('F (mN)')
        plt.minorticks_on()
        name_file = ''.join(['Force_vs_time_varV_', bd[name]['file_splitted_name'][0], '.pdf'])
        plt.savefig(name_file, bbox_inches='tight')
        plt.close()
        for name in sorted(names_by_plate[lame]):
            plt.plot(bd[name]['position'],bd[name]['force'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
            plt.legend(title='Vitesse (mm/s) :', prop={'size':11})
        plt.xlabel('p (mm)')
        plt.ylabel('F (mN)')
        plt.minorticks_on()
        name_file = ''.join(['Position_vs_force_varV_', bd[name]['file_splitted_name'][0], '.pdf'])
        plt.savefig(name_file, bbox_inches='tight')
        plt.close()
        # for name in sorted(names_by_plate[lame]):
        #     plt.plot(bd[name]['time'],bd[name]['stress'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
        #     plt.legend()
        # plt.xlabel('t (s)')
        # plt.ylabel(r'\tau (Pa)')
        # plt.minorticks_on()
        # name_file = ''.join(['Stress_vs_time_varV', bd[name]['file_splitted_name'][0], '.pdf'])
        # plt.savefig(name_file, bbox_inches='tight')
        # plt.close()
        # for name in sorted(names_by_plate[lame]):
        #     plt.plot(bd[name]['position'],bd[name]['stress'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
        #     plt.legend()
        # plt.xlabel('p (mm)')
        # plt.ylabel(r'\tau (Pa)')
        # plt.minorticks_on()
        # name_file = ''.join(['Stress_vs_position_varV', bd[name]['file_splitted_name'][0], '.pdf'])
        # plt.savefig(name_file, bbox_inches='tight')
        # plt.close()


def plot_for_each_file(bd):

    """
    Saves plots involving only one file each
    
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
        plot_boundaries(bd[name]['time'],bd[name]['force'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
        # la force en fonction de la position 
        plot_diff_regions_f_vs_p(bd[name]['position'],bd[name]['force'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
        # la force en fonction du temps
        plot_diff_regions_f_vs_t(bd[name]['time'],bd[name]['force'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
        # trace les subplots de force 
        #plot_f_vs_t_subplots(bd[name]['time'],bd[name]['force'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
        # trace la contrainte en fonction du temps 
        #plot_s_vs_t(bd[name]['time'],bd[name]['stress'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
        # trace la contrainte en fonction de la position 
        #plot_s_vs_p(bd[name]['position'],bd[name]['stress'],bd[name]['boundaries'],bd[name]['file_splitted_name'])


def plot_force_derivative(bd):
    """
    Saves plots of the force derivative vs the time for each file
    
    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.
    
    Returns
    -------
    Nothing, but saves png
    
    """
    for name in bd.keys():
        #sf = savgol_filter(bd[name]['force'], 9, 5, mode='mirror')
        sf2 = savgol_filter(bd[name]['force'], 9, 5, deriv=1, mode='mirror')
        sf2[sf2 < -0.05] = -0.05
        sf2[sf2 > 0.28] = 0.28
        #sf3 = savgol_filter(bd[name]['force'], 9, 5, deriv=2, mode='mirror')
        #plt.plot(bd[name]['time'], bd[name]['force'], label='data')
        #plt.plot(bd[name]['time'], sf, label = 'smoothed data')
        plt.plot(bd[name]['time'], sf2, label = 'derivative of smoothed data')
        #plt.plot(bd[name]['time'], sf3, label = 'second derivative of smoothed data')
        plt.legend(prop={'size':11})
        plt.setp(legend.get_title(),fontsize=13)
        plt.xlabel('t (s)')
        plt.ylabel('Dérivée de la force (mN/s)')
        plt.savefig('Force_derivative_' + name + '.pdf')
        plt.close()

def ratio_btw_slopes_all_plates(bd):
    """
    Saves plots of the ratio of the slopes btw penetration and fluidized regime 
    
    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.
    
    Returns
    -------
    Nothing, but saves png
    
    """

    names_by_plate = gt.get_names_files_same_plate(bd)
    for lame, _ in sorted(names_by_plate.items()):
        speed = []
        ratio = []
        error = []
        for name in sorted(names_by_plate[lame]):
            speed += [bd[name]['speed']]
            ratio += [bd[name]['fit_meniscus'][0]/bd[name]['fit_penetration'][0]]
            error += [bd[name]['fit_meniscus'][4]/bd[name]['fit_penetration'][0] - bd[name]['fit_meniscus'][0]*bd[name]['fit_penetration'][4]/(bd[name]['fit_penetration'][0]**2)]
        plt.plot(speed, ratio, label=bd[name]['file_splitted_name'][0], linestyle='', marker='o', ms = 4)
        plt.errorbar(speed, ratio, yerr=error, fmt='', marker='', linestyle='none', elinewidth=0.5, capthick=0.5, capsize=1, color='gray')
        legend = legend = plt.legend(title='Lame utilisée :', prop={'size':11})
        plt.setp(legend.get_title(),fontsize=13)
    plt.xlabel('V (mm/s')
    plt.ylabel(r'$\frac{\tau_1}{\tau_2}$')
    plt.ylim(-1.0021,-0.9985)
    plt.savefig('Ratio_btw_slopes_all_plates.pdf')
    plt.close()


def ratio_by_plate(bd):
    """
    Saves plots of the ratio of the slopes btw penetration and fluidized regime 
    
    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.
    
    Returns
    -------
    Nothing, but saves png
    
    """

    names_by_plate = gt.get_names_files_same_plate(bd)
    for lame, _ in sorted(names_by_plate.items()):
        speed = []
        ratio = []
        error = []
        for name in sorted(names_by_plate[lame]):
            speed += [bd[name]['speed']]
            ratio += [bd[name]['fit_meniscus'][0]/bd[name]['fit_penetration'][0]]
            error += [bd[name]['fit_meniscus'][4]/bd[name]['fit_penetration'][0] - bd[name]['fit_meniscus'][0]*bd[name]['fit_penetration'][4]/(bd[name]['fit_penetration'][0]**2)]
        plt.plot(speed, ratio, label=bd[name]['file_splitted_name'][0], linestyle='', marker='o', ms = 4)
        plt.errorbar(speed, ratio, yerr=error, fmt='', marker='', linestyle='none',  elinewidth=0.5, capthick=0.5, capsize=1, color='gray')
        legend = plt.legend(title='Lame utilisée :', prop={'size':11})
        plt.setp(legend.get_title(),fontsize=13)
        plt.xlabel('V (mm/s')
        plt.ylabel(r'$\frac{\tau_1}{\tau_2}$')
        plt.ylim(-1.0021,-0.9985)
        plt.savefig('Ratio_btw_slopes_' + bd[name]['file_splitted_name'][0] +  '.pdf')
        plt.close()


def log_plot_relaxation(bd):
    """
    Saves plots of the penetration part. Log plot of the force vs time. 
    
    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.
    
    Returns
    -------
    Nothing, but saves png
    
    """

    names_by_plate = gt.get_names_files_same_plate(bd)

    for lame, _ in sorted(names_by_plate.items()):

        for name in sorted(names_by_plate[lame]):
            bound1 = bd[name]['boundaries']['arg_force_max']
            bound2 = bd[name]['boundaries']['relaxation_to_elastic']
            time = bd[name]['time'][bound1:bound2] - bd[name]['time'][bound1]
            force = np.asarray(bd[name]['force'][bound1:bound2]) / bd[name]['force'][bound1]
            plt.plot(time,force , label=bd[name]['file_splitted_name'][1], linestyle='', marker='o', ms = 3)
        legend = plt.legend(title='Vitesse (mm/s) :', prop={'size':11})
        plt.setp(legend.get_title(),fontsize=13)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('t (s)')
        plt.ylabel(r'$ \frac{F}{F_0} (mN)$')
        plt.savefig( 'Force_log_relaxation_' + bd[name]['file_splitted_name'][0] + '.pdf')
        plt.close()


def plot_delta_vs_speed_by_plate(bd):
    """
    Saves plots of delta as a function of time by plate -delta being the position difference between the end of the
    relaxation and the beginning of the fluidized regime. 
    
    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.
    
    Returns
    -------
    Nothing, but saves png
    
    """
    names_by_plate = gt.get_names_files_same_plate(bd)

    for lame, _ in sorted(names_by_plate.items()):
        delta = []
        speed = []
        for name in sorted(names_by_plate[lame]):
            delta += [bd[name]['delta']]
            speed += [bd[name]['speed']]
        plt.plot(speed, delta, marker='o', linestyle='', label=bd[name]['file_splitted_name'][0])
        plt.legend(prop={'legend.fontsize':6})
        plt.setp(legend.get_title(),fontsize=13)
        plt.xlabel('V (mm/s')
        plt.ylabel(r'$\delta (mm)$')
        plt.savefig( 'Delta_vs_speed_by_plate' + bd[name]['file_splitted_name'][0] + '.pdf')
        plt.close()

def plot_penetration_f_vs_t(bd):
    names_by_plate = gt.get_names_files_same_plate(bd)


    for lame, _ in sorted(names_by_plate.items()):
        for name in sorted(names_by_plate[lame]):
            bound1 = bd[name]['boundaries']['contact_to_penetration']
            bound2 = bd[name]['boundaries']['arg_force_max']
            time = bd[name]['time'][bound1:bound2] - bd[name]['time'][bound1]
            force = bd[name]['force'][bound1:bound2] - bd[name]['force'][bound1]
            plt.plot(time, force , label=bd[name]['speed'], linestyle='', marker='o', ms = 3)
        legend = plt.legend(title='Vitesse (mm/s) :', prop={'size':11})
        plt.setp(legend.get_title(),fontsize=13)
        plt.xlabel('t (s)')
        plt.ylabel('F (mN)')
        plt.savefig( 'Force_penetration_' + bd[name]['file_splitted_name'][0] + '.pdf')
        plt.close() 

def plot_penetration_s_vs_t(bd):
    names_by_plate = gt.get_names_files_same_plate(bd)


    for lame, _ in sorted(names_by_plate.items()):
        for name in sorted(names_by_plate[lame]):
            bound1 = bd[name]['boundaries']['contact_to_penetration']
            bound2 = bd[name]['boundaries']['arg_force_max']
            time = bd[name]['time'][bound1:bound2] - bd[name]['time'][bound1]
            stress = bd[name]['stress'][bound1:bound2]
            plt.plot(time, stress , label=bd[name]['speed'], linestyle='', marker='o', ms = 3)
        legend = plt.legend(title='Vitesse (mm/s) :', prop={'size':11})
        plt.setp(legend.get_title(),fontsize=13)
        plt.xlabel('t (s)')
        plt.ylabel(r'$\tau (Pa)$')
        plt.savefig( 'Contrainte_penetration_' + bd[name]['file_splitted_name'][0] + '.pdf')
        plt.close() 

def plot_relaxation_f_vs_t(bd):
    names_by_plate = gt.get_names_files_same_plate(bd)


    for lame, _ in sorted(names_by_plate.items()):
        for name in sorted(names_by_plate[lame]):
            bound1 = bd[name]['boundaries']['arg_force_max']
            bound2 = bd[name]['boundaries']['relaxation_to_elastic']
            time = bd[name]['time'][bound1:bound2] - bd[name]['time'][bound1]
            force = bd[name]['force'][bound1:bound2]
            plt.plot(time, force , label=bd[name]['speed'], linestyle='', marker='o', ms = 3)
        legend = plt.legend(title='Vitesse (mm/s) :', prop={'size':11})
        plt.setp(legend.get_title(),fontsize=13)
        plt.xlabel('t (s)')
        plt.ylabel('F (mN)')
        plt.savefig( 'Force_relaxation_' + bd[name]['file_splitted_name'][0] + '.pdf')
        plt.close() 

def plot_relaxation_s_vs_t(bd):
    names_by_plate = gt.get_names_files_same_plate(bd)


    for lame, _ in sorted(names_by_plate.items()):
        for name in sorted(names_by_plate[lame]):
            bound1 = bd[name]['boundaries']['arg_force_max']
            bound2 = bd[name]['boundaries']['relaxation_to_elastic']
            time = bd[name]['time'][bound1:bound2] - bd[name]['time'][bound1]
            stress = bd[name]['stress'][bound1:bound2]
            plt.plot(time, stress , label=bd[name]['speed'], linestyle='', marker='o', ms = 3)
        legend = plt.legend(title='Vitesse (mm/s) :', prop={'size':11})
        plt.setp(legend.get_title(),fontsize=13)
        plt.xlabel('t (s)')
        plt.xscale("log")
        plt.yscale("log")


        plt.ylabel(r'$\tau (Pa)$')
        plt.savefig( 'Contrainte_relaxation_' + bd[name]['file_splitted_name'][0] + '.pdf')
        plt.close()


