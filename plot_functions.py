import numpy as np
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d
from scipy import constants
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import get_functions as gt
import yaml
from yaml import load, dump

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
    plt.xlabel('Time (s)')
    plt.ylabel('Force (mN)')
    plt.title(''.join(['Force_vs_tps_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))
    name = ''.join(['Force_vs_tps_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.png'])
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
    plt.xlabel('Position (mm)')
    plt.ylabel('Force (mN)')
    plt.title(''.join(['Force_vs_position_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))
    name = ''.join(['Force_vs_position_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.png'])
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

    plt.title(''.join(['Force_vs_time_subplots_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))
    name = ''.join(['Force_vs_time_subplots_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.png'])
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

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Stress (kPa)')
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

    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Stress (kPa)')
    ax[1].set_title(''.join(['Stress_vs_Time_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))


    ax[2].plot(time[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']],
             stress[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']], 
             'mediumaquamarine', marker='o', linestyle='')

    ax[2].plot(time[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']],
             stress[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']], 
             'cornflowerblue', marker='o', linestyle='')

    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Stress (kPa)')
    ax[2].set_title(''.join(['Stress_vs_Time_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))

    plt.title(''.join(['Stress_vs_Time_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))
    name = ''.join(['Stress_vs_Time_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.png'])
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
    plt.xlabel('Time (s)')
    plt.ylabel('Force (mN)')
    plt.minorticks_on()
    plt.title(''.join(['Force_vs_tps_boundaries_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))
    name = ''.join(['Force_vs_tps_boundaries_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.png'])
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

    ax[0].set_xlabel('Position (mm)')
    ax[0].set_ylabel('Stress (kPa)')
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

    ax[1].set_xlabel('Position (mm)')
    ax[1].set_ylabel('Stress (kPa)')
    ax[1].set_title(''.join(['Stress_vs_Position_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))


    ax[2].plot(position[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']],
             stress[boundaries['relaxation_to_elastic']:boundaries['elastic_to_fluidized']], 
             'mediumaquamarine', marker='o', linestyle='')

    ax[2].plot(position[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']],
             stress[boundaries['elastic_to_fluidized']:boundaries['fluidized_to_meniscus']], 
             'cornflowerblue', marker='o', linestyle='')

    ax[2].set_xlabel('Position (mm)')
    ax[2].set_ylabel('Stress (kPa)')
    ax[2].set_title(''.join(['Stress_vs_Time_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))
    plt.minorticks_on()
    plt.title(''.join(['Stress_vs_Position_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2]]))
    name = ''.join(['Stress_vs_Position_', file_splitted_name[0] , '_', file_splitted_name[1] ,'_',file_splitted_name[2],'.png'])
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
    plt.legend()


def plot_mean_stress_vs_speed(bd):
    """
    Saves a plot of all mean stress during penetration period for all velocities
    
    Parameters
    ----------
    bd : dictionnary
        Dictionnary containing informations about all *.txt files in the folder.
    
    Returns
    -------
    Nothing, but saves a .png
    
    """

    names_by_plate = gt.get_names_files_same_plate(bd)

    for lame in names_by_plate.keys():
        speed = []
        mean_stress = []
        std_stress = []
        for name in names_by_plate[lame]:
            speed += [bd[name]['speed']]
            mean_stress += [np.asarray(bd[name]['stress'][bd[name]['boundaries']['contact_to_penetration']+1:bd[name]['boundaries']['penetration_to_relaxation']]).mean()]
            std_stress += [np.std(bd[name]['stress'][bd[name]['boundaries']['contact_to_penetration']+1:bd[name]['boundaries']['penetration_to_relaxation']])]

        plt.plot(speed,mean_stress,label=bd[name]['file_splitted_name'][0], linestyle='', marker='o', ms = 2)
        plt.errorbar(speed,mean_stress, yerr=std_stress, fmt='o', ms = 4)
        plt.legend()

    plt.xlabel('Time (s)')
    plt.ylabel('Mean stress (kPa)')
    plt.title('Mean_stress_while_penetration_vs_speed_' )
    plt.savefig('Mean_stress_while_penetration_vs_speed_' + '.png')
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

    for lame in names_by_plate.keys():

        for name in names_by_plate[lame]:
            plt.plot(bd[name]['time'],bd[name]['force'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
            plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Force (mN)')
        plt.minorticks_on()
        plt.title('Force_vs_Time' + bd[name]['file_splitted_name'][0])
        name_file = ''.join(['Force_vs_time_varV_', bd[name]['file_splitted_name'][0], '.png'])
        plt.savefig(name_file, bbox_inches='tight')
        plt.close()
        for name in names_by_plate[lame]:
            plt.plot(bd[name]['position'],bd[name]['force'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
            plt.legend()
        plt.xlabel('Position (mm)')
        plt.ylabel('Force (mN)')
        plt.minorticks_on()
        plt.title('Force_vs_Position' + bd[name]['file_splitted_name'][0])
        name_file = ''.join(['Position_vs_force_varV_', bd[name]['file_splitted_name'][0], '.png'])
        plt.savefig(name_file, bbox_inches='tight')
        plt.close()
        # for name in names_by_plate[lame]:
        #     plt.plot(bd[name]['time'],bd[name]['stress'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
        #     plt.legend()
        # plt.xlabel('Time (s)')
        # plt.ylabel('Stress (kPa)')
        # plt.minorticks_on()
        # plt.title('Stress_vs_time' + bd[name]['file_splitted_name'][0])
        # name_file = ''.join(['Stress_vs_time_varV', bd[name]['file_splitted_name'][0], '.png'])
        # plt.savefig(name_file, bbox_inches='tight')
        # plt.close()
        # for name in names_by_plate[lame]:
        #     plt.plot(bd[name]['position'],bd[name]['stress'], marker='o', linestyle='-', ms ='2', label=bd[name]['file_splitted_name'][1])
        #     plt.legend()
        # plt.xlabel('Position (mm)')
        # plt.ylabel('Stress (kPa)')
        # plt.minorticks_on()
        # plt.title('Stress_vs_position' + bd[name]['file_splitted_name'][0])
        # name_file = ''.join(['Stress_vs_position_varV', bd[name]['file_splitted_name'][0], '.png'])
        # plt.savefig(name_file, bbox_inches='tight')
        # plt.close()


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
        plot_boundaries(bd[name]['time'],bd[name]['force'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
        # la force en fonction de la position 
        plot_diff_regions_f_vs_p(bd[name]['position'],bd[name]['force'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
        # trace les subplots de force 
        plot_f_vs_t_subplots(bd[name]['time'],bd[name]['force'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
        # trace la contrainte en fonction du temps 
        plot_s_vs_t(bd[name]['time'],bd[name]['stress'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
        # trace la contrainte en fonction de la position 
        plot_s_vs_p(bd[name]['position'],bd[name]['stress'],bd[name]['boundaries'],bd[name]['file_splitted_name'])
