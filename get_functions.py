import numpy as np
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d
from scipy import constants
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from itertools import compress

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
    threshold_force_in = np.std(force[:number_first_points])
    zero_load = force[:number_first_points].mean()
    far_from_threshold = np.abs((force - zero_load) / force) > threshold_force_in
    arg_approach_to_contact = far_from_threshold.nonzero()[0][0]
    
    
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
    width : float
        Plate width.
    
    Returns
    -------
    area : array
    """
    area = (position - position[boundaries['contact_to_penetration']]) * variables['plate_width']
    area[area < 0] = 0
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
    stress = [0]*(boundaries['contact_to_penetration'])
    stress.extend(force[boundaries['contact_to_penetration']:boundaries['fluidized_to_meniscus']])
    stress.extend([0]*(len(position) - boundaries['fluidized_to_meniscus']))
    penetrated_area = get_penetrated_area(position, boundaries, variables)
    stress[boundaries['contact_to_penetration']:boundaries['fluidized_to_meniscus']] /= penetrated_area[boundaries['contact_to_penetration']:boundaries['fluidized_to_meniscus']]
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

