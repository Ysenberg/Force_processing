import numpy as np
import matplotlib.pyplot as plt
import yaml
import re
from yaml import load, dump
import pandas as pd
import glob
import os

"""
for filename in glob.glob('*.txt'): # va servir à récupérer tous les chemins vers les vidéos splittées en images
    directory_name = os.path.splitext(filename)[0].split('.') # les images sont rangées dans un répertoire avec comme nom image-00%.png
    print(directory_name)
"""
directory_name = 'Lam2_V12_001'
print('./Videos/' + directory_name + '/*.png')
paths_to_images = glob.glob('./Videos/' + directory_name + '/*.png')
print(paths_to_images[0])
image = open(paths_to_images[0])
print(type(image))
