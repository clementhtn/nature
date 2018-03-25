# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:43:45 2018

@author: touss
"""

from sys import argv, path
path.append ("../ingestion_program") # Contains libraries you will need
from data_manager import DataManager  # such as DataManager

import prepro

preproc = prepro.Preprocessor()

print(preproc.select_best_features())