# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:17:34 2018

@author: touss
"""

from sys import argv, path
path.append ("../ingestion_program") # Contains libraries you will need
from data_manager import DataManager  # such as DataManager

import prepro

if __name__=="__main__":
    # We can use this to run this file as a script and test the Preprocessor
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../results" # Create this directory if it does not exist
    else:
        input_dir = argv[1]
        output_dir = argv[2];
    
    basename = 'air'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    # On effectue le test de notre preprocessing de façon très simple, en deux affichages :
    # Tout d'abord on vérifie que les données brutes ont 14 features contenant des valeurs entières.
    print("Nombre de features : {0}.".format(D.data['X_train'].shape[1]))
    print(D.data['X_train'][0,:])
    
    Prepro = prepro.Preprocessor()
 
    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    # Maintenant on vérifie que l'on a des données contenant plus de features (106) avec des valeurs
    # proches de 0.
    print("Nombre de features : {0}.".format(D.data['X_train'].shape[1]))
    print(D.data['X_train'][0,:])
    