#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 08:37:08 2018

@author: toussaint.cabeli
"""

from sys import argv, path
path.append ("../ingestion_program") # Contains libraries you will need
from data_manager import DataManager  # such as DataManager

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
import numpy as np

class Preprocessor(BaseEstimator):
    def __init__(self):
        self.transformer = preprocessing.PolynomialFeatures()
        # On réécrit à la main les résultats données par select_best_features (avec le fichier selectbest.py)
        # car on a besoin des données mais nous ne connaissons pas le chemin sur codalab.
        self.features = np.array([1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,19,21,23,25,26,29,30,31,32,33,34,35,36,38,39,40,41,42,43,44,45,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,67,69,73,74,78,84,86,93,98,99,101,102,110,111,114])
        #self.features = self.select_best_features()
        
    # On transforme les données, en les standardisant autour d'une moyenne égale à 0 pour chaque feature.
    # On utilise robust_scale plutôt que scale pour réduire l'impact des outliers.
    def normalize(self, X):
        return preprocessing.robust_scale(X)

    # Ici on utilise SelectKbest pour ne garder que les meilleures features, on prend 70 pour donner un compromis
    # entre score et rapidité.
    def select_best_features(self):
        path_to_training_data = "../ecolo_data/air_train.data"
        path_to_training_label = "../ecolo_data/air_train.solution"
        X_train = self.transformer.fit_transform(self.normalize(np.loadtxt(path_to_training_data)))
        y_train = np.loadtxt(path_to_training_label)
        kbest = SelectKBest(k = 70)
        test = kbest.fit_transform(X_train, y_train)
        return kbest.get_support(indices = True)
    
    def fit(self, X, y=None):
        res = self.transformer.fit(self.normalize(X), y)
        return res[:,self.features]

    def fit_transform(self, X, y=None):
        res = self.transformer.fit_transform(self.normalize(X))
        return res[:,self.features]

    def transform(self, X, y=None):
        res = self.transformer.transform(self.normalize(X))
        return res[:,self.features]