# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 23:23:15 2018

@author: touss
"""

from sys import argv, path
path.append ("../ingestion_program") # Contains libraries you will need
from data_manager import DataManager  # such as DataManager

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing

class Preprocessor(BaseEstimator):
    def __init__(self):
        self.transformer = preprocessing.PolynomialFeatures(interaction_only = True)
        # Ce transformer passe de 14 à 106 features, on aurait aimé pouvoir utiliser SelectKBest pour réduire
        # ce nombre mais je n'ai pas réussi à l'utiliser depuis ma machine (erreur indiquée directement
        # dans le code de la fonction dans le module sklearn).    
        #self.trueTrans = SelectKBest()  
        
    # On transforme les données, en les standardisant autour d'une moyenne égale à 0 pour chaque feature.
    # On utilise robust_scale plutôt que scale pour réduire l'impact des outliers.
    def normalize(self, X):
        return preprocessing.robust_scale(X)

    def fit(self, X, y=None):
        res = self.transformer.fit(self.normalize(X), y)
        return res
        #return self.trueTrans.fit(res)

    def fit_transform(self, X, y=None):
        res = self.transformer.fit_transform(self.normalize(X))
        return res
        #return self.trueTrans.fit_transform(res)

    def transform(self, X, y=None):
        res = self.transformer.transform(self.normalize(X))
        return res
        #return self.trueTrans.transform(res)
     