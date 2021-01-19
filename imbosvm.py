# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 09:39:49 2020

@author: David
"""


from sklearn.svm import OneClassSVM

class OneClassSVMClassifier:
    def __init__(self, majority, minority, **kwargs):
        self._majority, self._minority = majority, minority
        self._classifier = None
        self._kwargs = kwargs
        
    def fit(self, X_train, y_train, **fit_args):
        X_normal = X_train[y_train==self._majority]
        y_normal = y_train[y_train==self._majority]
        y_outliers = y_train[y_train==self._minority]
        outlier_prop = float(len(y_outliers)) / float(len(y_normal))
        kwargs = self._kwargs
        self._classifier = OneClassSVM(nu=outlier_prop, **kwargs)
        self._classifier.fit(X_normal, **fit_args)
        return self
    
    def predict(self, X_test, **kwargs):
        y_pred = self._classifier.predict(X_test, **kwargs)
        y_pred[y_pred==1] = self._majority
        y_pred[y_pred==-1] = self._minority
        return y_pred
    