#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:50:34 2020

@author: David Gil del Rosal
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from imneural import ANNClassifier
from imbbenchmark import ImbalancedBenchmark
from imbreports import StatsReport, SamplingReport, CrossValReport, EvalReport

class CreditCardFraud(ImbalancedBenchmark):
    def __init__(self):
        super().__init__('Datasets/CreditCardFraud', 'creditcard', 'Class', 0, 1)
    
    def preprocess(self, model_name, X_train, X_test):
        if model_name[:3] == 'ANN':
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train) 
            X_test = scaler.transform(X_test)            
        if model_name[:2] not in ['DT','RF','ET','AB']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train) 
            X_test = scaler.transform(X_test)
        else:
            X_train, X_test = super().preprocess(model_name, X_train, X_test)
        return X_train, X_test

if __name__ == '__main__':
    import sys
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from imbosvm import OneClassSVMClassifier

    SEED = 42
    TEST_SIZE = 0.3
    IMB_RATIO = 4.0
    np.random.seed(SEED)
    args = set(sys.argv[1:])
    cc = CreditCardFraud()
    cc.set_default_samplers()
    cc.set_default_metrics()
    majority, minority = cc.get_maj_min()
    #cc.add_model('BL', DummyClassifier(strategy='stratified'))
    #cc.add_model('LR', LogisticRegression(max_iter=1000))
    #cc.add_model('NB', GaussianNB())
    #cc.add_model('DT', DecisionTreeClassifier())
    #cc.add_model('ET', ExtraTreesClassifier())
    #cc.add_model('AB', AdaBoostClassifier())
    #cc.add_model('3NN', KNeighborsClassifier(n_neighbors=3))
    #hidden_layers = [('dense',64),('dense',16),('dropout',0.25)]
    #cc.add_model('ANN', ANNClassifier(29, hidden_layers))
    #cc.add_model('SGD', SGDClassifier())
    cc.add_model('OC-SVM', OneClassSVMClassifier(majority, minority, kernel='rbf'))
    if 'samples' in args:
        sampling_strategy = 1.0 / float(IMB_RATIO)
        cc.create_train_test(test_size=TEST_SIZE, drop_columns=['Time'])
        cc.create_train_samples(sampling_strategy=sampling_strategy)
    if 'tsne' in args:
        cc.create_tsne_samples()
    if 'stats' in args:
        # generamos las estadísticas descriptivas
        headers = ['Fraude']
        metrics = ['Media','StdDev','Min','Q1','Mediana','Q3','Max','IQR',
                   'Welch t','p-valor']
        caption = 'Credit Card Fraud: indicadores estadísticos'
        report = StatsReport(cc, 'CC')
        report.create_stats_table('creditcard.csv', headers, metrics, caption)
        # generamos las estadísticas sobre las muestras
        report = SamplingReport(cc, 'CC')
        report.create_sampling_table('Credit Card Fraud: muestras')
        report.plot_statistics('Samples.png', 'Observaciones', colors=['b','r'])
        #report.plot_tsne_samples('Samples-tSNE.png', colors=['b','r'])
    if 'train-crossval' in args or 'train-eval' in args:
        for model_name in cc.get_model_names():
            cc.eval_model(model_name, cv_folds=0)
    if 'eval-stats' in args:
        for model_name in cc.get_model_names():
            report = EvalReport(cc, model_name)
            report.create_scores_table('Sample', 'resultados de evaluación')
            if model_name in ['LR','DT','ET']:
               report.plot_feature_importances('Train')
            report.plot_confusion_matrices('No Fraude', 'Fraude', cols=3)
    if 'crossval-stats' in args:
        cc.eval_model('NB', cv_folds=5)
        report = CrossValReport(cc, 'NB')
        report.create_scores_table('Muestra', 'resultados de validación cruzada')
        report.plot_scores('F1')