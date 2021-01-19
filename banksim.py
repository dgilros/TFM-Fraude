# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 08:29:47 2020

@author: David
"""

from imbbenchmark import ImbalancedBenchmark
from imbreports import SamplingReport, StatsReport

class BankSim(ImbalancedBenchmark):
    def __init__(self):
        super().__init__('Datasets/BankSim', 'banksim_clean', 'fraud', 0, 1)
        with open(self.get_path('banksim.csv'),'r') as f_in:
            with open(self.get_path('banksim_clean.csv'),'w') as f_out:
                for line in f_in:
                    f_out.write(line.replace("'",""))
                    
                                       
banksim = BankSim()
banksim.create_train_test(test_size=0.3)
report = SamplingReport(banksim, 'BankSim')
report.create_sampling_table('BankSim: desequilibrio')
stats_report = StatsReport(banksim, 'BankSim')
headers = ['Fraude']
metrics = ['Media','StdDev','Min','Q1','Mediana','Q3','Max','IQR',
           'Welch t','p-valor']
caption = 'BankSim: indicadores estadísticos de Amount'
stats_report.create_stats_table('banksim_clean', headers, metrics, caption,
                                features=['amount'])