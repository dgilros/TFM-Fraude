#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:44:39 2020

@author: David Gil del Rosal
"""

#imbreports.py
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Genera una tabla LaTeX a partir de un diccionario con informes."""
def reports_to_tex(filename, reports, headers, metrics, caption, label, 
                   inc_report_name=False):
    def write_line(line, is_row=False):
        file.write(line)
        if is_row: 
            file.write(' \\\\\n')
            file.write('\\hline')
        file.write('\n')

    def tex_escape(value):
        if type(value)==float:
            text = '$%.4f$' % value
        else:
            text = str(value)
            text = text.replace('%', '\\%')
            text = text.replace('#', '\\#')
            text = text.replace('_', '\\_')
        return text
    
    def textbf(text):
        return '\\textbf{' + text + '}'
    
    def write_tex_row(row, header=False, first_row='', nrows=1):
        row = list([tex_escape(cell) for cell in row])
        if header == True:
            cells = [textbf(col) for col in row]
        elif inc_report_name and first_row != '':
            cells = ['\\multirow{' + str(nrows) + '}{*}{' + first_row + '}']
            cells+= row
        elif inc_report_name:
            cells = [' '] + row
        else:
            cells = row
        #print(cells)
        write_line(' & '.join(cells), is_row=True)
            
    n_headers = len(headers)
    n_metrics = len(metrics)
    with open(filename,'w') as file:
        write_line('\\begin{table}[H]')
        write_line('\\resizebox{\\textwidth}{!}{%\centering')
        write_line('\\begin{tabular}{|'+('l|'*n_headers)+('r|'*n_metrics)+'}')
        write_line('\\hline')
        write_tex_row(list(headers)+list(metrics), header=True)
        #write_line('\\hline')
        for report, data in reports.items():
            first_row, nrows = report, data.shape[0]
            for _,row in data.iterrows():
                write_tex_row(row, first_row=first_row, nrows=nrows)
                first_row = ''
        write_line('\\end{tabular}}')
        write_line('\\caption{'+ caption + '}')
        write_line('\\label{tab:' + label + '}')
        write_line('\\end{table}')
    print('Created', filename)
    

"""Superclase abstracta para la generación de informes a partir de un dataset."""
class ImbalancedReport:
    def __init__(self, dataset, report_title):
        self._dataset = dataset
        self._report_title = report_title
                
    """Devuelve la ruta de un fichero relativa al directorio del dataset."""
    def get_path(self, filename):
        return self._dataset.get_path(filename)
    
    """Genera un nombre de fichero prefijado por el nombre del informe."""
    def get_filename(self, filename):
        return self.get_path(self._report_title + '-' + filename)

    """Genera una tabla LaTex a partir de un diccionario con los informes."""    
    def create_tex_table(self, reports, headers, metrics,
                         caption, label, inc_report_name=False):
        filename = self.get_filename(label+'.tex')
        reports_to_tex(filename, reports, headers, metrics, caption, label, 
                       inc_report_name=inc_report_name)

    """Genera una tabla LaTex a partir de un DataFrame de Pandas."""
    def df_to_tex_table(self, df, *args, **kwargs):
        self.create_tex_table({'Report':df}, *args, **kwargs) 

    """Guarda la figura actual de MatPlotLib con el nombre dado."""
    def save_figure(self, filename):
        plt.savefig(self.get_filename(filename))
       
            
"""Clase para la generación de estadísticas descriptivas sobre un dataset."""
class StatsReport(ImbalancedReport):
    def __init__(self, dataset, report_title):
        super().__init__(dataset, report_title) 
        
    """Genera una tabla LaTeX con los estadísticos de los atributos numéricos."""
    def create_stats_table(self, ds_name, headers, metrics, caption, features=None):
         from scipy.stats import ttest_ind
         columns = headers + metrics
         df = self._dataset.load_dataframe(ds_name)
         label_attr = self._dataset.get_label_attr()
         labels = self._dataset.get_maj_min()
         df_class = {label:df[df[label_attr]==label] for label in labels}
         df_licit = df_class[labels[0]]
         df_fraud = df_class[labels[1]]
         print_test = True
         reports = {}
         if features is None: features = df.columns
         for column in features:
             if column != label_attr:
                 rows = []
                 for label,label_attr in zip(labels, ['0','1']):
                     values = df_class[label][column].to_numpy()
                     q1 = np.percentile(values, 25)
                     q3 = np.percentile(values, 75)
                     median = np.percentile(values, 50)
                     row = [label_attr, values.mean(), values.std(), 
                            values.min(), q1, median, q3, values.max(), q3-q1]
                     for i in range(1, len(row)):
                         row[i] = round(float(row[i]), 2)
                     if print_test:
                         tstat, pvalue = ttest_ind(df_licit[column], df_fraud[column])
                         tstat = round(tstat, 2)
                         pvalue = round(pvalue, 2)
                     row.append(tstat)
                     row.append(pvalue)
                     rows.append(row)
                     print_test = not print_test
                 reports[column] = pd.DataFrame(rows, columns=columns)
         self.create_tex_table(reports, headers, metrics, caption, 'Stats', 
                               inc_report_name=True)
         

"""Clase para la generación de estadísticas sobre las muestras"""
class SamplingReport(ImbalancedReport):
    def __init__(self, dataset, report_title):
        super().__init__(dataset, report_title)
        rows = []
        label_attr = dataset.get_label_attr()
        neg, pos = dataset.get_maj_min()
        for sample_name in dataset.get_sample_names(all_samples=True):
            df = dataset.load_dataframe(sample_name)
            y = df[label_attr].to_numpy()
            neg_count = len(y[y==neg])
            pos_count = len(y[y==pos])
            rows.append([sample_name, df.shape[0], neg_count, pos_count])
        df = pd.DataFrame(rows, columns=['Sample','#Obs','#Neg','#Pos'])
        df['%Pos'] = 100.0 * df['#Pos'] / df['#Obs']
        df['IR'] = df['#Neg'] / df['#Pos']
        self._stats_df = df

    def get_stats_df(self):
        return self._stats_df

    def plot_statistics(self, filename, xlabel, colors=['b','r']):
        stats_df = self.get_stats_df()
        fig, ax = plt.subplots(figsize=(12,6))
        ax.barh(stats_df['Sample'], stats_df['#Neg'], color=colors[0])
        ax.barh(stats_df['Sample'], stats_df['#Pos'], left=stats_df['#Neg'], 
                color=colors[1])
        ax.set_xlabel(xlabel)
        self.save_figure('Samples.png')

    def plot_tsne_samples(self, filename, cols=4, colors=['b','r']):
        maj, _ = self._dataset.get_maj_min()
        sample_names = self._dataset.get_sample_names(all=True)
        n = len(sample_names)
        rows = int(n / cols)
        if n % cols != 0: rows+= 1
        fig, axs = plt.subplots(rows, cols, figsize=(12,10))
        row, col = 0, 0
        for sampling_method in sample_names:
            df = self._dataset.load_dataframe('tsne-'+sampling_method)
            colors = [colors[0] if label == maj else colors[1]
                      for label in df[self._dataset.get_label_attr()]]
            axs[row,col].scatter(df['X1'], df['X2'], c=colors)
            axs[row,col].set_title(sampling_method)
            col = (col+1) % cols
            if col == 0: row+= 1
        for c in range(col,cols):
            try: fig.delaxes(ax=axs[row,c])
            except: pass
        fig.tight_layout()
        self.save_figure('Samples-tSNE.png')
        
    def create_sampling_table(self, caption):
        df = self.get_stats_df()
        self.df_to_tex_table(df, [df.columns[0]], df.columns[1:], caption,
                             'Samples', inc_report_name=False)
        

class CrossValReport(ImbalancedReport):
    def __init__(self, dataset, model_name):
        super().__init__(dataset, model_name+'-CrossVal')
        ds_name = model_name+'-Scores'
        self._model_name = model_name
        self._df = dataset.load_dataframe(ds_name)
        self._df_scores = self._df[self._df['Task']=='CrossVal']

    def create_scores_table(self, header, caption):
        rows = []
        sample_names = self._dataset.get_sample_names()
        metrics = self._df_scores.columns[4:]
        for sample_name in sample_names:
            df = self._df_scores
            df_sample = df[df['Sample']==sample_name]
            row = [sample_name]
            for metric in metrics:
                mean, std = df_sample[metric].mean(), df_sample[metric].std()
                row.append('$%.4f \pm %.2f$' % (mean,std))
            rows.append(row)
        df = pd.DataFrame(rows)
        tab_label = 'Scores'
        self.df_to_tex_table(df, [header], metrics, caption, tab_label)

    def plot_scores(self, metric, figsize=(12,10)):
        scores = defaultdict(list)
        for _,row in self._df_scores.iterrows():
           scores[row['Sample']].append(row[metric])
        plt.figure(figsize=figsize)
        plt.boxplot(scores.values(), labels=scores.keys(), showmeans=True)
        self.save_figure(metric+'.png')
        
class EvalReport(CrossValReport):
    def __init__(self, dataset, model_name):
        super().__init__(dataset, model_name)
        self._report_title = self._report_title.replace('-CrossVal','-Eval')
        self._df_scores = self._df[self._df['Task']=='Eval']
        
    def create_scores_table(self, header, caption):
        rows = []
        sample_names = self._dataset.get_sample_names()
        metrics = self._df_scores.columns[4:]
        for sample_name in sample_names:
            df = self._df_scores
            df_sample = df[df['Sample']==sample_name]
            row = [sample_name]
            for metric in metrics:
                row.append(df_sample[metric].values[0])
            rows.append(row)
        df = pd.DataFrame(rows)
        tab_label = 'Scores'
        self.df_to_tex_table(df, [header], metrics, caption, tab_label)
                
    def plot_tree_model(self, sample_name, figsize=(15,12), **kwargs):
        from sklearn import tree
        model = self._dataset.load_model(self._model_name, sample_name)
        df = self._dataset.load_dataframe(sample_name, nrows=1)
        df = df.drop([self._dataset.get_label_attr()], axis=1)
        plt.figure(figsize=(15,12))
        tree.plot_tree(model, feature_names=df.columns,
                       filled=True, label='none', **kwargs)
        self.save_figure(sample_name+'-Tree.png')

    def get_feature_importances(self, sample_name):
        model = self._dataset.load_model(self._model_name, sample_name)
        df = self._dataset.load_dataframe(sample_name, nrows=1)
        features = df.drop(['Class'], axis=1).columns
        yerr = None
        if hasattr(model, 'coef_'):
           # es un modelo de regresión, los coeficientes estiman la importancia
           importances = model.coef_[0]
        elif hasattr(model, 'feature_importances_'):
          # es un modelo basado en árboles de decisión
          importances = model.feature_importances_
          if hasattr(model, 'estimators_'):
             # es un modelo de ensamblaje: Random Forest, Extra Trees, etc.
             # mostrar también desviación estándar
             yerr = np.std([tree.feature_importances_ for tree in model.estimators_],
                            axis=0)
        else:
            raise ValueError('Model does not allow to estimate importances')
        return features, importances, yerr

    def plot_feature_importances(self, sample_name, **kwargs):
        # generar grafico de barras
        features, importances, yerr = self.get_feature_importances(sample_name)
        plt.figure(figsize=(18,10))
        plt.bar(features, importances, yerr=yerr)
        self.save_figure(sample_name+'-Importances.png')

    def feature_importance_table(self, sample_name, caption, n_feat=10):
        features, importances, _ = self.get_feature_importances(sample_name)
        df = pd.DataFrame({'Feature':features, 'Importance': importances})
        df = df.sort_values(by='Importance', ascending=False).head(n_feat)
        tab_label = self._model_name + '-' + sample_name + '-Importances'
        self.df_to_tex_table(df, ['Feature'], df.columns[1:], caption, tab_label)
        
    def plot_confusion_matrices(self, *labels,
                                cols=4, figsize=(12,10), normalize=False):
        import seaborn as sns
        df = self._df_scores
        n = df.shape[0]
        rows = int(n / cols)
        if n % cols != 0: rows+= 1
        fig, axs = plt.subplots(rows, cols, figsize=(12,10))
        row, col = 0, 0
        for _,orow in df.iterrows(): 
            sampling_method = orow['Sample']
            tn, fp, fn, tp = orow['TN'], orow['FP'], orow['FN'], orow['TP']
            if normalize:
                tnr = round(float(tn)/float(tn+fp), 2)
                fpr = 1.0 - tnr
                fnr = round(float(fn)/float(fn+tp), 2)
                tpr = 1.0 - fnr
                tn, fp, fn, tp = tnr, fpr, fnr, tpr
                fmt='2g'
            else:
                fmt='d'
            conf_mat = np.array([[tn,fp],[fn,tp]])
            sns.heatmap(conf_mat, cbar=False, annot=True, cmap='Blues',
                        xticklabels=labels, yticklabels=labels, 
                        ax=axs[row,col], fmt=fmt)
            axs[row,col].set_title(sampling_method)
            col = (col+1) % cols
            if col == 0: row+= 1
        for c in range(col,cols):
            try: fig.delaxes(ax=axs[row,c])
            except: pass
        fig.tight_layout()
        self.save_figure('-CM.png')        
