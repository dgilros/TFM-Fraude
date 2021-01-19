import os
import time
import joblib
import numpy as np
import pandas as pd
from imbaux import CallableFactory

DEFAULT_METRICS = ['TN','FP','FN','TP','Accuracy','Recall','TNR','Precision',
                   'F1','G-mean','MCC','ROC-AUC','PR-AUC']

# función auxiliar que concatena los arrays de atributos y
# etiquetas para formar un DataFrame de Pandas
def concat_X_y(X, y, columns):
    y = y[:,None]
    return pd.DataFrame(np.append(X,y,axis=1), columns=columns)

# función auxiliar que devuelve el 'scorer' asociado a una métrica
def conf_matrix_scorer(metric):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    from sklearn.metrics import average_precision_score, get_scorer
    from imblearn.metrics import geometric_mean_score
    
    def get_matrix(y_true, y_pred):
        tn,fp,fn,tp = confusion_matrix(y_true, y_pred).ravel()
        if metric == 'tn': return tn
        elif metric == 'fp': return fp
        elif metric == 'fn': return fn
        elif metric == 'tp': return tp
        elif metric == 'tnr': return tn/(tn+fp)

    if metric in ['tn','fp','fn','tp','tnr']:
        scorer = get_matrix
    elif metric == 'mcc':
        scorer = matthews_corrcoef
    elif metric == 'roc-auc':
        scorer = roc_auc_score
    elif metric == 'pr-auc':
        scorer = average_precision_score
    elif metric == 'g-mean':
        scorer = geometric_mean_score
    else:
        scorer = get_scorer(metric) 
        scorer = scorer.__dict__['_score_func']
    return scorer


"""Clase que define un benchmark desequilibrado."""
class ImbalancedBenchmark:
    def __init__(self, directory, name, label_attr, majority, minority,
                 label_type=np.uint8):
        self._directory = directory
        self._name = name
        self._label_attr = label_attr
        self._label_type = label_type
        self._majority = majority
        self._minority = minority
        self._samplers = CallableFactory()
        self._classifiers = CallableFactory()
        self._metrics = CallableFactory()
        
    """Aplica preproceso a los atributos en función del modelo."""
    def preprocess(self, model_name, X_train, X_test):
        return X_train, X_test
        
    """Devuelve la ruta de un fichero relativa al directorio del dataset."""
    def get_path(self, path):
        return os.path.join(self._directory, path)
        
    """Devuelve el nombre del dataset."""
    def get_name(self):
        return self._name
        
    """Devuelve el nombre del atributo de clase."""
    def get_label_attr(self):
        return self._label_attr
        
    """Devuelve las etiquetas de clase mayoritaria y minoritaria."""
    def get_maj_min(self):
        return self._majority, self._minority
        
    """Lee un DataFrame de Pandas de un fichero CSV."""
    def load_dataframe(self, ds_name, **load_args):
        if ds_name.find('.') < 0: ds_name+= '.csv'
        return pd.read_csv(self.get_path(ds_name), **load_args)

    """Guarda una dataframe de Pandas en formato CSV."""
    def save_dataframe(self, df, df_name, debug=True):
        filename = df_name + '.csv'
        df.to_csv(self.get_path(filename), index=False)
        if debug: print('Created', filename)

    """Guarda un dataset en un fichero CSV."""
    def save_dataset(self, ds_name, X, y, columns, shuffle=False):
        dataset = concat_X_y(X, y.astype(self._label_type), columns=columns)
        if shuffle: dataset = dataset.sample(frac=1.0)
        self.save_dataframe(dataset, ds_name)

    """Dado un dataset, devuelve los arrays de atributos, etiquetas y columnas."""
    def get_X_y_cols(self, dataset):
        label_attr = self.get_label_attr()
        y = dataset[label_attr].to_numpy().astype(self._label_type)
        X = dataset.drop([label_attr], axis=1).to_numpy()
        return X, y, dataset.columns

    """Lee un dataset."""
    def load_dataset(self, ds_name, drop_columns=None, **load_args):
        df = self.load_dataframe(ds_name, **load_args)
        if drop_columns is not None:
            df = df.drop(drop_columns, axis=1)
        return self.get_X_y_cols(df)
        
    """Registra un método de remuestreo para ser usado por el benchmark."""
    def set_sampler(self, name, builder):
        return self._samplers.set(name, builder)
        
    """Registra los métodos de remuestreo por defecto."""
    def set_default_samplers(self):
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.under_sampling import InstanceHardnessThreshold
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.over_sampling import SMOTE
        from imblearn.combine import SMOTETomek
        self.set_sampler('RUS', RandomUnderSampler)
        self.set_sampler('IHT', InstanceHardnessThreshold)
        self.set_sampler('ROS', RandomOverSampler)
        self.set_sampler('SMOTE', SMOTE)
        self.set_sampler('SMOTE-TL', SMOTETomek)
        
    """Registra un modelo."""
    def add_model(self, name, clf_obj):
        self._classifiers.set(name, clf_obj)

    """Devuelve la lista de modelos."""
    def get_model_names(self):
        return self._classifiers.key_list()

    """Registra una métrica."""
    def set_metric(self, metric):
        self._metrics.set(metric, conf_matrix_scorer(metric.lower()))

    """Devuelve la lista de métricas registradas."""
    def get_metric_names(self):
        return self._metrics.key_list()

    """Registra las métricas por defecto."""
    def set_default_metrics(self):
        for metric in DEFAULT_METRICS:
            self.set_metric(metric)

    """Devuelve el nombre de las muestras. Si all=True incluye el dataset y Test."""
    def get_sample_names(self, all_samples=False):
        sample_names = self._samplers.key_list()
        if all_samples == True: 
            sample_names = [self.get_name(),'Train','Test'] + sample_names
        else:
            sample_names = ['Train'] + sample_names
        return sample_names
        
    """Genera las muestras con el factor de desequilibrio pasado como argumento."""
    def create_train_test(self, test_size=0.3, drop_columns=None):
        from sklearn.model_selection import train_test_split
        X, y, columns = self.load_dataset(self.get_name(), drop_columns=drop_columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y
        )
        self.save_dataset('Train', X_train, y_train, columns=columns, shuffle=True)
        self.save_dataset('Test', X_test, y_test, columns=columns, shuffle=True)
        
    """Genera las muestras con el factor de desequilibrio pasado como argumento."""
    def create_train_samples(self, **sampling_args):
        X_train, y_train, columns = self.load_dataset('Train')
        sample_names = self.get_sample_names(all_samples=False)        
        for sample_name in sample_names:
            sampler = self._samplers.call(sample_name, **sampling_args)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            self.save_dataset(sample_name, X_res, y_res, 
                              columns=columns, shuffle=True)
            
    """Genera las transfromaciones t-SNE de los datasets."""
    def create_tsne_samples(self, preprocess_fn=None, max_size=500000):
        from sklearn.model_selection import train_test_split
        from sklearn.manifold import TSNE
        columns = ['X1','X2',self.get_label_attr()]
        sample_names = self.get_sample_names(all_samples=True)        
        for sample_name in sample_names:
            X, y, _ = self.load_dataset(sample_name)
            if len(y) > max_size:
                _,X,_,y = train_test_split(X, y, test_size=max_size, stratify=y)
            if preprocess_fn is not None: X = preprocess_fn(X)
            X = TSNE(n_components=2).fit_transform(X)
            self.save_dataset('tsne-'+sample_name, X, y, 
                              columns=columns, shuffle=True)      
        
    """Entrena y evalúa un modelo. Devuelve los resultados de evaluación."""
    def fit_eval_model(self, model_name, sample_name, task, 
                       X_train, y_train, X_test, y_test, **fit_args):
        builder = self._classifiers.get(model_name)
        start = time.time()
        model = builder.fit(X_train, y_train, **fit_args)
        if task == 'Eval':
            filename = self.get_path(model_name + '-' + sample_name + '.joblib')
            joblib.dump(model, filename)
        end = time.time() - start
        y_pred = model.predict(X_test)
        scores = []
        for metric in self.get_metric_names():
            scores.append(self._metrics.call(metric, y_test, y_pred))
        scores.append(end)
        return scores
        
    """Evalúa el modelo. Si cv_fold != 0, aplica validación cruzada estratificada."""
    def eval_model(self, model_name, cv_folds=0):
        from sklearn.model_selection import StratifiedKFold
        X_test, y_test, _ = self.load_dataset('Test')
        total_scores = []
        sample_names = self.get_sample_names(all_samples=False)
        for sample_name in sample_names:
            X_train, y_train, _ = self.load_dataset(sample_name)
            X_train, X_test_norm = self.preprocess(model_name, X_train, X_test)
            scores = [model_name,sample_name,'Eval',0]
            scores+= self.fit_eval_model(model_name, sample_name, 'Eval',
                                         X_train, y_train, X_test_norm, y_test) 
            total_scores.append(scores)
            if cv_folds == 0: continue
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=False)            
            for n_fold, indexes in enumerate(skf.split(X_train, y_train)):
                train_idx, test_idx = indexes
                Xf_train, yf_train = X_train[train_idx], y_train[train_idx]
                Xf_test, yf_test = X_train[test_idx], y_train[test_idx]
                scores = [model_name,sample_name,'CrossVal',n_fold+1]
                scores+= self.fit_eval_model(model_name, sample_name, 'CrossVal',
                                             Xf_train, yf_train, 
                                             Xf_test, yf_test)
                total_scores.append(scores)
        # guardar resultados en un fichero CSV con el nombre del modelo
        columns = ['Model','Sample','Task','Fold'] 
        columns+= self.get_metric_names() + ['Time']
        df_scores = pd.DataFrame(total_scores, columns=columns)
        self.save_dataframe(df_scores, model_name+'-Scores')
    
    """Carga el modelo entrenado con la muestra indicada."""
    def load_model(self, model_name, sample_name):
        filename = self.get_path(model_name + '-' + sample_name + '.joblib')
        return joblib.load(filename)