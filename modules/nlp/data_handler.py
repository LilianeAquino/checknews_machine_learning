import multiprocessing
import pandas as pd
import warnings
import numpy as np
from cloudpickle import dump
from collections import namedtuple, deque
from pandas.core.frame import DataFrame
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score, log_loss, roc_auc_score, roc_curve


cores = multiprocessing.cpu_count()
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True, precision=3)


stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
   

def labelEncoder(labels: list) -> namedtuple:
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    Labels = namedtuple('encoded_labels', ['labels', 'classes', 'label_encoder'])
    encoded_labels_tuple = Labels(encoded_labels, label_encoder.classes_.tolist(), label_encoder)
    return encoded_labels_tuple


def kFoldSplitTrainTest(data: pd.DataFrame, encoded_labels: tuple, columns: list) -> namedtuple:   
    k_fold = stratified_k_fold.split(data[columns], encoded_labels.labels)
    last_element = deque(k_fold, maxlen=1)
    train_idx, test_idx = last_element.pop()

    X_train = data[columns].iloc[train_idx]
    X_test =  data[columns].iloc[test_idx]
    y_train = encoded_labels.labels[train_idx]
    y_test = encoded_labels.labels[test_idx]
    return X_train, X_test, y_train, y_test


data_model = namedtuple('model_export', ['model', 'label_encoder'])

def persistModel(model: Pipeline, encoded_labels: LabelEncoder, model_name: str, data_model) -> None:
    model = data_model(model, encoded_labels)
    dump(model, open('model/'+model_name+'.sav', 'wb'))
    

def getMetrics(y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> tuple:
    """
        Gera as métricas do modelo
    """
    accuracy = accuracy_score(y_test, y_pred).round(3)
    recall = recall_score(y_test, y_pred, average='weighted').round(3)
    precision = precision_score(y_test, y_pred, average='weighted').round(3)
    f1 = f1_score(y_test, y_pred, average='weighted').round(3)
    log = log_loss(y_test, y_prob).round(3)
    auc = roc_auc_score(y_test, y_pred).round(3)
    return accuracy, recall, precision, f1, log, auc


def getClassificationReport(y_test: list, y_pred: list, encoded_labels_names: list, flag: bool = False) -> dict:
    """
        Gera o relatório de classificação
    """
    return classification_report(y_test, y_pred, output_dict=flag, target_names=encoded_labels_names)


def confusionMatrix(title: str, model: any, X_test: any, y_test: np.ndarray, encoded_labels_names: list) -> pyplot:
    """
        Gera a matriz de confusão
    """
    np.set_printoptions(precision=2)
    plt.rcParams['figure.figsize'] = [10, 5]

    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=encoded_labels_names,
        cmap=plt.cm.afmhot_r,
        normalize='true',
        xticks_rotation='vertical'
    )
    disp.ax_.set_title(title, y=1.05)
    return pyplot


def curvaRoc(label: str, y_test: np.ndarray, y_pred: np.ndarray, auc: float) -> pyplot:
    """
        Gera a curva ROC
    """   
    fig, ax = plt.subplots(figsize=(10, 5))
    np.set_printoptions(precision=2)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    ax.plot(fpr, tpr, label=label+'auc = '+str(auc))
    ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC')
    plt.legend()