# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:50:31 2022

@author: Kafein
"""
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler,SMOTE,BorderlineSMOTE,SVMSMOTE,ADASYN
from sklearn.decomposition import PCA
from numpy import where
from collections import Counter
from matplotlib import pyplot
import warnings
import streamlit as st
warnings.filterwarnings("ignore")


def control_df(df,cat=10):
    print("###############################################################")
    print("İlk 5 veri")
    print(df.head())
    print("###############################################################")
    print("Verinin boyutları:")
    print(df.shape)
    print("###############################################################")
    print("Verideki boş gözlem sayısı:")
    print(df.isnull().sum())
    cat_cols = [col for col in df.columns if df[col].dtype == "O"]
    num_but_cat_cols = [col for col in df.columns if df[col].nunique() < cat and df[col].dtype != "O"]
    cat_but_num_cols = [col for col in df.columns if df[col].nunique() > 20 and df[col].dtype == "O"]
    cat_cols = cat_cols + num_but_cat_cols
    num_cols = [col for col in df.columns if col not in cat_cols]
    print("###############################################################")
    print("Verideki kategorik değişkenler")
    print(cat_cols)
    print("###############################################################")
    print("Verideki sayısal değişkenler")
    print(num_cols)

    return cat_cols, num_cols

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
        
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
        

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def gg2(df, target,col_names):
    """
    Verilen col_names için target'a göre ortalamasının grafiğini çizer.
    :param df:
    :param target:
    :param col_names:
    :return:
    """
    col_names = col_names
    # col_names=df.columns[~df.columns.str.contains("charges")]
    print(col_names)
    for col in col_names:
        ss = df.groupby(col)[target].mean()
        plt.ylabel(target)
        plt.title(f"{col} adlı değişken kırılımında ortalama")
        plt.legend()
        # sns.set(rc={'figure.figsize': (8, 10)})
        sns.barplot(x=ss.index, y=ss.values)
        plt.show()
        print("\n")
        print(ss)
        print("######################\n\n")        

def plot_confusion_matrix(y, y_pred):
    from sklearn.metrics import accuracy_score
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


def scoring(model, x_train, y_train, x_test, y_test):


    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    #plot_confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    ss={"ACCURACY SCORE" : accuracy_score(y_test, y_pred),
        "PRECISION SCORE" : (cm[0, 0]) / (cm[0, 1] + cm[0, 0]),
        "RECALL SCORE" : recall_score(y_test, y_pred),
        "F1 SCORE" : f1_score(y_test, y_pred),
        "AUC SCORE" : roc_auc_score(y_test, y_prob)}
    #print(ss)
    return ss

def pca_plot(x,y):
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(x)
    counter = Counter(y)
    fig, ax = pyplot.subplots(figsize=(10, 5))
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        ax.scatter(pca_fit[row_ix, 0], pca_fit[row_ix, 1], label=str(label))
    ax.legend()
    #ax.show()
    return fig, ax


def preprocess_diabetes(df):
    cat_cols,num_cols=control_df(df,18)
    
    for col in num_cols:
        replace_with_thresholds(df,col)
    
    df["Is_Insufficent_Insulin"] = ["1" if val <126 else "0" for val in df["Insulin"]]

    df["Is_BMI_Over"] = ["1" if val >30 else "0" for val in df["BMI"]]
    
    for col in num_cols:
        ss = df[col].values.reshape(-1,1)
        transformer= MinMaxScaler().fit(ss)
        df[col] = transformer.transform(ss)
        
    df["Pregnancies"] = df["Pregnancies"].astype("O")
    #df.info()
    df = pd.get_dummies(df,drop_first=True)
    
    y=df["Outcome"]
    x=df.drop(["Outcome"],axis=1)
    
    return x, y


def normal(x, y):
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    log_model = LogisticRegression().fit(x_train, y_train)
    scores = scoring(log_model,x_train,y_train,x_test,y_test)
    fig, ax = pca_plot(x_train,y_train)
    st.subheader(f"Normal Recall Score: {scores['RECALL SCORE']}")
    st.pyplot(fig)
    
def random_oversampling(x, y):
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    log_model = LogisticRegression()
    oversample=RandomOverSampler(sampling_strategy="not majority")
    x_randomover,y_randomover = oversample.fit_resample(x_train,y_train)
    y_randomover.value_counts()
    log_model.fit(x_randomover,y_randomover)
    scores=scoring(log_model,x_randomover,y_randomover,x_test,y_test)
    fig, ax = pca_plot(x_randomover,y_randomover)
    st.subheader(f"Random Recall Score: {scores['RECALL SCORE']}")
    st.pyplot(fig)
    
def smote(x, y):
    oversample=SMOTE()
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    log_model = LogisticRegression()
    x_smote,y_smote = oversample.fit_resample(x_train,y_train)
    y_smote.value_counts()
    log_model.fit(x_smote,y_smote)
    scores=scoring(log_model,x_smote,y_smote,x_test,y_test)
    fig, ax = pca_plot(x_smote,y_smote)
    st.subheader(f"Smote Recall Score: {scores['RECALL SCORE']}")
    st.pyplot(fig)
    
def border(x, y):
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    log_model = LogisticRegression()
    oversample = BorderlineSMOTE()
    x_border,y_border = oversample.fit_resample(x_train,y_train)
    y_border.value_counts()
    log_model.fit(x_border,y_border)
    scores = scoring(log_model,x_border,y_border,x_test,y_test)
    fig, ax = pca_plot(x_border,y_border)
    st.subheader(f"Border Recall Score: {scores['RECALL SCORE']}")
    st.pyplot(fig)
    
def svm(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    log_model = LogisticRegression()
    oversample = SVMSMOTE()
    x_svm, y_svm = oversample.fit_resample(x_train.values, y_train)
    y_svm.value_counts()
    log_model.fit(x_svm, y_svm)
    scores = scoring(log_model,x_svm,y_svm,x_test,y_test)
    fig, ax = pca_plot(x_svm,y_svm)
    st.subheader(f"SVM Recall Score: {scores['RECALL SCORE']}")
    st.pyplot(fig)
    
def adasyn(x, y):
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    log_model = LogisticRegression()
    oversample=ADASYN()
    x_ada,y_ada = oversample.fit_resample(x_train,y_train)
    y_ada.value_counts()
    log_model.fit(x_ada,y_ada)
    scores = scoring(log_model,x_ada,y_ada,x_test,y_test)
    fig, ax = pca_plot(x_ada,y_ada)
    st.subheader(f"ADASYN Recall Score: {scores['RECALL SCORE']}")
    st.pyplot(fig)
        