# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:40:41 2022

@author: Kafein
"""

import pandas as pd
import numpy as np
import streamlit as st
from helpers import *


# %%

st.write(""" ### Dengesiz VeriSetinde OverSampling Yöntemleri
         
         """)
st.write("""
         Dengesiz veriseti sınıflandırılacak sınıflardan birinin diğerine oranla yetersiz olması durumunda model performasını doğrudan etkileyen problemlerden birisidir.
Bu çalışmada dengesiz veri setinde oversampling yöntemlerini ve etkileri gözlemlenecektir.
         """)
df = pd.read_csv("data/diabetes.csv")

st.subheader('Veri dağılımı')
st.write(df["Outcome"].value_counts())


x, y = preprocess_diabetes(df)



# %%

input_normal = st.checkbox("Normal")
input_ro = st.checkbox("Random")
input_smote = st.checkbox("SMOTE")
input_border = st.checkbox("Border")
input_svm = st.checkbox("SVM")
input_ada = st.checkbox("ADASYN")

if input_normal:
    normal(x, y)

if input_ro:
    random_oversampling(x, y)

if input_smote:
    smote(x, y)

if input_border:
    border(x, y)

if input_svm:
    svm(x, y)
    
if input_ada:
    adasyn(x, y)


    



