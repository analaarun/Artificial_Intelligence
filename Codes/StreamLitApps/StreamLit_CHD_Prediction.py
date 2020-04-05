import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import io
from copy import deepcopy
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

selOption = st.sidebar.radio(
    "Select Option", ('Brief Exploration', 'Correlation', 'Fix Outlier', 'Build Model'))

@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist=True)
def load_data():
    return pd.read_csv('saheart.dat', sep=',', header=13, names=['Sbp', 'Tobacco', 'Ldl', 'Adiposity', 'Famhist', 'Typea', 'Obesity', 'Alcohol', 'Age', 'Chd'])


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def train_model(X_train, y_train):
    logreg = LogisticRegression(solver='liblinear', random_state=0)
    return logreg.fit(X_train, y_train)

df = load_data()
df_1 = deepcopy(df)
df_1.Famhist.replace(['Absent', 'Present'], [0, 1], inplace=True)

if selOption == 'Brief Exploration':
    st.title('Coronary Heart Disease Dataset')
    st.text('''
            sbp		systolic blood pressure
            tobacco		cumulative tobacco(kg)
            ldl		low densiity lipoprotein cholesterol
            adiposity
            famhist		family history of heart disease(Present, Absent)
            typea		type-A behavior
            obesity
            alcohol		current alcohol consumption
            age		age at onset
            chd		response, coronary heart disease
            ''')

    st.header('Load Data')

    st.code('''
    df = pd.read_csv('saheart.dat', sep=',', header=13, 
    names=['Sbp', 'Tobacco', 'Ldl', 'Adiposity',
    'Famhist', 'Typea', 'Obesity', 'Alcohol', 'Age', 'Chd'])
    ''',language='python')
    st.table(df.head())
    st.write('Shape : ', df.shape)
    st.header('Brief Exploration')

    st.write('DF Info : ')
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write('Column Data Types : ', df.dtypes)
    st.write('Since Famhist type is object')
    st.code('''
    df_1.Famhist.replace(['Absent', 'Present'], [0, 1], inplace=True)
    ''', language='python')
    if st.checkbox('Describe'):
        st.table(df_1.describe())

elif selOption == 'Correlation':
    st.write('Correlation Tab')
    if st.checkbox('Pair Plot'):
        sns.pairplot(df_1, diag_kind='kde', hue='Chd')
        st.pyplot()


elif selOption == 'Fix Outlier':
    st.write('Fix Outlier Tab')
    plt.figure(figsize=(15, 10))
    pos = 1
    for i in df_1.drop(columns='Chd').columns:
        plt.subplot(3, 3, pos)
        sns.boxplot(df_1[i])
        pos += 1
    st.pyplot()
    if st.checkbox('Fix Outlier'):
        X = df_1.drop(columns='Chd')
        y = df_1.Chd
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=1)
        st.write('X_train.shape', X_train.shape, 'X_test.shape',
                 X_test.shape, 'y_train.shape', y_train.shape, 'y_test.shape', y_test.shape)
        for i in X_train.columns:
            q1, q2, q3 = X_train[i].quantile([0.25, 0.5, 0.75])
            IQR = q3 - q1
            a = X_train[i] > q3 + 1.5*IQR
            b = X_train[i] < q1 - 1.5*IQR
            X_train[i] = np.where(a | b, q2, X_train[i])
        plt.figure(figsize=(15, 10))
        pos = 1
        for i in X_train.columns:
            plt.subplot(3, 3, pos)
            sns.boxplot(X_train[i])
            pos += 1
        st.pyplot()

elif selOption == 'Build Model':
    X = df_1.drop(columns='Chd')
    y = df_1.Chd
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1)
    st.write('X_train.shape', X_train.shape, 'X_test.shape',
             X_test.shape, 'y_train.shape', y_train.shape, 'y_test.shape', y_test.shape)
    for i in X_train.columns:
        q1, q2, q3 = X_train[i].quantile([0.25, 0.5, 0.75])
        IQR = q3 - q1
        a = X_train[i] > q3 + 1.5*IQR
        b = X_train[i] < q1 - 1.5*IQR
        X_train[i] = np.where(a | b, q2, X_train[i])

    logreg = train_model(X_train, y_train)
    pred = logreg.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    lr_score = logreg.score(X_test, y_test)
    # Of all the people with Chd, how many were recognised to have Chd
    lr_recall = round(tp/(tp+fn), 3)
    # Of all the people predicted to have Chd, how many did have Chd
    lr_precision = round(tp/(tp+fp), 3)
    # Of all the people without Chds, how many were recognised to not have Chd
    lr_specificity = round(tn/(tn+fp), 3)

    result = pd.DataFrame({'Model': ['Logistic Regression'], 'Accuracy': [lr_score], 'Precision': [lr_precision],
                        'True positive rate': [lr_recall], 'True negative rate': [lr_specificity],
                        'False positive rate':  [1-lr_specificity]})
    st.table(result)

    Sbp = st.number_input('Sbp : ', value=130)
    Tobacco = st.number_input('Tobacco : ', value=2.5)
    Ldl = st.number_input('Ldl : ', value=3.66)
    Adiposity = st.number_input('Adiposity : ', value= 30.9)
    Famhist = st.radio('Famhist', (0, 1))
    Typea = st.number_input('Typea : ', value =54)
    Obesity = st.number_input('Obesity : ', value=27.59)
    Alcohol = st.number_input('Alcohol : ', value=15.11)
    Age = st.slider('Age',10,100)

    inputPrediction = logreg.predict([[Sbp, Tobacco, Ldl, Adiposity,
                    Famhist, Typea, Obesity, Alcohol, Age]])
    
    st.subheader('Prediction')
    st.write('Coronary Heart Disease : ', inputPrediction[0])
    if inputPrediction[0] == 1:
        st.write('Have Coronary Heart Disease')
    else:
        st.write('Not have Coronary Heart Disease')

