import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
# from sklearn import datasets
import matplotlib.pyplot as plt
import plotly.express as px
from numpy.random import RandomState


st.title('Decision Tree')
random_seed = st.sidebar.number_input('ID', value=1)

x1_label = 'Transaction Volume'
x2_label = 'AUM'
y_label = 'Outcome'

n1, n2, n3, n4 = 1000, 1000, 100, 100

@st.cache  #
def generate_Xy():
    X1 = np..multivariate_normal(mean=[-0.2, 1.0], cov=[[0.2, -0.12], [-0.12, 0.2]], size=n1)
    X2 = np.random.multivariate_normal([-1.5, 0.75], [[0.12, 0], [0, 0.13]], n2)
    X3 = np.random.multivariate_normal([+1.5, 1], [[0.1, 0.06], [0.06, 0.1]], n3)
    X4 = np.random.multivariate_normal([+0, -0.5], [[0.1, 0], [0, 0.1]], n4)
    X = np.concatenate((X1, X2, X3, X4))
    y = np.concatenate((np.zeros(n1 + n2), np.ones(n3 + n4)))

    return X, y

@st.cache  #
def draw_n_from_Xy(X, y, n=100):
    idx = np.random.choice(X.shape[0], n, replace=False)
    X_sample = X[idx]
    y_sample = y[idx]
    sort_ids = y_sample.argsort()
    X_sample = X_sample[sort_ids]
    y_sample = y_sample[sort_ids]
    return X_sample, y_sample

# Here: draw at random from X, y
X_, y_ = generate_Xy()
X, y = draw_n_from_Xy(X_, y_, n=random_seed)

rng = np.random.default_rng(12345)



df = pd.DataFrame({x1_label: X[:, 0], x2_label: X[:, 1], y_label: y})
df[y_label] = df[y_label].astype(int).astype(str)
fig = px.scatter(df,
                x=x1_label,
                y=x2_label,
                color=y_label,
                #hover_name='y',
                title='2-D Feature space with binary labels')
st.plotly_chart(fig)


# X, y = generate_Xy()
#
# fig, ax = plt.subplots(figsize=(6, 6))
# plt.xlabel("X0", fontsize=20)
# plt.ylabel("X1", fontsize=20)
# plt.scatter(X[:,0], X[:,1], s=60, c=y)
#
# st.pyplot(fig)
