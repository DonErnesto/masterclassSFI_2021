"""
TO-DO:
+ Get rid of 2 sample functions (makes no sense)
+ Tree depth: integer
- Single figure with the "training data" and the fitted classifier (
if you press "fit", you see the decision boundary and predictions.
Before only the training data. Predictions: indicate TN, TP, FP, FN
)
- Add good titles
- Second figure with predictions on all data  (or a different dataset: test set)
- Metrics in both figures: Recall, Precision, FPR, F1-score. Or a confusion matrix


"""

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
# from sklearn import datasets
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly
from numpy.random import RandomState
from sklearn.tree import DecisionTreeClassifier


st.title('Decision Tree')
n = st.sidebar.selectbox('dataset size', [100, 500, 1000])
random_seed = st.sidebar.number_input('ID', value=1)
max_depth = st.sidebar.number_input('max tree depth', min_value=1, max_value=25, value=1)

x1_label = 'Transaction Volume'
x2_label = 'AUM'
y_label = 'Outcome'


@st.cache
def generate_Xy(seed=1, n=100):
    n1, n2, n3, n4 = int(n*0.25), int(n*0.25), int(n*.05), int(n*.05)
    rng = np.random.default_rng(seed)
    X1 = rng.multivariate_normal(mean=[1.9, 3.0],
                cov=[[0.2, -0.12], [-0.12, 0.2]], size=n1)
    X2 = rng.multivariate_normal([1.25, 2.75], [[0.12, 0], [0, 0.13]], n2)
    X3 = rng.multivariate_normal([+3.5, 3], [[0.1, 0.06], [0.06, 0.1]], n3)
    X4 = rng.multivariate_normal([+2, 1.5], [[0.1, 0], [0, 0.1]], n4)
    X = np.concatenate((X1, X2, X3, X4))
    y = np.concatenate((np.zeros(n1 + n2), np.ones(n3 + n4)))
    return X, y

# @st.cache
# def draw_n_from_Xy(X, y, n=100, seed=1):
#     rng = np.random.default_rng(seed)
#     idx = np.random.choice(X.shape[0], n, replace=False)
#     X_sample = X[idx]
#     y_sample = y[idx]
#     sort_ids = y_sample.argsort()
#     X_sample = X_sample[sort_ids]
#     y_sample = y_sample[sort_ids]
#     return X_sample, y_sample




# Here: draw at random from X, y
# X_, y_ = generate_Xy()
X, y = generate_Xy(n=n, seed=random_seed)



df = pd.DataFrame({x1_label: X[:, 0], x2_label: X[:, 1], y_label: y})
df[y_label] = df[y_label].astype(int).astype(str)

def draw_data():
    fig = go.Figure(go.Scatter(x=X[:, 0], y=X[:, 1],
                        mode='markers',
                        showlegend=False,
                        marker=dict(size=10,
                                    color=y,
                                    colorscale='portland',
                                    line=dict(color='black', width=1))
                        )
                         )


    fig.update_layout(yaxis=dict(range=[0, 5]))
    fig.update_layout(xaxis=dict(range=[0, 5]))
    return fig




# Fitting a tree to the dataset

if st.sidebar.button('Fit classifier'):
    tree = DecisionTreeClassifier(max_depth=max_depth)
    h = 0.02
    x_min, x_max = 0, 5
    y_min, y_max = 0, 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h)
                         , np.arange(y_min, y_max, h))
    zz = tree.fit(X, y).predict(np.c_[xx.ravel(), yy.ravel()])
    y_hat = tree.fit(X, y).predict(X)

    trace1 = go.Heatmap(x=xx.ravel(), y=yy.ravel(), z=zz,
                  colorscale='portland',
                  showscale=False)
    trace2 = go.Scatter(x=X[:, 0], y=X[:, 1],
                    mode='markers',
                    showlegend=False,
                    marker=dict(size=10,
                                color=y_hat,
                                colorscale='portland',
                                line=dict(color='black', width=1))
                    )
    #fig = plotly.subplots.make_subplots(rows=1, cols=1,
    #                       subplot_titles=("Random Forest (Depth = 4)",))
    #fig = draw_data()
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    #fig.update_layout(yaxis=dict(range=[0, 5]))
    #fig.update_layout(xaxis=dict(range=[0, 5]))
    st.plotly_chart(fig)
else:
    fig = draw_data()
    st.plotly_chart(fig)



# create a meshgrid for visualizing the decision boundary
comment = r"""
x_min, x_max = 0, 5
y_min, y_max = 0, 5
h = 0.25

xx, yy = np.meshgrid(np.arange(x_min, x_max, h)
                     , np.arange(y_min, y_max, h))
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) """




# X, y = generate_Xy()
#
# fig, ax = plt.subplots(figsize=(6, 6))
# plt.xlabel("X0", fontsize=20)
# plt.ylabel("X1", fontsize=20)
# plt.scatter(X[:,0], X[:,1], s=60, c=y)
#
# st.pyplot(fig)
