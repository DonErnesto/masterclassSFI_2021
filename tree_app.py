"""
TO-DO:
+ Get rid of 2 sample functions (makes no sense)
+ Tree depth: integer
+ Single figure with the "training data" and the fitted classifier (
if you press "fit", you see the decision boundary and predictions.
Before only the training data. Predictions: indicate TN, TP, FP, FN
)
- adapt meshgrid size with max_depth
- Add titles
- Second figure with predictions on all data  (or a different dataset: test set)
- Metrics in both figures: Recall, Precision, FPR, F1-score. Or a confusion matrix
- Show the actual decision tree

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


# constants
y_label = 'Outcome'
x_title, y_title = 'Transaction Volume', 'AUM'

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

def update_figlayout(fig, x_title, y_title, x_min=0, x_max=5, y_min=0, y_max=5):
    fig.update_layout(xaxis=dict(range=[x_min, x_max]))
    fig.update_layout(yaxis=dict(range=[y_min, y_max]))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)

def draw_raw_data(X, y):
    fig = go.Figure(go.Scatter
                        (x=X[:, 0], y=X[:, 1],
                        mode='markers',
                        showlegend=False,
                        marker=dict(size=8,
                                    color=y,
                                    colorscale='portland',
                                    line=dict(color='black', width=1))
                        )
                    )
    update_figlayout(fig, x_title=x_title, y_title=y_title)
    return fig

def generate_2d_grid(x_min=0, x_max=5, y_min=0, y_max=5, h=0.02):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    return xx.ravel(), yy.ravel()

def make_classification_traces(X, y, y_hat):
    TN_idx = (y == 0) & (y_hat == 0)
    TP_idx = (y == 1) & (y_hat == 1)
    FN_idx = (y == 1) & (y_hat == 0)
    FP_idx = (y == 0) & (y_hat == 1)
    marker_list = ['circle', 'circle', 'x', 'x']
    color_list = ['blue', 'red', 'red', 'blue']
    size_list = [8, 8, 10, 10]
    lw_list = [1, 1, 2, 2]
    trace_list = []
    for marker, color, size, lw, idx in zip(
            marker_list,
            color_list,
            size_list,
            lw_list,
            [TN_idx, TP_idx, FP_idx, FN_idx]
                                ):
        trace_list.append(go.Scatter(x=X[idx, 0], y=X[idx, 1],
                    mode='markers',
                    marker_symbol=marker,
                    showlegend=False,
                    marker=dict(size=size,
                                color=color,
                                line=dict(color='black', width=lw))
                    ))
    return trace_list


# Widgets
st.title('Training Data')
n = st.sidebar.selectbox('Dataset size', [100, 500, 1000])
random_seed = st.sidebar.number_input('Population ID', value=1)
max_depth = st.sidebar.number_input('Max. Tree Depth', min_value=1, max_value=25, value=1)
fit_predict_clf = st.sidebar.button('Train and Predict')
remove_clf = st.sidebar.button('Remove Predictions')

# Main App
X, y = generate_Xy(n=n, seed=random_seed)

show_clf = False
if fit_predict_clf:
    show_clf = True
if remove_clf:
    show_clf = False

if show_clf:
    tree = DecisionTreeClassifier(max_depth=max_depth)
    h = 0.025 if max_depth > 5 else 0.05
    xx, yy = generate_2d_grid(h=h)
    zz = tree.fit(X, y).predict(np.c_[xx, yy])
    y_hat = tree.fit(X, y).predict(X)


    trace1 = go.Heatmap(x=xx, y=yy, z=zz,
                  colorscale='portland',
                  opacity=0.2,
                  showscale=False)
    trace_TN, trace_TP, trace_FP, trace_FN = make_classification_traces(
                                            X, y, y_hat)
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace_TN)
    fig.add_trace(trace_TP)
    fig.add_trace(trace_FN)
    fig.add_trace(trace_FP)

    update_figlayout(fig, x_title=x_title, y_title=y_title)
    st.plotly_chart(fig)
else:
    fig = draw_raw_data(X, y)
    update_figlayout(fig, x_title=x_title, y_title=y_title)
    st.plotly_chart(fig)
