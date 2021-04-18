"""
TO-DO:
+ Get rid of 2 sample functions (makes no sense)
+ Tree depth: integer
+ Single figure with the "training data" and the fitted classifier (
if you press "fit", you see the decision boundary and predictions.
Before only the training data. Predictions: indicate TN, TP, FP, FN
)
+ adapt meshgrid size with max_depth
+ Add titles

+ Radio button widget for training
+ Separate fitting from plotting
+ Separate train and test data
- Metrics (Confusion matrix, recall and precision)
- Second figure with predictions on all data  (or a different dataset: test set)
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
from sklearn.tree import DecisionTreeClassifier, plot_tree
import sklearn.metrics

# constants
local_grid = False
y_label = 'Outcome'
x_title, y_title = 'Transaction Volume', 'AUM'

@st.cache
def generate_Xy(seed=1, n=100):
    n1, n2, n3, n4 = int(n*0.45), int(n*0.45), int(n*.05), int(n*.05)
    rng = np.random.default_rng(seed)
    X1 = rng.multivariate_normal(mean=[1.9, 3.0],
                cov=[[0.2, -0.12], [-0.12, 0.2]], size=n1)
    X2 = rng.multivariate_normal([1.25, 2.75], [[0.12, 0], [0, 0.13]], n2)
    X3 = rng.multivariate_normal([+3.5, 3], [[0.1, 0.06], [0.06, 0.1]], n3)
    X4 = rng.multivariate_normal([+2, 1.5], [[0.1, 0], [0, 0.1]], n4)
    X = np.concatenate((X1, X2, X3, X4))
    y = np.concatenate((np.zeros(n1 + n2), np.ones(n3 + n4)))
    return X, y

def update_figlayout(fig, x_title, y_title, x_min=0, x_max=5, y_min=0, y_max=5,
        title=''):
    fig.update_layout(xaxis=dict(range=[x_min, x_max]))
    fig.update_layout(yaxis=dict(range=[y_min, y_max]))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    fig.update_layout(title=title, title_x=0.5)

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

def generate_local_2d_grid():
    """ Combines a rough global grid refined with the split points.
    Is faster, but somehow the plotly heatmap does not quite correspond.
    Are the values set on the edges or the cell centers?
    """
    xx_ = np.sort(np.concatenate( (np.arange(0, 5.1, 0.1),
                             tree.tree_.threshold[tree.tree_.feature == 0] - 1E-6
                    )
                  ))
    yy_ = np.sort(np.concatenate( (np.arange(0, 5.1, 0.1),
                             tree.tree_.threshold[tree.tree_.feature == 1] - 1E-6
                    )
                  ))
    xx, yy = np.meshgrid(xx_, yy_)
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

def make_basic_fig(X, y):
    fig = draw_raw_data(X, y)
    update_figlayout(fig, x_title=x_title, y_title=y_title)
    st.plotly_chart(fig)

# Widgets
st.title('Decision Tree on 2-D Data')
n_train = st.sidebar.selectbox('Train Data size', [100, 500, 1000])
random_seed_train = st.sidebar.number_input('Train Population ID', min_value=0, max_value=1000, value=1)
n_test = st.sidebar.selectbox('Test Data size', [100, 500, 1000])
random_seed_test = st.sidebar.number_input('Test Population ID', min_value=0, max_value=1000, value=1)


max_depth = st.sidebar.number_input('Max. Tree Depth', min_value=1, max_value=10, value=1)
fit_predict_clf = st.sidebar.checkbox('Train and Show Predictions', value=False)
show_traintest_data = st.sidebar.radio('Show Train or Test data', ['train', 'test'])
#local_grid = st.sidebar.checkbox('Local grid refinement', value=False)
fine_grid = st.sidebar.checkbox('Fine grid for rendering (slower)', value=False)
# remove_clf = st.sidebar.button('Remove Predictions')

# Main App
X_train, y_train = generate_Xy(n=n_train, seed=random_seed_train)
X_test, y_test = generate_Xy(n=n_test, seed=random_seed_test)


if fit_predict_clf:
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    clf = tree.fit(X_train, y_train)
    # Make X,y scatter traces
    if show_traintest_data == 'train':
        y_hat_plot = clf.predict(X_train)
        y_plot = y_train
        X_plot = X_train
    else:
        y_hat_plot = clf.predict(X_test)
        y_plot = y_test
        X_plot = X_test
    trace_TN, trace_TP, trace_FP, trace_FN = make_classification_traces(
                                                X_plot, y_plot, y_hat_plot)
    # Make Decisionboundary trace
    h = 0.02 if max_depth > 3 else 0.04
    if fine_grid:
        h = h * 0.5
    if local_grid:
        xx, yy = generate_local_2d_grid()
    else:
        xx, yy = generate_2d_grid(h=h)

    zz = clf.predict(np.c_[xx, yy])
    trace_db = go.Heatmap(x=xx, y=yy, z=zz,
                  colorscale='portland',
                  opacity=0.2,
                  showscale=False)

    recall = sklearn.metrics.recall_score(y_plot, y_hat_plot)
    precision = sklearn.metrics.precision_score(y_plot, y_hat_plot)
    accuracy = sklearn.metrics.accuracy_score(y_plot, y_hat_plot)



else:
    tree = None


if not tree is None:
    fig = go.Figure()
    fig.add_trace(trace_db)
    fig.add_trace(trace_TN)
    fig.add_trace(trace_TP)
    fig.add_trace(trace_FN)
    fig.add_trace(trace_FP)
    title_head = '    Train Data: ' if show_traintest_data == 'train' else '    Test Data: '
    update_figlayout(fig, x_title=x_title, y_title=y_title,
                    title=f'{title_head} Accuracy = {accuracy:.1%}, '\
                          f'Recall = {recall:.1%}, Precision = {precision:.1%}')
    st.plotly_chart(fig)

    y_actu = pd.Series(y_plot, name='Actual Label').astype(int)
    y_pred = pd.Series(y_hat_plot, name='Predicted Label').astype(int)

    df_confusion = pd.crosstab(y_actu, y_pred, margins=True)
    df_confusion.columns = ['Predicted N', 'Predicted P','All']
    df_confusion.index = ['Actual N', 'Actual P', 'All']

    st.dataframe(df_confusion, 400, 400)
else:
    if show_traintest_data == 'train':
        make_basic_fig(X_train, y_train)
    else:
        make_basic_fig(X_test, y_test)
