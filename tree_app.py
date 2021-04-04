import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt


st.title('Decision Tree')

X, y = datasets.make_classification(n_samples=200, n_features=2,
                                    n_informative=2, n_redundant=0,
                                    n_classes=2, n_clusters_per_class=2,
                                   weights=(0.9, 0.1), class_sep=1.0,
                                    random_state=5
                                   )


fig, ax = plt.subplots(figsize=(6, 6))
plt.xlabel("X0", fontsize=20)
plt.ylabel("X1", fontsize=20)
plt.scatter(X[:,0], X[:,1], s=60, c=y)
