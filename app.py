# """### Importing the libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.title("Fraud Detection Using SOM(Self-Organizing-Maps)")

# """## Importing the dataset"""

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

st.subheader('Dataset')
st.dataframe(dataset.head())
st.dataframe(dataset.tail())

# """## Feature Scaling"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# """##Training the SOM"""

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

st.subheader('Training the SOM')
code='''from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)'''
st.code(code, language='python')

# """##Visualizing the results"""
st.subheader('Visualizing the results')
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


# """## Finding the frauds"""

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,1)], mappings[(4,1)]), axis = 0)
frauds = sc.inverse_transform(frauds)


# """##Printing the Fraud Clients"""
st.subheader('Fraud Customer IDs')
print('Fraud Customer IDs')
for i in frauds[:, 0]:
    print(int(i))
    st.write(int(i))

# import pickle
# pickle_out = open("som.pkl", "wb")
# pickle.dump(som, pickle_out)
# pickle_out.close()
