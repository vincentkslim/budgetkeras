from budgetkeras import *

import pandas as pd
import seaborn as sns

# Read in sample datasets (This is the titanic dataset from kaggle)
titanic_X = pd.read_csv('titanic_X.csv', index_col='PassengerId')
titanic_Y = pd.read_csv('titanic_Y.csv', index_col='PassengerId')

titanic_train_Y = titanic_Y.to_numpy().reshape(1, 891) 
titanic_train_X = titanic_X.to_numpy() 

# set up the model
add, compile_model, summary = sequential()
add(dense(64, input_shape=(16,), activation=relu, initializer=kaiming))
add(dense(64, activation=relu, initializer=kaiming))
add(dense(1, activation=sigmoid, initializer=xavier))

fit = compile_model(gradient_descent(), binary_crossentropy) 

history, predict = fit(titanic_train_X, titanic_train_Y, epochs=500)
