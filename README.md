# budgetkeras
## A totally impractical but functional clone of the keras Sequential Model
### Introduction
This fun project is a clone of the keras Sequential Model but only
using numpy. tensorflow is used at all, and numpy is the only package that is
imported. The main purpose of this project was to help me understand all the
math behind a fully connected neural network.

This also features a few nifty features in Python that I learned during my time
in CS61A, including: 
- Higher Order Functions
- Mutable Functions (`nonlocal` keyword)
- Data Abstraction

### Supported Features
- Layers
	- Dense (fully connected)
- Weight Initializers
	- Random Normal
	- Xavier
	- Kaiming (He)
- Activation Functions
	- ReLU
	- sigmoid
	- tanh
- Optimizers
	- Batch Gradient Descent

### TODO
- Layers
	- Conv
	- Pooling
- Optimizers
	- Mini-batch Gradient Descent
	- RMSprop
	- Adam