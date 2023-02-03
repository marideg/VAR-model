from statsmodels.tsa.api import VARMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def varma_model(data,data_names,unit_names,parameters):
	"""
	* Fits a VAR(p) model to given data.  

	 CALLS FUNCTIONS:

	 INPUT:
	* data (dict): Input-data for each dimension in the final VAR model.
	* data_names (array): Keys of data (dict).
	* unit_names (dict): Unit of each input dataset from data (dict).
	* parameters (dict): Contain number of lags, p, for VAR model
	
	 OUTPUT:
	* 
	"""
	print(' ')
	print(' ')
	print('********** RESULTS FROM var_model() FOLLOWS **********')

	# Define parameters and dictionaries
	ar_lags = parameters['p']
	ma_lags = parameters['q']
	dict_residuals = {}
	dict_residuals_squared = {}

	# VARMA
	# Retrieve data
	data_values = pd.DataFrame.from_dict(data)

	# Fit VAR model
	model = VARMAX(data_values,order=(ar_lags,ma_lags))
	residuals = model.fit(maxiter=1000,disp=False)
	print(residuals.summary())
	estimated_parameter_matrices = residuals.params
	print("VAR model parameters: ",estimated_parameter_matrices.keys())

	# Store residuals
	"""
	for l in data_names:
		res = residuals.resid[l].values
		dict_residuals[l] = res
		dict_residuals_squared[l] = np.power(res,2)
	
	"""
	# Return VAR coefficients
	coefficients = estimated_parameter_matrices
	

	return coefficients #dict_residuals,dict_residuals_squared,











