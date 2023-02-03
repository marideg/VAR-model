from statsmodels.tsa.api import AutoReg,VAR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def var_model(data,data_names,unit_names,parameters):
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
	lags = parameters['p']
	dict_residuals = {}
	dict_residuals_squared = {}

	# Fit AR/VAR model
	if len(data_names) == 1:
		# Retrieve data
		data_values = data[data_names[0]]

		# Fit AR model
		model = AutoReg(data_values,lags,trend='n')
		residuals = model.fit()
		estimated_parameters = residuals.params
		print("AR model parameters: ",estimated_parameters)

		# Store residuals
		dict_residuals[data_names[0]] = residuals.resid
		dict_residuals_squared[data_names[0]] = np.power(residuals.resid,2)

		# Return AR coefficients
		coefficients = estimated_parameters

	else:
		# Retrieve data
		data_values = pd.DataFrame.from_dict(data)

		# Fit VAR model
		model = VAR(data_values)
		residuals = model.fit(lags,trend='n')
		estimated_parameter_matrices = residuals.params
		print("VAR fit: ",residuals.summary())

		# Store residuals
		for l in data_names:
			res = residuals.resid[l].values
			dict_residuals[l] = res
			dict_residuals_squared[l] = np.power(res,2)

		# Return VAR coefficients
		coefficients = estimated_parameter_matrices

	return dict_residuals,dict_residuals_squared











