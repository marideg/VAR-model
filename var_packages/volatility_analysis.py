import fourier_fit_functions as ff
import matplotlib.pyplot as plt
import numpy as np
import date_and_day as dd
from symfit import variables,CallableModel
import plot_functions as pl



def expected_value_year(data,days_of_year,num_of_years,num_of_lags=0):
	"""
	# Computes expected value of data every day during a year (of given length).
	  Only complete year datasets as input in this version. However, the function 
	  adjusts for first datapoints which are used as input to the VAR(p) model. 

	INPUT:
	# data_arrays (array of arrays): The datasets (of complete years)
	# data_names (arrays of strings): Names of the respective given datasets.
	# days_of_year (int): How long is your year
	# num_of_years (int): How many years of data
	# num_of_lags (int): Number of points used in a the VAR(p) model (p number of lags)

	OUTPUT:
	# expected_values (array): Expected value every day of a year
	# count (int): Test count (check if the correct number of datapoints used in calculation)
	"""
	expected_values = []
	count = 0
	for i in range(days_of_year):
		day = []
		for j in range(num_of_years):
			if j == 0 and i in range(num_of_lags,days_of_year):
				day.append(data[i-num_of_lags])
				count += 1
			if j != 0:
				day.append(data[days_of_year*j + (i-num_of_lags)])	
				count += 1    
		expected_values.append(np.mean(day))
	print("Number of terms added in expected value calculation: ",count)
	return expected_values,count







def volatility_function(data,data_names,unit_names,parameters,fourier_frequency,\
						fourier_terms,merge_parameters,figure_names,label_names):
	"""
	#Divides given dataset as a result of given cutoff restrictions and fits a Fourier series to each
	 part (three parts from two cuttoffs). Then two sigmoid functions are used to merge the 
	 three fitted Fourier series, such that the resulting (season) volatility function becomes smooth.
	 The resulting function is plotted. Then the given residual data are scaled by pointwise 
	 divinding the residual data by the smoothly constructed (season) volatility function. 

	CALLS FUNCTIONS: symfit functions
					 fourier_fit_functions: fourier_series(), plot_fourier_fit()
					 plot_functions: double_plot()

	INPUT:
	* data_arrays (dict of arrays): VAR model residuals
	* data_arrays_squared (dict of arrays): Squared VAR model residulas
	* data_names (array of strings): Names of the respective given datasets.
	* unit_names (array of strings): Units of data. Typically: ['[unit1]','[unit2]',...]
	* time_array (array): Array with days of year
	* days_of_year (int): Number of days in the year (the code assumes that each year is of equal length).
	* number_of_years (int): Number of years to consider. 
	* number_of_lags (int): Number of lags in VAR/MCAR model.
	* fourier_frequency (array of arrays with floats): Frequency of Fourier series of series 1, 2 and 3 respectively, 
										   for each variable. One array of frequencies for each array.
	* fourier_terms (array of arrays with int): Number of terms in each fitted Fourier series. One array of numbers
												for each array.
	* fourier_merge (array of arrays with floats, last element of outer array an int): scaling parameters in sigmoid 
					connection functions in arrays. Last integer is how many indexes the sigmoid connection function
					should grab hold of in the two functions that are going to be connected

	* volatility_fig_name (array of strings): One name for each dimension
	* figure_format (string): Name of figure format
	* dpi (int): Figure quality

	OUTPUT:
	# dict (dict of arrays): Squared scaled VAR model residuals.
	"""
	print(' ')
	print(' ')
	print('********** RESULTS FROM volatility_function() FOLLOWS **********')
	
	# Define parameters
	figure_format = parameters['figure format']
	dpi = parameters['figure dpi']
	number_of_years = parameters['number of years']
	days_of_year = parameters['days of year']
	lags = parameters['p']
	fourier_merge = parameters['merge constant']

	days = np.linspace(1,days_of_year,days_of_year)
	days = days.astype(int)
	
	#uwind = uwind_full_length[0:doy*noy]


	# Might be changed of moved to processing
	print("Date: ",dd.day_to_date(115))
	print("Date: ",dd.day_to_date(294-6))
	cut1 = dd.date_to_day("01.05")
	cut2 = dd.date_to_day("31.10")
	print("Index day: ",cut1)
	print("Index day: ",cut2)

	
	time_array = [days[0:cut1],\
					  days[cut1:cut2],\
					  days[cut2:]]
	
	dict_residuals_deseasoned = dict()
	dict_residuals_deseasoned_squared = dict()				  
	dict_expected_squared_reiduals = dict()

	for k in data_names:
		print(' ')
		print('Approximate volatility function for ',k,':')
		print(' ')
		residuals = data[k]
		residuals_squared = np.power(residuals,2)
		frequency = fourier_frequency[k]
		number_of_terms = fourier_terms[k]
		merge_parameter = merge_parameters[k]
		
		print(' ')
		print('** FUNCTION expected_value_year() CALLED IN volatility_function() **')
		print(' ')
		expected_values_squared_residuals,count =\
		expected_value_year(residuals_squared,days_of_year,number_of_years,lags)
		dict_expected_squared_reiduals[k] = expected_values_squared_residuals


		#Divide expected yearly squared residuals function into three parts
		expected_values = [expected_values_squared_residuals[0:cut1],\
						  expected_values_squared_residuals[cut1:cut2],\
						  expected_values_squared_residuals[cut2:]]

		

	
		#Fit fourier series to expected yearly squared residuals in three parts
		print(' ')
		print('** THREE PARTS FOURIER SEIRES THAT ARE FIT **')
		volatility_model_parts = []
		volatility_model_params = []
		
		for l in range(3):
			x, y = variables('x, y')
			part = CallableModel({y: ff.fourier_series(x,f=(frequency[l]*np.pi)/float(days_of_year),\
													   n=number_of_terms[l])})

			print(' ')
			print("Fourier series to be fitted: %s" % part)
			print(' ')
			

			volatility_model_part,volatility_model_param =\
			ff.plot_fourier_fit(part,time_array[l],expected_values[l],x,y,0,\
								' ',' ',(' ', ' '),days_of_year,' ',0,0,0,0,' ')
			volatility_model_parts.append(part)
			volatility_model_params.append(volatility_model_param)

		#Connect three functions fitted to expected yearly squared residuals (smoothly)
		def f_p1(x):
			f1 = volatility_model_parts[0]
			pars1 = volatility_model_params[0]
			f, = f1(x=x, **pars1) 
			return f

		def f_p2(x): 
			f2 = volatility_model_parts[1]
			pars2 = volatility_model_params[1]
			f, = f2(x=x, **pars2)
			return f

		def f_p3(x):
			f3 = volatility_model_parts[2]
			pars3 = volatility_model_params[2]
			f, = f3(x=x, **pars3)
			return f

		def function_blend1(x):
		    scale = merge_parameters[k]
		    sigma =  1. / (1 + np.exp(-(x - cut2)/scale[0]));
		    f = (1-sigma) * f_p2(x) + sigma * f_p3(x)
		    return f

		def function_blend2(x):
		    scale = merge_parameters[k]
		    sigma =  1. / (1 + np.exp(-(x - cut1)/scale[1]));
		    f = (1-sigma) * f_p1(x) + sigma * f_p2(x)
		    return f

		d1 = np.hstack([time_array[1],time_array[2]])
		d2 = np.hstack([time_array[0],time_array[1]])
		f1 = function_blend1(d1)
		f2 = function_blend2(d2)
		d3 = np.hstack([time_array[0],time_array[1],time_array[2]])
		variance_model = np.hstack([f2[0:cut1+fourier_merge],f1[fourier_merge:]])
	
		volatility_model = np.sqrt(variance_model)

		data_name = ''.join([label_names[k],' ',unit_names[k]])
		plt.figure(figsize=pl.cm2inch(17.4, 17.4/2.0))
		pl.double_plot(d3,expected_values_squared_residuals,variance_model,
						 ' ','Days',data_name,
						 ['Expected squared residuals','Squared volatility function'])
		plt.legend(loc='upper center')
		plt.savefig(figure_names[k],format=figure_format,dpi=dpi)
		#plt.show()

		#Merge volatility seasonality function to include all years and compute scaled residuals
		volatility_model_full = volatility_model[lags:days_of_year] #1st year
		for j in range(number_of_years-1): 								   #all years but 1st
			volatility_model_full = np.hstack([volatility_model_full,volatility_model])
		residuals_deseasoned = np.divide(data[k],volatility_model_full)  #deseasonalize residuals
		
		dict_residuals_deseasoned[k] = residuals_deseasoned
		dict_residuals_deseasoned_squared[k] = np.power(residuals_deseasoned,2)
	return dict_residuals_deseasoned, dict_residuals_deseasoned_squared






