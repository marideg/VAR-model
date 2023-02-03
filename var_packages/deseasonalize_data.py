"""
THIS DOCUMENT CONTAINS FUNCTIONS RELATED TO DESEASONALIZING TIME SERIES.
THE CONTAINED FUNCTIONS ARE THE FOLLOWING:
- deseasonalize_data()
"""

import numpy as np
import matplotlib.pyplot as plt
import plot_functions as pl
import fourier_fit_functions as ff
from symfit import variables
from matplotlib.ticker import FormatStrFormatter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def deseasonalize_data(data,data_names,parameters,label_names,unit_names,\
					   x_label_tick_names,start_plot_here,figure_names,number_of_fourier_terms,\
					   number_of_years_to_plot,ymin_plot,ymax_plot):
	"""
	* Reads data from given arrays. Fits a truncated Fourier series plus a function a+b*x 
	  to the given data, making a seasonality function. Deseasonalizes data by subtracting 
	  the fitted seasonality function. Plots the data with fitted seasonality function, for a 
	  given number of years.  

	 CALLS FUNCTIONS:

	 INPUT:
	* number_of_dimensions (int): Number of dimensions of MCARMA model.
	* data_arrays (array of arrays): Names of arrays containing ordered data (time series).
									 Each array contain input-data for each dimension in the
									 MCARMA model.
	* data_names (array of strings): Names of the respective given datasets.
	* unit_names (array of strings): Units of data. Typically: ['[unit1]','[unit2]',...]
	* time_step_name (string): Name of x-axis in plot of data and seasonality function.
	* x_label_tick_names (array of strings): Name of each year. For example: ['2010','2011',...]
	* figure_names (array of strings): Filenames of plot of data and seasonality function (without .type)
									   for each dimension.
	* days_of_year (int): Number of days in the year (the code assumes that each year is of equal length).
	* number_of_years (int): Number of years to consider. 
	* fourier_period_scale (float): Constant to adjust period of cos and sin in truncated Fourier series.
	* number_of_fourier_terms (int): Number of terms in the truncated Fourier series.
	* start_plot_here (int): Array element where to start the data and seasonality function plot. 
	* number_of_years_to_plot (int): How many years to plot from 'start_plot_here'
	* ymin_plot (array of ints): Minimum value of y-axis in the data and seasonality function plot. 
	* ymax_plot	(array of ints): Maximum value of y-axis in the data and seasonality function plot.
	
	 OUTPUT:
	* dict: deseasonalized data
	"""
	print(' ')
	print(' ')
	print('********** RESULTS FROM deseasonalize_data() FOLLOWS **********')

	# Define parameter names
	dict_ds = {}
	days_of_year = parameters['days of year']
	number_of_years = parameters['number of years']
	fourier_period_scale = parameters['period of year']
	time_step_name = parameters['time unit']

	# Make time array
	total_number_of_days = days_of_year*number_of_years
	time_vec = np.linspace(0,total_number_of_days-1,total_number_of_days)
	time = time_vec.astype(int)

	# Array for deseasonalized data
	data_ds = []

	# Deseasonalize data
	for j in data_names:
		data[j] = data[j][0:days_of_year*number_of_years]

		#Fit seasonality function to data
		x, y = variables('x, y')
		series,parameter_string_linear,parameter_string_cos,parameter_string_sin=\
		ff.fourier_series_linear(x,f=(np.pi)/(fourier_period_scale*days_of_year),n=number_of_fourier_terms[j])
		season_model = \
		{y: series}

		#Define names
		labels = [label_names[j],'Seasonality function']
		data_name = ''.join([label_names[j],' ',unit_names[j]])

		#Print info
		print(' ')
		print("Fourier series to fit to %s" % data_name, " is: ", season_model)
		print(' ')
		
		#Plot fit with data
		seasonality_function,params =\
		ff.plot_fourier_fit(season_model,time,data[j],x,y,1,time_step_name,data_name,\
					 		labels,days_of_year,x_label_tick_names,start=start_plot_here,\
					 		number_of_years_to_plot=number_of_years_to_plot,ymin_plot=ymin_plot[j],\
					 		ymax_plot=ymax_plot[j],figure_name=figure_names[j])

		#Print parameters
		linear_param_elements = [params[str(parameter_string_linear[i])] for i in range(0,len(parameter_string_linear))]
		cos_param_elements = [params[str(parameter_string_cos[i])] for i in range(0,len(parameter_string_cos))]
		sin_param_elements = [params[str(parameter_string_sin[i])] for i in range(0,len(parameter_string_sin))]

		print(' ')
		print("Linear parameters for %s" % data_name, ": ", parameter_string_linear, " = " , ["${0:0.5f}$".format(i) for i in linear_param_elements])
		print(' ')
		print("Cos parameters for %s" % data_name, ": ", parameter_string_cos, " = " , ["${0:0.5f}$".format(i) for i in cos_param_elements])
		print(' ')
		print("Sin parameters for %s" % data_name, ": ", parameter_string_sin, " = " , ["${0:0.5f}$".format(i) for i in sin_param_elements])
		print(' ')

		# Find change of mean value over given years of data
		start_mean = params['a0']
		slope = params['a1']
		end_mean = len(data[j])*slope + start_mean
		print(' ')
		print("Change of mean value over all given years for %s" % data_name, ":", end_mean - start_mean)
		print(' ')

		#Deseasonalize data
		data_ds_j_initial = data[j] - seasonality_function
		if unit_names[j] == '[B]':
			data_ds_j_corrected = (data_ds_j_initial + 180) % 360 - 180 #https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
			dict_ds[j] = data_ds_j_corrected
		else:
			dict_ds[j] = data_ds_j_initial

	return dict_ds


	


	








