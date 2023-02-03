import sys
sys.path.append("../var_packages")

from processing import *
import numpy as np
from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt
import deseasonalize_data as ds
import plot_functions as pl
import crosscorrelations as cc
import fit_var_model as var
import volatility_analysis as vol
import scaled_residuals_distribution as distr
pl.plot_props()

############################################################

def run():
	# 1) Deseasonalize data and plot data with fitted seasonality function
	dict_data_ds = ds.deseasonalize_data(data=dict_unprocessed_data,data_names=data_names,\
									parameters=dict_parameters,label_names=dict_label_names,\
									unit_names=dict_unit_names,x_label_tick_names=year_vector_to_plot,\
									start_plot_here=element_to_start_plot,figure_names=dict_seasonality_filenames,\
									number_of_fourier_terms=dict_parameters_fourier_terms,\
									number_of_years_to_plot=number_of_years_to_plot,\
									ymin_plot=dict_y_min_seasonality,ymax_plot=dict_y_max_seasonality)
	
	# 2) Fit a VAR model to data and retrieve residuals
	
	dict_residuals,\
	dict_residuals_squared = var.var_model(data=dict_data_ds,data_names=data_names,\
										  unit_names=dict_unit_names,parameters=dict_parameters)
	
	# 3) Compute seasonal volatility function and retrieve scaled residuals
	dict_residuals_deseasoned,\
	dict_residuals_deseasoned_squared = vol.volatility_function(data=dict_residuals,data_names=data_names,\
										unit_names=dict_unit_names,parameters=dict_parameters,\
										fourier_frequency=dict_parameters_fourier_frequency_volatility,\
										fourier_terms=dict_parameters_fourier_terms_volatility,\
										merge_parameters=dict_parameters_merge_volatility,\
										figure_names=dict_volatility_filenames,label_names=dict_label_names)

	# Check normality and/or NIG distribution of scaled residuals
	distr.fit_distribution(data=dict_residuals_deseasoned,data_names=data_names,parameters=dict_parameters,\
					 unit_names=dict_unit_names,label_names=dict_label_names,normal_fit=1,nig_fit=1,\
					 figure_names_normal_fit=dict_normal_distribution_filenames,\
					 figure_names_nig_fit=dict_nig_distribution_filenames)




	# Unprocessed data	
	cc.compute_and_plot_crosscorrelations(data=dict_unprocessed_data,data_names = data_names,\
							 parameters=dict_parameters,label_names=dict_label_names,\
							 unit_names=dict_unit_names,dataset_type=dataset_types[0],\
							 number_of_lags=dict_number_of_lags,y_axes_min=dict_y_min_crosscorr,\
							 y_axes_max=dict_y_max_crosscorr,figure_name_base=dict_crosscorrelation_filenames)
	# Deseasonalized data
	cc.compute_and_plot_crosscorrelations(data=dict_data_ds,data_names = data_names,\
							 parameters=dict_parameters,label_names=dict_label_names,\
							 unit_names=dict_unit_names,dataset_type=dataset_types[1],\
							 number_of_lags=dict_number_of_lags,y_axes_min=dict_y_min_crosscorr,\
							 y_axes_max=dict_y_max_crosscorr,figure_name_base=dict_crosscorrelation_filenames)
	# Expected value of squared residuals
	cc.compute_and_plot_crosscorrelations(data=dict_residuals_squared,data_names = data_names,\
							 parameters=dict_parameters,label_names=dict_label_names,\
							 unit_names=dict_unit_names,dataset_type=dataset_types[2],\
							 number_of_lags=dict_number_of_lags,y_axes_min=dict_y_min_crosscorr,\
							 y_axes_max=dict_y_max_crosscorr,figure_name_base=dict_crosscorrelation_filenames)
	# Scaled residuals
	cc.compute_and_plot_crosscorrelations(data=dict_residuals_deseasoned_squared,data_names = data_names,\
							 parameters=dict_parameters,label_names=dict_label_names,\
							 unit_names=dict_unit_names,dataset_type=dataset_types[3],\
							 number_of_lags=dict_number_of_lags,y_axes_min=dict_y_min_crosscorr,\
							 y_axes_max=dict_y_max_crosscorr,figure_name_base=dict_crosscorrelation_filenames)

	return 0

run()















