"""
* NAMES IN LIST OF DATANAMES MUST BE USED IN ALL DICTIONARIES DESCRIBING EACH DATASET
* ALL DATASETS CONTAINING ANGLE-VALUES MUST HAVE UNIT NAME [B]
"""
import sys
sys.path.append("../var_packages")

import numpy as np
import initialize_input_data as iid

############################################################

# 1) Read in data and make data dictionaries
# Lists containing data names 
## THE FOLLOWING DATANAMES MUST BE USED IN ALL FOLLOWING DICTIONARIES ##
## DESCRIBING EACH OF THE DATASETS 									  ##
data_names = ['u wind','temperature'] #LIST OF DATANAMES

#Read in data
uwind_txt = np.loadtxt('../data/u_79_19.txt')
uwind_data = np.array(uwind_txt)

temp_txt = np.loadtxt('../data/temp_79_19.txt')
temp_data = np.array(temp_txt)

#Make dictionaries with datasets (names as in list of datanames)
dict_unprocessed_data = {}
dict_unprocessed_data['u wind'] = uwind_data
dict_unprocessed_data['temperature'] = temp_data


############################################################

# 2) Define general variables
start_year = 1979 #start year of dataset (first point at 1 January)
stop_year = 2018  #stop year of dataset (last point at 31 December) 

dict_parameters = {}
dict_parameters['p'] = 4 			    #autoregressive parameter in VAR model
dict_parameters['days of year'] = 365   #number of days of your year
dict_parameters['period of year'] = 1.0 #scale: 1.0 assuming the year is 365 days
dict_parameters['number of years'] = stop_year - start_year + 1  #total number of years in dataset
dict_parameters['time unit'] = 'Days'
dict_parameters['figure dpi'] = 300
dict_parameters['figure format'] = 'pdf'
dict_parameters['merge constant'] = 100

############################################################

# 3) Define parameters for additive seasonality function
#Number of years to plot data with fitted seasonality function
number_of_years_to_plot = 10

#Number of fourier terms in each seasonality function
dict_parameters_fourier_terms = {} 
dict_parameters_fourier_terms['u wind'] = 10
dict_parameters_fourier_terms['temperature'] = 10

#Figure target file names
dict_seasonality_filenames = {} 
dict_seasonality_filenames['u wind'] = \
'resulting_figs/uwind_season.pdf'
dict_seasonality_filenames['temperature'] = \
'resulting_figs/temp_season.pdf'


#Max and min values of y-axes
dict_y_min_seasonality = {}
dict_y_min_seasonality['u wind'] = -50
dict_y_min_seasonality['temperature'] = 200

dict_y_max_seasonality = {}
dict_y_max_seasonality['u wind'] = 80
dict_y_max_seasonality['temperature'] = 240


## DO NOT CHANGE THE REMAINING OF THIS SECTION ##
#Vector containing all years of dataset
year_vector = iid.year_vector(start_year,stop_year)

#Vector containing all years to plot in final figure
year_vector_to_plot = iid.year_vector(stop_year-number_of_years_to_plot+1,stop_year) 
element_to_start_plot = len(uwind_data)-dict_parameters['days of year']*number_of_years_to_plot


############################################################


# 4) Data names and units 
dict_label_names = {}
dict_label_names['u wind'] = 'U wind'
dict_label_names['temperature'] = 'Temperature'

# Angle corrections in this program is dependent on unit names.
# That is: a dataset is considered to be angles if the unit name is '[B]'.
dict_unit_names = {}
dict_unit_names['u wind'] = '[m/s]'
dict_unit_names['temperature'] = '[K]'


############################################################


# 5) Define parameters for volatility seasonality function


#Number of fourier terms, fourier frequencies and merge parameters
#of fitted volatility functions
dict_parameters_fourier_terms_volatility = {} 
dict_parameters_fourier_terms_volatility['u wind'] = [2,3,4]
dict_parameters_fourier_terms_volatility['temperature'] = [2,2,2]


dict_parameters_fourier_frequency_volatility = {} 
dict_parameters_fourier_frequency_volatility['u wind'] = [0.3,0.5,0.05]
dict_parameters_fourier_frequency_volatility['temperature'] = [0.44,2.0,0.44]


dict_parameters_merge_volatility = {} 
dict_parameters_merge_volatility['u wind'] = [5.0,2.0]
dict_parameters_merge_volatility['temperature'] = [5.0,2.0]

#Figure target file names
dict_volatility_filenames = {} 
dict_volatility_filenames['u wind'] = \
'resulting_figs/uwind_variance.pdf'
dict_volatility_filenames['temperature'] = \
'resulting_figs/temp_variance.pdf'



############################################################


# 6) Define parameters for distribution fit

dict_normal_distribution_filenames = {}
dict_normal_distribution_filenames['u wind'] = \
'resulting_figs/uwind_normal_fit.pdf'
dict_normal_distribution_filenames['temperature'] = \
'resulting_figs/temp_normal_fit.pdf'

dict_nig_distribution_filenames = {}
dict_nig_distribution_filenames['u wind'] = \
'resulting_figs/uwind_nig_fit.pdf'
dict_nig_distribution_filenames['temperature'] = \
'resulting_figs/temp_nig_fit.pdf'



############################################################

# x) If you want crosscorrelation plots, figure target file names  and parameters are defined here
dataset_types = ['unprocessed','deseasonalized','squared residual','scaled residual']

#Max and min values of y-axes
dict_y_min_crosscorr = {}
dict_y_min_crosscorr['squared residual'] = -0.1
dict_y_min_crosscorr['scaled residual'] = -0.1

dict_y_max_crosscorr = {}
dict_y_max_crosscorr['squared residual'] = 0.25
dict_y_max_crosscorr['scaled residual'] = 0.25

#Number of crosscorrelations to plot
dict_number_of_lags = {}
dict_number_of_lags['unprocessed'] = 150
dict_number_of_lags['deseasonalized'] = 150
dict_number_of_lags['squared residual'] = 150
dict_number_of_lags['scaled residual'] = 25

#Target file names
dict_crosscorrelation_filenames = {}
dict_crosscorrelation_filenames['unprocessed'] =\
 'resulting_figs/crosscorrelations_unprocessed'
dict_crosscorrelation_filenames['deseasonalized'] =\
 'resulting_figs/crosscorrelations_deseasonalized'
dict_crosscorrelation_filenames['squared residual'] =\
 'resulting_figs/crosscorrelations_squaredresidual'
dict_crosscorrelation_filenames['scaled residual'] =\
 'resulting_figs/crosscorrelations_scaledresidual'
















