"""
THIS DOCUMENT CONTAINS FUNCTIONS RELATED TO FITTING AND PLOTTING FOURIER SERIES.
THE CONTAINED FUNCTIONS ARE THE FOLLOWING:
- fourier_series_linear()
- fourier_series()
- plot_fourier_fit()
"""

import numpy as np
import matplotlib.pyplot as plt
import plot_functions as pl
from symfit import parameters, variables, sin, cos, Fit, CallableModel
import datetime
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

def fourier_series_linear(x, f, n=0):
    """
    #Makes a symbolic Fourier series of order n with a constant term a0
     and a linear slope a1

    CALLS FUNCTIONS: symfit-functions
    
    INPUT:
    # x (array): independent variable, usually time
    # f (float): frequency of the Fourier series
    # n (int): order of the Fourier series

    OUTPUT:
    # series (symbolic element): a Fourier series  
    """
    # Make the parameter objects for all terms
    a0, a1, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 2)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    
    # Construct the series
    series = a0 + a1*x + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    parameter_string_linear = [a0,a1]
    parameter_string_cos = [ai for ai in cos_a]
    parameter_string_sin = [bi for bi in sin_b]
    return series,parameter_string_linear,parameter_string_cos,parameter_string_sin

def fourier_series(x, f, n=0):
    """
    #Makes a symbolic Fourier series of order n with a constant term a0
    
    CALLS FUNCTIONS: symfit-functions

    INPUT:
    # x (array): independent variable, usually time
    # f (float): frequency of the Fourier series
    # n (int): order of the Fourier series

    OUTPUT:
    # series (symbolic element): a Fourier series  
    """
    # Make the parameter objects for all terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))

    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

def plot_fourier_fit(model,time,values,x,y,plot_result,x_label,y_label,labels,days_of_year,x_label_tick_names,start,\
                     number_of_years_to_plot,ymin_plot,ymax_plot,figure_name):
    """
    b=1,c=0,info='yes',pmin=0,pmax=0,
    * Makes a fit from a feeded symbolic model (series). The model is fitted to given
      data (give both x- (time) and y-values). Then given data together with the 
      fitted model are plotted. This function might plot total length of the time series,
      or last part of the time series from a given start point.
    
    CALLS FUNCTIONS: symfit-functions, statsmodels.graphics.tsaplots: plot_pacf()

    INPUT:
    * model (collection): symbolic model to fit to function, given by the collection 
                          {y: symbolic element}, which might be given from 
                          fourier_series() or fourier_series_linear()
    * time (array): time values to be fitted (as x-values)
    * values (array): values to be fitted (as y-values), has same length as time array
    * x (symbolic element): given from x, y = variables('x, y') (symfit-package)
    * y (symbolic element): given from x, y = variables('x, y') (symfit-package)
    * plot_result (int): give 1 if to plot results
    * x_label (string): name of x-values
    * y_label (string): name of y-values
    * labels ((2x1) array, string element): ('name of given data','name of fitted model')
    * days_of_year (int): Number of days in the year (the code assumes that each year is of equal length).
    * x_label_tick_names (array of strings): Name of each year. For example: ['2010','2011',...]
    * start (int): index from where the 'short' plot is going to have its start value, 
                   with ERA Interim data set (temp/wind) start=11192 makes a plot from 
                   31 Aug 2009 to 2019. 10949 gives plot from 1 Jan 2009 to 31 Dec 2018.
    * number_of_years_to_plot (int): How many years to plot from 'start_plot_here'
    * ymin_plot (array of ints): Minimum value of y-axis in the data and seasonality function plot. 
    * ymax_plot (array of ints): Maximum value of y-axis in the data and seasonality function plot.
    * figure_name (array of strings): Filenames of plot of data and seasonality function (with .type)
                                       for each dimension.

    # info (string): 'yes' if model fit info should be printed to terminal 

    OUTPUT:
    # Plot of given data with fitted model (given time length)
    # fitted_function (array): the fitted function
    # fourier_fit_result.params (collection): parameters of fitted model
    """
    fourier_fit = Fit(model,x=time,y=values) 
    fourier_fit_result = fourier_fit.execute()
    fitted_function = fourier_fit.model(x=time,**fourier_fit_result.params).y
    #Print fit results
    #print("Fourier fit results for %s" % y_label, " is ", fourier_fit_result)

    # Plot fitted function to data
    if plot_result == 1:  
      num_days = days_of_year*number_of_years_to_plot
      lines = np.linspace(start,start+num_days,number_of_years_to_plot+1)
      fig = plt.figure(figsize=pl.cm2inch(17.4/1.0,17.4/2.0))
      ax = fig.add_subplot(1,1,1)
      plt.plot(time[start:start+num_days],values[start:start+num_days],label=labels[0])
      plt.plot(time[start:start+num_days],fitted_function[start:start+num_days],label=labels[1])
      plt.xlabel(x_label)
      plt.ylabel(y_label)
      plt.legend(loc='lower left')
      plt.vlines(lines,ymin_plot,ymax_plot,linestyles='dashed',colors='k',linewidth=0.5)
      ax.autoscale(enable=True, axis='x', tight=True)
      plt.xticks(lines[1:]-(days_of_year/2.0),x_label_tick_names)
      ax.tick_params(axis='x',length=0)
      plt.savefig(figure_name,format='pdf',dpi=300)
      #plt.show()
      plt.close(fig) 

    return fitted_function,fourier_fit_result.params


















