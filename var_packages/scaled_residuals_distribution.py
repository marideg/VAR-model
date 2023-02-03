import matplotlib.pyplot as plt
import numpy as np
import plot_functions as pl
from scipy.stats import norm,probplot,kstest,norminvgauss
from cycler import cycler


def fit_distribution(data,data_names,parameters,unit_names,label_names,\
					 normal_fit,nig_fit,figure_names_normal_fit,\
					 figure_names_nig_fit):
	"""
	#Does a normal distribution fit and a NIG distribution fit of given data. Plots results.
	 Also plots ACF and PACF.

	CALLS FUNCTIONS: plot_functions: norm_fit(), standard_plot(), standard_hist(), standard_acf(),
									 standard_pacf()
					 stochastic_model_const_lag: expected_residuals_year()
					 scipy.stats functions

	INPUT:
	* data_arrays (dict of arrays): scaled VAR model residuals
	* time_array (array): Array with days of year
	* days_of_year (int): Number of days in the year (the code assumes that each year is of equal length)
	* number_of_years (int): Number of years to consider
	* number_of_lags (int): Number of lags in VAR/MCAR model
	* plot_norm,plot_nig (int): 1 if plot normal fit, nig fit
	* nig_fig_name (array of strings): One name for each dimension
	* data_names (array of strings): Names of the respective given datasets.
	* unit_names (array of strings): Units of data. Typically: ['[unit1]','[unit2]',...]
	* figure_format (string): Name of figure format
	* dpi (int): Figure quality

	OUTPUT: 0
	"""
	print(' ')
	print(' ')
	print('********** RESULTS FROM deseasonalized_residuals_status() FOLLOWS **********')
	
	# Define parameter names
	days_of_year = parameters['days of year']
	number_of_years = parameters['number of years']
	figure_format = parameters['figure format']
	dpi = parameters['figure dpi']


	#Check distribution of deseasonalized residuals
	
	for k in data_names:
		data_name_unit = ''.join([label_names[k],' ',unit_names[k]])
		normfit = pl.norm_fit(data[k],-7,7,500)

		if normal_fit == 1:
			plt.rcParams['figure.figsize'] = pl.cm2inch(17.4/2.1, 17.4/2.6)
			fig = plt.figure()
			ax = fig.add_subplot(111)
			probplot(data[k], dist='norm',sparams=(normfit[2], normfit[3]),\
					 plot=plt)
			ax.get_lines()[0].set_markerfacecolor('steelblue')
			ax.get_lines()[0].set_markeredgecolor('steelblue')
			ax.get_lines()[1].set_color('k')
			plt.title(None)
			plt.ylabel(data_name_unit)
			plt.savefig(figure_names_normal_fit[k],format=figure_format,dpi=dpi)
			#plt.show()
			print(' ')
			print("Normal fit results for scaled VAR model residuals for ",label_names[k],": mu = %.2f,  std = %.2f"\
				  % (normfit[2], normfit[3]))
			print("KS-test results for normal fit of scaled VAR model residuals for ",label_names[k],":",\
				  kstest(data[k],'norm', args = (normfit[2], normfit[3])))

		if nig_fit == 1:
			nig1, nig2, nig3, nig4 = norminvgauss.fit(data[k])
			grid = np.linspace(-7, 7, 1000)
			nigfit = norminvgauss.pdf(grid,nig1,nig2,nig3,nig4)
			plt.figure(figsize=pl.cm2inch(17.4/2.1, 17.4/2.6))
			plt.rcParams['axes.prop_cycle'] = cycler(color=['k','steelblue'])
			pl.standard_plot(grid, nigfit,\
							 "NIG fit",data_name_unit,'Probability density',a=0)
			pl.standard_hist(data[k],200,a=0)
			plt.savefig(figure_names_nig_fit[k],format=figure_format,dpi=dpi)
			#plt.show()
			pl.plot_props()
			print(' ')
			print("NIG fit results for scaled VAR model residuals for ",label_names[k],\
				  ": Tail heaviness = %f, Asymmetry parameter = %f, Location = %f, Scale = %f" %\
				  (nig1,nig2,nig3,nig4))
			print("NIG KS-test results for scaled VAR model residuals for ",label_names[k],\
				  ": ", kstest(data[k],'norminvgauss', args = (nig1,nig2,nig3,nig4)))
	return 0