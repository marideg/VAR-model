"""
THIS DOCUMENT CONTAINS FUNCTIONS RELATED TO DERIVING A CAR(P) MODEL.
THE CONTAINED FUNCTIONS ARE THE FOLLOWING:
- fit_ar_model()
- expected_residuals_year()
- residuals_status()
- cutoff_restrictions()
- volatility_function()
- volatility_function_simple()
- deseasonalized_residuals_status()
"""

import numpy as np
import matplotlib.pyplot as plt
import plot_functions as pl
import fourier_fit_functions as ff
from symfit import parameters, variables, sin, cos, Fit, CallableModel
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from scipy.stats import norm, probplot, kstest, skew, kurtosis, skewnorm, norminvgauss
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import acf, pacf
from cycler import cycler
from matplotlib.ticker import FormatStrFormatter

def expected_residuals_year(data, days_of_year, num_of_years, p=0):
	"""
	# Computes expected value of data every day during a year (of given length).
	  Only complete years datasets as input in this version. However, the function 
	  adjusts for first datapoints which are used as input to a possible AR(p)-model,
	  if p is given. 

	INPUT:
	# data (array): The data set (complete years)
	# days_of_year (int): How long is your year
	# num_of_years (int): How many years of data
	# p (int): Number of points used in an AR(p) fit (p lags)

	OUTPUT:
	# expected_values (array): Expected value every day of a year
	# c (int): Test count (check if the correct number of datapoints used in calculation)
	"""
	print(' ')
	print('** FUNCTION expected_residuals_year() RUN INSIDE FUNCTION **')

	expected_values = []
	c=0
	for i in range(days_of_year):
		day = []
		for j in range(num_of_years):
			if j == 0 and i in range(p,days_of_year):
				day.append(data[i-p])
				c += 1
			if j != 0:
				day.append(data[days_of_year*j + (i-p)])	
				c += 1    
		expected_values.append(np.mean(day))
	print("Number of terms added in expected value calculation: ",c)
	return expected_values,c

def residuals_status(p,res,days_of_year,num_of_years,plot_acf=0,plot_mean_res=0,plot_mean_res_compare=0,acf_plot_name='no_name'):
	"""
	#Computes expected squared residuals every day of the year. This function only works for datasets 
	 of complete years, since the function expected_residuals_year() is run. ACF of residuals and 
	 squared residuals are plotted. Also the expected squared residuals over a year is plotted, as well
	 as its ACF function

	CALLS FUNCTIONS: stochastic model_const_lag: expected_residuals_year()
					 plot_functions: standard_acf(), standard_plot(), double_plot()

	INPUT:
	# p (int): Number of lags in AR(p)-model fit to (deseasonalized) data
	# res (array (two columns)): One column with residual data and one column with 
								 squared residual data
	# plot_acf (int): 1 if plot ACF of residuals and squared residuals
	# plot_mean_res (int): 1 if plot expected values of residuals and squared residuals
	# plot_mean_res_compare (int): 1 if you want to check if variance approximation is ok 

	OUTPUT:
	# expected_values (array): Expected squared residuals every day of the year (365 values)
	# days (array): Days of the year (day 1 to day 365)
	"""
	print(' ')
	print(' ')
	print('********** RESULTS FROM residuals_status() FOLLOWS **********')
	#Residuals data and its ACF
	days = np.linspace(1,days_of_year,days_of_year)
	days = days.astype(int)

	residuals = res[0]
	residuals_sq = res[1]

	if plot_acf == 1:
		plt.rcParams['figure.figsize'] = pl.cm2inch(17.4/2.1, 17.4/2.6)
		#pl.standard_acf(residuals,800,'ACF of residuals','Lags','ACF')
		#plt.show()
		fig,ax = plt.subplots()
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		pl.standard_acf(residuals_sq,730,' ','Days','ACF',ax=ax)
		plt.savefig(acf_plot_name, format='jpeg', dpi=1200)
		plt.show()

	#Compute daily expected values during a years
	expected_values,num_of_data = expected_residuals_year(residuals, days_of_year, num_of_years, p)
	expected_values2,num_of_data2 = expected_residuals_year(residuals_sq, days_of_year, num_of_years, p)
	expected_values_sq = np.power(expected_values,2)
	print(' ')
	print("If",len(res[0]),"=",num_of_data,": number of data points in residual vector is correct!")
	print(' ')

	if plot_mean_res == 1:
		pl.standard_plot(days,expected_values2,'Expected squared residuals during a year',
							'Days of year','Expected squared residuals [K]')
		pl.standard_plot(days,expected_values,'Expected residuals during a year',
							'Days of year','Expected residuals [K]')

	#Check whether variance approximation is ok or not
	if plot_mean_res_compare == 1:
		print('Test variance approximation:')
		print("Mean of computed E[res(t)^2]: ",np.mean(expected_values2))
		print("Mean of computed E[res(t)]^2: ",np.mean(expected_values_sq))
		rel_error = np.abs(-expected_values_sq/(expected_values2-expected_values_sq))
		print("len:",len(rel_error))
		print("Mean absolute percentage error", np.sum(rel_error)/len(rel_error))
		#pl.standard_plot(days,(expected_values2-expected_values_sq),'Var def',
		#					'Days of year','Variance')
		plt.rcParams['figure.figsize'] = pl.cm2inch(17.4/1.0, 17.4/2.0)
		fig, (ax1,ax2) = plt.subplots(2,sharex=True)
		ax1.plot(days,expected_values2)
		ax2.plot(days,rel_error)
		ax1.set(ylabel="Variance")
		ax2.set(ylabel="Relative error")
		ax2.set(xlabel="Days")
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		#pl.double_plot(days,expected_values2,expected_values2-expected_values_sq," ",\
		#			   "Days","Variance",["Approximation","Variance"])
		#plt.legend(loc='upper center')
		plt.savefig('resulting_figs/fig_3.jpeg', format='jpeg', dpi=1200)
		plt.show()
		
	return expected_values2, days

def cutoff_restrictions(function,variance_limit,n,plot_loc_var=1):
	"""
	#Find cutoff restriction for separating a function based on local variance.
	 Plots local variance to be analysed visually, and prints index and stability coefficient
	 of local variance within the given limit. Printed elements represents days of the year
	 (1-365, not 0-364)!

	CALLS FUNCTIONS: 0
	
	INPUT:
	# function (array): Data of function to be evaluated 
	# variance_limit (float): The variance limit 
	# n (int, odd number): Number of terms to be included in calculation of local variance
	# plot_loc_var (int): 1 if plot rolling value of local variance

	OUTPUT:
	# 0
	"""
	print(' ')
	print(' ')
	print('********** RESULTS FROM cutoff_restrictions() FOLLOWS **********')
	h = int(((n-1)/2))

	print(' ')
	print("Given variance limit is %f, and number of elements in local variance is %i" % (variance_limit,n))

	#Compute local variance
	k = np.linspace(1+h,365-h,365-n+1)
	local_var = []
	elements = []
	for i in k:
		center = int(i)
		j = int(i-(h+1))
		index1 = center - h
		index2 = center + h
		m = np.mean([function[index1-1:index2]])
		d = np.sum((function[index1-1:index2]-m)**2)
		local_var.append(d/n)
		if local_var[j] < variance_limit:
			elements.append(center)
	if plot_loc_var == 1:
		plt.plot(local_var)
		plt.title('Local variance')
		plt.show()

	#Fill index array for stability considerations
	elements_diff = [[elements[i+1] - elements[i] for i in range(len(elements)-1)],
					 [elements[i+1] for i in range(len(elements)-1)]]
	elements_diff = list(zip(elements_diff[0],elements_diff[1]))

	#Print elements that do not follow the cutoff variance limit continuously (non-stability)
	print(' ')
	print("Results from cutoff computation:")
	for j in range(len(elements_diff)-1):
		if elements_diff[j][0] != 1:
			print("Elements that do not follow cutoff continuously: ",elements_diff[j])
	print(' ')
	return 0;

def volatility_function(p,function,days,res,cut1,cut2,ff1,ff2,ff3,alpha1,alpha2,merge_number,\
						doy,noy,plot_ff1=0,plot_ff2=0,plot_ff3=0,plot_volatility=0,\
						volatility_plot_name='no_name',measurement_name='no_name',info1=' ',\
						info2=' ',info3=' '):
	"""
	#Divides given dataset as a result of given cutoff restrictions and fits a Fourier series to each
	 part (must be three parts from two cuttoffs). Then two sigmoid functions are used to merge the 
	 three fitted Fourier series, such that the resulting (season) volatility function becomes smooth.
	 The resulting function is plotted. Then the given residual data are deseasonalized by pointwise 
	 divinding the residual data by the smoothly constructed (season) volatility function. 

	CALLS FUNCTIONS: symfit functions
					 fourier_fit_functions: fourier_series(), plot_fourier_fit()
					 plot_functions: double_plot()

	INPUT:
	# p (int): Numer of lags in AR(p)-model estimated earlier
	# function (array): Expected values every day of the year of squared residuals (time varying variance)
	# days (array): Days during a year (1 to 365)
	# res (array): Residual data
	# cut1, cut2 (int, int): Cutoff index set from cutoff_restrictions()
	# ff1, ff2, ff3 (2x1 arrrays): First element is frequency of Fourier series of series 1, 2 and 3
								   respectively, second element is number of cos() and sin() terms
								   in the respective series  
	# plot_ff1, plot_ff2, plot_ff3 (int): 1 if fitted Fourier series on part 1/2/3 should be plotted
	# alpha1, alpha2 (float, float): scaling parameters in sigmoid connection functions
	# merge_number (int): How many indexes should sigmoid connection function grab hold of in the two 
						  functions that are going to be connected
	# doy, noy (int): Days of year and number of years in residual dataset
	# plot_volatility (int): 1 if plot fitted variance (vol sq) with data

	OUTPUT:
	# residuals_deseasoned (array): Deseasonalized residuals
	"""
	print(' ')
	print(' ')
	print('********** RESULTS FROM volatility_function() FOLLOWS **********')
	#Divide expected yearly squared residuals function into three parts
	expected_values_p1 = function[0:cut1]
	expected_values_p2 = function[cut1:cut2]
	expected_values_p3 = function[cut2:]
	days_p1 = days[0:cut1]
	days_p2 = days[cut1:cut2]
	days_p3 = days[cut2:]

	#Fit fourier series to expected yearly squared residuals in three parts
	print(' ')
	print('** THREE PARTS FOURIER SEIRES THAT ARE FIT **')
	x, y = variables('x, y')
	model_vol_p1 = CallableModel({y: ff.fourier_series(x, f=(ff1[0]*np.pi)/365.0, n=ff1[1])})
	print(' ')
	print("Fourier series to be fitted: %s" % model_vol_p1)
	print(' ')
	variance_season_model_p1,params_ff_p1 = ff.plot_fourier_fit(model_vol_p1, days_p1, 
																  expected_values_p1, x, y, 
																  'Fit of expected squared residuals during a year (p1)',
						   										  'Days of year','Expected squared residual [K]',
						   										  ('Squared residuals', 'Fitted function'),
						   									 	  b=plot_ff1,info=info1)

	x, y = variables('x, y')
	model_vol_p2 = CallableModel({y: ff.fourier_series(x, f=(ff2[0]*np.pi)/365.0, n=ff2[1])})
	print(' ')
	print("Fourier series to be fitted: %s" % model_vol_p2)
	print(' ')
	variance_season_model_p2,params_ff_p2 = ff.plot_fourier_fit(model_vol_p2, days_p2,
																  expected_values_p2, x, y,
														 		  'Fit of expected squared residuals during a year (p2)',
						   										  'Days of year','Expected squared residual [K]',
						   										  ('Squared residuals', 'Fitted function'),
						   										  b=plot_ff2,info=info2)

	x, y = variables('x, y')
	model_vol_p3 = CallableModel({y: ff.fourier_series(x, f=(ff3[0]*np.pi)/365.0, n=ff3[1])})
	print(' ')
	print("Fourier series to be fitted: %s" % model_vol_p3)
	print(' ')
	variance_season_model_p3,params_ff_p3 = ff.plot_fourier_fit(model_vol_p3, days_p3,
																  expected_values_p3, x, y,
																  'Fit of expected squared residuals during a year (p3)',
						   										  'Days of year','Expected squared residual [K]',
						   										  ('Squared residuals', 'Fitted function'),
						   										  b=plot_ff3,info=info3)


	#Connect three functions fitted to expected yearly squared residuals (smoothly)
	def f_p1(x):
		f, = model_vol_p1(x=x, **params_ff_p1) 
		return f

	def f_p2(x): 
		f, = model_vol_p2(x=x, **params_ff_p2)
		return f

	def f_p3(x):
	 	f, = model_vol_p3(x=x, **params_ff_p3)
	 	return f

	def function_blend1(x):
	    sigma =  1. / (1 + np.exp(-(x - cut2)/alpha1));
	    f = (1-sigma) * f_p2(x) + sigma * f_p3(x)
	    return f

	def function_blend2(x):
	    sigma =  1. / (1 + np.exp(-(x - cut1)/alpha2));
	    f = (1-sigma) * f_p1(x) + sigma * f_p2(x)
	    return f

	d1 = np.hstack([days_p2,days_p3])
	d2 = np.hstack([days_p1,days_p2])
	f1 = function_blend1(d1)
	f2 = function_blend2(d2)
	d3 = np.hstack([days_p1,days_p2,days_p3])
	f3 = np.hstack([f2[0:cut1+merge_number],f1[merge_number:]])
	
	volatility_season_function = np.sqrt(f3)
	if plot_volatility == 1:
		plt.figure(figsize=pl.cm2inch(17.4, 17.4/2.0))
		pl.double_plot(d3,function,f3,
						 'Expected squared residuals with season fit','Days',measurement_name,
						 ['Expected squared residuals','Squared volatility function'])
		plt.legend(loc='upper center')
		plt.savefig(volatility_plot_name, format='pdf', dpi=300)
		plt.show()

	#Merge volatility seasonality function to include all years
	volatility_seasonality = volatility_season_function[p:doy] #1st year
	for j in range(noy-1): 									   #all years but 1st
		volatility_seasonality = np.hstack([volatility_seasonality,volatility_season_function])
	residuals_deseasoned = np.divide(res,volatility_seasonality)  #deseasonalize residuals
	return residuals_deseasoned

def volatility_function_simple(p,function,days,res,ff_parm,plot_ff=0,plot_volatility=0):
	"""
	#Fits Fourier series to given function. The resulting function is plotted if told to.
	 Then the given residual data are deseasonalized by pointwise divinding the residual 
	 data by the constructed (season) volatility function. 

	CALLS FUNCTIONS: symfit functions
					 fourier_fit_functions: fourier_series(), plot_fourier_fit()
					 plot_functions: double_plot()

	INPUT:
	# p (int): Numer of lags in AR(p)-model estimated earlier
	# function (array): Expected values every day of the year of squared residuals (time varying variance)
	# days (array): Days during a year (1 to 365)
	# res (array): Residual data
	# plot_ff (int): 1 if fitted Fourier series should be plotted
	# plot_volatility (int): 1 if plot fitted variance (vol sq) with data


	OUTPUT:
	# Plot of resulting seasonality function together with residual data
	# residuals_deseasoned (array): Deseasonalized residuals
	"""
	print(' ')
	print(' ')
	print('********** RESULTS FROM volatility_function_simple() FOLLOWS **********')
	#Fit fourier series to expected yearly squared residuals in three parts
	print(' ')
	print('** FOURIER SEIRES THAT IS FIT **')
	x, y = variables('x, y')
	model_vol = CallableModel({y: ff.fourier_series(x, f=(ff_parm[0]*np.pi)/365.0, n=ff_parm[1])})
	print(' ')
	print("Fourier series to be fitted: %s" % model_vol)
	print(' ')
	variance_season_model,params_ff = ff.plot_fourier_fit(model_vol, days, 
														  function, x, y, 
														  'Fit of expected squared residuals during a year (p1)',
						   								  'Days of year','Expected squared residual [K]',
						   								  ('Squared residuals', 'Fitted function'),
						   								  b=plot_ff,info='no')
	volatility_season_function = np.sqrt(variance_season_model)

	if plot_volatility == 1:
		pl.double_plot(days,function,variance_season_model,
						 'Expected squared residuals with season fit','Days of year','Temperature [K]',
						 ['Expected squared residuals','Seasonality function'])

	#Merge volatility seasonality function to include all years
	volatility_seasonality = volatility_season_function[p:doy]    #1st year
	for j in range(noy-1): 										  #all years but 1st
		volatility_seasonality = np.hstack([volatility_seasonality,volatility_season_function])
	residuals_deseasoned = np.divide(res,volatility_seasonality)  #deseasonalize residuals
	return residuals_deseasoned


def deseasonalized_residuals_status(r_ds,p,days,days_of_year,num_of_years,plot_norm=0,plot_nig=0,plot_res_exp=0,\
									plot_res_sq_exp=0,plot_res_acf=0,plot_res_sq_acf=0,\
									nig_density_fig_name='resulting_figs/none.jpeg',measurement_name='none'):
	"""
	#Does a normal distribution fit and a NIG distribution fit of given data. Plots results.
	 Also plots ACF and PACF.

	CALLS FUNCTIONS: plot_functions: norm_fit(), standard_plot(), standard_hist(), standard_acf(),
									 standard_pacf()
					 stochastic_model_const_lag: expected_residuals_year()
					 scipy.stats functions

	INPUT:
	# r_ds (array): The data you want status of (deseasonalized residuals)
	# p (int): Number of lags of AR(p) model estimated earlier
	# days (array): The days of the year (1 to 365) 
	# plot_norm,plot_nig,plot_res_exp,plot_res_sq_exp,plot_res,plot_res_sq (int): 
	  1 if plot normal fit, nig fit, daily expected value of residuals/squared residuals during
	  a year and ACF/PACF, and ACF/PACF of deseasonalized residuals and squared residulas 

	OUTPUT: 0
	"""
	print(' ')
	print(' ')
	print('********** RESULTS FROM deseasonalized_residuals_status() FOLLOWS **********')
	#Check distribution of deseasonalized residuals
	residuals_deseasoned = r_ds
	normfit_residuals_ds = pl.norm_fit(residuals_deseasoned,-7,7,500)

	if plot_norm == 1:
		plt.rcParams['figure.figsize'] = pl.cm2inch(17.4/2.1, 17.4/2.6)
		#pl.standard_plot(normfit_residuals_ds[0],normfit_residuals_ds[1],
		#					"Fit results: mu = %.2f,  std = %.2f" % (normfit_residuals_ds[2], normfit_residuals_ds[3]),
		#					'Deseasonalized residuals [K]','Probability',a=0)
		#pl.standard_hist(r_ds,200,a=0)
		#plt.show()
		fig = plt.figure()
		ax = fig.add_subplot(111)
		probplot(r_ds, dist='norm',sparams=(normfit_residuals_ds[2], normfit_residuals_ds[3]),\
				 plot=plt)
		ax.get_lines()[0].set_markerfacecolor('steelblue')
		ax.get_lines()[0].set_markeredgecolor('steelblue')
		ax.get_lines()[1].set_color('k')
		#ax.get_lines()[0].set_markerfacecolor('gray')
		#ax.get_lines()[0].set_markeredgecolor('gray')
		#ax.get_lines()[1].set_color('k')
		plt.title(None)
		plt.ylabel('Temperature [K]')
		plt.savefig('resulting_figs/fig_6a.jpeg', format='jpeg', dpi=1200)
		plt.show()
		print("Normal fit results r_ds: mu = %.2f,  std = %.2f" % (normfit_residuals_ds[2], normfit_residuals_ds[3]))
		print("KS-test results norm: ", kstest(r_ds,'norm', args = (normfit_residuals_ds[2], normfit_residuals_ds[3])))

	nig1, nig2, nig3, nig4 = norminvgauss.fit(r_ds)
	grid = np.linspace(-10, 10, 1000)
	prob = norminvgauss.pdf(grid,nig1,nig2,nig3,nig4)

	if plot_nig == 1:
		plt.figure(figsize=pl.cm2inch(17.4/2.1, 17.4/2.6))
		plt.rcParams['axes.prop_cycle'] = cycler(color=['k','steelblue'])
		#plt.rcParams['axes.prop_cycle'] = cycler(color=['k','gray'])
		pl.standard_plot(grid, prob,
							"Fit results: Tail heaviness = %f, Asymmetry parameter = %f, Loc. = %f, Scale = %f" %\
							 (nig1,nig2,nig3,nig4),measurement_name,'Probability density',a=0)
		pl.standard_hist(r_ds,200,a=0)
		plt.savefig(nig_density_fig_name, format='jpeg', dpi=1200)
		plt.show()
		pl.plot_props()
		#probplot(residuals_deseasoned, plot=plt, dist='norminvgauss',sparams=(nig1,nig2,nig3,nig4))
		#plt.show()
		print(' ')
		print("Fit results NIG r_ds: Tail heaviness = %f, Asymmetry parameter = %f, Loc. = %f, Scale = %f" %\
							 (nig1,nig2,nig3,nig4))
		print("KS-test results NIG: ", kstest(r_ds,'norminvgauss', args = (nig1,nig2,nig3,nig4)))

	#Look at expected values of deseasonalized residuals and sq residuals each year 
	r_ds_sq = np.power(r_ds,2)

	expected_values_ds,c_ds = expected_residuals_year(r_ds,days_of_year,num_of_years,p)
	expected_values_ds2,c_ds2 = expected_residuals_year(r_ds_sq,days_of_year,num_of_years,p)

	if plot_res_exp == 1:
		pl.standard_plot(days,expected_values_ds,'Expected deseasonalized residuals during a year',
							'Days of year','Expected deseasonalized residuals [K]')
		pl.standard_acf(expected_values_ds,350,'ACF of expected deseasonalized residuals during a year',
							'Lags','ACF',1)
		pl.standard_pacf(expected_values_ds,150,'PACF of expected deseasonalized residuals during a year',
							'Lags','PACF',1)
	if plot_res_sq_exp == 1:
		pl.standard_plot(days,expected_values_ds2,'Expected squared deseasonalized residuals during a year',
							'Days of year','Expected squared deseasonalized residuals [K]')
		pl.standard_acf(expected_values_ds2,350,'ACF of expected squared deseasonalized residuals during a year',
							'Lags','ACF',1)
		pl.standard_pacf(expected_values_ds2,150,'PACF of expected squared deseasonalized residuals during a year',
							'Lags','PACF',1)
	if plot_res_acf == 1:
		pl.standard_acf(r_ds,400,'ACF of current residuals',
							'Lags','ACF',1)
		pl.standard_pacf(r_ds,400,'PACF of current residuals',
							'Lags','PACF',1)
	if plot_res_sq_acf == 1:
		plt.rcParams['figure.figsize'] = pl.cm2inch(17.4/2.1, 17.4/2.6)
		pl.standard_acf(r_ds_sq,40,'ACF of current squared residuals',
							'Days','ACF')
		plt.ylim(top=0.27)
		plt.savefig('resulting_figs/none.jpeg', format='jpeg', dpi=1200)
		plt.show()
		pl.standard_pacf(r_ds_sq,40,'PACF of current squared residuals',
							'Days','PACF')
		plt.ylim(top=0.27)
		plt.savefig('resulting_figs/none.jpeg', format='jpeg', dpi=1200)
		plt.show()
	return nig4