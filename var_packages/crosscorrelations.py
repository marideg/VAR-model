import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf
import plot_functions as pl
import itertools


def cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)

def compute_and_plot_crosscorrelations(data,data_names,parameters,label_names,unit_names,\
										dataset_type,number_of_lags,y_axes_min,y_axes_max,figure_name_base):
	print(' ')
	print(' ')
	print('********** RESULTS FROM compute_and_plot_crosscorrelations() FOLLOWS **********')

	# Compute crosscorrelations and plot
	fig = plt.figure(figsize=cm2inch(17.4/1.0, 17.4/1.5),constrained_layout=False)
	n = 0
	number_of_dimensions = len(data_names)
	#unit_name = 'xy'
	for k,l in itertools.product(data_names,repeat=2):
		corr = ccf(data[k],data[l])

		#Output
		print("Correlation between ",k," and ",l," is computed for ",dataset_type," data")
		ax = plt.subplot(number_of_dimensions,number_of_dimensions,n + 1)
		ax.plot(corr[0:number_of_lags[dataset_type]], 'o')
		ax.vlines(range(len(corr[0:number_of_lags[dataset_type]])),[0],corr[0:number_of_lags[dataset_type]])
		ax.axhline()
		subtitle = ''.join([label_names[k],' - ',label_names[l]])
		#ax.set(title=subtitle)
		if dataset_type in y_axes_max:
			ax.set_ylim(bottom=y_axes_min[dataset_type],top=y_axes_max[dataset_type])
		if unit_names[k] == '[B]':
			unit_name = 'angle'
		n += 1
	#fig.tight_layout()
	plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
	figure_name = ''.join([figure_name_base[dataset_type],'_','unit','.',parameters['figure format']])
	plt.savefig(figure_name,format=parameters['figure format'],dpi=parameters['figure dpi'])	
	#plt.show()
	plt.close(fig) 
	print("See corresponding crosscorrelation plot!")
	return 0


