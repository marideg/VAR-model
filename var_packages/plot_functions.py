"""
THIS DOCUMENT CONTAINS FUNCTIONS RELATED TO PLOTTING.
THE CONTAINED FUNCTIONS ARE THE FOLLOWING:
- norm_fit()
- standard_plot()
- double_plot()
- standard_hist()
- standard_acf()
- standard_pacf()
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from cycler import cycler

#SET PLOT PROPERTIES:
def cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)

def plot_props():
    #np.random.seed(19680801)
    plt.style.use('grayscale')
    plt.rcParams['figure.constrained_layout.use'] = True
    #print(plt.style.available)
    plt.rcParams['axes.prop_cycle'] = cycler(linestyle=['solid', 'dashed'],\
                                             color=['steelblue','k'],\
                                             linewidth=[1.6,1.6]) #color=['gray','k']
    # specify the custom font type
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # specify the custom font size
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.handlelength'] = 1.4
    plt.rcParams['figure.figsize'] = cm2inch(17.4/3.1, 17.4/3.6)
    return 0

def norm_fit(data,xmin,xmax,n):
    """
    #Fits a normal distribution to a given data set. 

    INPUT:
    # data (array): A data set that might be close to narmally distributed
    # xmin (int): Minimum x-value of fitted normal distribution (lower extreme of data)
    # xmax (int): Maximum x-value of fitted normal distribution (upper extreme of data)
    # n (int): Number of spaces between xmin and xmax 

    OUTPUT:
    # x (array): range of data
    # prob (array): normal pdf over range x
    # mu (float): mean value of fitted normal distribution
    # std (float): standard deviation of fitted normal distribution
    """
    mu, std = norm.fit(data)
    x = np.linspace(xmin, xmax, n)
    prob = norm.pdf(x, mu, std)
    return x, prob, mu, std



def standard_plot(x,y,title,xlabel,ylabel,a=1,b=0):
    """
    #Makes a standard plot.

    INPUT: 
    # x (array): x-values
    # y (array): y-values
    # title (string): Title of plot
    # xlabel (string): name of x-values
    # ylabel (string): name of y-values
    # a (int): 1 if command plt.show() should be executed
    # b (int): 1 if plotting y-value without given x-value

    OUTPUT:
    # Plot
    """
    if b == 1:
        plt.plot(y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)
    else:
        plt.plot(x,y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)
    if a==1:
        plt.show()

def double_plot(x,y1,y2,title,xlabel,ylabel,labels,legend_loc='best',legend_size='medium',b=0):
    """
    #Makes a plot with two functions.

    INUPT:
    # x (array): x-values
    # y1, y2 (arrays): y-values of two functions
    # title, xlabel, ylabel (strings): title and names of x- and y-values
    # labels (2x1 array, string elements): names of the two plotted functions
    # legend_loc, legend_size (strings): location and size of legends

    OUTPUT:
    # One plot with two functions
    """
    if b == 1:
        plt.plot(y1,label=labels[0])
        plt.plot(y2,label=labels[1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)
        plt.legend(loc=legend_loc, fontsize=legend_size)
        plt.show()
    else:
        plt.plot(x,y1,label=labels[0])
        plt.plot(x, y2, label=labels[1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)
        plt.legend(loc=legend_loc, fontsize=legend_size)
        #plt.show()

def standard_hist(x,n,title=' ',xlabel=' ',ylabel=' ',a=1,b=0):
    """
    #Makes a standard histogram.

    INPUT: 
    # x (array): x-values
    # n (integer): number of bins
    # title (string): title of plot
    # xlabel (string): name of x-values
    # ylabel (string): name of y-values
    # a (int): 1 if command plt.show() should be executed 
    # b (int): 1 if histogram should have title and labels

    OUTPUT:
    # Histogram plot
    """
    plt.hist(x,n,density=True)
    if b==1:
        #plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    if a==1:
        plt.show()

def standard_acf(x,n,title,xlabel,ylabel,a=0,b='plot',show=1,ax=None):
    """
    #Makes a standard acf-plot (single- or multi-frame), (99-percentile intervals).

    CALLS FUNCTIONS: statsmodels.graphics.tsaplots: plot_acf()

    INPUT: 
    # x (array): x-values
    # n (int): number of lags
    # title (string): title of acf-plot
    # xlabel (string): name of lags
    # ylabel (string): name of acf-values
    # a (int): number of plot-frames
    # b (string): 'plot' for one single acf-plot (not multi-frame)
    # show (int): 1 when to print all plots in a multi-frame acf-plot

    OUTPUT:
    # ACF-plot
    """
    if b == 'plot':
        plot_acf(x, lags = n,ax=ax,alpha=0.01,title=None)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)
        #plt.show()
    else:
        plot_acf(x, lags = n, ax=ax,alpha=0.01,title=None)
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        #a.set_title(title)
        #if show == 1:
        #   plt.show()
    if a == 1:
        plt.show()

def standard_pacf(x,n,title,xlabel,ylabel,a=0,b='plot',show=1):
    """
    #Makes a standard pacf-plot (single- or multi-frame), (99-percentile intervals).

    CALLS FUNCTIONS: statsmodels.graphics.tsaplots: plot_pacf()

    INPUT: 
    # x (array): x-values
    # n (int): number of lags
    # title (string): title of pacf-plot
    # xlabel (string): name of lags
    # ylabel (string): name of pacf-values
    # a (int): number of plot-frames
    # b (string): 'plot' for one single pacf-plot (not multi-frame)
    # show (int): 1 when to print all plots in a multi-frame pacf-plot

    OUTPUT:
    # PACF-plot
    """
    if b == 'plot':    
        plot_pacf(x, lags = n,alpha=0.01,title=None,method='ywm')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(title)
        #plt.show()
    else:
        plot_pacf(x, lags = n, ax=a,alpha=0.01,title=None,method='ywm')
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        #a.set_title(title)
    if a == 1:
        plt.show()

def corr_plot_2dims(x1,x2,x3,x4,nlags,\
                    fig_name,name1,name2,name3,name4,\
                    start_lag1,start_lag2,start_lag3,start_lag4,\
                    y_axes_limit,bottom,top):
    fig = plt.figure(figsize=cm2inch(17.4/1.0, 17.4/1.5),constrained_layout=False)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    ax1.plot(x1[start_lag1:nlags], 'o')
    ax1.vlines(range(len(x1[start_lag1:nlags])),[0],x1[start_lag1:nlags])
    ax1.axhline()
    ax1.set(title=name1)
    if y_axes_limit == 1:
        ax1.set_ylim(bottom=bottom,top=top)

    ax2.plot(x2[start_lag2:nlags], 'o')
    ax2.vlines(range(len(x2[start_lag2:nlags])),[0],x2[start_lag2:nlags])
    ax2.axhline()
    ax2.set(title=name2)
    if y_axes_limit == 1:
        ax2.set_ylim(bottom=bottom,top=top)
    
    ax3.plot(x3[start_lag3:nlags], 'o')
    ax3.vlines(range(len(x3[start_lag3:nlags])),[0],x3[start_lag3:nlags])
    ax3.axhline()
    ax3.set(title=name3)
    if y_axes_limit == 1:
        ax3.set_ylim(bottom=bottom,top=top)

    ax4.plot(x4[start_lag4:nlags], 'o')
    ax4.vlines(range(len(x4[start_lag4:nlags])),[0],x4[start_lag4:nlags])
    ax4.axhline()
    ax4.set(title=name4)
    if y_axes_limit == 1:
        ax4.set_ylim(bottom=bottom,top=top)

    fig.tight_layout()
    plt.savefig(fig_name, format='jpeg', dpi=1200)
    plt.show()

def ten_years_plot(yvalues,time,fitted_function,start,n,labels,x_label,y_label,pmin,pmax,savefig_name):
    ten_years = 365*10
    lines = np.linspace(start,start+ten_years,n+1)
    print("Check here if correct vertical lines of short legth plot is given: ",lines)
    fig = plt.figure(figsize=cm2inch(17.4/1.0,17.4/2.0))
    ax = fig.add_subplot(1,1,1)
    plt.plot(time[start:start+ten_years],yvalues[start:start+ten_years],label=labels[0])
    plt.plot(time[start:start+ten_years],fitted_function[start:start+ten_years],label=labels[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='lower left')
    plt.vlines(lines,pmin,pmax,linestyles='dashed',colors='k',linewidth=0.5)
    ax.autoscale(enable=True, axis='x', tight=True)

    ax_labels = ["2009","2010","2011","2012","2013","2014","2015","2016","2017","2018"]

    #ax.set_xticklabels(ax_labels)
    plt.xticks([11133,11498,11863,12228,12593,12958,13323,13688,14053,14418],ax_labels)
    ax.tick_params(axis='x',length=0)

    plt.savefig(savefig_name, format='jpeg', dpi=1200)
    plt.show()


