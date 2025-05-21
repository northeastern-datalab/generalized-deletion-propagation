import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pprint import pprint

# mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': [u'Arial', u'Liberation Sans']})
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['grid.color'] = '777777'  # grid color
mpl.rcParams['xtick.major.pad'] = 2  # padding of tick labels: default = 4
mpl.rcParams['ytick.major.pad'] = 1  # padding of tick labels: default = 4


plt.rc('font', size=14)          # controls default text size
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

expt_markers = {
    'lp_results':'s',
    'ilp_results':'*',
    'lp_results_old':'^',
    'ilp_results_old':'<',
    'specialized_dpvs':'H',
    'specialized_adp':'H',
    'specialized_swp':'H',
}
expt_colors = {
    'lp_results':'tab:blue',
    'ilp_results':'tab:orange',
    'lp_results_old':'greenyellow',
    'ilp_results_old':'tab:purple',
    'specialized_dpvs':'tab:olive',
    'specialized_adp':'tab:olive',
    'specialized_swp':'tab:olive',
}
expt_colors_light = {
    'lp_results':'tab:blue',
    'ilp_results':'tab:orange',
    'lp_results_old':'greenyellow',
    'ilp_results_10':'tab:purple',
    'specialized_dpvs':'tab:olive',
    'specialized_adp':'tab:olive',
    'specialized_swp':'tab:olive',
}
expt_labels = {
    'lp_results':'LP',
    'ilp_results':'ILP',
    'lp_results_old':'LP (naive)',
    'ilp_results_old':'ILP (naive)',
    'specialized_dpvs':'DPVS-S',
    'specialized_adp':'ADP-S',
    'specialized_swp':'SWP-S',
}

timing_lines = ['lp_results', 'ilp_results', 'lp_results_old','ilp_results_old','specialized_swp','specialized_dpvs','specialized_adp']
gdp_lines = ['lp_results', 'ilp_results', 'lp_results_old','ilp_results_old','specialized_swp','specialized_dpvs','specialized_adp']

def plot_case_study(EXPT_DATA_FILE = 'data/synthetic_expt/expt-data-case-study.csv', PLOT_OUTPUT_FILE = None, xmin= 1, xmax = 1e4, expt_type="GDP", g01_arrow_point = None):
    # Read the data
    expt_data = pd.read_csv(EXPT_DATA_FILE, on_bad_lines='warn')
    # Drop rows where lp_results: GDP is na
    expt_data = expt_data.dropna(subset=['lp_results: '+expt_type])

    # Sort with increasing number of witnesses
    expt_data = expt_data.sort_values(by=['lp_results: number_of_witnesses'])


    fig, axes = plt.subplots(1, 2, figsize=(9,4))
    plt.subplots_adjust(wspace=-1, hspace=-1)

    # # Add line for linear scalability
    # linear_delta = 0.9999*2
    # linear_x = np.linspace(2,1e9,2)
    # linear_y = linear_x - linear_delta 
    # axes[0].plot(linear_x, linear_y, linestyle='dashed',color='black')

    expt_data_gb = expt_data.groupby(['lp_results: number_of_witnesses'], as_index = False).agg({'lp_results: Solve Time': 'median', 'ilp_results: Solve Time': 'median'})

    expt_data = expt_data.merge(expt_data_gb, on = ['lp_results: number_of_witnesses'], how = 'left', suffixes = ('', '_median'))
 
    for expt in gdp_lines:
        if expt+': '+expt_type in expt_data.columns:
            expt_data[expt+': '+expt_type+': Delta'] = np.abs(expt_data[expt+': '+expt_type]) / np.abs(expt_data['ilp_results: '+expt_type])


    # Plot 1: Timing - Actual Values
    for expt in timing_lines:
        if expt+': Solve Time' in expt_data.columns:
            axes[0].scatter(expt_data['lp_results: number_of_witnesses'], expt_data[expt+': Solve Time_median'], label = expt_labels[expt],  marker = expt_markers[expt], color = expt_colors[expt])
            # Draw line of best fit in the plot
            z = np.polyfit(expt_data['lp_results: number_of_witnesses'], expt_data[expt+': Solve Time_median'], 2)
            p = np.poly1d(z)
            axes[0].plot(expt_data['lp_results: number_of_witnesses'],p(expt_data['lp_results: number_of_witnesses']), color = expt_colors[expt], linestyle='dashed')

    # Plot 2: RES - Actual Values
    for expt in gdp_lines:
        if expt+': GDP' in expt_data.columns:
            axes[1].scatter(expt_data['lp_results: number_of_witnesses'], expt_data[expt+': GDP: Delta'],  label = expt_labels[expt], marker = expt_markers[expt], color = expt_colors[expt])
            z = np.polyfit(expt_data['lp_results: number_of_witnesses'], expt_data[expt+': GDP: Delta'], 1)
            p = np.poly1d(z)
            axes[1].plot(expt_data['lp_results: number_of_witnesses'],p(expt_data['lp_results: number_of_witnesses']), color = expt_colors[expt], linestyle='dashed')


    # Set y axis to log scale
    axes[0].set_yscale('log')
    axes[1].set_xscale('log')

    xMinorLocator = LogLocator(base=10, subs=[0.1 * n for n in range(1, 10)], numticks = 40)   
    yMinorLocator = LogLocator(base=10, subs=[0.1 * n for n in range(1, 10)], numticks = 40)   
    grids = [0,1]
    for i in grids:
        axes_i = axes[i]
        axes_i.set_xscale('log')
        axes_i.set_yscale('log')
        axes_i.xaxis.set_minor_locator(xMinorLocator)
        axes_i.yaxis.set_minor_locator(yMinorLocator)
        axes_i.grid(True, which='both', axis='both', alpha=0.05, linestyle='-',linewidth=1, zorder=1)  # linestyle='dashed', which='minor', axis='y',
        axes_i.set_xticks([10**i for i in range(int(math.log(xmax,10)))])
        axes_i.set_xlim(xmax=xmax)    
        axes_i.set_xlim(xmin=xmin)   

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc= 'best',
                                handlelength=1.5,
                                labelspacing=0,             # distance between label entries
                                handletextpad=0.3,          # distance between label and the line representation
                                borderaxespad=0.1,        # distance between legend and the outer axes
                                borderpad=0.1,                # padding inside legend box
                                numpoints=1,)

    axes[0].set_ylabel('Solve Time (s)')
    axes[1].set_ylabel('GDP')
    axes[0].set_xlabel('Number of witnesses')
    axes[1].set_xlabel('Number of witnesses')

    expt1 = 'ilp_results'
    expt2 = 'lp_results'

    # if g01_arrow_point is not None:
    #     median_x = expt_data['number_of_witnesses'].iloc[g01_arrow_point]
    #     median_ymin = expt_data[expt1+': GDP: Delta'].iloc[g01_arrow_point]
    #     median_ymax = expt_data[expt2+': GDP: Delta'].iloc[g01_arrow_point]
    #     median_ymid = math.sqrt(median_ymax*median_ymin)
    #     axes[1].annotate("", xy=(median_x, median_ymin), 
    #                     xytext=(median_x, median_ymax), 
    #                     arrowprops=dict(arrowstyle="<->"))
    #     axes[1].annotate(str(round(median_ymax / median_ymin, 1))+'x', 
    #                     xy=(median_x, median_ymin), 
    #                     xytext=(median_x*1.1, median_ymid), 
    #                     )

    
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc= 'upper left',
                            handlelength=1.5,
                            labelspacing=0,             # distance between label entries
                            handletextpad=0.3,          # distance between label and the line representation
                            borderaxespad=0.1,        # distance between legend and the outer axes
                            borderpad=0.1,                # padding inside legend box
                            numpoints=1,)
    
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc= 'lower right',
                        handlelength=1.5,
                        labelspacing=0,             # distance between label entries
                        handletextpad=0.3,          # distance between label and the line representation
                        borderaxespad=0.1,        # distance between legend and the outer axes
                        borderpad=0.1,                # padding inside legend box
                        numpoints=1,)

    fig.tight_layout()

    if PLOT_OUTPUT_FILE == None:
        plt.show()
    else:
        print('Saving plot to '+PLOT_OUTPUT_FILE)
        plt.savefig(PLOT_OUTPUT_FILE, bbox_inches="tight")


def plot_gdp_expt(case_no, EXPT_DATA_FILE = 'data/synthetic_expt/expt-data-case-{}.csv', PLOT_OUTPUT_FILE = 'data/synthetic_expt/plots/expt-data-case-{}.pdf', expt_type="GDP", 
BUCKET_SIZE_DENOMINATOR = 4, DATA_ALPHA = 0.1, xmax = 1e4, xmin = 1,g00_ymax = 1e3, g01_ymax = 1e5, 
g10_ymax= 1e5, g11_ymax = 4, hide_top_plots = False, hide_bottom_plots = False, plot_legend = True, g00_ymin = 1e-2, g00_arrow_point = None, g00_arrow_expt1 = None, g00_arrow_expt2 = None, single_plot = True, no_bins = False, g01_arrow_point = None, g01_arrow_expt1 = None, g01_arrow_expt2 = None,):
    """
    Generate and save a plot from the data available for a synthetic expt case

    Args:
        case_no (int): Experiment case number
        BUCKET_SIZE_DENOMINATOR(int): the size of buckets from which we plot median. Size 4 implies that buckets are of size 10^(1/4)
        DATA_ALPHA (float): alpha value of all points that are not the median
    """

    # Read the data
    expt_data = pd.read_csv(EXPT_DATA_FILE.format(case_no), on_bad_lines='warn')

    # Rename some columns
    if 'specialized_adp: ADP' in expt_data.columns:
        expt_data['specialized_adp: GDP'] = expt_data['specialized_adp: ADP']
    if 'specialized_dpvs: DPVS' in expt_data.columns:
        expt_data['specialized_dpvs: GDP'] = expt_data['specialized_adp: DPVS']
    if 'specialized_swp: SWP' in expt_data.columns:
        # Make GDP positive for SWP
        for expt in gdp_lines:
            if expt+': '+expt_type in expt_data.columns:
                expt_data[expt+': '+expt_type] = expt_data[expt+': number_of_tuples'] + expt_data[expt+': '+expt_type]

        expt_data['specialized_swp: GDP'] = expt_data['specialized_swp: SWP']
    
    for expt in gdp_lines:
        if expt+': GDP' in expt_data.columns:
            expt_data[expt+': GDP'] = np.abs(expt_data[expt+': GDP'])


    # Accentuate data with deltas:
    for expt in timing_lines:
        if expt+': Solve Time' in expt_data.columns:
            expt_data[expt+': Solve Time'+': Delta'] = expt_data[expt+': Solve Time']/expt_data['lp_results: Solve Time']
    for expt in gdp_lines:
        if expt+': '+expt_type in expt_data.columns:
            expt_data[expt+': '+expt_type+': Delta'] = expt_data[expt+': '+expt_type] / expt_data['ilp_results: '+expt_type]
    
    if not single_plot:
        fig, axes = plt.subplots(1,2, figsize=(9,4))
        plt.subplots_adjust(wspace=-1, hspace=-1)
    else:
        fig, axes = plt.subplots(1,1, figsize=(5,4))

    if single_plot:
        axes_zero = axes
    else:
        axes_zero = axes[0]

    # Add line for linear scalability
    linear_delta = 0.9999*2
    linear_x = np.linspace(2,1e9,2)
    linear_y = linear_x - linear_delta 
    axes_zero.plot(linear_x, linear_y, linestyle='dashed',color='black')



    # Plot 1: Timing - Actual Values
    for expt in timing_lines:
        if expt+': Solve Time' in expt_data.columns:
            axes_zero.scatter(expt_data['number_of_witnesses'], expt_data[expt+': Solve Time'], marker = expt_markers[expt], color = lighten_color(expt_colors[expt]), alpha = DATA_ALPHA )
    if not single_plot:
        # Plot 2: RES - Actual Values
        for expt in gdp_lines:
            if expt+': '+expt_type in expt_data.columns:
                axes[1].scatter(expt_data['number_of_witnesses'], expt_data[expt+': '+expt_type+': Delta'], marker = expt_markers[expt], color = lighten_color(expt_colors[expt]), alpha = DATA_ALPHA )

    # Plot median points
    
    # Bucket and find the median of values
    if not no_bins: 
        labels = [(10**(1/BUCKET_SIZE_DENOMINATOR))**i for i in range(100)]
        bins = [0]+[(labels[i]+labels[i+1])/2 for i in range(len(labels)-1)]+[10**math.ceil(math.log(labels[-1],10))]
        expt_data['binned_witnesses'] = pd.cut(expt_data['number_of_witnesses'], bins=bins, labels= labels)
        expt_data_bin = pd.DataFrame()
        for col in expt_data.columns:
            if is_numeric_dtype(expt_data[col]):
                expt_data_bin[col] = expt_data.groupby(['binned_witnesses'], observed=False)[col].agg('median').values
        expt_data_bin['number_of_witnesses'] = labels

        for expt in timing_lines:
            if expt+': Solve Time' in expt_data_bin.columns:
                axes_zero.plot(expt_data_bin['number_of_witnesses'], expt_data_bin[expt+': Solve Time'], label = expt_labels[expt], marker = expt_markers[expt], color = expt_colors[expt], markerfacecolor='none')

        if not single_plot:
            # Plot 2: RES - Actual Values
            for expt in gdp_lines:
                if expt+': '+expt_type in expt_data.columns:
                    axes[1].plot(expt_data_bin['number_of_witnesses'], expt_data_bin[expt+': '+expt_type+': Delta'], label = expt_labels[expt], marker = expt_markers[expt], color = expt_colors[expt], markerfacecolor='none')

    # Draw an arrow between last median points
    # axes[0].arrow(expt_data_bin['number_of_witnesses'].iloc[13], 10, 0, 1e2, head_width= 1e6, head_length = 2*1e1, shape ='full')

    xMinorLocator = LogLocator(base=10, subs=[0.1 * n for n in range(1, 10)], numticks = 40)   
    yMinorLocator = LogLocator(base=10, subs=[0.1 * n for n in range(1, 10)], numticks = 40)   
    grids = [0,1]
    if single_plot:
        grids = [0]
    for i in grids:
        if single_plot:
            axes_i = axes_zero
        else:
            axes_i = axes[i]
        axes_i.set_xscale('log')
        axes_i.set_yscale('log')
        axes_i.xaxis.set_minor_locator(xMinorLocator)
        axes_i.yaxis.set_minor_locator(yMinorLocator)
        axes_i.grid(True, which='both', axis='both', alpha=0.05, linestyle='-',linewidth=1, zorder=1)  # linestyle='dashed', which='minor', axis='y',
        axes_i.set_xticks([10**i for i in range(int(math.log(xmax,10)))])
        axes_i.set_xlim(xmax=xmax)    
        axes_i.set_xlim(xmin=xmin)    


    axes_zero.set_ylim(ymin=g00_ymin)    
    axes_zero.set_ylim(ymax=g00_ymax)
    axes_zero.set_xlabel('Number of witnesses')
    axes_zero.set_ylabel('Solve Time (s)')
    
    if not single_plot:
        axes[1].set_ylim(ymin=1)    
        axes[1].set_ylim(ymax=g01_ymax) 
        axes[1].set_xlabel('Number of witnesses')
        axes[1].set_ylabel('Δ'+expt_type)
    

    handles, labels = axes_zero.get_legend_handles_labels()
    if plot_legend:
        if not single_plot:
            axes[1].legend(handles, labels, loc= 'best',
                                handlelength=1.5,
                                labelspacing=0,             # distance between label entries
                                handletextpad=0.3,          # distance between label and the line representation
                                borderaxespad=0.1,        # distance between legend and the outer axes
                                borderpad=0.1,                # padding inside legend box
                                numpoints=1,)
        
        handles, labels = axes_zero.get_legend_handles_labels()
        axes_zero.legend(handles, labels, loc= 'upper left',
                            handlelength=1.5,
                            labelspacing=0,             # distance between label entries
                            handletextpad=0.3,          # distance between label and the line representation
                            borderaxespad=0.1,        # distance between legend and the outer axes
                            borderpad=0.1,                # padding inside legend box
                            numpoints=1,)
    fig.tight_layout()
    
    expt1 = g00_arrow_expt1
    expt2 = g00_arrow_expt2
    if g00_arrow_point is not None:
        median_x = expt_data_bin['number_of_witnesses'].iloc[g00_arrow_point]
        median_ymin = expt_data_bin[expt1+': Solve Time'].iloc[g00_arrow_point]
        median_ymax = expt_data_bin[expt2+': Solve Time'].iloc[g00_arrow_point]
        median_ymid = math.sqrt(median_ymax*median_ymin)
        axes_zero.annotate("", xy=(median_x, median_ymin), 
                        xytext=(median_x, median_ymax), 
                        arrowprops=dict(arrowstyle="<->"))
        axes_zero.annotate(str(round(median_ymax / median_ymin, 1))+'x', 
                        xy=(median_x, median_ymin), 
                        xytext=(median_x*1.1, median_ymid), 
                        )
    expt1 = g01_arrow_expt1
    expt2 = g01_arrow_expt2        
    if g01_arrow_point is not None:
        median_x = expt_data_bin['number_of_witnesses'].iloc[g01_arrow_point]
        median_ymin = expt_data_bin[expt1+': GDP: Delta'].iloc[g01_arrow_point]
        median_ymax = expt_data_bin[expt2+': GDP: Delta'].iloc[g01_arrow_point]
        median_ymid = math.sqrt(median_ymax*median_ymin)
        axes[1].annotate("", xy=(median_x, median_ymin), 
                        xytext=(median_x, median_ymax), 
                        arrowprops=dict(arrowstyle="<->"))
        axes[1].annotate(str(round(median_ymax / median_ymin, 1))+'x', 
                        xy=(median_x, median_ymin), 
                        xytext=(median_x*1.1, median_ymid), 
                        )
        
    if PLOT_OUTPUT_FILE == None:
        plt.show()
    else:
        print('Saving plot to '+PLOT_OUTPUT_FILE.format(case_no))
        plt.savefig(PLOT_OUTPUT_FILE.format(case_no), bbox_inches="tight")
        # allows to automatically show the PDF. Can be commented out if it gets too annoying
        # Does not work on Windows
        # os.system("open " + PLOT_OUTPUT_FILE.format(case_no))            


def plot_gdp_expt_with_deltas(case_no, EXPT_DATA_FILE = 'data/synthetic_expt/expt-data-case-{}.csv', PLOT_OUTPUT_FILE = 'data/synthetic_expt/plots/expt-data-case-{}.pdf', expt_type='GDP',
BUCKET_SIZE_DENOMINATOR = 4, DATA_ALPHA = 0.1, xmin = 0, xmax = 1e4, g00_ymax = 1e3, g01_ymax = 1e5, g10_ymax= 1e5, 
g11_ymax = 4, g01_ymin = 1, g11_ymin = 0.1, hide_top_plots = False, hide_bottom_plots = False, g00_arrow_point = None, 
g11_arrow_points = None):
    """
    Generate and save a plot from the data available for a synthetic expt case

    Args:
        case_no (int): Experiment case number
        BUCKET_SIZE_DENOMINATOR(int): the size of buckets from which we plot median. Size 4 implies that buckets are of size 10^(1/4)
        DATA_ALPHA (float): alpha value of all points that are not the median
    """

    # Read the data
    expt_data = pd.read_csv(EXPT_DATA_FILE.format(case_no), on_bad_lines='warn')

    # Pre-process: rename column
    if expt_type == "GDP":
        for expt in gdp_lines:
            if expt+': '+expt_type in expt_data.columns:
                expt_data[expt+': '+expt_type] = -expt_data[expt+': '+expt_type]

    # Accentuate data with deltas:
    for expt in timing_lines:
        if expt+': Solve Time' in expt_data.columns:
            expt_data[expt+': Solve Time'+': Delta'] = expt_data[expt+': Solve Time']/expt_data['lp_results: Solve Time']
    for expt in gdp_lines:
        if expt+': '+expt_type in expt_data.columns:
            expt_data[expt+': '+expt_type+': Delta'] = expt_data[expt+': '+expt_type] / expt_data['ilp_results: '+expt_type]
    
    fig, axes = plt.subplots(2, 2, figsize=(5*2,4*2))


    # Add line for linear scalability
    linear_delta = 0.99*2
    linear_x = np.linspace(2,1e9,2)
    linear_y = linear_x - linear_delta 
    axes[0][0].plot(linear_x, linear_y, linestyle='dashed',color='black')

    # Plot 1: Timing - Actual Values
    for expt in timing_lines:
        if expt+': Solve Time' in expt_data.columns:
            axes[0][0].scatter(expt_data['lp_results: number_of_witnesses'], expt_data[expt+': Solve Time'], marker = expt_markers[expt], color = lighten_color(expt_colors[expt]), alpha = DATA_ALPHA )
    # Plot 2: RES - Actual Values
    for expt in gdp_lines:
        if expt+': '+expt_type in expt_data.columns:
            axes[0][1].scatter(expt_data['lp_results: number_of_witnesses'], expt_data[expt+': '+expt_type], marker = expt_markers[expt], color = lighten_color(expt_colors[expt]), alpha = DATA_ALPHA )

    if not hide_bottom_plots:
        # Plot 3: Timing - Relative Values
        for expt in timing_lines:
            if expt+': Solve Time' in expt_data.columns and expt != 'lp_results':
                axes[1][0].scatter(expt_data['lp_results: number_of_witnesses'], expt_data[expt+': Solve Time: Delta'], marker = expt_markers[expt], color = lighten_color(expt_colors[expt]), alpha = DATA_ALPHA )
        # Plot 4: RES - Relative Values 
        for expt in gdp_lines:
            if expt+': '+expt_type in expt_data.columns and expt != 'ilp_results':
                axes[1][1].scatter(expt_data['lp_results: number_of_witnesses'], expt_data[expt+': '+expt_type+': Delta'], marker = expt_markers[expt], color = lighten_color(expt_colors[expt]), alpha = DATA_ALPHA)


    # Plot median points
    
    # Bucket and find the median of values
    labels = [(10**(1/BUCKET_SIZE_DENOMINATOR))**i for i in range(100)]
    bins = [0]+[(labels[i]+labels[i+1])/2 for i in range(len(labels)-1)]+[10**math.ceil(math.log(labels[-1],10))]
    expt_data['binned_witnesses'] = pd.cut(expt_data['lp_results: number_of_witnesses'], bins=bins, labels= labels)
    expt_data_bin = pd.DataFrame()
    for col in expt_data.columns:
        if is_numeric_dtype(expt_data[col]):
            expt_data_bin[col] = expt_data.groupby(['binned_witnesses'], observed = False)[col].agg('median').values
            pprint(expt_data_bin)

    expt_data_bin['lp_results: number_of_witnesses'] = labels

    for expt in timing_lines:
        if expt+': Solve Time' in expt_data_bin.columns:
            axes[0][0].plot(expt_data_bin['lp_results: number_of_witnesses'], expt_data_bin[expt+': Solve Time'], label = expt_labels[expt], marker = expt_markers[expt], color = expt_colors[expt], markerfacecolor='none')
    # Plot 2: RES - Actual Values
    for expt in gdp_lines:
        if expt+': '+expt_type in expt_data.columns:
            axes[0][1].plot(expt_data_bin['lp_results: number_of_witnesses'], expt_data_bin[expt+': '+expt_type], label = expt_labels[expt], marker = expt_markers[expt], color = expt_colors[expt], markerfacecolor='none')
    if not hide_bottom_plots:
        # Plot 3: Timing - Relative Values
        axes[1][0].axhline(y = 1, marker = expt_markers['lp_results'], color = expt_colors['lp_results'],)
        for expt in timing_lines:
            if expt+': Solve Time' in expt_data_bin.columns and expt != 'lp_results':
                axes[1][0].plot(expt_data_bin['lp_results: number_of_witnesses'], expt_data_bin[expt+': Solve Time: Delta'], label = expt_labels[expt], marker = expt_markers[expt], color = expt_colors[expt], markerfacecolor='none')
        # Plot 4: RES - Relative Values
        for expt in gdp_lines:
            if expt+': '+expt_type in expt_data.columns and expt != 'ilp_results':
                axes[1][1].plot(expt_data_bin['lp_results: number_of_witnesses'], expt_data_bin[expt+': '+expt_type+': Delta'], label = expt_labels[expt], marker = expt_markers[expt], color = expt_colors[expt], markerfacecolor='none')

    xMinorLocator = LogLocator(base=10, subs=[0.1 * n for n in range(1, 10)], numticks = 40)   
    yMinorLocator = LogLocator(base=10, subs=[0.1 * n for n in range(1, 10)], numticks = 40)   
    for i,j in [[0,0],[0,1],[1,0]]:
        axes[i][j].set_xscale('log')
        axes[i][j].set_yscale('log')
        axes[i][j].xaxis.set_minor_locator(xMinorLocator)
        axes[i][j].yaxis.set_minor_locator(yMinorLocator)
        axes[i][j].grid(True, which='both', axis='both', alpha=0.05, linestyle='-',linewidth=1, zorder=1)  # linestyle='dashed', which='minor', axis='y',
        axes[i][j].set_xticks([10**i for i in range(int(math.log(xmax,10)))])
        axes[i][j].set_xlim(xmax=xmax)    
        axes[i][j].set_xlim(xmin=xmin)    

    axes[1][1].set_xscale('log')
    axes[1][1].xaxis.set_minor_locator(xMinorLocator)
    axes[1][1].set_xticks([10**i for i in range(int(math.log(xmax,10)))])
    axes[1][1].set_xlim(xmax=xmax)    
    axes[1][1].set_xlim(xmin=xmin)    
    axes[1][1].grid(True, which='both', axis='both', alpha=0.05, linestyle='-',linewidth=1, zorder=1)  # linestyle='dashed', which='minor', axis='y',


    axes[0][0].set_ylim(ymin=1e-2)    
    axes[0][0].set_ylim(ymax=g00_ymax)    
    axes[0][1].set_ylim(ymin=g01_ymin)    
    axes[0][1].set_ylim(ymax=g01_ymax)    
    axes[1][0].set_ylim(ymax=g10_ymax)    
    
    axes[1][1].set_ylim(ymin=g11_ymin)    
    axes[1][1].set_ylim(ymax=g11_ymax)    

    axes[0][0].set_ylabel('Solve Time (s)')
    axes[0][1].set_ylabel(expt_type)
    axes[1][0].set_ylabel('Δ Solve Time')
    axes[1][1].set_ylabel('Δ'+expt_type)
    axes[1][0].set_xlabel('Number of witnesses')
    axes[1][1].set_xlabel('Number of witnesses')

    

    handles, labels = axes[0][1].get_legend_handles_labels()
    axes[0][1].legend(handles, labels, loc= 'lower right',
                            handlelength=1.5,
                            labelspacing=0,             # distance between label entries
                            handletextpad=0.3,          # distance between label and the line representation
                            borderaxespad=0.1,        # distance between legend and the outer axes
                            borderpad=0.1,                # padding inside legend box
                            numpoints=1,)

    
    expt1 = 'ilp_results'
    expt2 = 'ilp_results_old'
    if g00_arrow_point is not None:  
        median_x = expt_data_bin['lp_results: number_of_witnesses'].iloc[g00_arrow_point]
        median_ymin = expt_data_bin[expt1+': Solve Time'].iloc[g00_arrow_point]
        median_ymax = expt_data_bin[expt2+': Solve Time'].iloc[g00_arrow_point]
        print(expt_data_bin[expt1+': Solve Time'])
        print(float(median_ymin))
        print(float(median_ymax))
        median_ymid = math.sqrt(median_ymax*median_ymin)
        axes[0][0].annotate("", xy=(median_x, median_ymin), 
                        xytext=(median_x, median_ymax), 
                        arrowprops=dict(arrowstyle="<->"))
        axes[0][0].annotate(str(round(median_ymax / median_ymin))+'x', 
                        xy=(median_x, median_ymin), 
                        xytext=(median_x*1.1, median_ymid), 
                        )
    
    if g11_arrow_points is not None:

        for (arrow_point, expt1, expt2) in g11_arrow_points:
            median_x = expt_data_bin['lp_results: number_of_witnesses'].iloc[arrow_point]
            median_ymin = expt_data_bin[expt1+': '+expt_type+': Delta'].iloc[arrow_point]
            median_ymax = expt_data_bin[expt2+': '+expt_type+': Delta'].iloc[arrow_point]
            median_ymid = math.sqrt(median_ymax*median_ymin)
            axes[1][1].annotate("", xy=(median_x, median_ymin), 
                            xytext=(median_x, median_ymax), 
                            arrowprops=dict(arrowstyle="<->"))
            axes[1][1].annotate(str(round(median_ymax / median_ymin,1))+'x', 
                            xy=(median_x, median_ymin), 
                            xytext=(median_x*1.1, median_ymid), 
                            )

    fig.tight_layout()
    if PLOT_OUTPUT_FILE == None:
        plt.show()
    else:
        print('Saving plot to '+PLOT_OUTPUT_FILE.format(case_no))
        plt.savefig(PLOT_OUTPUT_FILE.format(case_no), bbox_inches="tight")
        # allows to automatically show the PDF. Can be commented out if it gets too annoying
        # Does not work on Windows
        # os.system("open " + PLOT_OUTPUT_FILE.format(case_no))            

def lighten_color(color, amount=0.3):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])

if __name__ == '__main__':
    for i in range(31, 140):
        try:
            plot_resilience_expt(i)
        except:
            pass