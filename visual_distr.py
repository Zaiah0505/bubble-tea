""" 
The purpose of this file is to illustrate the concepts of distributions.

You are not required to understand the code.
"""

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from scipy.stats import norm
import pandas as pd

def update_x(x):
    """Function for updating the chart with the new X value """
    
    xs = np.arange(-3.5, 3.51, 0.01)
    ys = norm.pdf(xs)
    x_cdf = np.arange(-3.5, x+0.01, 0.01)
    y_cdf = norm.pdf(x_cdf) 
    x_conn = x
    
    plt.figure(figsize=(6, 3.5))
    plt.plot(xs, ys, linewidth=3, c='k', alpha=0.4, label='Density curve')
    plt.fill_between(x_cdf, y_cdf, color='b', alpha=0.4, 
                     label='Cumulative probability')
    plt.scatter(x, 0, c='r', s=80, alpha=0.5)
    
    plt.plot([x_conn, 3], [0.5*norm.pdf(x_conn), 0.35], 
             c='b', linewidth=2.5, alpha=0.4)
    plt.plot([max(-3.8, x-0.8), x-0.01], 
             [norm.pdf(x), norm.pdf(x)], 
             c='k', linewidth=2.5, alpha=0.5)
    
    plt.text(x+0.2, -0.03, '$X=$' + '{0:4.2f}'.format(x), c='r', fontsize=11)
    plt.text(1.5, 0.36, 
             '$P(X\leq$' + '{0:5.2f}'.format(x) + '$)=$' + 
             '{0:0.3f}'.format(norm.cdf(x)), 
             c='b', fontsize=11)
    plt.text(max(-5, x-2), norm.pdf(x)-0.01, '{0:0.3f}'.format(norm.pdf(x)), 
             fontsize=12)
    tx = 'norm.pdf({0:6.3f}, loc=0, scale=1)'.format(x)
    tx += ' = {0:6.3f}\n'.format(norm.pdf(x))
    tx += 'norm.cdf({0:6.3f}, loc=0, scale=1)'.format(x)
    tx += ' = {0:6.3f}\n'.format(norm.cdf(x))
    tx += 'norm.ppf({0:6.3f}, loc=0, scale=1)'.format(norm.cdf(x))
    tx += ' = {0:6.3f}'.format(x)
    attr = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.text(-5, -0.32, tx, fontsize=15, bbox=attr)
    
    plt.xlabel('Random variable $X$', fontsize=12)
    plt.ylabel('Probability density function', fontsize=12)
    plt.ylim([-0.04, 0.58])
    plt.xlim([-5.2, 5.2])
    plt.legend(fontsize=12, loc='upper left')
    plt.grid()
    plt.show()

    
def plot_normal():
    x_sld = widgets.FloatSlider(value=-1, min=-3.5, max=3.5, step=0.01,
                                description='Value of $X$: ', disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True, readout_format='.2f')
    ui = widgets.HBox([x_sld])
    out = widgets.interactive_output(update_x, {'x': x_sld})
    display(ui, out)

    
def plot_sdistr(population, n, repeats):
    
    sample_means = []
    for i in range(repeats):
        sample = population.sample(n, replace=True)     # Create a sample with size to be n  
        sample_means.append(sample.mean())              # Sample mean for each experiment
    
    sample_means = pd.Series(sample_means)              # Convert sample_means to a series
    
    msg = 'Population mean $\mu$: ' 
    msg += '{0:0.2f}\n'.format(population.mean())
    msg += 'Population STD. $\sigma$:   ' 
    msg += '{0:0.2f}\n\n'.format(population.std(ddof=0))
    msg += 'Mean of $\overline{x}$: '
    msg += '{0:0.2f}\n'.format(sample_means.mean())
    msg += 'STD. of $\overline{x}$:  '
    msg += '{0:0.2f}'.format(sample_means.std())


    # Display the histogram of the sample means
    plt.hist(sample_means, bins=20)
    attr = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.text(1371.5, 2, msg, fontsize=15, bbox=attr)

    plt.xlabel(r'Sample means $\bar{X}$ (hours)', fontsize=12)
    plt.ylabel(r'Frequency', fontsize=15)
    plt.grid()
    plt.xlim([1335, 1370])
    plt.show()