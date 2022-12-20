'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"
(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''

import pandas as pd
import numpy as np

def liftChart(predicted, title='Decile Lift Chart', labelBars=True, ax=None, figsize=None):
    """ Create a lift chart using predicted values
    Input:
        predictions: must be sorted by probability
        ax (optional): axis for matplotlib graph
        title (optional): set to None to suppress title
        labelBars (optional): set to False to avoid mean response labels on bar chart
    """
    # group the sorted predictions into 10 roughly equal groups and calculate the mean
    groups = [int(10 * i / len(predicted)) for i in range(len(predicted))]
    meanPercentile = predicted.groupby(groups).mean()
    # divide by the mean prediction to get the mean response
    meanResponse = meanPercentile / predicted.mean()
    meanResponse.index = (meanResponse.index + 1) * 10

    ax = meanResponse.plot.bar(color='C0', ax=ax, figsize=figsize)
    ax.set_ylim(0, 1.12 * meanResponse.max() if labelBars else None)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Lift')
    if title:
        ax.set_title(title)

    if labelBars:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', (p.get_x(), p.get_height() + 0.1))
    return ax


def gainsChart(gains, title='Gains Chart', color='C0', label=None, ax=None, figsize=None):
    """ Create a gains chart using predicted values
    Input:
        gains: must be sorted by probability
        color (optional): color of graph
        ax (optional): axis for matplotlib graph
        figsize (optional): size of matplotlib graph
    """
    nTotal = len(gains)  # number of records
    nActual = gains.sum()  # number of desired records

    # get cumulative sum of gains and convert to percentage
    cumGains = pd.concat([pd.Series([0]), gains.cumsum()])  # Note the additional 0 at the front
    gains_df = pd.DataFrame({'records': list(range(len(gains) + 1)), 'cumGains': cumGains})

    ax = gains_df.plot(x='records', y='cumGains', color=color, label=label, legend=False,
                       ax=ax, figsize=figsize)

    # Add line for random gain
    ax.plot([0, nTotal], [0, nActual], linestyle='--', color='k')
    ax.set_xlabel('# records')
    ax.set_ylabel('# cumulative gains')
    if title:
        ax.set_title(title)
    return ax

#Example
# data = pd.Series([7] * 10 + [2.5] * 10 + [0.5] * 10 + [0.25] * 20 + [0.1] * 50)
# liftChart(data)
# gainsChart(data)
