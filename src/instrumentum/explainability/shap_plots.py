import shap
from shap.plots import(    
    bar as Bar,
    waterfall as Waterfall,
    scatter as Scatter,
    heatmap as Heatmap,
    beeswarm as Beeswarm,
    force as Force,
    text as Text,
    image as Image,
    partial_dependence as PartialDependence,
    violin as Violin
)

import matplotlib.pyplot as plt

BAR_LEGEND = '''Bar graph allows two options: when is on GLOBAL (shows all shap_values) it give us the mean of the absolute value of the SHAP value allowing us to see
which variables have contributed the most. This prediction include positive and negative controntribution, where negative push the prediction to
0 (0%) and positive to 1 (100%). In other hands when it on LOCAL (shows a specific shap_value, example shap_value[0]) it is specified for a single client, or person who that row
is related to, in this case the values are shown on red moving to the right, indicating pushing the prediction to 1, and are on blue bars moving to the left, which means is pushing
the prediction to 0.'''
WATERFALL_LEGEND = '''Waterfall it is defined as a LOCAL graph, which means it only predict values decicated to a client, person or whatever a row identify. When graph are red moving to the
right indicate is moving the prediction to 1 (100%), when they are blue means it is moving the prediction to 0 (0%), this  graph also in the left of the variables allow us to see the current value
of all the variables connected with the current row, which give a better idea of what is happening.'''
SCATTER_LEGEND = 'With the scatter we can create a dependence scatter plot to allows see what effect a single variable have ascross the whole dataset.'
HEATMAP_LEGEND = '''Heatmap allow us to see the instances of the shap_values in the x-axis and we can see the the variables related with the model in the y-axis.
The output of the model can be seen above in the f(x) function.'''
BEESWARM_LEGEND = '''Beeswarm plot are points placed in a way to avoid overlaps of the data and understand better the distribution.
For Shapley values it summarizes the effects of all the features.'''
FORCE_LEGEND = '''Force plots simulate the interactions between features and arrange them in a way they can represent their influences in the prediction.
For Shapley values we can see how much each variable contributed to the prediction, where red represent moving the contribution `higher`
toward 1 and blue moving `lower` toward 0 (considering a probability from 0 to 1). To get a prediction of shap_values[0]
-- representing the first row -- will have a force corresponding a prediction for that row. But passing the whole shap_values will report
in the contributions into a horizontally positioned stack of force plots per each row.\n'''
TEXT_LEGEND = 'SHAP package orignally have support for natural language, so we are able to explain analyzed text using the shap_values.'
IMAGE_LEGEND = 'Using Deep SHAP to explain models related with images can help with the creation of shap_values which can be expressed using this type of graph.'
PARTIAL_DEPENDENCE_LEGEND = 'Partial dependence plot allow us to understand how the change of the variable impacts the final result.'
SUMMARY_LEGEND = """Summary is a special plot, which allow is to have a dot plot (by default) or a violin plot (by adding plot_type = 'violin' as part of the parameters).
Both plots help summarizing the effects of all the features. The color represents the average feature value at that position."""

# '''

# '''
class Plots:
    """ This is an extension from `shap` package plots, create plots predefined on this package.

    Source:
    -------
    See `API Examples <https://shap.readthedocs.io/en/latest/api_examples.html#plots>`_
    
    """

    """ pseudocode
        - call select_plot
        - use name from select_plot to call a method of that name using getattr()
        - the method being called is going to use a call_shap_visualialization
            who is a wrapper that is going to call the that visualization method
            correspoding with that graph.
    """

    def plot_values(self, *args, vis: str, title = None, ylabel = None, ylabel_size = 16, title_size = 16, **kwargs):
        """ Depending on user's input, pick a plot from the shap library

        Parameters
        ----------
        vis : str
            Specify the visualization sent to plot.

        shap_values : Shapley Values
            contains the Shapley Values produced by the explainer.

        color_by: None
            Define parameters to use to color the graphs (when available).

        Returns
        --------
            The specified plot depending on given arguments.
        """

        if vis == 'force':
            print("Force plot it is not yet supported with this method, please call force for example with `plot.force(shap_values)` for a global explanation, or `plot.force(shap_values[0])`, where 0 represent the client or row to analyze, and `plot` is the initialization of the class Plot().")
            return

        if title:
            plt.title(title, fontsize=title_size)
        if ylabel:
            plt.ylabel(ylabel, fontsize=ylabel_size)

        getattr(self, vis)(*args, **kwargs)

    def bar(self, *args, **kwargs):
        ''' Create a bar plot of a set of SHAP values.'''
        Bar(*args, **kwargs)
        self.__print_legend(BAR_LEGEND)
    
    def waterfall(self, *args, **kwargs):
        ''' Plots an explantion of a single prediction as a waterfall plot.'''
        Waterfall(*args, **kwargs)
        self.__print_legend(WATERFALL_LEGEND)

    def scatter(self, *args, **kwargs):
        ''' Create a SHAP dependence scatter plot, colored by an interaction feature.'''
        Scatter(*args, **kwargs)
        self.__print_legend(SCATTER_LEGEND)

    def heatmap(self, *args, **kwargs):
        ''' Create a heatmap plot of a set of SHAP values.'''
        Heatmap(*args, **kwargs)
        self.__print_legend(HEATMAP_LEGEND)

    def beeswarm(self, *args, **kwargs):
        ''' Plot the summary of the effects of all the features'''
        Beeswarm(*args, **kwargs)
        self.__print_legend(BEESWARM_LEGEND)

    def force(self, *args, **kwargs):
        ''' Visualize the given SHAP values with an additive force layout.'''
        shap.initjs()
        self.__print_legend(FORCE_LEGEND)
        return Force(*args, **kwargs)

    def text(self, *args, **kwargs):
        ''' Plots an explanation of a string of text using coloring and interactive labels.'''
        Text(*args, **kwargs)
        self.__print_legend(TEXT_LEGEND)

    def image(self, *args, **kwargs):
        ''' Plots SHAP values for image inputs.'''
        Image(*args, **kwargs)
        self.__print_legend(IMAGE_LEGEND)

    def partial_dependence(self, *args, **kwargs):
        ''' Plot a basic partial dependence plot function.'''
        PartialDependence(*args, **kwargs)
        self.__print_legend(PARTIAL_DEPENDENCE_LEGEND)

    def summary(self, *args, **kwargs):
        '''  Display the distribution of importance for each variable.'''
        shap.summary_plot(*args, **kwargs)
        self.__print_legend(SUMMARY_LEGEND)
    
    def __print_legend(self, LEGEND):
        print(LEGEND.replace("\n",' '))