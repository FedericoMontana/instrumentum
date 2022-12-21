'''
        Example:
        explain = ShapExtension(
                model = model                       # We pass the resulting model from our prediction to be analyzed by the explainability classes
                x_var = X                           # We pass the X variables from the dataset used to create the model
                explainer = 'agnostic',             # (optional -- default: 'agnostic'): correspond to the type of explainers from the `shap` library we are going to use -- OPTIONS: agnostic, tree, gpu_tree, linear, permutation, partition, sampling, additive, coefficient, random, lime_tabular, maple, tree_maple, tree_gain
                vis = 'bar'                         # (optional -- default: 'bar'): given vis variable will plot the visualization corresponding with that variable, otherwise 'bar' is the default visualization
        )
        explain.features_importances()              # runs the first explanation example based on the previous input variables

'''

import pickle
from .explainers import Explainers
from .shap_plots import Plots

class ShapExtension(Plots, Explainers):
    """ ShapExtension() is based on SHAP package, given a model and its X, it creates the shap values and returns the explanability of given model.
    
    """
    def __init__(
        self, *,
        model = None,
        x_var = None,
        explainer = 'agnostic',
        vis = 'bar',
        title = None,
        ylabel = None
    ):
        self.model = model
        self.independent_var = x_var
        self.explainer = explainer 
        self.vis = vis
        self.shap_vals = None
        self.title = title
        self.ylabel = ylabel
        self.expected_val = None
        super().__init__()

    def shap_values(self):
        ''' returns the shap values. '''
        return self.shap_vals

    def expected_value(self):
        ''' returns the expected value. '''
        return self.expected_val

    def save_values(self, filename):
        ''' save shap values into a binary file. '''
        with open(filename + '.bin', 'wb') as file:
            pickle.dump(self.shap_vals, file)

    def load_values(self, filename):
        ''' load shap values from a binary file, requires extension to be added to the file name otherwise will return a file not found error. '''
        with open(filename, 'rb') as file:
            self.shap_vals = pickle.load(file)
        return self.shap_vals                

    def features_importance(self):
        ''' given the input this method create the explainability. '''
        exp, self.shap_vals = self.set_explainer()
        legend = self.plot_values(vis = self.vis, title = self.title, ylabel = self.ylabel, shap_values = self.shap_vals)

    def set_explainer(self):
        ''' based on the initialization of the ShapExtension class, set the explainer to be used. '''
        selected_explainer = shap_values = None
        
        if self.explainer == 'agnostic' or self.explainer == 'additive':
            print(f'You selected an {self.explainer} explainer.')
        else:
            print(f'You selected a {self.explainer} explainer.')

        exp = Explainers()
        selected_explainer = getattr(exp, self.explainer)(self.model, self.independent_var)
        shap_values = selected_explainer(self.independent_var)
        self.expected_val = selected_explainer.expected_value

        return selected_explainer, shap_values