from shap import (
    Explainer,
    TreeExplainer,
    GPUTreeExplainer,
    LinearExplainer,
    PermutationExplainer,
    PartitionExplainer,
    SamplingExplainer,
    AdditiveExplainer,
    other
)
import functools
# '''
# Explainers class references from the SHAP API explainers to allow run specific explainers as needed.

# '''
# TODO: verify shap API for additional parameters could be required for specific explainers.
class Explainers:
    """ Explainers class references from the SHAP API explainers to allow run specific
    explainers as needed. This is an extension from `shap` package explainers,
    create shapley explanations predefined on this package.

    Source:
    -------
    See `API Examples <https://shap.readthedocs.io/en/latest/api_examples.html#plots>`_
    
    """

    def call_shap_explainer(self, function, *args, **kwargs):
        """ This is a helper method to allow other methods to accept the same parameters
        created on shap package and be able to keep accepting them even if the shap library
        is updated, this way this extention requires less updates and works for a long period.
        """
        def run(*args, **kwargs):
            return function(*args, **kwargs)
        functools.wraps(function)(run)
        return run(*args, **kwargs)

    def agnostic(self, *args, **kwargs):
        ''' Uses Shapley values to explain any machine learning model or python function.'''
        return self.call_shap_explainer(Explainer, *args, **kwargs)

    def tree(self, *args, **kwargs):
        ''' Uses Tree SHAP algorithms to explain the output of ensemble tree models.'''
        return self.call_shap_explainer(TreeExplainer, *args, **kwargs)

    def gpu_tree(self, *args, **kwargs):
        ''' Experimental GPU accelerated version of TreeExplainer.'''
        return self.call_shap_explainer(GPUTreeExplainer, *args, **kwargs)

    def linear(self, *args, **kwargs):
        ''' Computes SHAP values for a linear model, optionally accounting for inter-feature correlations.'''
        return self.call_shap_explainer(LinearExplainer, *args, **kwargs)

    def permutation(self, *args, **kwargs):
        ''' This method approximates the Shapley values by iterating through permutations of the inputs.'''
        return self.call_shap_explainer(PermutationExplainer, *args, **kwargs)

    def partition(self, *args, **kwargs):
        ''' Uses the Partition SHAP method to explain the output of any function.'''
        return self.call_shap_explainer(PartitionExplainer, *args, **kwargs)

    def sampling(self, *args, **kwargs):
        ''' This is an extension of the Shapley sampling values explanation method.'''
        return self.call_shap_explainer(SamplingExplainer, *args, **kwargs)

    def additive(self, *args, **kwargs):
        ''' Computes SHAP values for generalized additive models.'''
        return self.call_shap_explainer(AdditiveExplainer, *args, **kwargs)

    # # 'other' Explainers below

    def _coefficient(self, *args, **kwargs):
        ''' Returns the model coefficents as the feature attributions.'''
        return self.call_shap_explainer(other.Coefficient, *args, **kwargs)

    def _random(self, *args, **kwargs):
        ''' Returns random (normally distributed) feature attributions.'''
        return self.call_shap_explainer(other.Random, *args, **kwargs)

    def _lime_tabular(self, *args, **kwargs):
        ''' Wrap of lime.lime_tabular.LimeTabularExplainer into the common shap interface.'''
        return self.call_shap_explainer(other.LimeTabular, *args, **kwargs)

    def _maple(self, *args, **kwargs):
        ''' Modifying Tree Ensembles to get Local Explanations -- (Wraps MAPLE into the common SHAP interface).'''
        return self.call_shap_explainer(other.Maple, *args, **kwargs)

    def _tree_maple(self, *args, **kwargs):
        ''' Simply tree MAPLE into the common SHAP interface.'''
        return self.call_shap_explainer(other.Maple, *args, **kwargs)

    def _tree_gain(self, *args, **kwargs):
        ''' Returns the global gain/gini feature importances for tree models.'''
        return self.call_shap_explainer(other.Maple, *args, **kwargs)