from __future__ import annotations

import copy
import functools

from river import base
from river.utils.math import minkowski_distance


class PerOutputClassifier(base.MultiLabelClassifier):
    """Multi-target classification.
    
    This class implements a classification strategy where a classifier is 
    fitting multiple targets separately.
    

    Parameters
    ----------
    model
        The classifier used for learning.

    Examples
    --------


    """
    
    def __init__(self, model: base.Classifier):
        super().__init__()
        self.model = model

    @classmethod
    def _unit_test_params(cls):
        from river import linear_model, neighbors

        yield {"model": linear_model.LogisticRegression()}  # binary classifier
        yield {
            "model": neighbors.KNNClassifier(
                n_neighbors=1,
                engine=neighbors.LazySearch(
                    window_size=10, dist_func=functools.partial(minkowski_distance, p=2)
                ),
            )
        }  # multi-class classifier

    @property
    def _multiclass(self):
        return self.model._multiclass
    
    def learn_one(self, x, y):
        pass
    
    #def predict_one(self, x, y):
    #    pass

    def predict_proba_one(self, x, **kwargs):
        x = copy.copy(x)
        y_pred = {} # in case the model hasn't seen data yet 

        return y_pred
