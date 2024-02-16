from __future__ import annotations

import functools

from river import base
from river.utils.math import minkowski_distance


class PerOutputClassifier(base.MultiLabelClassifier):
    """Placeholder class docstring

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
        from river import neighbors

        yield {
            "model": neighbors.KNNClassifier(
                n_neighbors=1,
                engine=neighbors.LazySearch(
                    window_size=10, dist_func=functools.partial(minkowski_distance, p=2)
                ),
            )
        }  # multi-class classifier

    def learn_one(self, x, y):
        pass
    
    def predict_one(self, x, y):
        pass

    def predict_proba_one(self, x, **kwargs):
        pass
