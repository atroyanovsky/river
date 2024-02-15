from river import base


class PerOutputClassifier(base.Wrapper):
    def __init__(self, model: base.Classifier):
        super().__init__()
        pass

    def learn_one(self, x, y):
        pass
    
    def predict_one(self, x, y):
        pass

    def predict_proba_one(self, x, **kwargs):
        pass

    @classmethod
    def _unit_test_params(cls):
        from river import tree

        yield {"model": tree.HoeffdingTreeClassifier()}
