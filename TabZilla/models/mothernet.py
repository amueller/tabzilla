from models.basemodel import BaseModel
from mothernet.prediction import MotherNetClassifier, EnsembleMeta
import torch


class MotherNet(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)
        if args.use_gpu:
            device = f"cuda:{args.gpu_ids[0]}"
            n_jobs = 1
        else:
            device = "cpu"
            torch.set_num_threads(4)
            n_jobs = 3
        if args.objective == "regression":
            raise NotImplementedError("Does not support")
        elif args.objective == "classification":
            self.model = EnsembleMeta(MotherNetClassifier(device=device), n_estimators=3, n_jobs=n_jobs)
        elif args.objective == "binary":
            self.model = EnsembleMeta(MotherNetClassifier(device=device), n_estimators=3, n_jobs=n_jobs)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return [], []

    def predict_helper(self, X):
        return self.model.predict(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = dict()
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        params = dict()
        return params

    @classmethod
    def default_parameters(cls):
        params = dict()
        return params

    def get_classes(self):
        return self.model.classes_