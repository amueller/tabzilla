from models.basemodel import BaseModel
from mothernet.prediction import MotherNetClassifier, EnsembleMeta
from mothernet.utils import get_mn_model

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
        elif args.objective in ["binary", "classification"]:
            print(f"device: {device}")
            model_string = "mn_Dclass_average_03_25_2024_17_14_32_epoch_2910.cpkt"
            model_path = get_mn_model(model_string)
            self.model = EnsembleMeta(MotherNetClassifier(device=device, inference_device=device, path=model_path), n_estimators=8, n_jobs=n_jobs, onehot=True)

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