from types import MethodType
import torch
import torch.nn as nn
import models
from .classification import Learner_Classification as Learner_Template
from modules.pairwise import PairEnum


class Learner_DensePairSimilarity(Learner_Template):

    @staticmethod
    def create_model(model_type,model_name,out_dim):
        # Create Similarity Prediction Network (SPN) by model surgery
        model = models.__dict__[model_type].__dict__[model_name](out_dim=out_dim)
        n_feat = model.last.in_features

        # Replace task-dependent module
        model.last = nn.Sequential(
            nn.Linear(n_feat*2, n_feat*4),
            nn.BatchNorm1d(n_feat*4),
            nn.ReLU(inplace=True),
            nn.Linear(n_feat*4, 2)
        )

        # Replace task-dependent function
        def new_logits(self, x):
            feat1, feat2 = PairEnum(x)
            featcat = torch.cat([feat1, feat2], 1)
            out = self.last(featcat)
            return out
        model.logits = MethodType(new_logits, model)

        return model
