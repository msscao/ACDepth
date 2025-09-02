import torch.nn as nn
from collections import OrderedDict
import torch

class ContrastLoss(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.loss = nn.L1Loss()
        self.c_w1 = cfg.LOSS.CONTRAST_LOSS_W1
        self.c_w2 = cfg.LOSS.CONTRAST_LOSS_W2
    def forward(self, inputs,outputs,teacher_feature):
        """
        Computes the contrastive loss between augmented features and teacher features.
        """
        batch_size = len(inputs['weather_depth'])
        mask = torch.zeros(batch_size, dtype=torch.bool)
        indices = [index for index, value in enumerate(inputs['weather_depth']) if value in ['night-translated','day-rain-translated']]
        mask[indices] = True
        mask_M = ~mask
        loss_SS = 0
        loss_TS1 = 0
        loss_TS2 = 0
        for i in range(5):
            if len(indices) > 0:
                loss_SS += self.loss(outputs['feature_aug'][i][mask],outputs['feature'][i][mask].detach())
                loss_TS2 += self.loss(outputs['feature_aug'][i][mask], teacher_feature[i][mask].detach())
            if len(indices) < batch_size:
                loss_TS1 += self.loss(outputs['feature_aug'][i][mask_M], teacher_feature[i][mask_M].detach())
        return self.c_w1 * (loss_SS/5) + self.c_w2 * (loss_TS1/5 + loss_TS2/5)
