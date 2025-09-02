
import torch
import torch.nn as nn
from models.DepthAnything.dpt import DepthAnythingV2
from data.transforms import NormalizeDynamic
from utils.image import interpolate_scales,match_scales
from utils.camera import Warping    
import torch.nn.functional as F
import numpy as np
from matplotlib.cm import get_cmap

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
def change_size(image,size):
    return F.interpolate(image, size=size, mode='bilinear')

class RankingLoss(nn.Module):
    """
    Distillation loss for inverse depth maps.
    Parameters
    """
    def __init__(self, cfg):
        super().__init__()
        self.scales = cfg.DATASET.SCALES
        self.temp_context = cfg.DATASET.TEMP_CONTEXT
        self.warping = Warping(self.scales)
        self.min_depth = cfg.MODEL.DEPTH.MIN_DEPTH
        self.max_depth = cfg.MODEL.DEPTH.MAX_DEPTH
        self.encoder = cfg.LOAD.DV2_ENCODER
        self.DepthAnythingV2 = DepthAnythingV2(**model_configs[self.encoder])
        self.DepthAnythingV2.load_state_dict(torch.load(cfg.LOAD.DV2_PATH, map_location='cpu'))
        self.DepthAnythingV2.cuda()
        self.ranking_windows_gt = cfg.LOSS.RANKING_WINDOWS_GT
        self.ranking_windows_pre = cfg.LOSS.RANKING_WINDOWS_PRE
        for param in self.DepthAnythingV2.parameters():
            param.requires_grad = False
        self.normalize = NormalizeDynamic(cfg)
        self.sample_ratio = cfg.LOSS.SAMPLE_RATIO
        self.filter_depth = 1e-8
        self.local_sample_ratio = cfg.LOSS.LOCAL_SAMPLE_RATIO
        
    def get_differ_region(self,mask):
        """
        Generates a mask for unreliable regions based on the input mask.
        """
        B, C, H, W = mask.shape
        unreliable_p = 0.95
        weight_e = torch.quantile(mask.flatten(1), unreliable_p, dim=1)  
        invalidMask_temp = mask < weight_e.view(-1, 1, 1, 1)
        rand = torch.rand_like(mask).cuda()
        mask_rand = rand < (1 - self.sample_ratio)
        return torch.logical_or(mask_rand, invalidMask_temp)

    def get_ordinal_label_local(self,depth, pred, invalid_mask, theta=0.15): # 0.1 0.05 0.15，0.2，0.001，ro 0.15
        """
        Generates ordinal labels for local depth estimation.
        """
        B, C, H, W = depth.shape
        depth[depth==0] = 1e-5
        valid_mask = ~invalid_mask
        rand = torch.rand(B,C, H, W).cuda()
        mask_rand = torch.full((B,C, H, W), True, dtype=torch.bool).cuda()
        
        mask_rand[rand < (1 - self.local_sample_ratio)] = False
        invalid_mask = torch.logical_and(invalid_mask, mask_rand)

        # change invalid_mask
        gt_inval, gt_val, pred_inval, pred_val = None, None, None, None
        for bs in range(B):
            gt_invalid = depth[bs, :, :, :]
            pred_invalid = pred[bs, :, :, :]
            # select the area which belongs to invalid/occlusion
            mask_invalid = invalid_mask[bs, :, :, :]
            # gt_invalid is the normal，gt_valid is the abnormal
            gt_valid = depth[bs, :, :, :]
            pre_valid = pred[bs, :, :, :]
            # select the area which belongs to valid/reliable
            mask_valid = valid_mask[bs, :, :, :]

            if self.ranking_windows_gt:
                gt_invalid = self.compute_average_depth_local(gt_invalid,mask_invalid)
                gt_valid = self.compute_average_depth_local(gt_valid,mask_valid)
            else:
                gt_invalid = gt_invalid[mask_invalid]
                gt_valid = gt_valid[mask_valid]

            if self.ranking_windows_pre:   
                pred_invalid = self.compute_average_depth_local(pred_invalid,mask_invalid) 
                pre_valid = self.compute_average_depth_local(pre_valid,mask_valid)
            else:
                pre_valid = pre_valid[mask_valid]
                pred_invalid = pred_invalid[mask_invalid]
            
            # generate the sample index. index range -> (0, len(gt_valid)). The amount -> gt_invalid.size()
            idx = torch.randint(0, len(gt_invalid), gt_valid.size())

            # Take a number from 0 to the total number of night areas to fill an area the size of the total number of day areas
            # gt_valid = gt_valid[idx]
            # pre_valid = pre_valid[idx]

            gt_invalid = gt_invalid[idx]
            pred_invalid = pred_invalid[idx]
            if bs == 0:
                gt_inval, gt_val, pred_inval, pred_val = gt_invalid, gt_valid, pred_invalid, pre_valid
                continue
            gt_inval = torch.cat((gt_inval, gt_invalid), dim=0)
            gt_val = torch.cat((gt_val, gt_valid), dim=0)
            pred_inval = torch.cat((pred_inval, pred_invalid), dim=0)
            pred_val = torch.cat((pred_val, pre_valid), dim=0)

        za_gt = gt_inval
        zb_gt = gt_val

        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt
        mask1 = flag1 >= 1 + theta
        mask2 = flag2 > 1 + theta
        target = torch.zeros(za_gt.size()).cuda()
        target[mask1] = -1
        target[mask2] = 1

        return pred_inval, pred_val, target

    def compute_average_depth_local(self,depth, mask, patch_size=5):
        """
        Computes the average depth value in a local neighborhood defined by a patch size.
        """  
        C, H, W = depth.shape
        padding = patch_size // 2
        kernel = torch.ones(1, 1, patch_size, patch_size, device=depth.device, dtype=depth.dtype)
        kernel /= (patch_size * patch_size) 
        mean_depth_map = F.conv2d(depth.reshape(C, 1, H, W), kernel,padding=padding).squeeze(1)  
        mask_flat = mask.view(-1).bool()
        mean_depth_flat = mean_depth_map.permute(1, 2, 0).reshape(-1, C)
        valid_depths = mean_depth_flat[mask_flat]
        return valid_depths.view(-1)

    def compute_average_depth_global(self, depth_map, mask, patch_size=5):
        """
        Computes the average depth value in a local neighborhood defined by a patch size.
        Inputs:
            depth_map: [B, C, H, W] 
            mask: [B, 1, H, W] 
            patch_size: int, size of the patch to compute the average depth
        Returns:
            [N] depth values, where N is the number of valid pixels in the mask
        """
        assert patch_size % 2 == 1, "Patch size should be odd"
        half_pad = patch_size // 2
        padded_depth = F.pad(depth_map, (half_pad, half_pad, half_pad, half_pad), mode='constant', value=0) 
        patches = F.unfold(padded_depth, kernel_size=patch_size, stride=1)
        avg_depths = patches.mean(dim=1)
        mask_flat = mask.view(mask.shape[0], -1)
        selected_depths = avg_depths[mask_flat > 0] 
        return selected_depths


    def get_ordinal_label_global(self,depth,pred,theta=0.15,sample_ratio = 0.01):
        """ 
        Generates ordinal labels for global depth estimation.
        """
        B, C, H, W = depth.shape
        mask_A = torch.rand(C, H, W).cuda()
        mask_A[mask_A >= (1 - sample_ratio)] = 1
        mask_A[mask_A < (1 - sample_ratio)] = 0
        idx = torch.randperm(mask_A.nelement())
        mask_B = mask_A.view(-1)[idx].view(mask_A.size())
        mask_A = mask_A.repeat(B, 1, 1).view(depth.shape) == 1
        mask_B = mask_B.repeat(B, 1, 1).view(depth.shape) == 1
        if self.ranking_windows_gt:
            za_gt = self.compute_average_depth_global(depth,mask_A)
            zb_gt = self.compute_average_depth_global(depth,mask_B)
        else:
            za_gt = depth[mask_A]
            zb_gt = depth[mask_B]
        mask_ignoreb = zb_gt > self.filter_depth
        mask_ignorea = za_gt > self.filter_depth
        mask_ignore = mask_ignorea | mask_ignoreb
        za_gt = za_gt[mask_ignore]
        zb_gt = zb_gt[mask_ignore]

        flag1 = za_gt / zb_gt
        flag2 = zb_gt / za_gt
        mask1 = flag1 > 1 + theta
        mask2 = flag2 > 1 + theta
        target = torch.zeros(za_gt.size()).cuda()
        target[mask1] = -1
        target[mask2] = 1
        # mask_ignore = mask_ignore.squeeze(1)
        if self.ranking_windows_pre:
            pred_A = self.compute_average_depth_global(pred,mask_A)
            pred_B = self.compute_average_depth_global(pred,mask_B)
            return pred_A[mask_ignore], pred_B[mask_ignore], target
        else:
            return pred[mask_A][mask_ignore], pred[mask_B][mask_ignore], target

    def cal_ranking_loss(self,A,B,label):
        """
        Calculates the loss of ranking ordinal 
        """
        pred_depth = A - B
        log_loss = torch.sum(
            torch.log(1 + torch.exp(-label[label != 0] * pred_depth[label != 0])))
        pointNum = len(label[label != 0])
        return log_loss, pointNum
    
    def forward(self, inputs, outputs):
        """
        Calculates the loss
        """
        image = inputs[('color', 0)]
        resize_image = change_size(image,(322,574))
        depthV2 = self.DepthAnythingV2((self.normalize(resize_image, inputs['weather'])))
        size = outputs[("disp", 0, 0)].shape[2:]
        depthV2 = change_size(depthV2, size)
        depth_mask = outputs['Mask']
        depth_pred = 1/outputs[("disp", 0, 0)]
        # Local
        unreliableMask = self.get_differ_region(depth_mask)
        A_1,B_1,ordinal_label_1 = self.get_ordinal_label_local(depthV2,depth_pred,unreliableMask) #
        loss_local,local_count = self.cal_ranking_loss(A_1,B_1,ordinal_label_1)
        # Global
        A_2,B_2,ordinal_label_2 = self.get_ordinal_label_global(depthV2,depth_pred)
        loss_global,global_count = self.cal_ranking_loss(A_2,B_2,ordinal_label_2)
        # Sum loss
        loss = (loss_local + loss_global)/(global_count+local_count)
        return loss