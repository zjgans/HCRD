import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import SupCluLoss,DIST

class HCRD(nn.Module):
    def __init__(self,args,num_classes, encoder_q,encoder_k,**kwargs):
        super().__init__()
        self.args = args
        self.criterion_clu = SupCluLoss(temperature=args.sup_t)
        self.criterion_agg = SupCluLoss(temperature=args.contrast_t)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.dist_loss = DIST()

        self.base_m = args.mvavg_rate
        self.alpha = args.alpha
        self.beta = args.beta
        self.lamda = args.lamda
        self.temperature = args.mix_t
        self.soft_t = args.distill_t
        self.num_classes = num_classes
        self.use_global_feat = args.use_global_feat
        self.global_w = args.global_w

        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.k2q_mapping = self.para_keys()

        state_dict_q = self.encoder_q.state_dict()
        state_dict_k = self.encoder_k.state_dict()
        for name_k, name_q in self.k2q_mapping.items():
            param_q = state_dict_q[name_q]
            param_k = state_dict_k[name_k]
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def para_keys(self):
        k2q_mapping = {k_name: q_name for q_name, k_name in
                       zip(self.encoder_q.state_dict().keys(), self.encoder_k.state_dict().keys())}
        return k2q_mapping

    @torch.no_grad()
    def _momentum_update_encoder(self):
        """
        Momentum update of the key encoder

        """
        state_dict_q = self.encoder_q.state_dict()
        state_dict_k = self.encoder_k.state_dict()
        for name_k, name_q in self.k2q_mapping.items():
            param_k = state_dict_k[name_k]
            param_q = state_dict_q[name_q]
            param_k.data.copy_(param_k.data * self.base_m + param_q.data * (1. - self.base_m))

    @torch.no_grad()
    def _montage_opera(self, images):
        n, c, h, w = images[0].shape
        permute = torch.randperm(n * 4).cuda()

        un_shuffle_permute = torch.argsort(permute)
        images_gather = torch.cat(images, dim=0)
        images_gather = images_gather[permute, :, :, :]

        col1 = torch.cat([images_gather[0:n], images_gather[n:2 * n]], dim=3)
        col2 = torch.cat([images_gather[2 * n:3 * n], images_gather[3 * n:]], dim=3)
        images_gather = torch.cat([col1, col2], dim=2).cuda()
        return images_gather, permute, n,un_shuffle_permute

    @torch.no_grad()
    def _multi_montage_opera(self, images,m):
        n, c, h, w = images[0].shape
        num_patch = m * m
        permute = torch.randperm(n * num_patch).cuda()
        un_shuffle_permute = torch.argsort(permute)
        images_gather = torch.cat(images, dim=0)
        images_gather = images_gather[permute, :, :, :]

        col_list = []
        for i in range(m):  # column
            col_i = []
            for j in range(m):
                col_i.append(images_gather[(i * m + j) * n:(i * m + j + 1) * n])
            col_cat_i = torch.cat(col_i, dim=3)
            col_list.append(col_cat_i)
        images_gather = torch.cat(col_list, dim=2).cuda()
        return images_gather, permute, n, un_shuffle_permute

    def _multi_decouple_feature(self, feature,m):
        if m == 2:
            num_patch = int(m*m)
            n, c, h, w = feature.shape
            c1, c2 = feature.split([1, 1], dim=2)
            f1, f2 = c1.split([1, 1], dim=3)
            f3, f4 = c2.split([1, 1], dim=3)
            f_gather = torch.cat([f1, f2, f3, f4], dim=0)
            mix_gather = f_gather.view(n * num_patch, -1)
            return mix_gather
        elif m == 3:
            num_patch = int(m*m)
            n, c, h, w = feature.shape
            c1, c2, c3 = feature.split([1, 1, 1], dim=2)
            f_c1 = c1.split([1, 1, 1], dim=3)
            f_c2 = c2.split([1, 1, 1], dim=3)
            f_c3 = c3.split([1, 1, 1], dim=3)
            f_gather = torch.cat([f_c1[0], f_c1[1], f_c1[2], f_c2[0], f_c2[1], f_c2[2], f_c3[0], f_c3[1], f_c3[2]],
                                 dim=0)
            mix_gather = f_gather.view(n * num_patch, -1)
            return mix_gather
        else:
            num_patch = int(m*m)
            n, c, h, w = feature.shape
            c1, c2, c3, c4 = feature.split([1, 1, 1, 1], dim=2)
            f_c1 = c1.split([1, 1, 1, 1], dim=3)
            f_c2 = c2.split([1, 1, 1, 1], dim=3)
            f_c3 = c3.split([1, 1, 1, 1], dim=3)
            f_c4 = c3.split([1, 1, 1, 1], dim=3)
            f_gather = torch.cat(
                [f_c1[0], f_c1[1], f_c1[2], f_c1[3], f_c2[0], f_c2[1], f_c2[2], f_c2[3], f_c3[0], f_c3[1], f_c3[2],
                 f_c3[3],
                 f_c4[0], f_c4[1], f_c4[2], f_c4[3]], dim=0)
            mix_gather = f_gather.view(n * num_patch, -1)
            return mix_gather

    def local_forward(self,jig_images,targets):

        # region_feat, reg_contra_loss = self.RegionAggregation(reg_x, targets, k, weights)
        m = int(math.sqrt(len(jig_images)))
        num_patch = int(m*m)
        mix_images, permute, bs_all, un_shuffle_permute = self._multi_montage_opera(jig_images,m)

        # compute features
        mix_features = self.encoder_q(mix_images, use_clu=True)
        mix_gather = self._multi_decouple_feature(mix_features,m)
        order_targets = targets.repeat(num_patch)
        mix_targets = order_targets[permute]
        # region_targets = targets.repeat(2)

        local_feat = self.encoder_q.local_head(mix_gather)
        local_feat_norm = nn.functional.normalize(local_feat, dim=1)
        local_agg_loss = self.criterion_clu(local_feat_norm, mix_targets)

        with torch.no_grad():
            mix_features_t = self.encoder_k(mix_images,use_clu=True)
            mix_gather_t = self._multi_decouple_feature(mix_features_t, m)
            local_feat_t = self.encoder_k.local_head(mix_gather_t)
            local_feat_t = nn.functional.normalize(local_feat_t,dim=1)
        distill_loss = self.dist_loss(local_feat_norm,local_feat_t)
        return local_agg_loss,distill_loss

    def mse_loss(self,pred,target):
        N = pred.size(0)
        loss = 1-(pred * target).sum()/ N
        return loss

    def kl_loss(self,y_s,y_t):
        y_s_norm = nn.functional.normalize(y_s,dim=-1)
        y_t_norm = nn.functional.normalize(y_t,dim=-1)
        y_s_soft = F.log_softmax(y_s_norm,dim=-1)
        y_t_soft = F.softmax(y_t_norm.detach(),dim=-1)
        loss_div = F.kl_div(y_s_soft,y_t_soft,reduction='batchmean')
        return loss_div

    def forward(self,data_x,x_mix=None,target=None):
        img_k = data_x[1]
        batch_size = img_k.size(0)
        input_data = torch.cat(data_x,dim=0)
        _,logits,_,out_s = self.encoder_q(input_data)

        with torch.no_grad():
            # self.momentum_update_encoder_head()
            _,logits_t,_,out_t = self.encoder_k(input_data)

        # global distillation

        logit_loss_div = self.kl_loss(logits,logits_t)
        feat_distill_dist = self.dist_loss(out_s,out_t)
        loss_local, loss_local_div = self.local_forward(x_mix, target)
        mix_loss, mix_loss_div = self.mixup_forward(data_x, target)
        targets = target.repeat(2)
        single_loss = F.cross_entropy(logits,targets)

        # global aggregation loss
        global_agg_loss = self.criterion_agg(out_s,targets)

        distill_loss = (loss_local_div + mix_loss_div  + logit_loss_div + feat_distill_dist ) * self.alpha
        loss = single_loss + loss_local + (mix_loss + global_agg_loss) * self.global_w + distill_loss
        self._momentum_update_encoder()
        return loss,logits

    def mixup_forward(self, data_x, target, sub_feat=None, sub_num=None):

        img1, img2 = data_x[0], data_x[1]
        mix_x1, y_a1, y_b1, index1, lam1 = self.mix_data_lab(img1, target)
        mix_x2, y_a2, y_b2, index2, lam2 = self.mix_data_lab(img1, target)
        batch_size = mix_x1.size(0)
        inputs = torch.cat((mix_x1, mix_x2), dim=0)
        _,_,_, mix_feature = self.encoder_q(inputs)

        with torch.no_grad():
            _,_,_, mix_feature_t = self.encoder_k(inputs)
        mix_loss_div = self.dist_loss(mix_feature, mix_feature_t)

        target_1 = target.contiguous().view(-1, 1)
        mask1 = torch.eq(target_1, target_1.t()).cuda()

        anchor_dot_contrast = torch.div(
            torch.matmul(mix_feature, mix_feature.t()),
            self.temperature)

        if sub_feat is not None:
            sub_feat = sub_feat.detach()
            global_dot_contrast = torch.div(torch.matmul(mix_feature, sub_feat.t()), self.temperature)
            anchor_dot_contrast = torch.cat((anchor_dot_contrast, global_dot_contrast), dim=1)
            sub_mask1 = mask1.repeat(2, sub_num)
            sub_logits_mask1 = torch.ones_like(sub_mask1)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask1 = mask1.repeat(2, 2)

        logits_mask1 = torch.scatter(
            torch.ones_like(mask1),
            1,
            torch.arange(batch_size * 2).view(-1, 1).cuda(),
            0
        )
        # logits_mask1 = mask1[torch.eye(mask1.shape[0]) == 1] = 0

        mask1 = mask1 * logits_mask1
        if sub_feat is not None:
            mask1 = torch.cat((mask1, sub_mask1), dim=1)
            logits_mask1 = torch.cat((logits_mask1, sub_logits_mask1), dim=1)

        exp_logits_1 = torch.exp(logits) * logits_mask1
        log_prob = logits - torch.log(exp_logits_1.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos1 = (mask1 * log_prob).sum(1) / mask1.sum(1)
        mix_target_a = target[index1]
        target_a = mix_target_a.contiguous().view(-1, 1)
        mask2_a_sup_pre = torch.eq(target_a, target_a.t()).cuda()
        mask2_a_sup = torch.cat((mask2_a_sup_pre, mask2_a_sup_pre), dim=1)

        mix_target_b = target[index2]
        target_b = mix_target_b.contiguous().view(-1, 1)
        mask2_b_sup_pre = torch.eq(target_b, target_b.t()).cuda()
        mask2_b_sup = torch.cat((mask2_b_sup_pre, mask2_b_sup_pre), dim=1)
        mask2 = torch.cat((mask2_a_sup, mask2_b_sup), dim=0)

        logits_mask2 = torch.scatter(
            torch.ones_like(mask2),
            1,
            torch.arange(batch_size * 2).view(-1, 1).cuda(),
            0
        )
        mask2 = mask2 * logits_mask2

        if sub_feat is not None:
            sub_mask2_a = torch.eq(target_a, target_a.t()).cuda()
            sub_mask2_b = torch.eq(target_b, target_b.t()).cuda()
            sub_mask2_a_sup = sub_mask2_a.repeat(1, sub_num)
            sub_mask2_b_sup = sub_mask2_b.repeat(1, sub_num)
            sub_mask2_sup = torch.cat((sub_mask2_a_sup, sub_mask2_b_sup), dim=0)
            sub_logits_mask2 = torch.ones_like(sub_mask2_sup)
            mask2 = torch.cat((mask2, sub_mask2_sup), dim=1)
            logits_mask2 = torch.cat((logits_mask2, sub_logits_mask2), dim=1)

        exp_logits_1 = torch.exp(logits) * logits_mask2
        log_prob = logits - torch.log(exp_logits_1.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos2 = (mask2 * log_prob).sum(1) / mask2.sum(1)

        loss1a = -lam1 * mean_log_prob_pos1[:batch_size]
        loss1b = -lam2 * mean_log_prob_pos1[batch_size:]

        loss1 = torch.cat((loss1a, loss1b))

        loss2a = -(1.0 - lam1) * mean_log_prob_pos2[:batch_size]
        loss2b = -(1.0 - lam2) * mean_log_prob_pos2[batch_size:]
        loss2 = torch.cat((loss2a, loss2b))

        loss = loss1 + loss2
        loss = loss.view(2, batch_size).mean(dim=0)
        return loss.mean(), mix_loss_div

    def mix_data_lab(self, x, y, alpha=1.0, device='cuda'):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)

        lam = max(lam, 1 - lam)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, index, lam


