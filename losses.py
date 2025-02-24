import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def simple_contrstive_loss(vi_batch, vi_t_batch, mn_arr, temp_parameter=0.1):
    """
    Returns the probability that feature representation for image I and I_t belong to same distribution.
    :param vi_batch: Feature representation for batch of images I
    :param vi_t_batch: Feature representation for batch containing transformed versions of I.
    :param mn_arr: Memory bank of feature representations for negative images for current batch
    :param temp_parameter: The temperature parameter
    """

    # Define constant eps to ensure training is not impacted if norm of any image rep is zero
    eps = 1e-6

    # L2 normalize vi, vi_t and memory bank representations
    vi_norm_arr = torch.norm(vi_batch, dim=1, keepdim=True)
    vi_t_norm_arr = torch.norm(vi_t_batch, dim=1, keepdim=True)
    mn_norm_arr = torch.norm(mn_arr, dim=1, keepdim=True)

    vi_batch = vi_batch / (vi_norm_arr + eps)
    vi_t_batch = vi_t_batch/ (vi_t_norm_arr + eps)
    mn_arr = mn_arr / (mn_norm_arr + eps)

    # Find cosine similarities
    sim_vi_vi_t_arr = (vi_batch @ vi_t_batch.t()).diagonal()
    sim_vi_t_mn_mat = (vi_t_batch @ mn_arr.t())

    # Fine exponentiation of similarity arrays
    exp_sim_vi_vi_t_arr = torch.exp(sim_vi_vi_t_arr / temp_parameter)
    exp_sim_vi_t_mn_mat = torch.exp(sim_vi_t_mn_mat / temp_parameter)

    # Sum exponential similarities of I_t with different images from memory bank of negatives
    sum_exp_sim_vi_t_mn_arr = torch.sum(exp_sim_vi_t_mn_mat, 1)

    # Find batch probabilities arr
    batch_prob_arr = exp_sim_vi_vi_t_arr / (exp_sim_vi_vi_t_arr + sum_exp_sim_vi_t_mn_arr + eps)

    neg_log_img_pair_probs = -1 * torch.log(batch_prob_arr)
    loss_i_i_t = torch.sum(neg_log_img_pair_probs) / neg_log_img_pair_probs.size()[0]
    return loss_i_i_t
    
# used in our work
def mixup_supcontrastive_loss(feature,target,temperature,batch_size,index1,index2,lam1,lam2,sub_feat=None,sub_num=None):

        target_1 = target.contiguous().view(-1,1)
        mask1 = torch.eq(target_1,target_1.t()).cuda()

        anchor_dot_contrast = torch.div(
            torch.matmul(feature, feature.t()),
            temperature)

        if sub_feat is not None:
            sub_feat = sub_feat.detach()
            global_dot_contrast = torch.div(torch.matmul(feature, sub_feat.t()), temperature)
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
        mask2_a_sup = torch.eq(target_a, target_a.t()).cuda()
        mask2_a_sup = torch.cat((mask2_a_sup, mask2_a_sup), dim=1)

        mix_target_b = target[index2]
        target_b = mix_target_b.contiguous().view(-1, 1)
        mask2_b_sup = torch.eq(target_b, target_b.t()).cuda()
        mask2_b_sup = torch.cat((mask2_b_sup, mask2_b_sup), dim=1)

        mask2 = torch.cat((mask2_a_sup, mask2_b_sup), dim=0)

        logits_mask2 = torch.scatter(
            torch.ones_like(mask2),
            1,
            torch.arange(batch_size * 2).view(-1, 1).cuda(),
            0
        )
        mask2 = mask2 * logits_mask2

        if sub_feat is not None:
            sub_mask2_a = torch.eq(target_a, target.t()).cuda()
            sub_mask2_b = torch.eq(target_b, target.t()).cuda()
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
        return loss.mean()

def Align_Loss(pred,target):
    pred_norm = F.normalize(pred,p=2,dim=-1)
    target_norm = F.normalize(target,p=2,dim=-1)
    loss = -torch.sum(F.softmax(pred_norm, dim=-1) * F.log_softmax(target_norm, dim=-1), dim=-1).mean()
    return loss

class ContrastByClassCalculator(nn.Module):
    def __init__(self,T,num_classes,trans,num_samples,momentum,dim=128,K=4096):
        super(ContrastByClassCalculator, self).__init__()
        self.T =T
        self.K =K
        self.trans = trans
        self.num_classes=num_classes
        self.m = momentum
        self.register_buffer("queue",torch.randn(num_classes,dim,K))
        self.queue = F.normalize(self.queue,dim=1)
        self.register_buffer("queue_ptr",torch.zeros(num_classes,dtype=torch.long))

    def forward(self,q,k,weight,cls_labels,criterion,use_angle=False):
        queue = self.queue[:self.num_classes,:,:].detach().clone()
        # self.dequeue_and_enqueue(k,cls_labels)

        fin_weight =weight.unsqueeze(2)
        if use_angle:
            class_weight_by_label = weight[cls_labels]
            q = self.angle_preprocess(q,class_weight=class_weight_by_label)
            k = self.angle_preprocess(k,class_weight=class_weight_by_label)
            queue = self.angle_preprocess(queue,class_weight=fin_weight)
        l_pos = torch.einsum('nc,nc->n',[q,k]).unsqueeze(-1)

        labels_onehot= torch.zeros((cls_labels.size(0),self.num_classes)).cuda().scatter(
            1,cls_labels.unsqueeze(1),1)
        q_onehot = labels_onehot.unsqueeze(-1) * q.unsqueeze(1)
        l_neg = torch.einsum('ncd,cdk->nk',q_onehot,queue)

        logits = torch.cat([l_pos,l_neg],dim=1)
        logits /=self.T
        labels = torch.zeros(logits.size(0),dtype=torch.long).cuda()

        loss = criterion(logits,labels)

        return loss

    @ torch.no_grad()
    def dequeue_and_enqueue(self,keys,cls_label):
        for cls_id in torch.unique(cls_label):
            cls_keys = keys[cls_label==cls_id]
            num_keys = cls_keys.size(1)
            batch_size = cls_keys.size(0)
            ptr = int(self.queue_ptr[cls_id])

            if ptr + batch_size >= self.K:
                self.queue[cls_id][:,ptr:]= cls_keys.T[:,:self.K - ptr]
                self.queue[cls_id][:,:(ptr+batch_size) % self.K] = cls_keys.T[:,self.K-ptr:]
            else:
                self.queue[cls_id][:,ptr:ptr + batch_size] =cls_keys.T
            ptr = (ptr + batch_size)%self.K
            self.queue_ptr[cls_id]=ptr

    def angle_preprocess(self,v,class_weight):
        v = F.normalize(F.normalize(v)-F.normalize(class_weight))
        return v

