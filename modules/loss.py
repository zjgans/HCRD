from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def kl_loss(y_s, y_t):
    y_s_norm = nn.functional.normalize(y_s, dim=-1)
    y_t_norm = nn.functional.normalize(y_t, dim=-1)
    y_s_soft = F.log_softmax(y_s_norm, dim=-1)
    y_t_soft = F.softmax(y_t_norm.detach(), dim=-1)
    loss_div = F.kl_div(y_s_soft, y_t_soft, reduction='batchmean')
    return loss_div

def mse_loss(pred,target):
    N = pred.size(0)
    loss = 1-(pred * target).sum()/ N
    return loss

def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

def DIST_LOSS(y_s, y_t,beta=1,gamma=2):
    assert y_s.ndim in (2, 4)
    if y_s.ndim == 4:
        num_classes = y_s.shape[1]
        y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
        y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
    # if logit:
    y_s = y_s.softmax(dim=1)
    y_t = y_t.softmax(dim=1)
    # else:
    # y_s = F.normalize(y_s, dim=1)
    # y_t = F.normalize(y_t, dim=1)
    inter_loss = inter_class_relation(y_s, y_t)
    intra_loss = intra_class_relation(y_s, y_t)
    loss = beta * inter_loss + gamma * intra_loss
    return loss


class DIST(nn.Module):
    def __init__(self, beta=1., gamma=2.):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t,logit=False):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)

        y_s = y_s.softmax(dim=1)
        y_t = y_t.softmax(dim=1)
        # y_s = F.normalize(y_s, dim=1)
        # y_t = F.normalize(y_t, dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss

"""Reference: Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
class SupCluLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupCluLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, memory_features=None,mem_labels=None,mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = 1
        contrast_feature = features
        if self.contrast_mode == 'one':
            assert False
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        mask = mask.repeat(anchor_count, contrast_count)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        if memory_features is not None:
            global_dot_contrast = torch.div(torch.matmul(anchor_feature,memory_features.T),self.temperature)
            anchor_dot_contrast = torch.cat((anchor_dot_contrast,global_dot_contrast),dim=1)
            memory_mask = torch.eq(labels, mem_labels).float().to(device)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        if memory_features is not None:
            mem_logits_mask = torch.ones_like(memory_mask)
            logits_mask = torch.cat((logits_mask,mem_logits_mask),dim=1)
            mask = torch.cat((mask, memory_mask), dim=1)
        #logits_mask = mask[torch.eye(mask.shape[0])==1]=0 # mask-out self-contrast cases
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class SoftSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SoftSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, max_probs, labels=None, mask=None, reduction="mean", select_matrix=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None and select_matrix is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs, max_probs.T)
            # Some may find that the line 59 is different with eq(6)
            # Acutuall the final mask will set weight=0 when i=j, following Eq(8) in paper
            # For more details, please see issue 9
            # https://github.com/TencentYoutuResearch/Classification-SemiCLS/issues/9
            mask = mask.mul(score_mask) * select_matrix

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            #max_probs = max_probs.reshape((batch_size,1))
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs,max_probs.T)
            mask = mask.mul(score_mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        if reduction == "mean":
            loss = loss.mean()

        return loss

class cos_centroidsLoss(nn.Module):
    def __init__(self, n_class, out_dim, n_centroids, device, T, margin):
        super(cos_centroidsLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.centroids = nn.Parameter(torch.Tensor(n_class * n_centroids, out_dim))
        nn.init.xavier_uniform_(self.centroids)
        self.T = T
        self.margin = margin
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.device = device

    def forward(self, z):
        norm_z = F.normalize(z)
        norm_c = F.normalize(self.centroids)
        return norm_z, norm_c

    def pred(self, z):
        norm_z = F.normalize(z)
        norm_c = F.normalize(self.centroids)
        logits = F.linear(norm_z, norm_c)
        return logits

    def loss(self, norm_z, norm_c, y):
        cosine_theta = F.linear(norm_z, norm_c)
        sine_theta = torch.sqrt(1.0 - torch.pow(cosine_theta, 2))
        cons_theta_m = cosine_theta * self.cos_m - sine_theta * self.sin_m
        cons_theta_m = torch.where(cosine_theta > 0, cons_theta_m, cosine_theta)
        y_1Hot = torch.zeros_like(cons_theta_m)
        y_1Hot.scatter_(1, y.view(-1, 1), 1)
        logits = (y_1Hot * cons_theta_m) + ((1.0 - y_1Hot) * cosine_theta)
        return self.criterion(logits / self.T, y)


class NT_xent(nn.Module):
    def __init__(self,t):
        super(NT_xent, self).__init__()
        self.t = t

    def forward(self,x1,x2):
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        batch_size = x1.size(0)
        out = torch.cat([x1, x2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / self.t)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

class JSLoss(nn.Module):
    def __init__(self,t=0.1,t2=0.01):
        super(JSLoss, self).__init__()
        self.t = t
        self.t2 = t2

    def forward(self,x1,x2,xa):
        pred_sim1 = torch.mm(F.normalize(x1, dim=1), F.normalize(xa, dim=1).t())
        inputs1 = F.log_softmax(pred_sim1 / self.t, dim=1)
        pred_sim2 = torch.mm(F.normalize(x2, dim=1), F.normalize(xa, dim=1).t())
        inputs2 = F.log_softmax(pred_sim2 / self.t, dim=1)
        target_js = (F.softmax(pred_sim1 / self.t2, dim=1) + F.softmax(pred_sim2 /self.t2, dim=1)) / 2
        js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
        js_loss2 = F.kl_div(inputs2, target_js, reduction="batchmean")
        return (js_loss1 + js_loss2) / 2.0


class ContrastAllCalculator(nn.Module):
    def __init__(self, T, metric):
        super(ContrastAllCalculator, self).__init__()
        self.T = T
        self.metric = metric
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, q_outs, k=None, queue=None, **kwargs):
        # if encoder_q outputs both encodings and class preds, take encodings
        q = q_outs[0] if isinstance(q_outs, tuple) else q_outs
        queue = queue[-1, :, :] if len(queue.shape) > 2 else queue

        q = nn.functional.normalize(q)
        k = nn.functional.normalize(k)
        queue = nn.functional.normalize(queue, dim=0)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, queue])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.criterion(logits,labels)

        return loss

class ContrastByClassCalculator(nn.Module):
    def __init__(self, T, metric, num_classes):
        super(ContrastByClassCalculator, self).__init__()
        self.T = T
        self.metric = metric
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, q_outs, k=None,  queue=None, cls_labels=None,class_weights=None, **kwargs):
        q = q_outs[0] if isinstance(q_outs, tuple) else q_outs
        queue = queue[:self.num_classes, :, :]
        class_weights_by_label = class_weights[cls_labels]
        q = self.metric.preprocess(q, class_weights=class_weights_by_label, cls_labels=cls_labels, track=True)
        k = self.metric.preprocess(k, class_weights=class_weights_by_label, cls_labels=cls_labels, track=False)
        queue = self.metric.preprocess(queue, class_weights=class_weights.unsqueeze(2), cls_labels=cls_labels,
                                       track=False)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        labels_onehot = torch.zeros((cls_labels.shape[0], self.num_classes)).cuda().scatter(
            1, cls_labels.unsqueeze(1), 1)
        q_onehot = labels_onehot.unsqueeze(-1) * q.unsqueeze(1)
        l_neg = torch.einsum('ncd,cdk->nk', q_onehot, queue)

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.criterion(logits,labels)
        return loss

class Metric(nn.Module):
    def preprocess(self, v):
        raise NotImplementedError


class Norm(Metric):
    def preprocess(self, v, **kwargs):
        v = nn.functional.normalize(v)
        return v


class Angular(Metric):
    def preprocess(self, v, class_weights=None, **kwargs):
        v = nn.functional.normalize(nn.functional.normalize(v) - nn.functional.normalize(class_weights))
        return v
