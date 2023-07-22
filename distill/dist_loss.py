import torch.nn as nn
import torch.nn.functional as F
import torch

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
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def Feat_Loss(x_s,x_t):
    batch_size, dim = x_s.size()
    x_s_abs = x_s.norm(dim=1)
    x_t_abs = x_t.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x_s, x_t) / torch.einsum('i,j->ij', x_s_abs, x_t_abs)

    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    intra_loss = 2 * pos_sim.sum().div(batch_size * batch_size) - sim_matrix.sum().div(batch_size * batch_size)

    c = x_s.T @ x_t

    c.div_(batch_size)
    r = c.max() - c.min()
    r = r.detach()
    c = c.div(r)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()

    inter_loss = 1.0 * on_diag + 1.0 * off_diag
    inter_loss = inter_loss.div(batch_size * batch_size)

    loss = inter_loss + 2* intra_loss

    return loss