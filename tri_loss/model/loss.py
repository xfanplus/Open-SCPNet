from __future__ import print_function
import torch
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def global_loss(tri_loss, global_feat, labels, normalize_feature=True):
    """
    Args:
      tri_loss: a `TripletLoss` object
      global_feat: pytorch Variable, shape [N, C]
      labels: pytorch LongTensor, with shape [N]
      normalize_feature: whether to normalize feature to unit length along the
        Channel dimension
    Returns:
      loss: pytorch Variable, with shape [1]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
      ==================
      For Debugging, etc
      ==================
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
    """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels, return_inds=True)
    loss = tri_loss(dist_ap, dist_an)
    return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat


def triplet_hard_loss(feat, P, K):
    sh0 = feat.size(0)  # number of samples
    sh1 = feat.size(1)  # feature dimension
    assert P * K == sh0, "Error: Dimension does not match! P={},K={},sh0={}".format(P, K, sh0)
    feat1 = feat.view(sh0, 1, sh1).repeat(1, sh0, 1)
    feat2 = feat.view(1, sh0, sh1).repeat(sh0, 1, 1)
    delta = feat1 - feat2
    l2 = torch.sqrt((delta * delta).sum(dim=2) + 1e-8)
    positive = [l2[i * K:i * K + K, i * K:i * K + K] for i in range(P)]
    positive = torch.cat(positive, dim=0)
    positive, _ = positive.max(dim=1)
    negative = []
    for i in range(P):
        tmp = [l2[i * K:i * K + K, j * K:j * K + K] for j in range(P) if i != j]
        tmp = torch.cat(tmp, dim=1)
        negative.append(tmp)
    negative = torch.cat(negative, dim=0)
    negative, _ = negative.min(dim=1)
    _loss = F.relu(positive - negative + 0.3).mean()
    return _loss, positive.mean().data[0], negative.mean().data[0]


def scp_loss(feats, targets, criterion_cls, criterion_feature, P, K):
    scores, features, std_features = feats
    # cls
    loss_cls1 = criterion_cls(scores[0], targets)
    loss_cls2 = criterion_cls(scores[1], targets)
    loss_cls3 = criterion_cls(scores[2], targets)
    loss_cls4 = criterion_cls(scores[3], targets)
    loss_cls = loss_cls1 + loss_cls2 + loss_cls3 + loss_cls4
    # tri
    loss_tri1, pos1, neg1 = triplet_hard_loss(features[0], P, K)
    loss_tri2, pos2, neg2 = triplet_hard_loss(features[1], P, K)
    loss_tri3, pos3, neg3 = triplet_hard_loss(features[2], P, K)
    loss_tri4, pos4, neg4 = triplet_hard_loss(features[3], P, K)
    loss_tri = loss_tri1 + loss_tri2 + loss_tri3 + loss_tri4
    pos = pos1 + pos2 + pos3 + pos4
    neg = neg1 + neg2 + neg3 + neg4
    # feat
    loss_feat1 = criterion_feature(features[0], std_features[0])
    loss_feat2 = criterion_feature(features[1], std_features[1])
    loss_feat3 = criterion_feature(features[2], std_features[2])
    loss_feat4 = criterion_feature(features[3], std_features[3])
    loss_feat = loss_feat1 + loss_feat2 + loss_feat3 + loss_feat4
    loss = loss_cls + loss_tri + loss_feat * 10

    # acc
    _, idx = sum(scores).max(dim=1)
    acc = (idx == targets).float().mean().data[0]
    return loss, acc
