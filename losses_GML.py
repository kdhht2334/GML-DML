import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import linalg
from sklearn.manifold import MDS, Isomap, SpectralEmbedding, TSNE, LocallyLinearEmbedding
import math
import random

from pytorch_metric_learning import miners, losses
from similarity import pairwise_distance
from fabulous.color import fg256


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)


def power_iteration(A):
    A = A.detach().cpu().numpy()
    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.01: break

        v = v_new
        ev = ev_new
    return ev_new, v_new


def Linear_discriminant_analysis(batch, labels, dimension, regularizer=None, method="naive"):
    unique_labels = np.unique(labels)
    mean_vectors = []
    for cl in range(np.shape(unique_labels)[0]):  # [25,]; uint8
        ids = torch.tensor(np.array([labels==unique_labels[cl]], dtype=np.uint8), dtype=torch.bool).squeeze(0).cuda()
        aaa = batch[ids,:]  # [4, 128]
        mean_vectors.append(torch.mean(aaa, dim=0))
    mean_vectors = torch.stack(mean_vectors)  # [25, 128]

    S_W = torch.zeros((dimension,dimension))
    for cl in range(np.shape(unique_labels)[0]):  # [1 ~ 25]; uint8
        ids = torch.tensor(np.array([labels==unique_labels[cl]], dtype=np.uint8), dtype=torch.bool).squeeze(0).cuda()
        class_sc_mat = torch.zeros((dimension,dimension))
        for row in batch[ids,:]:
            row, mv = row.reshape(dimension,1).detach().cpu(), mean_vectors[cl,:].reshape(dimension,1).detach().cpu()
            class_sc_mat += (row-mv) @ torch.t(row-mv)
        S_W += class_sc_mat

    overall_mean = torch.mean(batch, dim=0).reshape(dimension,1).detach().cpu()
    S_B = torch.zeros((dimension,dimension))
    for i, mean_vec in enumerate(mean_vectors):
        mean_vec = mean_vec.reshape(dimension,1).detach().cpu()
        S_B += 4.0 * (mean_vec - overall_mean) @ torch.t(mean_vec - overall_mean)

    if regularizer:
        fisher_criterion = torch.inverse(S_W + torch.tensor(1e-6*torch.eye(S_W.size()[0]) * regularizer)) @ S_B
    else:
        fisher_criterion = torch.inverse(S_W + 1e-3*torch.eye(S_W.size()[0])) @ S_B

    if method == "naive":
        eig = torch.eig(fisher_criterion, eigenvectors=True)
        eigval = eig[0]; eigvec = eig[1]
        eigval, ind = torch.sort(eigval[:,0], 0, descending=True)
        eigvec = eigvec[:,ind[0]].unsqueeze(1)
        
    elif method == "PI":
        eigval, eigvec = power_iteration(fisher_criterion)
        eigval = torch.tensor([eigval]).type(torch.FloatTensor)
        eigvec = torch.tensor([eigvec]).type(torch.FloatTensor).t()
        print(fg256("yellow", 'PI-based eigval is {}'.format(eigval)))
        
    elif method == "LOBPCG":
        eigval, eigvec = torch.lobpcg(fisher_criterion)
        eigval, ind = torch.sort(eigval, 0, descending=True)
        eigvec = eigvec[:,ind[0]].unsqueeze(1)
#    eig = torch.eig(fisher_criterion, eigenvectors=True)
#    eigval = eig[0]; eigvec = eig[1]
#    eigval, ind = torch.sort(eigval[:,0], 0, descending=True)
    return eigval, eigvec  #, ind


def lifting_map_on_Sphere(q, x, shrink=True):
    x = x.unsqueeze(1).repeat(1,q.size(0))
    m = x @ q

    qq = np.eye(q.size(1))*2.0
    b = np.eye(q.size(1))
    r = np.eye(q.size(1))

    s = torch.from_numpy(linalg.solve_continuous_are(-m.detach().cpu().numpy(), b, qq, r)).type(torch.cuda.FloatTensor)  # [3,3]

    lift_map = q @ s - x.t()
    return lift_map


def lifting_map_on_Stiefel(X_proj, x, shrink=True):

    rlist = []
    x = x.unsqueeze(0).repeat(X_proj.size(2),1)
    for i in range(X_proj.size(0)):
        q = X_proj[i]
        m = x @ q
    
        qq = np.eye(q.size(1))*2.0
        b = np.eye(q.size(1))*1.0
        r = np.eye(q.size(1))*1.0
    
        s = torch.from_numpy(linalg.solve_continuous_are(-m.detach().cpu().numpy(), b, qq, r)).type(torch.cuda.FloatTensor)  # [2,2]
    
        lift_map = q @ s - x.t()
        rlist.append(lift_map)
    lift_map = torch.stack(rlist)
    return lift_map


def Retraction(x, the, shrink=True):
#    rr = (x + the) @ torch.pinverse((torch.eye(x.size(0)).cuda() + the.t()@the).pow(0.5))
    # In the case of the orthogonal group
    q, r = torch.qr(x + the)
    if shrink:
        rr = torch.mean(q, dim=1, keepdim=True)
    else:
        rr = q
    return rr


def kronecker(A, B):
#        return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))
    return torch.einsum("ab,cd->acbd", A, B).reshape(A.size(0)*B.size(0),  A.size(1)*B.size(1))


class PPGML_Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda()/8)
        nn.init.orthogonal_(self.proxies)  # Orthogonal parameter initialization
        self.projector = torch.nn.Sequential(torch.nn.Linear(1,2)).type(torch.cuda.FloatTensor)

        self.keys = range(3)
        values = [i+2 for i in range(1,len(self.keys)+1)]
        self.LDS_dim = dict(zip(self.keys, values))
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

        self.p = {}
        for i in range(len(self.keys)):
            self.p[i] = torch.cat([torch.zeros(self.LDS_dim[i]-1), torch.ones(1)]).cuda()

        self.offset = 0.50
        
    def forward(self, X, T):
        P = self.proxies

        ### GML ###
        cos = F.linear(l2_norm(X), l2_norm(P), bias=torch.ones([1]).cuda()*1e-3)  # Calcluate cosine similarity; [180, 100] (x_dim, p_dim)
#        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        X_proj = {}
        for i in range(len(self.keys)):
            X_proj[i] = torch.from_numpy(Isomap(n_components=self.LDS_dim[i]).fit_transform(X.detach().cpu().numpy())).type(torch.cuda.FloatTensor)  # [180, 3]

        pip_loss = {}
        for i in range(len(self.keys)):
            pip_loss[i] = 0.001 * (X.size(1) - X_proj[i].size(1)) + 2.0 * torch.norm(X_proj[i].t() @ X[:,X.size(1)-(self.LDS_dim[i])], p=2).pow(2)
        min_ind = min(pip_loss, key=pip_loss.get)

        rlist = []
        _X_proj = X_proj[min_ind].unsqueeze(2)
        for i in range(_X_proj.size(0)):
            for j in range(_X_proj.size(1)):
                rlist.append(self.projector(_X_proj[i,j]))
        rtensor = torch.stack(rlist)
        rtensor = torch.reshape(rtensor, (_X_proj.size(0), _X_proj.size(1), dict(self.projector.named_children())['0'].out_features))
        del rlist
        X_proj = F.normalize(rtensor)
        X_log_map = lifting_map_on_Stiefel(X_proj, self.p[min_ind])
        k = self.LDS_dim[min_ind] * X_log_map.size(2)
        
        
        klist = []
        for i in range(X_log_map.size(0)): klist.append(kronecker(X_log_map[i], X_log_map[i].t())[0,:])  # Product Stiefel manifold
        X_log_map = torch.stack(klist)
#        X_log_map = torch.reshape(X_log_map, (X_log_map.size(0), X_log_map.size(1)*X_log_map.size(2)))
#        X_log_map = X_log_map @ X_log_map.t()  # using kernel
        sss_eigval, sss_eigvec = Linear_discriminant_analysis(X_log_map, T.detach().cpu().numpy(),
                                                              X_log_map.size(1), regularizer=None,
                                                              method="naive")  # "naive"
#        sss_eigvec = sss_eigvec[:,sss_ind[0]].unsqueeze(1)
        
        mean_log = torch.abs(sss_eigval.mean()) * 1e-4
        center = torch.abs(cos.mean()) + self.offset

        # Uniform distribution
        unif = torch.distributions.uniform.Uniform(center-mean_log, center+mean_log)
        unif_ref = torch.zeros(X.size(0), k).cuda()
        for i in range(X.size(0)):
            for j in range(k):
                unif_ref[i,j] = unif.rsample()
        ref = unif_ref @ sss_eigvec.cuda()
        ref = torch.sum(torch.abs(ref))/cos.size(1)  # absolute sum

        ref_pos = torch.clamp(torch.sqrt(ref), 0.5, 1.5)  # technical part: value clipping for gradient exploding
        ref_neg = torch.clamp(torch.sqrt(ref), 0.5, 1.5)

        ###################################################

        pos_exp = torch.exp(-48.0 * (cos - 0.10))
        neg_exp = torch.exp(48.0 * (cos + 0.10))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)
        num_valid_proxies = len(with_pos_proxies)

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = (torch.log(1.0 + P_sim_sum*ref_pos).sum()) / num_valid_proxies
        neg_term = (torch.log(1.0 + N_sim_sum*ref_neg).sum()) / self.nb_classes

        loss = pos_term + neg_term - 1e-6 * sss_eigval.mean().cuda()
        return loss


class PPGML_MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(PPGML_MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.projector = torch.nn.Sequential(torch.nn.Linear(1, 4)).type(torch.cuda.FloatTensor)

        self.LDS_dim_1 = 3
        self.LDS_dim_2 = 4
        self.LDS_dim_3 = 5

        self.p0 = torch.tensor([0., 0., 1.]).cuda()
        self.p1 = torch.cat([torch.zeros(self.LDS_dim_2-1), torch.ones(1)]).cuda()  # 7
        self.p2 = torch.cat([torch.zeros(self.LDS_dim_3-1), torch.ones(1)]).cuda()  # 15

        self.offset = 0.75

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):

        X = embeddings
        T = labels
        
        #########################

        X_proj_1 = torch.from_numpy(Isomap(n_components=self.LDS_dim_1).fit_transform(X.detach().cpu().numpy())).type(torch.cuda.FloatTensor)
        X_proj_2 = torch.from_numpy(Isomap(n_components=self.LDS_dim_2).fit_transform(X.detach().cpu().numpy())).type(torch.cuda.FloatTensor)
        X_proj_3 = torch.from_numpy(Isomap(n_components=self.LDS_dim_3).fit_transform(X.detach().cpu().numpy())).type(torch.cuda.FloatTensor)

        pip_loss_1 = 0.001 * (X.size(1) - X_proj_1.size(1)) + 2.0 * torch.norm(X_proj_1.t() @ X[:,X.size(1)-(self.LDS_dim_1-1)], p=2).pow(2)
        pip_loss_2 = 0.001 * (X.size(1) - X_proj_2.size(1)) + 2.0 * torch.norm(X_proj_2.t() @ X[:,X.size(1)-(self.LDS_dim_2-1)], p=2).pow(2)
        pip_loss_3 = 0.001 * (X.size(1) - X_proj_3.size(1)) + 2.0 * torch.norm(X_proj_3.t() @ X[:,X.size(1)-(self.LDS_dim_3-1)], p=2).pow(2)
        min_ind = torch.argmin(torch.FloatTensor([pip_loss_1, pip_loss_2, pip_loss_3]))
        print("\n")
        print(fg256("orange", '{} | {} | {}'.format(pip_loss_1, pip_loss_2, pip_loss_3)))
        print(fg256("magenta", 'min_ind: {}'.format(min_ind)))
        
        if min_ind == 0:
            X_proj_1 = X_proj_1.unsqueeze(2)
            rlist = []
            for i in range(X_proj_1.size(0)):
                for j in range(X_proj_1.size(1)):
                    rlist.append(self.projector(X_proj_1[i,j]))
            rtensor = torch.stack(rlist)  # [180*3, 2]
            rtensor = torch.reshape(rtensor, (X_proj_1.size(0), X_proj_1.size(1), rtensor.size(1)))  # [180, 3, 2]
            del rlist
            X_proj = F.normalize(rtensor)
            X_log_map = lifting_map_on_Stiefel(X_proj, self.p0)
            k = self.LDS_dim_1 * X_log_map.size(2)
        elif min_ind == 1:
            X_proj_2 = X_proj_2.unsqueeze(2)
            rlist = []
            for i in range(X_proj_2.size(0)):
                for j in range(X_proj_2.size(1)):
                    rlist.append(self.projector(X_proj_2[i,j]))
            rtensor = torch.stack(rlist)  # [180*3, 2]
            rtensor = torch.reshape(rtensor, (X_proj_2.size(0), X_proj_2.size(1), rtensor.size(1)))  # [180, 3, 2]
            del rlist
            X_proj = F.normalize(rtensor)
            X_log_map = lifting_map_on_Stiefel(X_proj, self.p1)
            k = self.LDS_dim_2 * X_log_map.size(2)
        else:
            X_proj_3 = X_proj_3.unsqueeze(2)
            rlist = []
            for i in range(X_proj_3.size(0)):
                for j in range(X_proj_3.size(1)):
                    rlist.append(self.projector(X_proj_3[i,j]))
            rtensor = torch.stack(rlist)  # [180*3, 2]
            rtensor = torch.reshape(rtensor, (X_proj_3.size(0), X_proj_3.size(1), rtensor.size(1)))  # [180, 3, 2]
            del rlist
            X_proj = F.normalize(rtensor)
            X_log_map = lifting_map_on_Stiefel(X_proj, self.p2)
            k = self.LDS_dim_3 * X_log_map.size(2)
        

        klist = []
        for i in range(X_log_map.size(0)): klist.append(kronecker(X_log_map[i], X_log_map[i].t())[0,:])  # Product Stiefel manifold
        X_log_map = torch.stack(klist)
#        X_log_map = torch.reshape(X_log_map, (X_log_map.size(0), X_log_map.size(1)*X_log_map.size(2)))
        sss_eigval, sss_eigvec, sss_ind = Linear_discriminant_analysis(X_log_map, T.detach().cpu().numpy(),
                                                                       X_log_map.size(1), regularizer=None)
        sss_eigvec = sss_eigvec[:,sss_ind[0]].unsqueeze(1)
        
        mean_log = torch.abs(sss_eigval.mean()) * 1e-4
        center = torch.abs(embeddings.mean()) + self.offset
        print("mean log is ", mean_log)
        print("center is ", center)

        # Uniform distribution
        unif = torch.distributions.uniform.Uniform(center-mean_log, center+mean_log)
        unif_ref = torch.zeros(X.size(0), k).cuda()
        for i in range(X.size(0)):
            for j in range(k):
                unif_ref[i,j] = unif.rsample()
        ref = unif_ref @ sss_eigvec.cuda()
        ref = torch.mean(torch.abs(ref))
#        ref = torch.sum(torch.abs(ref))/embeddings.size(0)  # absolute sum

        ref_pos = torch.clamp(torch.sqrt(ref), 0.8, 1.5)
        ref_neg = torch.clamp(torch.sqrt(ref), 0.8, 1.5)
        print(fg256("yellow", 'ref_pos is {}'.format(ref_pos)))

#        ref_pos = ref_neg = 1.0
        hard_pairs = self.miner(embeddings, labels)
#        loss = self.loss_func(embeddings, labels, hard_pairs, ref_pos, ref_neg)  - 1e-4 * sss_eigval.mean().cuda()
        loss = self.loss_func(embeddings, labels, hard_pairs, ref_pos, ref_neg) - 1e-10 * sss_eigval.mean().cuda()
        return loss