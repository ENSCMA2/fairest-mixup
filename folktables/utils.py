import torch
import numpy as np
from numpy.random import beta
from sklearn.metrics import *
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

best = {'employment': {'base': {'lam': 0.5, 'd': 10, 'k': 100},
                       'dp': {'lam': 0.5, 'd': 10, 'k': 100},
                       'eo': {'lam': 0.25, 'd': 10, 'k': 100},
                       'mpmc': {'lam': 0.25, 'd': 10, 'k': 100},
                       'mpma': {'lam': 0.5, 'd': 10, 'k': 100},
                       'base_mixup': {'lam': 0.25, 'd': 10, 'k': 3},
                       'eo_mixup': {'lam': 0.25, 'd': 10, 'k': 100},
                       'mpmc_mixup': {'lam': 0.25, 'd': 10, 'k': 40},
                       'mpma_mixup': {'lam': 0.25, 'd': 10, 'k': 3},
                       'enforce_ma': {'lam': 0.5, 'd': 10, 'k': 3},
                       'enforce_mc': {'lam': 0.5, 'd': 10, 'k': 3},
                       'fairbase': {'lam': 0, 'd': 10, 'k': 3},
                       "mixup_enforce_mc": {'lam': 0.25, 'd': 10, 'k': 3},
                       "dp_enforce_mc": {'lam': 0.5, 'd': 10, 'k': 100},
                       "eo_enforce_mc": {'lam': 0.25, 'd': 10, 'k': 100},
                       "fairbase_enforce_mc": {'lam': 0, 'd': 10, 'k': 3},
                       "mpmc_enforce_mc": {'lam': 0.25, 'd': 10, 'k': 100},
                       "mpma_enforce_mc": {'lam': 0.5, 'd': 10, 'k': 100}},
        'income': {'base': {'lam': 0.5, 'd': 10, 'k': 100},
                   'dp': {'lam': 0.25, 'd': 10, 'k': 3},
                   'eo': {'lam': 0.5, 'd': 10, 'k': 3},
                   'mpmc': {'lam': 0.25, 'd': 10, 'k': 3},
                   'mpma': {'lam': 0.5, 'd': 10, 'k': 3},
                   'base_mixup': {'lam': 0.25, 'd': 10, 'k': 40},
                   'eo_mixup': {'lam': 0.5, 'd': 10, 'k': 40},
                   'mpmc_mixup': {'lam': 0.5, 'd': 10, 'k': 40},
                   'mpma_mixup': {'lam': 0.25, 'd': 10, 'k': 40},
                   'enforce_ma': {'lam': 0.5, 'd': 10, 'k': 3},
                   'enforce_mc': {'lam': 0.5, 'd': 10, 'k': 3},
                   'fairbase': {'lam': 0, 'd': 10, 'k': 3},
                   "mixup_enforce_mc": {'lam': 0.25, 'd': 10, 'k': 40},
                   "dp_enforce_mc": {'lam': 0.25, 'd': 10, 'k': 3},
                   "eo_enforce_mc": {'lam': 0.5, 'd': 10, 'k': 3},
                   "fairbase_enforce_mc": {'lam': 0, 'd': 10, 'k': 3},
                   "mpmc_enforce_mc": {'lam': 0.25, 'd': 10, 'k': 3},
                   "mpma_enforce_mc": {'lam': 0.5, 'd': 10, 'k': 3}}}

def sample_batch_sen_idx(X, A, y, batch_size, s, dpma = False):    
    cands = np.where(A==s)[0]
    if X.shape[0] < 306029 * 0.6 and dpma:
        batch_idx = np.random.choice(cands, size=batch_size, replace=batch_size > len(cands)).tolist()
    else:
        batch_idx = np.random.choice(cands, size=min(batch_size, len(cands)), replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).float()

    return batch_x, batch_y

def sample_batch_sen_idx_y(X, A, y, batch_size, s):
    batch_idx = []
    for i in range(2):
        idx = set(np.where(A==s)[0]) 
        idx = list(idx.intersection(set(np.where(y==i)[0])))
        batch_idx += np.random.choice(idx, size=min(batch_size, len(idx)), replace=False).tolist()

    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y

def select_pred(X, y, f, bsz, i, d):
    batch_size = int(2 * bsz / d)
    idx = set(np.where(y >= i / d)[0])
    idx = list(idx.intersection(set(np.where(y < (i + 1) / d)[0])))
    if len(idx) == 0:
        return [], []
    batch_idx = np.random.choice(idx, size=min(batch_size, len(idx)), replace = False).tolist()
    batch_x = X[batch_idx].cuda().float()
    batch_y = y[batch_idx].cuda().float()

    return batch_x, batch_y

def train_dpma(model, criterion, optimizer, X_train, group_crits, y_train, lam, d = None, batch_size=500, niter=100, thresh = 0.0001, alpha = 1, k = 3, normal = False, mode = "dp"):
    start_time = time.time()
    model.train()
    curr_loss = float("inf")
    bsz = int(batch_size / len(group_crits))
    for it in range(niter):
        loss_regs = []
        batch_x, batch_y = None, None
        for crit in group_crits:
            A = np.ones(X_train.shape[0])
            for n, c, v in crit:
                A *= X_train[:, c] == v

            # Gender Split
            batch_x_0, batch_y_0 = sample_batch_sen_idx(X_train, A, y_train, bsz, 0, dpma = True)
            batch_x_1, batch_y_1 = sample_batch_sen_idx(X_train, A, y_train, bsz, 1, dpma = True)

            smaller_group = min(len(batch_x_1), len(batch_x_0))
            batch_x_0, batch_y_0 = batch_x_0[:smaller_group], batch_y_0[:smaller_group]
            batch_x_1, batch_y_1 = batch_x_1[:smaller_group], batch_y_1[:smaller_group]

            batch_x = torch.cat((batch_x_0, batch_x_1), 0) if batch_x is None else torch.cat((batch_x, batch_x_0, batch_x_1), 0)
            batch_y = torch.cat((batch_y_0, batch_y_1), 0) if batch_y is None else torch.cat((batch_y, batch_y_0, batch_y_1), 0)

            if lam > 0:
                # Fair Mixup
                alpha = alpha
                gamma = beta(alpha, alpha)

                # Interpolate batch_size datapoints, only 1 interpolation
                batch_x_mix = batch_x_0 * gamma + batch_x_1 * (1 - gamma)
                batch_x_mix = batch_x_mix.requires_grad_(True)

                # generate predictions on interpolated data f(i)
                output = model(batch_x_mix)

                if mode == "ma":
                    batch_y_mix = batch_y_0 * gamma + batch_y_1 * (1 - gamma)

                # gradient regularization
                # gradient of E[f(T(x_0, x_1, t))] * batch_size
                if not normal:
                    gradx = torch.autograd.grad(output.sum() - (batch_y_mix.sum() if mode == "ma" else 0), batch_x_mix, create_graph=True)[0]

                    # second component of inner product from Section 5
                    batch_x_d = batch_x_1 - batch_x_0
                    # inner product from section 5, approximate integral
                    grad_inn = (gradx * batch_x_d).sum(1)
                    # then take the expectation for another integral
                    E_grad = grad_inn.mean(0)
                    # absolute value to match equation
                    loss_regs.append(float(torch.abs(E_grad).cpu()))
                    del output
                else:
                    loss_reg = gamma * criterion(output.squeeze(), batch_y_0.cuda().float()) + (1 - gamma) * criterion(output.squeeze(), batch_y_1.cuda().float())
                    loss_regs.append(loss_reg.item())

        output = model(batch_x)
        loss_sup = criterion(output.squeeze(), batch_y.cuda())

        # final loss from loss minimization algorithm
        if lam > 0:
            fairness_loss = np.mean(sorted(loss_regs, reverse = True)[:k])
        else:
            fairness_loss = 0
        loss = loss_sup + lam * fairness_loss
        del output

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if abs(curr_loss - loss) < thresh:
            return time.time() - start_time, it + 1, [], []
    return time.time() - start_time, niter, [], [], model

def train_mpma(model, criterion, optimizer, X_train, group_crits, y_train, lam, d = None, batch_size=500, niter=100, thresh = 0.0001, alpha = 1, k = 3, normal = False, seed = None):
    return train_dpma(model, criterion, optimizer, X_train, group_crits, 
                      y_train, lam, d = None, batch_size=500, niter=100, 
                      thresh = 0.0001, alpha = 1, k = 3, normal = False, 
                      mode = "ma")

def train_dp(model, criterion, optimizer, X_train, group_crits, y_train, lam, d = None, batch_size=500, niter=100, thresh = 0.0001, alpha = 1, k = 3, normal = False, seed = None):
    return train_dpma(model, criterion, optimizer, X_train, group_crits, 
                      y_train, lam, d = None, batch_size=500, niter=100, 
                      thresh = 0.0001, alpha = 1, k = 3, normal = False, 
                      mode = "dp")

def train_mpmc(model, criterion, optimizer, X_train, group_crits, y_train, lam, d = None, batch_size=500, niter=100, thresh = 0.0001, alpha = 1, k = 3, normal = False, seed = None):
    start_time = time.time()
    model.train()
    bsz = int(batch_size / len(group_crits))
    curr_loss = float("inf")
    for it in range(niter):
        loss_regs = []
        batch_x, batch_y = None, None
        for crit in group_crits:
            A = np.ones(X_train.shape[0])
            for n, c, v in crit:
                A *= X_train[:, c] == v

            # Group Split
            batch_x_0, batch_y_0 = sample_batch_sen_idx(X_train, A, y_train, bsz, 0)
            batch_x_1, batch_y_1 = sample_batch_sen_idx(X_train, A, y_train, bsz, 1)

            smaller_group = min(len(batch_x_1), len(batch_x_0))
            batch_x_0, batch_y_0 = batch_x_0[:smaller_group], batch_y_0[:smaller_group]
            batch_x_1, batch_y_1 = batch_x_1[:smaller_group], batch_y_1[:smaller_group]

            batch_x = torch.cat((batch_x_0, batch_x_1), 0) if batch_x is None else torch.cat((batch_x, batch_x_0, batch_x_1), 0)
            batch_y = torch.cat((batch_y_0, batch_y_1), 0) if batch_y is None else torch.cat((batch_y, batch_y_0, batch_y_1), 0)

            if lam > 0:
                output_0 = model(batch_x_0).cpu()
                output_1 = model(batch_x_1).cpu()

                # Fair Mixup
                alpha = alpha
                group_loss_reg = 0
                for i in range(d):
                    batch_x_0_, batch_y_0_ = select_pred(batch_x_0, batch_y_0, output_0, bsz, i, d)
                    batch_x_1_, batch_y_1_ = select_pred(batch_x_1, batch_y_1, output_1, bsz, i, d)
                    smaller = min(len(batch_y_0_), len(batch_y_1_))
                    if len(batch_y_0_) == 0 or len(batch_y_1_) == 0:
                        continue
                    gamma = beta(alpha, alpha)

                    batch_x_0_, batch_y_0_ = batch_x_0_[:smaller], batch_y_0_[:smaller]
                    batch_x_1_, batch_y_1_ = batch_x_1_[:smaller], batch_y_1_[:smaller]

                    # Interpolate batch_size datapoints, only 1 interpolation
                    batch_x_mix = batch_x_0_ * gamma + batch_x_1_ * (1 - gamma)
                    batch_x_mix = batch_x_mix.requires_grad_(True)

                    # generate predictions on interpolated data f(i)
                    output = model(batch_x_mix)

                    # interpolate labels
                    batch_y_mix = batch_y_0_ * gamma + batch_y_1_ * (1 - gamma)

                    # gradient regularization
                    if not normal:
                        gradx = torch.autograd.grad(output.sum() - batch_y_mix.sum(), batch_x_mix, create_graph=True)[0]

                        # second component of inner product from Section 5
                        batch_x_d = batch_x_1_ - batch_x_0_
                        # inner product from section 5, approximate integral
                        grad_inn = (gradx * batch_x_d).sum(1)
                        # then take the expectation for another integral
                        E_grad = grad_inn.mean(0)
                        # absolute value to match equation
                        group_loss_reg += float(torch.abs(E_grad).cpu())
                    else:
                        gampart = gamma * criterion(output.squeeze(), batch_y_0_.squeeze().cuda().float())
                        onepart = (1 - gamma) * criterion(output.squeeze(), batch_y_1_.squeeze().cuda().float())
                        group_loss_reg = gampart + onepart
                        group_loss_reg = group_loss_reg.item()
                    del output
                loss_regs.append(group_loss_reg / d)

        output = model(batch_x)
        loss_sup = criterion(output.squeeze(), batch_y.cuda())
        del output

        # final loss from loss minimization algorithm
        if lam > 0:
            fairness_loss = np.mean(sorted(loss_regs, reverse = True)[:k])
        else:
            fairness_loss = 0
        # final loss from loss minimization algorithm
        loss = loss_sup + lam * fairness_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if abs(curr_loss - loss) < thresh:
            return time.time() - start_time, it + 1
    return time.time() - start_time, niter, [], [], model

def train_eo(model, criterion, optimizer, X_train, group_crits, y_train, lam, d = None, batch_size=500, niter=100, thresh = 0.0001, alpha = 1, k = 3, normal = False, seed = None):
    start_time = time.time()
    model.train()
    curr_loss = float("inf")
    bsz = int(batch_size / len(group_crits))
    for it in range(niter):
        loss_regs = []
        batch_x, batch_y = None, None
        for crit in group_crits:
            A = np.ones(X_train.shape[0])
            for n, c, v in crit:
                A *= X_train[:, c] == v

            batch_x_0, batch_y_0 = sample_batch_sen_idx(X_train, A, y_train, bsz, 0)
            batch_x_1, batch_y_1 = sample_batch_sen_idx(X_train, A, y_train, bsz, 1)

            batch_x = torch.cat((batch_x_0, batch_x_1), 0) if batch_x is None else torch.cat((batch_x, batch_x_0, batch_x_1), 0)
            batch_y = torch.cat((batch_y_0, batch_y_1), 0) if batch_y is None else torch.cat((batch_y, batch_y_0, batch_y_1), 0)

            if lam > 0:
                # separate class
                idx_0_0 = np.where(batch_y_0 == 0)[0]
                idx_0_1 = np.where(batch_y_0 == 1)[0]
                
                idx_1_0 = np.where(batch_y_1 == 0)[0]
                idx_1_1 = np.where(batch_y_1 == 1)[0]

                smallest_group = min(len(idx_0_0), min(len(idx_0_1),
                                                       min(len(idx_1_0),
                                                           len(idx_1_1))))
                batch_x_0_ = [batch_x_0[idx_0_0][:smallest_group], batch_x_0[idx_0_1][:smallest_group]]
                batch_x_1_ = [batch_x_1[idx_1_0][:smallest_group], batch_x_1[idx_1_1][:smallest_group]]

                batch_y_0_ = [batch_y_0[idx_0_0][:smallest_group], batch_y_0[idx_0_1][:smallest_group]]
                batch_y_1_ = [batch_y_1[idx_1_0][:smallest_group], batch_y_1[idx_1_1][:smallest_group]]

                alpha = alpha
                loss_reg = 0
                for i in range(2):
                    gamma = beta(alpha, alpha)
                    batch_x_0_i = batch_x_0_[i]
                    batch_x_1_i = batch_x_1_[i]
                    batch_y_0_i = batch_y_0_[i]
                    batch_y_1_i = batch_y_1_[i]

                    if len(batch_y_0_i) == 0 or len(batch_y_1_i) == 0:
                        continue

                    batch_x_mix = batch_x_0_i * gamma + batch_x_1_i * (1 - gamma)
                    batch_x_mix = batch_x_mix.requires_grad_(True)
                    output = model(batch_x_mix)

                     # gradient regularization
                    if not normal:
                        gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]
                        batch_x_d = batch_x_1_i - batch_x_0_i
                        grad_inn = (gradx * batch_x_d).sum(1)
                        loss_reg += float(torch.abs(grad_inn.mean()).cpu()) / 2
                    else:
                        loss_reg = gamma * criterion(output.squeeze(), batch_y_0_i.squeeze().cuda().float()) + (1 - gamma) * criterion(output.squeeze(), batch_y_1_i.squeeze().cuda().float())
                        loss_reg = loss_reg.item()
                    del output
                loss_regs.append(loss_reg)

        output = model(batch_x)
        loss_sup = criterion(output.squeeze(), batch_y.cuda())
        del output

        # final loss
        if lam > 0:
            fairness_loss = np.mean(sorted(loss_regs, reverse = True)[:k])
        else:
            fairness_loss = 0
        loss = loss_sup + lam * fairness_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if abs(curr_loss - loss) < thresh:
            return time.time() - start_time, it + 1
    return time.time() - start_time, niter, [], [], model

def train_base(model, criterion, optimizer, X_train, group_crits, y_train, lam, d = None, batch_size=500, niter=100, thresh = 0.0001, alpha = 1, k = 3, normal = False, seed = None):
    start_time = time.time()
    model.train()
    bsz = int(batch_size / len(group_crits))
    curr_loss = float("inf")
    for it in range(niter):
        batch_idx = np.random.choice(range(len(y_train)), size = batch_size, replace = False)
        batch_x, batch_y = torch.tensor(X_train[batch_idx]).cuda().float(), torch.tensor(y_train[batch_idx])

        if normal:
            gamma = beta(alpha, alpha)
            index = torch.randperm(batch_size)
            mixed_x = gamma * batch_x + (1 - gamma) * batch_x[index, :]
            y_a, y_b = batch_y.cuda().float(), batch_y[index].cuda().float()
            mixed_out = model(mixed_x)
            loss = gamma * criterion(mixed_out.squeeze(), y_a) + (1 - gamma) * criterion(mixed_out.squeeze(), y_b)
            del mixed_out
        else:
            output = model(batch_x)
            batch_y = batch_y.cuda().float()
            loss = criterion(output.squeeze(), batch_y)
            del output

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if abs(curr_loss - loss) < thresh:
            return time.time() - start_time, it + 1
    return time.time() - start_time, niter, [], [], model

def enforce_mc(model, X_v, group_crits, y_v, alpha = 0.01):
    start_time = time.time()
    def calculate_calibration(p, S, data):
        # split S into deciles
        S_deciles = [[] for _ in range(11)]
        for i in range(0, 11):
            for elem in S:
                if elem is not None:
                    if (0.1 * i <= p[elem] < 0.1 * (i + 1)):
                        S_deciles[i].append(elem)
        
        # compute violations               
        S_violations = [0 for _ in range(11)]
        for decile in range(0, len(S_deciles)):
            n = len(S_deciles[decile])
            if (n != 0):
                predictor_sum = 0.
                data_sum = 0.
                for elem in S_deciles[decile]:
                    if elem is not None:
                        predictor_sum += p[elem]
                        data_sum += data[elem]
                S_violations[decile] = (data_sum - predictor_sum) / n

        # return the violation posed by the decile with the maximum violation
        j_max = S_violations.index(max(S_violations))
        j_min = S_violations.index(min(S_violations))

        if (abs(S_violations[j_max]) > abs(S_violations[j_min])):
            j = j_max
        else:
            j = j_min

        return j, abs(S_violations[j]), S_violations, S_deciles

    def multicalibrate(X_val, p, group_crits, data, alpha):
        p_new = p.detach().clone().cpu().numpy()
        done = False
        seen = []
        updates = []
        n = 0
        while not done:
            n += 1
            ind = np.random.choice(len(group_crits), size=1, replace=False)[0]
            A = np.ones(X_val.shape[0])
            for _, c, v in group_crits[ind]:
                A *= X_val[:, c] == v
            curr_set = np.where(A == 1)[0]
            seen.append(ind)
            if len(curr_set) > 0:
                # compute deciles of S
                j, violation, S_violations, S_deciles = calculate_calibration(p_new, curr_set, data)
                # violation too high!
                if (violation > alpha):
                    # update p by nudging
                    seen = []
                    updates.append((group_crits[ind], j, S_violations[j]))
                    for elem in S_deciles[j]:
                        if elem is not None:
                            p_new[elem] += S_violations[j]
                            if (p_new[elem] > 1):
                                p_new[elem] = 1
                            elif (p[elem] < 0):
                                p_new[elem] = 0
            if (len(seen) == len(group_crits)):
                done = True
        return updates, n
    updates, n = multicalibrate(X_v, 
                                model(torch.from_numpy(X_v).float().cuda()).squeeze(), 
                                group_crits, 
                                y_v, 
                                alpha = 0.01)
    return time.time() - start_time, n, updates

def enforce_ma(model, X_v, group_crits, y_v, alpha = 0.01):
    start_time = time.time()
    
    def calculate_accuracy(p, S, data):
        predictor_sum = 0
        data_sum = 0
        # compute violations           
        for elem in S:
            if elem is not None:
                predictor_sum += p[elem]
                data_sum += data[elem]
        
        return (data_sum - predictor_sum) / len(S)

    def enforce_multiaccuracy(X_val, p, group_crits, data, alpha):
        p_new = p.detach().clone().cpu().numpy()
        done = False
        seen = []
        updates = []
        n = 0
        while not done:
            n += 1
            ind = np.random.choice(len(group_crits), size=1, replace=False)[0]
            A = np.ones(X_val.shape[0])
            for _, c, v in group_crits[ind]:
                A *= X_val[:, c] == v
            curr_set = np.where(A == 1)[0]
            seen.append(ind)
            # compute deciles of S
            if len(curr_set) > 0:
                violation = calculate_accuracy(p_new, curr_set, data)
                # violation too high!
                if (abs(violation) > alpha):
                    # update p by nudging
                    seen = []
                    updates.append((group_crits[ind], violation))
                    for elem in curr_set:
                        if elem is not None:
                            p_new[elem] += violation
                            if (p_new[elem] > 1):
                                p_new[elem] = 1
                            elif (p[elem] < 0):
                                p_new[elem] = 0
            
            if (len(seen) == len(group_crits)):
                done = True
        
        return updates, n

    updates, n = enforce_multiaccuracy(X_v, 
                                       model(torch.from_numpy(X_v).float().cuda()).squeeze(), 
                                       group_crits, 
                                       y_v, 
                                       alpha = 0.01)
    return time.time() - start_time, n, updates
    
def evaluate_mcma(model, X_test, y_test, group_crits, d, ma_updates, mc_updates):
    model.eval()

    X_test_cuda = torch.tensor(X_test).cuda().float()
    output = model(X_test_cuda).cpu()

    def update_test_preds_MC(X_test, p, updates):
        p_new = p.copy()
        for update in updates:
            set_to_update, decile_to_update, violation = update
            A = np.ones(X_test.shape[0])
            for _, c, v in set_to_update:
                A *= X_test[:, c] == v
            curr_set = np.where(A == 1)[0]
            for elem in curr_set:
                if elem is not None:
                    elem = int(elem)
                    if (0.1 * (decile_to_update) <= p_new[elem] < 0.1 * (decile_to_update + 1)):
                        p_new[elem] += violation
                        if(p_new[elem] > 1):
                            p_new[elem] = 1
                        elif(p_new[elem] < 0):
                            p_new[elem] = 0
            
        return np.array(p_new)

    def update_test_preds_MA(X_test, p, updates):
        p_new = p.detach().clone().cpu().numpy()
        for update in updates:
            set_to_update, violation = update
            A = np.ones(X_test.shape[0])
            for _, c, v in set_to_update:
                A *= X_test[:, c] == v
            curr_set = np.where(A == 1)[0]
            for elem in curr_set:
                if elem is not None:
                    elem = int(elem)
                    p_new[elem] += violation
                    if(p_new[elem] > 1):
                        p_new[elem] = 1
                    elif(p_new[elem] < 0):
                        p_new[elem] = 0
                
        return np.array(p_new)

    output = update_test_preds_MA(X_test, output.squeeze(), ma_updates)
    output = update_test_preds_MC(X_test, output, mc_updates)
    mc_violations = []
    ma_violations = []

    for crit in group_crits:
        A = np.ones(X_test.shape[0])
        for n, c, v in crit:
            A *= X_test[:, c] == v
        # calculate DP gap
        idx_1 = np.where(A==1)[0]

        X_test_1 = torch.tensor(X_test[idx_1]).cuda().float()
        y_test_1 = torch.tensor(y_test[idx_1]).to(int)
        pred_1 = model(X_test_1).cpu()
        violation_1 = abs((pred_1 - y_test_1).mean())
        ma_violations.append(float(violation_1))
        del X_test_1; del pred_1

        violation_1 = 0

        for i in range(d + 1):
            idx_1 = set(np.where(A==1)[0])
            idx_1 = idx_1.intersection(set(np.where(output >= i / d)[0]))
            idx_1 = list(idx_1.intersection(set(np.where(output < (i + 1) / d)[0])))
            if len(idx_1) == 0:
                continue

            X_test_1 = torch.tensor(X_test[idx_1]).cuda().float()
            y_test_1 = torch.tensor(y_test[idx_1]).to(int)
            pred_1 = model(X_test_1).cpu()
            new_v = abs((pred_1 - y_test_1).mean()).item()
            violation_1 = max(violation_1, new_v)

            del X_test_1; del pred_1
        mc_violations.append(float(violation_1))

    # calculate average precision
    
    preds = (output >= 0.5).astype(int)
    y_test = y_test.astype(int)
    acc = balanced_accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    del output

    return acc, prec, rec, f1, mc_violations, ma_violations

def train_mixup_enforce_mc(model, criterion, optimizer, X_train, group_crits, y_train, lam, d = None, batch_size=500, niter=100, thresh = 0.0001, alpha = 1, k = 3, normal = False, seed = None):
    return train_base(model, criterion, optimizer, X_train, group_crits, y_train, lam, d, batch_size, niter, thresh, alpha, k, normal = True, seed = seed)

def train_fairbase_enforce_mc(model, criterion, optimizer, X_train, group_crits, y_train, lam, d = None, batch_size=500, niter=100, thresh = 0.0001, alpha = 1, k = 3, normal = False, seed = None):
    return train_mpmc(model, criterion, optimizer, X_train, group_crits, y_train, 0, d, batch_size, niter, thresh, alpha, k, normal, seed)

funcs = {
            "dp": train_dp,
            "eo": train_eo,
            "mpma": train_mpma,
            "mpmc": train_mpmc,
            "base": train_base,
            "enforce_mc": train_base,
            "enforce_ma": train_base,
            "mixup_enforce_mc": train_mixup_enforce_mc,
            "dp_enforce_mc": train_dp,
            "eo_enforce_mc": train_eo,
            "fairbase_enforce_mc": train_fairbase_enforce_mc,
            "mpmc_enforce_mc": train_mpmc,
            "mpma_enforce_mc": train_mpma
        }

def crit_to_name(crit):
    return "-".join([f"{n}_{c}" for n, c, v in crit])