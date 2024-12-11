import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from dataset import *
from model import Net
from utils import *
import pandas as pd
import os

def log(message):
    with open("log.txt", "a") as o:
        o.write(f"{message}\n")

def run_experiments(dsname, state, year, 
                    mode='mpma', 
                    lam = 0.5, num_exp = 10,
                    d = 10, alpha = 1, niter = 100, k = 3,
                    setting = 'all', normal = False):
    '''
    Retrain each model for 10 times and report performance and fairness metrics.
    '''
    if os.path.exists(f"results/{dsname}/{state}/{year}/{mode}{'' if not normal else '_mixup'}/lam_{lam}_d_{d}_alpha_{alpha}_niter_{niter}_k_{k}_{setting}.csv"):
        return

    acc = []
    prec = []
    rec = []
    f1 = []
    mc = []
    ma = []
    idxs = []
    wallclocks = []
    niters = []

    for i in range(num_exp):
        print(f'On experiment {i}')
        # get train/test data
        ds = folktables_ds(seed = i, name = dsname, 
                           state = state, year = year,
                           setting = setting)
        print("created ds")
        X_train, X_val, X_test, y_train, y_val, y_test, group_crits, smallest_race = ds
        if os.path.exists(f"results/{dsname}/{state}/{year}/{mode}{'' if not normal else '_mixup'}/lam_{lam}_d_{d}_alpha_{alpha}_niter_{niter}_k_{k}_{setting}.csv"):
            return

        # initialize model
        model = Net(input_size=len(X_train[0])).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        mc_val_epoch = []
        ma_val_epoch = []
        mc_test_epoch = []
        ma_test_epoch = []
        acc_test_epoch = []
        prec_test_epoch = []
        rec_test_epoch = []
        f1_test_epoch = []
        wallclock_epoch = []
        niter_epoch = []


        # run experiments
        try:
            sh = pd.read_csv(f"results/{dsname}/CA/2022/mean/lam_{lam}_d_{d}_alpha_{alpha}_niter_{niter}_k_{k}_{setting}.csv")
            num_epochs = sh[sh["Mode"] == mode]["Max Best Epoch"].tolist()[0]
        except Exception as e:
            num_epochs = 10
        if 'enforce' in mode:
            X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size = 0.25,
                                                  random_state = i)
        else:
            X_t, y_t = X_train, y_train
            X_v, y_v = None, None
        for j in tqdm(range(num_epochs)):
            if os.path.exists(f"results/{dsname}/{state}/{year}/{mode}{'' if not normal else '_mixup'}/lam_{lam}_d_{d}_alpha_{alpha}_niter_{niter}_k_{k}_{setting}.csv"):
                return
            print(f"epoch {j + 1}")
            train = funcs[mode]
            

            wallclock, niter, ma_updates, mc_updates, model = train(model, 
                                                                    criterion, 
                                                                    optimizer, 
                                                                    X_t,
                                                                    group_crits, 
                                                                    y_t, lam, d, 
                                                                    alpha = alpha, 
                                                                    niter = niter, 
                                                                    k = k, 
                                                                    normal = normal,
                                                                    seed = i)
            if "enforce" in mode:
                if not os.path.exists(f"models/{dsname}/{state}/{year}/{setting}/{mode}/"):
                    os.makedirs(f"models/{dsname}/{state}/{year}/{setting}/{mode}/")
                torch.save(model.state_dict(), f"models/{dsname}/{state}/{year}/{setting}/{mode}/{j}.pt")

            acc_val, pre_val, rec_val, f1_val, mc_val, ma_val = evaluate_mcma(model, X_val, y_val, group_crits, d, ma_updates, mc_updates)
            acc_test, prec_test, rec_test, f1_test, mc_test, ma_test = evaluate_mcma(model, X_test, y_test, group_crits, d, ma_updates, mc_updates)

            acc_test_epoch.append(acc_test)
            prec_test_epoch.append(prec_test)
            rec_test_epoch.append(rec_test)
            f1_test_epoch.append(f1_test)
            mc_val_epoch.append(mc_val)
            mc_test_epoch.append(mc_test)
            ma_test_epoch.append(mc_test)
            wallclock_epoch.append(wallclock)
            niter_epoch.append(niter)

        mc_val_epoch = np.array(mc_val_epoch)
        # best model based on validation performance
        idx = np.argmin(np.mean(mc_val_epoch, axis = 1))
        idxs.append(idx + 1)
        if "enforce" in mode:
            model.load_state_dict(torch.load(f"models/{dsname}/{state}/{year}/{setting}/{mode}/{idx}.pt"))
            ma_updates, mc_updates = [], []
            if 'enforce_ma' in mode:
                enforce_time, n, updates = enforce_ma(model, X_v, group_crits, y_v)
                ma_updates = updates
            elif "enforce_mc" in mode:
                enforce_time, n, updates = enforce_mc(model, X_v, group_crits, y_v)
                mc_updates = updates
            acc_test, prec_test, rec_test, f1_test, mc_test, ma_test = evaluate_mcma(model, X_test, y_test, group_crits, d, ma_updates, mc_updates)
            mc_test_epoch[idx] = mc_test
            ma_test_epoch[idx] = ma_test
            acc_test_epoch[idx] = acc_test
            prec_test_epoch[idx] = prec_test
            rec_test_epoch[idx] = rec_test
            f1_test_epoch[idx] = f1_test
        else:
            enforce_time = 0
            n = 0

        mc.append(mc_test_epoch[idx])
        ma.append(ma_test_epoch[idx])
        acc.append(acc_test_epoch[idx])
        prec.append(prec_test_epoch[idx])
        rec.append(rec_test_epoch[idx])
        f1.append(f1_test_epoch[idx])
        wallclocks.append(np.mean(wallclock_epoch) + enforce_time / num_epochs)
        niters.append(np.mean(n))

        path = f"transformed_data/{dsname}/{state}/{year}/{i}/"
        def remove_if_exists(p):
            if os.path.exists(p):
                os.remove(p)

        for p in [f"{path}X_train.npy", f"{path}X_val.npy", f"{path}X_test.npy",
                  f"{path}y_train.npy", f"{path}y_val.npy", f"{path}y_test.npy"] + [f"models/{dsname}/{state}/{mode}/{j}.pt" for j in range(num_epochs)]:
            remove_if_exists(p)

    df = pd.DataFrame(data = 
                        [[i, idxs[i], wallclocks[i], niters[i],
                          acc[i], prec[i], rec[i], f1[i]] 
                          + ma[i] + mc[i] for i in range(num_exp)],
                      columns = ["Seed", "Best Epoch", 
                                 "Mean Wall Clock Time Per Epoch",
                                 "Mean #Iters Per Epoch",
                                 "Balanced Accuracy", "Precision", 
                                 "Recall", "F1"] 
                                + [f"MA_{crit_to_name(crit)}" for crit in group_crits] 
                                + [f"MC_{crit_to_name(crit)}" for crit in group_crits])
    Path(f"results/{dsname}/{state}/{year}/{mode}{'' if not normal else '_mixup'}").mkdir(parents = True, exist_ok = True)
    df.to_csv(f"results/{dsname}/{state}/{year}/{mode}{'' if not normal else '_mixup'}/lam_{lam}_d_{d}_alpha_{alpha}_niter_{niter}_k_{k}_{setting}.csv")

parser = argparse.ArgumentParser(description='Folktables Experiment')
parser.add_argument('--mode', default='dp', type=str, help='dp/eo/mpma/mpmc')
parser.add_argument('--lam', default=0.5, type=float, help='Lambda for regularization')
parser.add_argument('--dsname', type=str, help='name of folktables DS')
parser.add_argument('--setting', default='all', type=str, help='what groups to calibrate')
parser.add_argument('--state', type=str, help='2-letter state code')
parser.add_argument('--year', type=int, help='year of ACS survey')
parser.add_argument('--num_exp', default=10, type=int, help='number of experimental runs')
parser.add_argument('--d', default=10, type=int, help='discretization parameter (# intervals)')
parser.add_argument('--niter', default=100, type=int, help='number of batch iterations per epoch')
parser.add_argument('--k', default=3, type=int, help='number of worst groups to consider')
parser.add_argument('--alpha', default=1, type=float, help='interpolation rate')
parser.add_argument('--best', action='store_true', help='best hparams')
parser.add_argument('--mixup', action='store_true', help='mixup or fair mixup')

args = parser.parse_args()

if not args.best:
    run_experiments(args.dsname, args.state, args.year, args.mode, args.lam,
                    num_exp = args.num_exp, d = args.d, 
                    niter = args.niter, k = args.k,
                    alpha = args.alpha, setting = args.setting,
                    normal = args.mixup)
else:
    run_experiments(args.dsname, args.state, args.year, args.mode, 
                    lam = best[args.dsname][args.mode]['lam'],
                    num_exp = args.num_exp, 
                    d = best[args.dsname][args.mode]['d'], 
                    niter = args.niter, k = best[args.dsname][args.mode]['k'],
                    alpha = args.alpha, setting = args.setting,
                    normal = args.mixup)
