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

def run_experiments(dsname, state, year, 
                    mode='mpma', 
                    lam = 0.5, num_exp = 10, num_epochs = 10,
                    d = 10, alpha = 1, niter = 100):
    '''
    Retrain each model for 10 times and report performance and fairness metrics.
    '''

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
        print('On experiment', i)
        # get train/test data
        ds = folktables_ds(seed = i, name = dsname, 
                           state = state, year = year)
        X_train, X_val, X_test, y_train, y_val, y_test, group_crits, smallest_race = ds

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
        for j in tqdm(range(num_epochs)):
            train = funcs[mode]
            # change line 68 to be whatever baseline training method
            wallclock, niter = train(model, criterion, optimizer, 
                                     X_train, group_crits, y_train, 
                                     lam, d, alpha = alpha, niter = niter)
            acc_val, pre_val, rec_val, f1_val, mc_val, ma_val = evaluate_mcma(model, X_val, y_val, group_crits, d)
            acc_test, prec_test, rec_test, f1_test, mc_test, ma_test = evaluate_mcma(model, X_test, y_test, group_crits, d)

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
        mc.append(mc_test_epoch[idx])
        ma.append(ma_test_epoch[idx])
        acc.append(acc_test_epoch[idx])
        prec.append(prec_test_epoch[idx])
        rec.append(rec_test_epoch[idx])
        f1.append(f1_test_epoch[idx])
        wallclocks.append(np.mean(wallclock_epoch))
        niters.append(np.mean(niter_epoch))

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
    Path(f"results/{dsname}/{state}/{year}/{mode}").mkdir(parents = True, 
                                                          exist_ok = True)
    df.to_csv(f"results/{dsname}/{state}/{year}/{mode}/lam_{lam}_d_{d}_alpha_{alpha}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adult Experiment')
    parser.add_argument('--mode', default='dp', type=str, help='dp/eo/mpma/mpmc')
    parser.add_argument('--lam', default=0.5, type=float, help='Lambda for regularization')
    parser.add_argument('--dsname', type=str, help='name of folktables DS')
    parser.add_argument('--state', type=str, help='2-letter state code')
    parser.add_argument('--year', type=int, help='year of ACS survey')
    parser.add_argument('--num_exp', default=10, type=int, help='number of experimental runs')
    parser.add_argument('--d', default=10, type=int, help='discretization parameter (# intervals)')
    parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('--niter', default=100, type=int, help='number of batch iterations per epoch')
    args = parser.parse_args()

    run_experiments(args.dsname, args.state, args.year, args.mode, args.lam,
                    num_exp = args.num_exp, d = args.d, 
                    num_epochs = args.num_epochs, niter = args.niter)

