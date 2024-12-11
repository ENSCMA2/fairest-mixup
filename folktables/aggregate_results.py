import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from utils import *
import os

def create_mean_csv(ds, state, year, alpha = 1, niter = 100,
					root_dir = '.', setting = 'all'):
	
	modes = ["base", "base_mixup", "dp", "eo", "eo_mixup", "fairbase", 
			 "mpma", "mpma_mixup", "mpmc", "mpmc_mixup", 
			 "enforce_ma", "enforce_mc", "mixup_enforce_mc"]

	pth = f"{root_dir}/results/{ds}/{state}/{year}"
	columns = ["Mode"]
	data = []

	for mode in modes:
		portion = best[ds][mode] if 'base' not in mode else best[ds]['mpma']
		lam_ = portion['lam'] if "fairbase" not in mode else 0.0
		d_ = portion['d']
		k_ = portion['k'] if "fairbase" not in mode else 3
		description = f"lam_{lam_}_d_{d_}_alpha_{alpha}_niter_{niter}_k_{k_}_{setting}"
		sheet = pd.read_csv(f"{pth}/{mode}/{description}.csv")
		cols = [c for c in sheet.columns if c != "Seed" and "Unnamed" not in c]
		if len(columns) == 1:
			columns += cols
		to_average = sheet[cols].to_numpy()
		means = np.mean(to_average, axis = 0)
		data.append([mode] + means.tolist() + [np.max(sheet["Best Epoch"])])
	df = pd.DataFrame(data = data, columns = columns + ["Max Best Epoch"])
	if not os.path.exists(f"{pth}/mean"):
		os.mkdir(f"{pth}/mean")
	df.to_csv(f"{pth}/mean/{setting}.csv")

	baselines = ["base", "base_mixup", "dp", "eo", "enforce_ma", "enforce_mc"]
	novel_methods = ["fairbase", "mpma", 
					 "mpma_mixup", "mpmc", "mpmc_mixup",]
	columns = ["Baseline", "New Method"]
	tdata = []

	for baseline in baselines:
		for method in novel_methods:
			portion = best[ds][method]
			lam_ = portion['lam'] if 'fairbase' not in method else 0.0
			d_ = portion['d']
			k_ = portion['k'] if 'fairbase' not in method or method == 'base' else 3
			description_method = f"lam_{lam_}_d_{d_}_alpha_{alpha}_niter_{niter}_k_{k_}_{setting}"
			portion = best[ds][baseline] if baseline != 'base' else best[ds]['mpma']
			lam_ = portion['lam'] if 'fairbase' not in baseline else 0.0
			d_ = portion['d']
			k_ = portion['k'] if 'fairbase' not in baseline else 3
			description_base = f"lam_{lam_}_d_{d_}_alpha_{alpha}_niter_{niter}_k_{k_}_{setting}"
			baseline_sheet = pd.read_csv(f"{pth}/{baseline}/{description_base}.csv")
			method_sheet = pd.read_csv(f"{pth}/{method}/{description_method}.csv")
			cols = [c for c in method_sheet.columns if c != "Seed" and "Unnamed" not in c]
			if len(columns) == 2:
				columns += cols
			baseline_results = baseline_sheet[cols].to_numpy()
			method_results = method_sheet[cols].to_numpy()
			pvals = []
			for i in range(baseline_results.shape[1]):
				pval = ttest_ind(baseline_results[:, i], method_results[:, i]).pvalue
				pvals.append(pval)
			tdata.append([baseline, method] + pvals)

	df = pd.DataFrame(data = tdata, columns = columns)
	if not os.path.exists(f"{pth}/t"):
		os.mkdir(f"{pth}/t")
	df.to_csv(f"{pth}/t/{setting}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Result Aggregation')
    parser.add_argument('--lam', default=0.5, type=float, help='Lambda for regularization')
    parser.add_argument('--ds', type=str, help='name of folktables DS')
    parser.add_argument('--state', type=str, help='2-letter state code')
    parser.add_argument('--year', type=int, help='year of ACS survey')
    parser.add_argument('--d', default=10, type=int, help='discretization parameter (# intervals)')
    parser.add_argument('--alpha', default=1, type=float, help='alpha interpolation parameter')
    parser.add_argument('--niter', default=100, type=int, help='alpha interpolation parameter')
    parser.add_argument('--k', default=3, type=int, help='number of worst groups to consider')
    parser.add_argument('--root_dir', default='.', type=str, help='root dir of results within results')
    parser.add_argument('--setting', default='all', type=str, help='group crit setting')
    parser.add_argument('--fairbase', action='store_true', help='fairbase or lam > 0')
    parser.add_argument('--mixup', action='store_true', help='mixup or regular')
    args = parser.parse_args()

    create_mean_csv(ds = args.ds, state = args.state, year = args.year,
	    			alpha = args.alpha, niter = args.niter,
	    			root_dir = args.root_dir, setting = args.setting)