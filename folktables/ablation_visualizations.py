import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import json
import numpy as np
import os
from collections import Counter
from utils import *
from dataset import *

def visualize_race_ablation(name, state, year, lam = 0.5, d = 10, alpha = 1, niter = 20, k = 3):
	print(name, state, year)
	dpath = f"transformed_data/{name}/{state}/{year}/"
	statspath = f"{dpath}race_stats.json"
	m2c = {"base": "black",
		   "eo": "blue",
		   "dp": "red",
		   "mpma": "orange",
		   "mpmc": "green"}
	with open(statspath) as o:
		stats = json.load(o)

	results = pd.read_csv(f"results/{name}/{state}/{year}/mean_lam_{lam}_d_{d}_alpha_{alpha}_niter_{niter}_k_{k}.csv")
	macols = [i for i in results.columns if "MA_RAC3P" in i and "DIS" not in i]
	mccols = [i for i in results.columns if "MC_RAC3P" in i and "DIS" not in i]
	dmacols = [i for i in results.columns if "MA_RAC3P" in i and "DIS" in i]
	dmccols = [i for i in results.columns if "MC_RAC3P" in i and "DIS" in i]
	print(macols)

	fig = plt.figure()
	fig, (ma, mc) = plt.subplots(2, 1)

	ma.set_xlabel("Group Size")
	ma.set_ylabel("MA Violation")
	mc.set_xlabel("Group Size")
	mc.set_ylabel("MC Violation")

	mcconv = [int(i.replace("MC_RAC3P_", "").split("-")[0]) for i in mccols]
	maconv = [int(i.replace("MA_RAC3P_", "").split("-")[0]) for i in macols]

	x = [stats[str(float(i + 1))]["race"] for i in mcconv]
	x += [stats[str(float(i + 1))]["disabled"] for i in mcconv]

	df = pd.DataFrame(data = [x], columns = macols + dmacols)
	df.to_csv(f"results/{name}/{state}/{year}/group_sizes.csv")
	for mode in m2c:
		row = results[results["Mode"] == mode]
		ma_row = row[macols + dmacols].iloc[0].tolist()
		mc_row = row[mccols + dmccols].iloc[0].tolist()
		ma.scatter(x, ma_row, color = m2c[mode], label = mode, s = 5)
		mc.scatter(x, mc_row, color = m2c[mode], label = mode, s = 5)
	mc.legend(loc = 'best')
	ma.legend(loc = 'best')

	fig.savefig(f"results/{name}/{state}/{year}/race_ablation_lam_{lam}_d_{d}_alpha_{alpha}_niter_{niter}_k_{k}.png")
	plt.close(fig)

def visualize_hparam_ablation(name, state, year, mixup = False, ks = None, lams = None, ds = None):
	if ks is None:
		ks = [1, 
			  3, 
			  40, 
			  100
			  ]
	if lams is None:
		lams = [0.25, 
				0.5
				]
	if ds is None:
		ds = [10, 55, 100]
	mixupstr = "mixup_" if mixup else ""
	pth = f"results/{name}/{state}/{year}"
	table = []
	labels = ["lam", "d", "k",]
	for lam in lams:
		for d in ds:
			for k in ks:
				mixupstr2 = "_mixup" if mixup else ""
				rpath = f"{pth}/mean/lam_{lam}_d_{d}_alpha_1_niter_100_k_{k}_all.csv"
				result = pd.read_csv(rpath)
				base_row = result[result["Mode"] == "base"]
				base_bacc = base_row["Balanced Accuracy"].tolist()[0]
				macols = [i for i in result.columns if "MA" in i]
				mccols = [i for i in result.columns if "MC" in i]
				base_ma = np.max(base_row[macols].iloc[0].tolist())
				base_mc = np.max(base_row[mccols].iloc[0].tolist())

				rpath = f"{pth}/mean/lam_{lam}_d_{d}_alpha_1_niter_100_k_{k}_all{mixupstr2}.csv"
				result = pd.read_csv(rpath)

				modes = list(set(result["Mode"]))

				agg_row = [lam, d, k]

				for mode in modes:
					row = result[result["Mode"] == mode]
					bacc = row["Balanced Accuracy"].tolist()[0]
					ma = np.max(row[macols].iloc[0].tolist())
					mc = np.max(row[mccols].iloc[0].tolist())
					bacc_diff = (bacc - base_bacc) / base_bacc
					ma_diff = (ma - base_ma) / base_ma
					mc_diff = (mc - base_mc) / base_mc
					agg_row.extend([bacc_diff, ma_diff, mc_diff])
					if f"{mixupstr}{mode} BAcc PChange" not in labels:
						labels.append(f"{mixupstr}{mode} BAcc PChange")
					if f"{mixupstr}{mode} MA PChange" not in labels:
						labels.append(f"{mixupstr}{mode} MA PChange")
					if f"{mixupstr}{mode} MC PChange" not in labels:
						labels.append(f"{mixupstr}{mode} MC PChange")

				table.append(agg_row)

	df = pd.DataFrame(data = table, columns = labels)
	df.to_csv(f"{pth}/{mixupstr}hparam_ablation.csv")

def visualize_stateyear_ablation(name, setting, modes = None):
	print(name, setting)
	if setting == 'all':
		summfig, ((summ_ma, summ_mc), (summ_bacc, summ_dis), (summ_srace, summ_dispoc), (summ_big, summ_small)) = plt.subplots(4, 2, figsize = (25, 30))
	else:
		summfig, ((summ_ma, summ_mc), (summ_bacc, summ_dis), (summ_srace, summ_dispoc)) = plt.subplots(3, 2, figsize = (25, 30))
	if setting != 'dis':
		worstfig, worstsumm = plt.subplots(1, 1, figsize = (20, 20))

	years = [2018, 2019, 2021, 2022]
	modes = ["base", "base_mixup", "dp", "eo", "eo_mixup", "fairbase", 
			 "mpma", "mpma_mixup", "mpmc", "mpmc_mixup", 
			 "enforce_ma", "enforce_mc", "mixup_enforce_mc"] if modes is None else modes
	mpkeys = list(matplotlib.colors.cnames.keys())
	modes = {modes[i]: mpkeys[i * 2] for i in range(len(modes))}

	tally_ma = {mode: 0 for mode in modes}
	tally_mc = {mode: 0 for mode in modes}
	tally_bacc = {mode: 0 for mode in modes}
	tally_dis = {mode: 0 for mode in modes}
	tally_srace = {mode: 0 for mode in modes}
	tally_dispoc = {mode: 0 for mode in modes}
	tally_big = {mode: 0 for mode in modes}
	tally_small = {mode: 0 for mode in modes}
	tally_worst = {mode: 0 for mode in modes}
	
	for year in years:
		if not os.path.exists(f"results/{name}/stateyear_ablations/{year}/{setting}"):
			os.makedirs(f"results/{name}/stateyear_ablations/{year}/{setting}", exist_ok = True)
		states = {"CA": {}, 
				  "TX": {}, 
				  "FL": {}, 
				  "NY": {}, 
				  "PA": {}, 
				  "IL": {}, 
				  "OH": {}, 
				  "GA": {}, 
				  "NC": {}, 
				  "MI": {},}
		if setting == 'all':
			for state in states:
				bigs = folktables_ds(seed = 0, name = name, state = state, 
									 year = year, setting = 'big', 
									 full_set = False)
				smalls = folktables_ds(seed = 0, name = name, state = state, 
									   year = year, setting = 'small', 
									   full_set = False)
				states[state]["bigs"] = bigs
				states[state]["smalls"] = smalls
		included_states = list(states.keys())
		if setting == 'all':
			fig, ((ma, mc), (bacc, dis), (srace, dispoc), (big, small)) = plt.subplots(4, 2, figsize = (25, 30))
		else:
			fig, ((ma, mc), (bacc, dis), (srace, dispoc)) = plt.subplots(3, 2, figsize = (25, 30))
		if setting != 'dis':
			fig2, worstmc = plt.subplots(1, 1, figsize = (20, 20))
			all_worst = []
		all_ma, all_mc, all_bacc, all_dis, all_srace, all_dispoc, all_big, all_small = [], [], [], [], [], [], [], []
		for mode in modes:	
			data_ma, data_mc, data_bacc, data_dis, data_srace, data_dispoc, data_big, data_small = [], [], [], [], [], [], [], []
			if setting != 'dis':
				data_worst = []

			def get_mean(row):
				return np.mean(row.iloc[0].tolist())

			def get_worst(row):
				return np.max(row.iloc[0].tolist())

			for state in states:
				if setting == 'all':
					bigcols = [crit_to_name(ncv) for ncv in states[state]["bigs"]]
					smallcols = [crit_to_name(ncv) for ncv in states[state]["smalls"]]
					bigmccols = [f"MC_{i}" for i in bigcols if "DIS" not in i]
					smallmccols = [f"MC_{i}" for i in smallcols if "DIS" not in i]
					bigdmccols = [f"MC_{i}" for i in bigcols if "DIS" in i and "RAC3P" in i]
					smalldmccols = [f"MC_{i}" for i in smallcols if "DIS" in i and "RAC3P" in i]

				pth = f"results/{name}/{state}/{year}/mean/"
				stng = f"{setting}.csv"
				results = pd.read_csv(f"{pth}{stng}")

				if setting != 'dis':
					macols = [i for i in results.columns if "MA_RAC3P" in i and "DIS" not in i]
					mccols = [i for i in results.columns if "MC_RAC3P" in i and "DIS" not in i]
					dmacols = [i for i in results.columns if "MA_RAC3P" in i and "DIS" in i]
					dmccols = [i for i in results.columns if "MC_RAC3P" in i and "DIS" in i]
				else:
					macols = ["MA_DIS_-1"]
					mccols = ["MC_DIS_-1"]
					dmacols = []
					dmccols = []

				row = results[results["Mode"] == mode]

				row_ma = get_mean(row[macols + dmacols + ["MA_DIS_-1"]])
				data_ma.append(row_ma)

				row_mc = get_mean(row[mccols + dmccols + ["MC_DIS_-1"]])
				data_mc.append(row_mc)

				if setting != 'dis':
					row_worst = get_worst(row[mccols + dmccols + ["MC_DIS_-1"]])
					data_worst.append(row_worst)

				row_bacc = row["Balanced Accuracy"].iloc[0]
				data_bacc.append(row_bacc)

				if len(dmccols) > 0:
					row_srace = row[mccols[-1]].iloc[0]
					data_srace.append(row_srace)

					row_dispoc = np.mean(row[dmccols])
					data_dispoc.append(row_dispoc)
				else:
					included_states = [s for s in included_states if s != state]

				row_dis = row["MC_DIS_-1"].iloc[0]
				data_dis.append(row_dis)

				if setting == 'all':
					bigstuff = [i for i in bigmccols + bigdmccols if i in results.columns]
					row_big = get_mean(row[bigstuff])
					data_big.append(row_big)

					smallstuff = [i for i in smallmccols + smalldmccols if i in results.columns]
					row_small = get_mean(row[smallstuff])
					data_small.append(row_small)

			def make_scatter(plot, data, st = list(states.keys())):
				plot.scatter(st, data, color = modes[mode], label = mode, s = 75)

			for (plot, data, title) in [(ma, data_ma, "ma"), 
										(mc, data_mc, "mc"), 
								 		(bacc, data_bacc, "bacc"), 
								 		(dis, data_dis, "dis")]:
				if len(data) > 0:
					make_scatter(plot, data)
					df = pd.DataFrame(data = [data], 
									  columns = list(states.keys()))
					df.to_csv(f"results/{name}/stateyear_ablations/{year}/{setting}/{title}_{mode}.csv")
					plot.set_xlabel("State")

			if len(data_srace) > 0:
				for (plot, data, title) in [(srace, data_srace, "srace"), 
										    (dispoc, data_dispoc, "dispoc")]:
					make_scatter(plot, data, included_states)
					df = pd.DataFrame(data = [data], 
									  columns = included_states)
					df.to_csv(f"results/{name}/stateyear_ablations/{year}/{setting}/{title}_{mode}.csv")
					plot.set_xlabel("State")

			if setting == 'all':
				for (plot, data, title) in [(big, data_big, "big"), 
											(small, data_small, "small")]:
					make_scatter(plot, data)
					df = pd.DataFrame(data = [data], 
									  columns = list(states.keys()))
					df.to_csv(f"results/{name}/stateyear_ablations/{year}/{setting}/{title}_{mode}.csv")
					plot.set_xlabel("State")
					plot.set_ylabel("MC Violation")
					plot.legend()

			if setting != 'dis':
				make_scatter(worstmc, data_worst)
				df = pd.DataFrame(data = [data_worst], columns = list(states.keys()))
				df.to_csv(f"results/{name}/stateyear_ablations/{year}/{setting}/worst_{mode}.csv")
				worstmc.set_xlabel("State")
				worstmc.set_ylabel("MC Violation")
				worstmc.legend()

			for plot in (mc, dis, srace, dispoc):
				plot.set_ylabel("MC Violation")

			all_ma.append(data_ma)
			all_mc.append(data_ma)
			all_bacc.append(data_bacc)
			all_dis.append(data_dis)
			if setting != 'dis':
				all_worst.append(data_worst)
			if len(data_srace) > 0:
				all_srace.append(data_srace)
				all_dispoc.append(data_dispoc)
			if setting == 'all':
				all_big.append(data_big)
				all_small.append(data_small)

		titles = ["ma", "mc", "bacc", "dis"]
		if setting != 'dis':
			titles += ["srace", "dispoc", "worst"]
		if setting == 'all':
			titles += ["big", "small"]
		for title in titles:
			big_data = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
			big_data = [[s] for s in big_data]
			labels = ["State"] + list(modes.keys())
			for mode in modes:
				tm = pd.read_csv(f"results/{name}/stateyear_ablations/{year}/{setting}/{title}_{mode}.csv")
				for i in range(len(big_data)):
					state = big_data[i][0]
					if state in tm.columns:
						stat = tm[state].iloc[0]
						big_data[i].append(stat)
					else:
						big_data[i].append("N/A")
			df = pd.DataFrame(data = big_data, columns = labels)
			df.to_csv(f"results/{name}/stateyear_ablations/{year}/{setting}/{title}.csv")

		for title in titles:
			for mode in modes:
				os.remove(f"results/{name}/stateyear_ablations/{year}/{setting}/{title}_{mode}.csv")

		ma.set_ylabel("MA Violation")
		bacc.set_ylabel("Balanced Accuracy")

		ma.set_title("Mean MA Violation Across Racial Groups and Race x Disability Groups")
		mc.set_title("Mean MC Violation Across Racial Groups and Race x Disability Groups")
		bacc.set_title("Balanced Accuracy")
		dis.set_title("MC Violation For Disabled Group")
		if setting != 'dis':
			worstmc.set_title("Worst MC Violation")
		if len(all_srace) > 0:
			srace.set_title("MC Violation of Least Frequent Racial Group")
			dispoc.set_title("Mean MC Violation Across Groups of Disabled POC")
		if setting == 'all':
			big.set_title("Mean MC Violation Across Groups Bigger than 0.25pct of Population")
			small.set_title("Mean MC Violation Across Groups Smaller than 0.25pct of Population")

		for p in (ma, mc, bacc, dis):
			p.legend()
		if len(all_srace) > 0:
			srace.legend()
			dispoc.legend()
		if setting != 'dis':
			fig2.savefig(f"results/{name}/stateyear_ablations/{year}/{setting}/worst.png")
			plt.close(fig2)
		fig.savefig(f"results/{name}/stateyear_ablations/{year}/{setting}/plots.png")
		plt.close(fig)

		def update_tally(tally, arr, maximum = False):
			argmaxes = Counter((np.argmax if maximum else np.argmin)(arr, axis = 0))
			for i in range(len(modes)):
				tally[list(modes.keys())[i]] += argmaxes[i]

		for (tally, arr) in [(tally_ma, all_ma), (tally_mc, all_mc),
							 (tally_dis, all_dis),] + ([(tally_srace, all_srace),
							 (tally_dispoc, all_dispoc), (tally_worst, all_worst)] if setting != 'dis' and len(all_srace) > 0 else []):
			update_tally(tally, arr)

		if setting == 'all':
			for (tally, arr) in [(tally_big, all_big), 
								 (tally_small, all_small)]:
				update_tally(tally, arr)

		update_tally(tally_bacc, all_bacc, maximum = True)

	if not os.path.exists(f"results/{name}/tallies"):
		os.mkdir(f"results/{name}/tallies")

	ptt = [(summ_ma, tally_ma, "ma"), (summ_mc, tally_mc, "mc"), 
		   (summ_bacc, tally_bacc, "bacc"), (summ_dis, tally_dis, "dis")] 
	if setting != 'dis' :
		ptt += [(worstsumm, tally_worst, "worst")]
		if tally_srace['base'] > 0:
			ptt += [(summ_srace, tally_srace, "srace"), (summ_dispoc, tally_dispoc, "dispoc")]
	if setting == 'all':
		ptt += [(summ_big, tally_big, "big"), (summ_small, tally_small, "small")]

	for (plot, tally, title) in ptt:
		with open(f"results/{name}/tallies/{name}_{setting}.json", "w") as o:
			json.dump(tally, o)
		k = list(tally.keys())
		v = list(tally.values())
		plot.bar(k, v)
		plot.set_xlabel("Mode")
		plot.set_ylabel("Frequency of Best Relative Value")

	summ_ma.set_title("Mean MA Violation Across Race & Race x Disability")
	summ_mc.set_title("Mean MC Violation Across Race & Race x Disability")
	summ_bacc.set_title("Balanced Accuracy")
	summ_dis.set_title("MC Violation for Disabled Group")
	if setting != 'dis':
		summ_srace.set_title("MC Violation for Least Frequent Racial Group")
		summ_dispoc.set_title("Mean MC Violation Across Race x Disabled Groups")
		worstsumm.set_title("Worst MC Violation")
	if setting == 'all':
		summ_big.set_title("Mean MC Violation Across Groups Bigger than 0.25pct of Population")
		summ_small.set_title("Mean MC Violation Across Groups Smaller than 0.25pct of Population")

	if not os.path.exists(f"results/{name}/summary"):
		os.mkdir(f"results/{name}/summary")
	summfig.savefig(f"results/{name}/summary/{setting}.png")
	plt.close(summfig)
	if setting != 'dis':
		worstfig.savefig(f"results/{name}/summary/worst_{setting}.png")
		plt.close(worstfig)

def visualize_setting_ablation(name, state, year, lam = None, missing = None):
	settings = ["all", "big", "small", "dis", "one"]
	colors = ["red", "orange", "green", "brown", "black"]
	modes = ["dp", "eo", "mpma", "mpmc"] if missing is None else ["eo", "mpma", "mpmc"]
	labels = ["base"] + modes
	pth = f"results/{name}/{state}/{year}/mean"
	baccs = {}
	mcs = {}
	fig, (bacc_plt, mc_plt) = plt.subplots(2, 1, figsize = (10, 10))
	for setting in settings:
		results = pd.read_csv(f"{pth}/{setting}.csv")
		hparams = best[name]['base']
		base = pd.read_csv(f"{pth}/lam_{hparams['lam']}_d_{hparams['d']}_alpha_1_niter_100_k_{hparams['k']}_all.csv")
		base = base[base["Mode"] == 'base']
		base_bacc = base["Balanced Accuracy"].iloc[0]
		mccols = [c for c in base.columns if "MC_" in c]
		base_mc = np.mean(base[mccols].iloc[0].tolist())
		baccs["base"] = [base_bacc] if "base" not in baccs.keys() else baccs["base"] + [base_bacc]
		mcs["base"] = [base_mc] if "base" not in mcs.keys() else mcs["base"] + [base_mc]
		
		for mode in modes:
			mccols = [c for c in results.columns if "MC_" in c]
			res = results[results["Mode"] == mode]
			mc = np.mean(res[mccols].iloc[0].tolist())
			bacc = res["Balanced Accuracy"].iloc[0]
			baccs[mode] = [bacc] if mode not in baccs.keys() else baccs[mode] + [bacc]
			mcs[mode] = [mc] if mode not in mcs.keys() else mcs[mode] + [mc]
	for i in range(len(labels)):
		bacc_plt.scatter(settings, baccs[labels[i]], color = colors[i], label = labels[i])
		mc_plt.scatter(settings, mcs[labels[i]], color = colors[i], label = labels[i])
	bacc_plt.legend()
	mc_plt.legend()
	bacc_plt.set_title("Balanced Accuracy For Various Group Settings")
	mc_plt.set_title("Mean MC Violation for Various Group Settings")
	for p in (bacc_plt, mc_plt):
		p.set_xlabel("Setting")
	mc_plt.set_ylabel("MC Violation")
	bacc_plt.set_ylabel("Balanced Accuracy")


	fig.savefig(f"{pth}/setting_ablation.png")
	plt.close(fig)

def visualize_efficiency(name, setting):
	modes = ["enforce_mc", "mixup_enforce_mc"]
	states = {"CA": {}, "TX": {}, "FL": {}, "NY": {}, "PA": {}, "IL": {}, 
			  "OH": {}, "GA": {}, "NC": {}, "MI": {},}
	years = [2018, 2019, 2021, 2022]
	data1 = []
	data2 = []
	for year in years:
		for state in states:
			point1 = [year, state]
			point2 = [year, state]
			pth = f"results/{name}/{state}/{year}/mean/{setting}.csv"
			result = pd.read_csv(pth)
			col1 = "Mean Wall Clock Time Per Epoch"
			col2 = "Mean #Iters Per Epoch"
			for mode in modes:
				row = result[result["Mode"] == mode]
				wt = row[col1].iloc[0]
				point1.append(wt)
				niter = row[col2].iloc[0]
				point2.append(niter)
			data1.append(point1)
			data2.append(point2)
	df1 = pd.DataFrame(data = data1, columns = ["Year", "State"] + modes)
	df2 = pd.DataFrame(data = data2, columns = ["Year", "State"] + modes)
	df1.to_csv(f"results/{name}/stateyear_ablations/{setting}/wallclock.csv")
	df2.to_csv(f"results/{name}/stateyear_ablations/{setting}/num_epochs.csv")
# visualize_setting_ablation("employment", "CA", 2022)
# visualize_setting_ablation("income", "CA", 2022)
for setting in ['big', 'small', 'dis', 'one', 'all']:
	visualize_stateyear_ablation("employment", setting)
	visualize_stateyear_ablation("income", setting)
# visualize_hparam_ablation("employment", "CA", 2022)
# visualize_hparam_ablation("income", "CA", 2022)
# visualize_race_ablation("income", "CA", 2022)
# visualize_race_ablation("employment", "CA", 2022)
# for ds in ["income", "employment"]:
# 	visualize_hparam_ablation(ds, "CA", 2022, mixup = True, ds = [10])
for ds in ["income", "employment"]:
	for setting in ["big", "small", "dis", "one", "all"]:
		visualize_efficiency(ds, setting)