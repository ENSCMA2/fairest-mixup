import pandas as pd
import os

modes = ["base", "base_mixup", "dp", "eo", "eo_mixup", "fairbase", "mpma", "mpma_mixup", 
		 "mpmc", "mpmc_mixup", "enforce_ma", "enforce_mc", "mixup_enforce_mc"]
years = [2018, 2019, 2021, 2022]

for name in ["income", "employment"]:
	inputs = pd.read_csv(f"transformed_data/{name}_stats.csv")
	for title in ["mc", "bacc", "dis", "srace", "dispoc", "worst", "big", "small"]:
		for setting in ["all", "big", "small", "dis", "one"]:
			data = [[setting, name] for i in range(40)]
			for y in range(len(years)):
				year = years[y]
				pth = f"results/{name}/stateyear_ablations/{year}/{setting}/{title}.csv"
				if os.path.exists(pth):
					outputs = pd.read_csv(pth)[modes]
					for i, tem in outputs.iterrows():
						data[i * 4 + y].extend(tem.tolist())
			if len(data[0]) > 2:
				cols = ["Setting", "Dataset"] + modes
				inputs[cols] = data
				os.makedirs(f"results/{name}/stateyear_ablations/{setting}/", exist_ok = True)
				inputs.to_csv(f"results/{name}/stateyear_ablations/{setting}/{title}.csv")
		dfs = []
		for setting in ["all", "big", "small", "dis", "one"]:
			if os.path.exists(f"results/{name}/stateyear_ablations/{setting}/{title}.csv"):
				comp = pd.read_csv(f"results/{name}/stateyear_ablations/{setting}/{title}.csv")
				dfs.append(comp)
		if len(dfs) > 0:
			bigdf = pd.concat(dfs)
			bigdf.to_csv(f"results/{name}/stateyear_ablations/{title}.csv")

					

