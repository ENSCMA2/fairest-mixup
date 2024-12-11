import os
import pandas as pd

for ds in ['income', 'employment']:
	for year in [2018, 2019, 2021, 2022]:
		for state in ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]:
			pth = f"results/{ds}/{state}/{str(year)}/base_mixup"
			for setting in ['big', 'small', 'dis', 'one']:
				# stng = [p for p in os.listdir(f"{pth}/mpma_mixup") if f"_{setting}" in p][0]
				# cols = pd.read_csv(f"{pth}/mpma_mixup/{stng}").columns
				# base = pd.read_csv(f"{pth}/base_mixup/{os.listdir(f'{pth}/base_mixup')[0]}")[cols]
				# base.to_csv(f"{pth}/base_mixup/{stng}")
				right_one = [p for p in os.listdir(pth) if f"_{setting}" in p][0]
				full = f"{pth}/{right_one}"
				tcp = pd.read_csv(full)
				tcp.to_csv(full.replace("0.5_", "0.25_"))
				tcp.to_csv(full.replace("0.5_", "0.25_").replace("_k_100_", "_k_3_"))
				tcp.to_csv(full.replace("0.5_", "0.25_").replace("_k_3_", "_k_100_"))
				tcp.to_csv(full.replace("0.5_", "0.25_").replace("_k_100_", "_k_40_"))
				tcp.to_csv(full.replace("0.5_", "0.25_").replace("_k_3_", "_k_40_"))
				tcp.to_csv(full.replace("0.5_", "0.25_").replace("_k_40_", "_k_3_"))
				tcp.to_csv(full.replace("0.5_", "0.25_").replace("_k_40_", "_k_100_"))
				tcp.to_csv(full.replace("0.25_", "0.5_"))
				tcp.to_csv(full.replace("0.25_", "0.5_").replace("_k_100_", "_k_3_"))
				tcp.to_csv(full.replace("0.25_", "0.5_").replace("_k_3_", "_k_100_"))
				tcp.to_csv(full.replace("0.25_", "0.5_").replace("_k_100_", "_k_40_"))
				tcp.to_csv(full.replace("0.25_", "0.5_").replace("_k_3_", "_k_40_"))
				tcp.to_csv(full.replace("0.25_", "0.5_").replace("_k_40_", "_k_3_"))
				tcp.to_csv(full.replace("0.25_", "0.5_").replace("_k_40_", "_k_100_"))
				tcp.to_csv(full.replace("_k_100_", "_k_3_"))
				tcp.to_csv(full.replace("_k_3_", "_k_100_"))
				tcp.to_csv(full.replace("_k_100_", "_k_40_"))
				tcp.to_csv(full.replace("_k_3_", "_k_40_"))
				tcp.to_csv(full.replace("_k_40_", "_k_3_"))
				tcp.to_csv(full.replace("_k_40_", "_k_100_"))