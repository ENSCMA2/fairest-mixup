from folktables import *
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import folktables
import requests
import json
from pathlib import Path
import os

yeses = ["Female", "Yes", "Native", "With a disability"]

# source: https://github.com/ProgBelarus/BatchMultivalidConformal/blob/main/experiments/FolktablesExperiment.ipynb
def adult_and_race_filter(data):
    """Mimic the filters in place for Adult data.
    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['PINCP'] > 100]
    df = df[df['WKHP'] > 0]
    return df

median_incomes = {"CA": {2022: 39812,
						 2018: 30797,
						 2019: 31960,
						 2020: 33719,
						 2021: 36281},
				  "TX": {2022: 36538,
				  		 2021: 33644,
				  		 2020: 31462,
				  		 2019: 30596,
				  		 2018: 29460},
				  "FL": {2022: 34062,
				  		 2021: 31169,
				  		 2020: 29159,
				  		 2019: 27936,
				  		 2018: 26895},
				  "PA": {2022: 37284,
				  		 2021: 34229,
				  		 2020: 32156,
				  		 2019: 31099,
				  		 2018: 30142},
				  "NY": {2022: 39551,
				  		 2021: 36302,
				  		 2020: 34386,
				  		 2019: 32320,
				  		 2018: 31248},
				  "IL": {2022: 40112,
				  		 2021: 36729,
				  		 2020: 34489,
				  		 2019: 32495,
				  		 2018: 31597},
				  "OH": {2022: 35981,
				  		 2021: 32980,
				  		 2020: 31078,
				  		 2019: 30171,
				  		 2018: 28814},
				  "NC": {2022: 34850,
				  		 2021: 31840,
				  		 2020: 30106,
				  		 2019: 28836,
				  		 2018: 27458},
				  "GA": {2022: 35753,
				  		 2021: 32657,
				  		 2020: 30916,
				  		 2019: 29958,
				  		 2018: 28283},
				  "MI": {2022: 35350,
				  		 2021: 32451,
				  		 2020: 30723,
				  		 2019: 29672,
				  		 2018: 28178}}

def bin_transform(arr, i, info):
	for j in range(len(arr[:, i])):
		arr[j, i] = info[arr[j, i]]

def _one_hot(a, num_bins=10):
    return np.squeeze(np.eye(num_bins)[a.reshape(-1).astype(np.int32)])

def folktables_ds(seed, name, state, year, setting = 'all', full_set = True):
	dpath = f"transformed_data/{name}/{state}/{year}/{seed}/"
	Path(dpath).mkdir(parents = True, exist_ok = True)
	data_source = ACSDataSource(survey_year=str(year), 
								horizon='1-Year', 
								survey='person')
	acs_data = data_source.get_data(states=[state], download=True)
	Employment = folktables.BasicProblem(features = ['RAC3P', # categorical
									        		 'RELSHIPP' if year != 2018 else 'RELP',
									        		 'MIG',
									        		 'MIL',
									        		 'ANC',
									        		 'ESP',
									        		 'CIT',
									        		 'MAR',
									        		 'DREM',
									        		 'AGEP', # continuous
									        		 'SCHL',
									        		 'SEX', # binary
									        		 'DEAR',
									        		 'DEYE',
									        		 'NATIVITY',
									        		 'DIS',],
									     target = 'ESR',
									     target_transform = lambda x: x == 1,
									     preprocess = lambda x: x,
									     postprocess = lambda x: np.nan_to_num(x, nan = 0),)
	Income = folktables.BasicProblem(features=['RAC3P', # categorical
		    								   'COW',
										       'MAR',
										       'OCCP',
										       'RELSHIPP' if year != 2018 else 'RELP',
										       'POBP',
										       'AGEP', # continuous
										       'SCHL',
										       'WKHP',
										       'SEX', # binary
										       'DIS',],
									 target ='PINCP',
									 target_transform = lambda x: x > median_incomes[state][year],
									 preprocess = adult_and_race_filter,
									 postprocess = lambda x: np.nan_to_num(x, -1),)

	name_to_ds = {"employment": (Employment, 9, 11),
			  	  "income": (Income, 6, 9)}
	ds, first_cont_column, first_bin_column = name_to_ds[name]
	for feature in ds.features:
		pth = f"features/{feature}.json"
		if not os.path.exists(pth):
			url = f"https://api.census.gov/data/{year}/acs/acs1/pums/variables/{feature}.json"
			info = requests.get(url).json()
			if not os.path.exists("features"):
				os.mkdir("features")
			with open(pth, "w") as o:
				json.dump(info, o)

	features, label, group = ds.df_to_numpy(acs_data)
	race_set = Counter(features[:, 0])

	# is white alone
	ordered = [item for item in race_set.most_common(len(race_set)) if int(item[0]) != 1]
	dis_counts = []
	for item in ordered:
		this_race = [p for p in features if p[0] == item[0]]
		num_dis = len([p for p in this_race if int(p[-1]) == 1])
		dis_counts.append(num_dis)
	ordered = [ordered[i] for i in range(len(ordered)) if dis_counts[i] >= 20]
	bigger = [ordered[i] for i in range(len(ordered)) if ordered[i][1] >= 0.0025 * features.shape[0]]
	smaller = [ordered[i] for i in range(len(ordered)) if ordered[i][1] < 0.0025 * features.shape[0]]

	if not os.path.exists(f"transformed_data/{name}/{state}/{year}/race_stats.json"):
		with open(f"transformed_data/{name}/{state}/{year}/race_stats.json", "w") as o:
			json.dump({ordered[i][0]: {"race": ordered[i][1], "disabled": dis_counts[i]}for i in range(len(ordered))}, o)
	smallest_race = int(ordered[-1][0])

	if setting == 'all':
		group_crits = [[("RAC3P", int(race[0] - 1), 1)] for race in ordered] + [[("DIS", -1, 1)]] + [[("RAC3P", int(race[0] - 1), 1), ("DIS", -1, 1)] for race in ordered]
	elif setting == 'dis':
		group_crits = [[("DIS", -1, 1)]]
	elif setting == 'big':
		group_crits = [[("RAC3P", int(race[0] - 1), 1)] for race in bigger] + [[("DIS", -1, 1)]] + [[("RAC3P", int(race[0] - 1), 1), ("DIS", -1, 1)] for race in bigger]
	elif setting == 'small':
		group_crits = [[("RAC3P", int(race[0] - 1), 1)] for race in smaller] + [[("DIS", -1, 1)]] + [[("RAC3P", int(race[0] - 1), 1), ("DIS", -1, 1)] for race in smaller]
	else:
		group_crits = [[("RAC3P", smallest_race - 1, 1)], [("DIS", -1, 1)], [("RAC3P", smallest_race - 1, 1), ("DIS", -1, 1)]]

	if not full_set:
		return group_crits

	if not (os.path.exists(f"{dpath}X_train")
		    and os.path.exists(f"{dpath}y_train")
		    and os.path.exists(f"{dpath}X_val")
		    and os.path.exists(f"{dpath}y_val")
		    and os.path.exists(f"{dpath}X_test")
		    and os.path.exists(f"{dpath}y_test")):
		X_trainval, X_test, y_trainval, y_test = train_test_split(features, label,
																  test_size = 0.2,
																  random_state = seed)
		X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,
														  test_size = 0.25,
														  random_state = seed)

		categorical_features_train = X_train[:, :first_cont_column].astype(int)
		categorical_features_val = X_val[:, :first_cont_column].astype(int)
		categorical_features_test = X_test[:, :first_cont_column].astype(int)

		onehot_features_train = None
		onehot_features_val = None
		onehot_features_test = None

		for i in range(categorical_features_train.shape[1]):
			num_cats = max(np.max(categorical_features_train[:, i]),
						   np.max(categorical_features_val[:, i]),
						   np.max(categorical_features_test[:, i]))
			onehot_train = np.zeros((categorical_features_train.shape[0], num_cats))
			onehot_val = np.zeros((categorical_features_val.shape[0], num_cats))
			onehot_test = np.zeros((categorical_features_test.shape[0], num_cats))
			for j in range(len(categorical_features_train[:, i])):
				val = categorical_features_train[j, i]
				onehot_train[j, val - 1] = 1
			for j in range(len(categorical_features_val[:, i])):
				val = categorical_features_val[j, i]
				onehot_val[j, val - 1] = 1
			for j in range(len(categorical_features_test[:, i])):
				val = categorical_features_test[j, i]
				onehot_test[j, val - 1] = 1
			onehot_features_train = onehot_train if onehot_features_train is None else np.hstack((onehot_features_train, onehot_train))
			onehot_features_val = onehot_val if onehot_features_val is None else np.hstack((onehot_features_val, onehot_val))
			onehot_features_test = onehot_test if onehot_features_test is None else np.hstack((onehot_features_test, onehot_test))

		# scale continuous features for unit variance
		continuous_features_train = X_train[:, first_cont_column:first_bin_column]
		ss = StandardScaler().fit(continuous_features_train)
		scaled_features_train = ss.transform(continuous_features_train)
		continuous_features_val = X_val[:, first_cont_column:first_bin_column]
		scaled_features_val = ss.transform(continuous_features_val)
		continuous_features_test = X_test[:, first_cont_column:first_bin_column]
		scaled_features_test = ss.transform(continuous_features_test)

		binary_features_train = X_train[:, first_bin_column:].astype(int)
		binary_features_val = X_val[:, first_bin_column:].astype(int)
		binary_features_test = X_test[:, first_bin_column:].astype(int)

		bin_idx_to_name = {i - first_bin_column: ds.features[i] for i in range(first_bin_column, len(ds.features))}
		for i in range(binary_features_train.shape[1]):
			with open(f"features/{bin_idx_to_name[i]}.json") as o:
				info = json.load(o)["values"]["item"]
			info = {int(key): int(info[key] in yeses) for key in info}
			bin_transform(binary_features_train, i, info)
			bin_transform(binary_features_val, i, info)
			bin_transform(binary_features_test, i, info)

		X_train = np.hstack((onehot_features_train, scaled_features_train, binary_features_train))
		X_val = np.hstack((onehot_features_val, scaled_features_val, binary_features_val))
		X_test = np.hstack((onehot_features_test, scaled_features_test, binary_features_test))

		np.save(f"{dpath}X_train", X_train)
		np.save(f"{dpath}y_train", y_train)
		np.save(f"{dpath}X_val", X_val)
		np.save(f"{dpath}y_val", y_val)
		np.save(f"{dpath}X_test", X_test)
		np.save(f"{dpath}y_test", y_test)

	X_train = np.load(f"{dpath}X_train.npy")
	y_train = np.load(f"{dpath}y_train.npy")
	X_val = np.load(f"{dpath}X_val.npy")
	y_val = np.load(f"{dpath}y_val.npy")
	X_test = np.load(f"{dpath}X_test.npy")
	y_test = np.load(f"{dpath}y_test.npy")

	return X_train, X_val, X_test, y_train, y_val, y_test, group_crits, smallest_race
