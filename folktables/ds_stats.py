import pandas as pd
import os
import json
import numpy as np
from dataset import *

def get_stats(name, state, year):
	data_source = ACSDataSource(survey_year=str(year), 
								horizon='1-Year', 
								survey='person')
	acs_data = data_source.get_data(states = [state], download = True)
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
	features, label, group = ds.df_to_numpy(acs_data)
	size = len(label)
	race_set = Counter(features[:, 0])

	# is white alone
	ordered = [item for item in race_set.most_common(len(race_set)) if int(item[0]) != 1]
	dis_count = len([p for p in features if int(p[-1]) == 1])
	dis_counts = []
	for item in ordered:
		this_race = [p for p in features if p[0] == item[0]]
		num_dis = len([p for p in this_race if int(p[-1]) == 1])
		dis_counts.append(num_dis)
	ordered = [ordered[i] for i in range(len(ordered)) if dis_counts[i] >= 20]
	smallest_race = int(ordered[-1][0])
	srace_num = ordered[-1][1]
	srace_dis = [item for item in dis_counts if item >= 20]
	srace_dis_num = srace_dis[-1]

	bigger = [ordered[i] for i in range(len(ordered)) if ordered[i][1] >= 0.0025 * features.shape[0]] + [i for i in srace_dis if i >= 0.0025 * features.shape[0]]
	smaller = [ordered[i] for i in range(len(ordered)) if ordered[i][1] < 0.0025 * features.shape[0]] + [i for i in srace_dis if i < 0.0025 * features.shape[0]]
	numbig = len(bigger)
	numsmall = len(smaller)

	url = f"https://api.census.gov/data/{year}/acs/acs1/pums/variables/RAC3P.json"
	info = requests.get(url).json()["values"]["item"]
	info = {int(key): info[key] for key in info}
	srace_string = info[smallest_race]

	return size, numbig, numsmall, dis_count, sum([item[1] for item in ordered]), smallest_race, srace_string, srace_num, srace_dis_num

labels = ["State", "Year", "Total Size", 
		  "Num Groups Bigger than 0.25%% of Population",
		  "Num Groups Smaller than 0.25%% of Population",
		  "Total Num Minority Groups",
		  "Num Disabled Individuals",
		  "Num Nonwhite People",
		  "Least Frequent Race (Number)",
		  "Least Frequent Race (String)",
		  "Least Frequent Race (Count)",
		  "Disabled x Least Frequent Race (Count)"]

for ds in ["income", "employment"]:
	data = []
	for state in median_incomes:
		for year in [2018, 2019, 2021, 2022]:
			print(ds, state, year)
			size, numbig, numsmall, dis_count, num_poc, smallest_race, srace_string, srace_num, srace_dis_num = get_stats(ds, state, year)
			data.append([state, year, size, numbig, numsmall, numbig + numsmall, dis_count, num_poc, smallest_race, srace_string, srace_num, srace_dis_num])
	df = pd.DataFrame(data = data, columns = labels)
	df.to_csv(f"transformed_data/{ds}_stats.csv")
