for ds in income employment;
do
	for setting in all big small dis one;
	do
		python aggregate_results.py --ds $ds \
									--state CA \
									--year 2022 \
									--setting $setting \
									--best
	done
done