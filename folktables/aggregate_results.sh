for setting in 'one' 'big' 'small' 'dis' 'all';
do
	for year in 2018 2019 2021 2022;
	do
		for state in IL OH GA NC TX FL NY PA MI CA;
		do
			for ds in income employment;
			do
				echo "$setting, $year, $state, $ds"
				python aggregate_results.py --ds $ds --state $state --year $year --setting $setting
			done
		done
	done
done

# for k in 1 3 40 100;
# do
# 	for lam in 0.25 0.5;
# 	do
# 		for ds in income employment;
# 		do
# 				python aggregate_results.py --ds $ds --state CA --year 2022 \
# 											--lam $lam --k $k
# 		done
# 	done
# done