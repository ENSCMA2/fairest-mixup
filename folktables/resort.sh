for ds in income employment;
do
	for state in CA TX FL NY PA IL OH GA NC MI;
	do
		for year in 2022 2019 2021 2018;
		do
			for mode in mpma mpmc eo;
			do
				# mkdir results/$ds/$state/$year/"$mode"_fairbase
				ls results/$ds/$state/$year/"$mode"_fairbase
				rm -r results/$ds/$state/$year/"$mode"_fairbase
			done
			# mv results/$ds/$state/$year/eo_fairbase results/$ds/$state/$year/fairbase
		done
	done
done
