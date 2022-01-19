kernel:
	gcc -O3 kmean.c -o kmean.x -mavx -mfma -std=c99 -fopenmp -DKERNEL=1
	cat ./data/dataset_40k_points_16d | ./kmean.x 3 4 16 40000

kmean:
	gcc -O3 kmean.c -o kmean.x -mavx -mfma -std=c99
	cat ./data/dataset_40k_points_16d | ./kmean.x 3 4 16 40000

kmean_par:
	gcc -O3 kmean_par.c -o kmean_par.x -mavx -mfma -std=c99 -fopenmp
	cat ./data/dataset_40k_points_16d | ./kmean_par.x 3 4 16 40000 1
	cat ./data/dataset_40k_points_16d | ./kmean_par.x 3 4 16 40000 2
	cat ./data/dataset_40k_points_16d | ./kmean_par.x 3 4 16 40000 4
	cat ./data/dataset_40k_points_16d | ./kmean_par.x 3 4 16 40000 8
	cat ./data/dataset_40k_points_16d | ./kmean_par.x 3 4 16 40000 16
	cat ./data/dataset_40k_points_16d | ./kmean_par.x 3 4 16 40000 20

kmean_par_schedule:
	gcc -O3 kmean_par_schedule.c -o kmean_par_schedule.x -mavx -mfma -std=c99 -fopenmp
	cat ./data/dataset_40k_points_64d_4C | ./kmean_par_schedule.x 3 4 64 40000 1
	cat ./data/dataset_40k_points_64d_4C | ./kmean_par_schedule.x 3 4 64 40000 2
	cat ./data/dataset_40k_points_64d_4C | ./kmean_par_schedule.x 3 4 64 40000 4
	cat ./data/dataset_40k_points_64d_4C | ./kmean_par_schedule.x 3 4 64 40000 8
	cat ./data/dataset_40k_points_64d_4C | ./kmean_par_schedule.x 3 4 64 40000 16
	cat ./data/dataset_40k_points_64d_4C | ./kmean_par_schedule.x 3 4 64 40000 20
	cat ./data/dataset_40k_points_64d_4C | ./kmean_par_schedule.x 3 4 64 40000 24
	cat ./data/dataset_40k_points_64d_4C | ./kmean_par_schedule.x 3 4 64 40000 30
	cat ./data/dataset_40k_points_64d_4C | ./kmean_par_schedule.x 3 4 64 40000 35
	cat ./data/dataset_40k_points_64d_4C | ./kmean_par_schedule.x 3 4 64 40000 40

correctness:
	gcc -O3 kmean_par_schedule.c -DCORRECTNESS_CHECK -o kmean_par_schedule.x -mavx -mfma -std=c99 -fopenmp
	cat ./data/dataset_40k_points_16d | ./kmean_par_schedule.x 3 4 16 40000 1
	cat ./data/dataset_40k_points_16d | ./kmean_par_schedule.x 3 4 16 40000 2
	cat ./data/dataset_40k_points_16d | ./kmean_par_schedule.x 3 4 16 40000 4
	cat ./data/dataset_40k_points_16d | ./kmean_par_schedule.x 3 4 16 40000 8
	cat ./data/dataset_40k_points_16d | ./kmean_par_schedule.x 3 4 16 40000 16
	cat ./data/dataset_40k_points_16d | ./kmean_par_schedule.x 3 4 16 40000 20
	
clean:
	rm -rf *.x *~
