gcc train_nn.c -lm -g -o train
gcc test_nn.c -lm -g -o test
gcc openmp.c -lm -g -fopenmp -o openmp
gcc cilk.c -lm -g -fcilkplus -o cilk
