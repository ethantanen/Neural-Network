Artificial Neural Network and Testing Suite -Ethan Tanen
#############################################################
# Build instructions

To build/run on mcscn you will need to load:
	gcc-7.2.0
	slurm
To run the compilation use the compile script:
./compile.sh


#############################################################
# Running the program

to train the net run the shell command ./train for the serial version, ./cilk for the cilk version and ./openmp for the openmp version.

-to change hidden node count, open the .c file with vim and change the hid definition

outputs from each program is saved in binary files titled serial.bin, cilk.bin and openmp.bin respectively. To test the net, run ./train name_of_binary_file.bin

