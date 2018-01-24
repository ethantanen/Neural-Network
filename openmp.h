//
//  openmp.h
//  
//
//  Created by Ethan Tanen on 10/15/17.
//
//

#ifndef openmp_h
#define openmp_h

#include <stdio.h>

#define MNIST_DOUBLE
#define USE_MNIST_LOADER
#define MNIST_STATIC
#include "mnist.h"
#include <time.h>


extern const int in;
extern const int hid;
extern const int out;

double sigmoid(double activation);
double sig_prime(double activation);
int get_mnist(double *input,double *target);
int save_net(double weights_ih[in+1][hid+1],double weights_ho[hid+1][out+1],char *file_name);
long get_elapsed_time(struct timespec start_time,struct timespec end_time);
int train_net(double weights_ih[in+1][hid+1],double weights_ho[hid+1][out+1],double **images, double **targets,int thread_id);
int train_net(double weights_ih[in+1][hid+1],double weights_ho[hid+1][out+1],double **images, double **targets,int thread_id);
int combine_weights(double weights_ih_i[in+1][hid+1],double weights_ih_ii[in+1][hid+1], double weights_ho_i[hid+1][out+1],double weights_ho_ii[hid+1][out+1]);


#endif /* openmp_h */
