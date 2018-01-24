//
//  test_nn.h
//  
//
//  Created by Ethan Tanen on 10/9/17.
//
//

#ifndef test_nn_h
#define test_nn_h

#include <stdio.h>
#define MNIST_DOUBLE
#define USE_MNIST_LOADER
#define MNIST_STATIC
#include "mnist.h"

double sigmoid(double activation);
double sig_prime(double activation);
int get_mnist(double *input,double *target);



#endif /* test_nn_h */
