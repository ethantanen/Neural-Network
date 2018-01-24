//
//  train_nn.h
//  
//
//  Created by Ethan Tanen on 10/10/17.
//
//

#include "train_nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define USE_MNIST_LOADER
#define MNIST_STATIC
#define MNIST_DOUBLE
#include "mnist.h"

#define in (28*28)
#define hid 5
#define out 10

int main (int argc, char **argv){
    
    struct timespec start_time;
    struct timespec end_time;
    
    char *file_name = "serial.bin";
    
    const int TRAIN_TOTAL = 3;
    const int IMAGE_SIZE = (28*28+1);
    const int TARGET_SIZE = (10+1);
    
    double *_images = malloc(sizeof(double)*60000*IMAGE_SIZE);
    double *_targets = malloc(sizeof(double)*60000*IMAGE_SIZE);
    get_mnist(_images,_targets);
    
    double **images = malloc(sizeof(double *)*TRAIN_TOTAL);
    double **targets = malloc(sizeof(double *)*TRAIN_TOTAL);
    
    for(int i=0; i<TRAIN_TOTAL;i++){
        images[i] = &_images[i*IMAGE_SIZE];
        targets[i] = &_targets[i*TARGET_SIZE];
    }

    printf("\nMNIST data parsed into image and target arrays...\n");
    
    
    int i=0,j=0,train_index;
    
    double input[IMAGE_SIZE];
    double target[TARGET_SIZE];
    
    double Error = 0, error_threshold = 0.001, learning_rate = .1;
    
    double hidden_activation[hid+1];
    double output_activation[out+1];
    
    double hidden_output[hid+1];
    double output_output[out+1];
    
    double hidden_ld[hid+1];
    double output_ld[out+1];
    
    
    double weights_ih[in+1][hid+1];
    double weights_ho[hid+1][out+1];
    
    /**************
     Randomize Weights
     ***************/
    for(i=0; i<=in; i++){
        for(j=0; j<=hid; j++){
            double rand_w = (double)rand()/(double)RAND_MAX;
            if(rand_w>.5)
                rand_w -= .7;
            weights_ih[i][j] = rand_w;
        }
    }
    
    for(i=0; i<=hid; i++){
        for(j=0; j<=out; j++){
            double rand_w = (double)rand()/(double)RAND_MAX;
            if(rand_w>.5)
                rand_w -= .7;
            weights_ho[i][j] = rand_w;
        }
    }
    
    
    
    //Begin Timer
    clock_gettime(CLOCK_MONOTONIC,&start_time);
    
    
    for(int epoch = 0; epoch <20000; epoch++){
        
        /***********
         Forward Propagate
         ***********/
        
        //printf("Epoch: %d, Error: %f\n",epoch,Error);
        
        Error = 0;
        
        int train_index = 0;
        
        int rand_index = rand() % TRAIN_TOTAL;
        
       for(int c = 0; c < TRAIN_TOTAL; c++){
        
           int train_index = (c + rand_index) % TRAIN_TOTAL;
           
           //printf("Train Index: %d\n",train_index);
           
            //create input vector
            for(i=0; i<IMAGE_SIZE;i++){
                input[i] = *(images[c]+i);
            }
            
            //create target vector
            for(i = 0; i<(TARGET_SIZE);i++){
                    target[i] = *(targets[c]+i);
            }
           
           //printf("Image and Target vectors created...\n");
            
            
            //calc hidden_activaton & hidden_output
            for(i=1; i<=hid; i++){
                hidden_activation[i] = weights_ih[0][i];
                for(int j=1; j<=in; j++){
                    hidden_activation[i] += weights_ih[j][i] * input[j];
                }
                hidden_output[i] = sigmoid(hidden_activation[i]);
            }
            
            
            
            //calc output_activatin & output_output
            for(i=1; i<=out; i++){
                output_activation[i] = weights_ho[0][i];
                for(int j=1; j<=hid; j++){
                    output_activation[i] += weights_ho[j][i] * hidden_output[j];
                }
                output_output[i] = sigmoid(output_activation[i]);
            }
            
            /*****************
             Backpropagation
             ******************/
            
            //calc output_ld & system error
            for(i=1; i<=out; i++){
                output_ld[i] =  sig_prime(output_output[i])*(output_output[i]-target[i]);
                Error += .5 * (target[i]-output_output[i])*(target[i]-output_output[i]);
            }
            
            //calc hidden_ld
            for(i=1; i<=hid;i++){
                double sum = 0;
                for(j=1; j<=out; j++){
                    sum += output_ld[j] * weights_ho[i][j];
                }
                hidden_ld[i] = sig_prime(hidden_output[i]) * sum;
            }
            
            
            //calc hidden_bg and update weights
            for(i=1; i<=hid; i++){
                weights_ih[0][i] -= learning_rate * hidden_ld[i];
                for(j=1; j<=in; j++){
                    weights_ih[j][i] -= learning_rate * hidden_ld[i] * input[j];
                }
            }
            
            //calc out_bd and update weights
            for(i=1; i<=out; i++){
                weights_ho[0][i] -= learning_rate * output_ld[i];
                for(j=1; j<=hid; j++){
                    weights_ho[j][i] -= learning_rate * output_ld[i] * hidden_output[j];
                }
            }
        
           /*
            if(epoch % 1000){
                printf("\nError: %f, Epoch: %d\n",Error,epoch);
                
                
                for(i=0; i<=out;i++){
                    printf("output: %f, target: %f\n",output_output[i],target[i]);
                }
                printf("\n\n");
            }
           */
        }
        
        
        
        if(Error < error_threshold ){
            printf("Network Trained, Error: %f, Epoch: %d\n",Error,epoch);
            clock_gettime(CLOCK_MONOTONIC,&end_time);
            get_elapsed_time(start_time,end_time);
            save_net(weights_ih,weights_ho,file_name);
            return 0;
        }
    
    }
    
    clock_gettime(CLOCK_MONOTONIC,&end_time);
    get_elapsed_time(start_time,end_time);
    printf("Error did not reach threshold before the last epoch\n");
    save_net(weights_ih,weights_ho,file_name);
    
    return 0;
    
}

/*********
UTILITIES
************/
double sigmoid(double activation){
    return 1/(1+exp(-1*activation));
}

double sig_prime(double activation){
    return activation*(1-activation);
}


int get_mnist(double *input,double *target){
    mnist_data *data;
    int input_index = 0;
    int label_index = 0;
    unsigned int image_count;
    int labels[60000];
    int ret;
    printf("Loading MNIST dataset...\n");
    //load data from files
    if ((ret = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &data, &image_count))) {
        printf("An error occured: %d\n", ret);
    } else {
        printf("Images Loaded: %d\n", image_count);
    }
    //flatted image data into huge array aswell as add bias node to the beginning of each image & store labels in int format in another array for further processing
    for(int i=0; i<60000; i++){
        
        input[input_index] = 0;
        input_index++;
        
        for(int j=0; j<28; j++){
            for(int k=0; k<28; k++){
                input[input_index] = data[i].data[j][k];
                input_index++;
            }
        }
        labels[i] = data[i].label;
    }
    for(int i=0; i<60000; i++){
        target[label_index] = 0;
        label_index++;
        int label = labels[i];
        for(int j=1; j<=10; j++){
            if(j==label+1){
                target[label_index] = 1;
            }else{
                target[label_index] = 0;
            }
            label_index++;
        }
    }
    return image_count;
}



int save_net(double weights_ih[in+1][hid+1],double weights_ho[hid+1][out+1],char *file_name){
    
    
    printf("Network being saved to %s...\n",file_name);
    
    FILE *f=fopen(file_name,"wb");
    
    if(f==NULL){
        printf("file failed to open. unfortunetly the network data is lost...");
        return 0;
    }
    
    
    
    int net_dim[3] = {in,hid,out};
    
     int a = fwrite(net_dim, sizeof(int),3, f);
    int b = fwrite(weights_ih,sizeof(double),(in+1)*(hid+1),f);
    int c = fwrite(weights_ho,sizeof(double),(hid+1)*(out+1),f);
    
    printf("%d of the %d network dimensiosn were saved\n",a,3);
    printf("%d of the %d weights in weights_ih were saved\n",b,(in+1)*(hid+1));
    printf("%d of the %d wegihts in weights_ho were saved\n",c,(hid+1)*(out+1));

    return 1;
    
    
}

long get_elapsed_time(struct timespec start_time,struct timespec end_time){
    long msec = (end_time.tv_sec - start_time.tv_sec)*1000 + (end_time.tv_nsec - start_time.tv_nsec)/1000000;
    printf("Total time to train: %ld ms...\n",msec);
    return msec;
}


