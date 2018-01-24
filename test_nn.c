//
//  test_nn.c
//  
//
//  Created by Ethan Tanen on 10/9/17.
//
//

#include "test_nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main (int argc, char **argv){
    
    
    int in;
    int hid;
    int out;
    
    
    //check if there is a binary file for aquiring weights
    if(argc < 2){
        printf("Include binary file of weights as argument");
        return 1;
    }
    
    //create a var of type FILE from the 1st argument
    printf("Using weights from file %s\n",argv[1]);
    FILE *f = fopen(argv[1],"rb");
    if(f == NULL){
        printf("Error opening file.\n");
    }
    
    //create an array to hold number of nodes in the in,hid and out layers
    int node_count_info[3];
    fread(node_count_info,sizeof(int),3,f);
    
    //store information on constructing the net
    in = node_count_info[0];
    hid = node_count_info[1];
    out = node_count_info[2];
    
    
    printf("\nIN: %d, HID: %d, OUT: %d\n",in,hid,out);
    
    //create arrays to hold the nn weights
    printf("\nReading weights from %s...\n",argv[1]);
    double weights_ih[in+1][hid+1];
    double weights_ho[hid+1][out+1];
    
    size_t c = fread(weights_ih,sizeof(double),(in+1)*(hid+1),f);
    size_t d = fread(weights_ho,sizeof(double),(hid+1)*(out+1),f);
    
    printf("%zu elements read of %d elements in weights_ih\n",c,(in+1)*(hid+1));
    printf("%zu elements read of %d elements in weights_ho\n\n",d,(hid+1)*(out+1));

    //train total is how many images the net should be trained on
    const int TRAIN_TOTAL = 10;
    const int IMAGE_SIZE = (28*28+1);
    const int TARGET_SIZE = (10+1);
    
    double *_images = malloc(sizeof(double)*10000*IMAGE_SIZE);
    double *_targets = malloc(sizeof(double)*10000*TARGET_SIZE);
    
    //retrieves the entrie mnist testing set
    get_mnist(_images,_targets);
    
    double **images = malloc(sizeof(double *)*TRAIN_TOTAL);
    double **targets = malloc(sizeof(double *)*TRAIN_TOTAL);
    
    for(int i=0; i<TRAIN_TOTAL; i++){
        images[i] = &_images[i*IMAGE_SIZE];
        targets[i]= &_targets[i*TARGET_SIZE];
    }
    
    printf("\nMNIST data parsed into image and target arrays...\n");
    
    int i=0,j=0;
    
    double input[IMAGE_SIZE];
    double target[TARGET_SIZE];
    
    double Error=0;
    
    double hidden_activation[hid+1];
    double output_activation[out+1];
    
    double hidden_output[hid+1];
    double output_output[out+1];
    
    //array to hold the output of the net only for nodes that correspond to the expected target
    double error_x[TRAIN_TOTAL];
    
    
    for(int c=0; c<TRAIN_TOTAL; c++){
        //create input vector
        for(i=0; i<IMAGE_SIZE;i++){
            input[i] = *(images[c]+i);
        }
        
        //create target vector
        for(i = 0; i<(TARGET_SIZE);i++){
            target[i] = *(targets[c]+i);
        }
        
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
        
        //calc output_ld & system error
        for(i=1; i<=out; i++){
            Error += .5 * (target[i]-output_output[i])*(target[i]-output_output[i]);
        }
        
        int count = 1;
        for(i=1; i<TARGET_SIZE; i++){
            if(target[i] == 1){
                break;
            }
            count++;
        }
        
        error_x[c] = output_output[count];
        
        printf("\nOUTPUT over TARGET\n");
        for(int i=0; i<=out;i++){
            printf("%f ",output_output[i]);
        }
        printf("\n");
        for(int i=0; i<=out;i++){
            printf("%f ",target[i]);
        }
        printf("\n");
    }
    
    //calculates error for the targets
    double error_target = 0;
    for(int i=0; i<TRAIN_TOTAL; i++){
        
        error_target+= .5*(error_x[i]-1)*(error_x[i]-1);
        
    }
    printf("Error: %f  Error Targ: %f\n",Error,error_target);
}


/*************
 UTILITIES
 *************/

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
    int labels[10000];
    int ret;
    printf("Loading MNIST dataset...\n");
    //load data from files
    if ((ret = mnist_load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &data, &image_count))) {
        printf("An error occured: %d\n", ret);
    } else {
        printf("Images Loaded: %d\n", image_count);
    }
    //flatted image data into huge array aswell as add bias node to the beginning of each image & store labels in int format in another array for further processing
    for(int i=0; i<10000; i++){
        
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
    for(int i=0; i<10000; i++){
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



