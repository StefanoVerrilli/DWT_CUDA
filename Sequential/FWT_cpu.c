#include <stdio.h>
#include "../Shared/utils.c"

#define ALPHA -1.586134342
#define BETA -0.05298011854
#define GAMMA 0.8829110762
#define DELTA 0.4435068522
#define K 1.230174105
#define IK 0.812893066

void lifting97(float*, int);
void Ilifting97(float*,int);
void forwardDWT97(float*, int, int,int);
void InverseDWT97(float*, int, int ,int );



float* temp=0;

void lifting97(float* data, int size) {
    int i;

    // Predict 1
    for (i = 1; i < size-2; i += 2) {
        data[i] += ALPHA * (data[i - 1] + data[i + 1]);
    }
    data[size-1] += ALPHA*data[size-2];

    // Update 1
    for (i = 2; i < size-1; i += 2) {
        data[i] += BETA * (data[i - 1] + data[i + 1]);
    }
    data[0]+=BETA*data[1];

    // Predict 2
    for (i = 1; i < size - 2; i += 2) {
        data[i] += GAMMA * (data[i - 1] + data[i + 1]);
    }
    data[size-1] += GAMMA*data[size-2];

    // Update 2
    for (i = 2; i < size-1; i += 2) {
        data[i] += DELTA * (data[i - 1] + data[i + 1]);
    }
    data[0] += DELTA*data[1];

    for (i = 0; i < size; i++) {
        if(i%2) data[i]*=K;  // Scaling factor for the 9/7 wavelet transform
        else data[i]*=IK;
    }

  if (temp==0) temp = (float*)malloc(size*sizeof(float));
    for(i=0;i<size;i++){
        if(i%2==0) temp[i/2] = data[i];
        else temp[size/2+ i/2] = data[i];
    }
    for(i=0;i<size;i++) data[i] = temp[i];
}


void Ilifting97(float* data,int size){
    

    if(temp == 0) temp = (float *) malloc(size*sizeof(float));
    for(int i=0;i<size/2;i++){
        temp[i*2] = data[i];
        temp[i*2+1] = data[i+size/2];
    }
    for(int i=0;i<size;i++) data[i] = temp[i];

    //Undo scaling
    for(int i=0;i<size;i++){
        if(i%2) data[i] /= K;
        else data[i]/= IK;
    }
    //Undo update 2
    for(int i=2;i<size-1;i+=2){
        data[i]+=(-DELTA)*(data[i-1]+ data[i+1]);
    }
    data[0] += (-DELTA)*data[1];

    //Undo predict 2
    for(int i=1;i<size-2;i+=2){
        data[i]+= (-GAMMA)*(data[i-1]+data[i+1]);
    }
    data[size-1] +=(-GAMMA)*data[size-2];

    //Undo update 1
    for(int i=2;i<size-1;i+=2){
        data[i]-= BETA*(data[i-1]+data[i+1]);
    }
    data[0]+=(-BETA)*data[1];


    //Undo predict 1
    for(int i=1;i<size-2;i+=2){
        data[i] += (-ALPHA)*(data[i-1]+data[i+1]);
    }
    data[size-1] += (-ALPHA)*data[size-2];

}


// Forward 9/7 wavelet transform on a 2D image
void forwardDWT97(float* image, int width, int height,int levels) {
    int i, j;

for(int k = 0;k<levels;k++){

        //Decompose rows
         for (i = 0; i < height; i++) {
            lifting97(image + i * width,width);
        }

        saveImage("Results/direct1.bmp",image,width,height);

        // Decompose columns
        for (j = 0; j < width; j++) {
            float column[height];
            for (i = 0; i < height; i++) {
                column[i] = image[i * width + j];
            }

            lifting97(column, height);

            for (i = 0; i < height; i++) {
                image[i * width + j] = column[i];
            }
        }
        saveImage("Results/direct2.bmp",image,width,height);
}
}

void InverseDWT97(float* image, int width, int height,int levels) {
    int i, j;

for(int k=levels-1;k>=0;k--){


    // Inverse rows
        for (i = 0; i < height; i++) {
            Ilifting97(image + i * width, width);
        }

        saveImage("Results/inverse1.bmp",image,width,height);


    for (j = 0; j < width; j++) {
            float column[height];
            for (i = 0; i < height; i++) {
                column[i] = image[i * width + j];
            }

            Ilifting97(column, height);

            for (i = 0; i < height; i++) {
                image[i * width + j] = column[i];
            }
        }

        saveImage("Results/inverse2.bmp",image,width,height);

}
}