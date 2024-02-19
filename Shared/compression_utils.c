#pragma once
#include <stdio.h>
#include <math.h>

typedef struct {
    int run_length;
    int value;
} CompressedSegment;

void quantize_image(float* coefficients, int width,int height, float quantization_step) {
    printf("Running Quantization Step\n");
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            coefficients[i*width+j] = round(coefficients[i*width+j]/quantization_step)*quantization_step;
        }
    }
}

void threshold_image(float* image, int width, int height, float threshold) {
    printf("Running Thresholing on image\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Access the DWT coefficient at position (x, y) in the image
            float coefficient = image[i * width + j];

            // Apply thresholding
            if (fabs(coefficient) < threshold) {
                image[i * width + j] = 0.0;  // Set coefficients below threshold to zero
            }
        }
    }
}

void count_non_zero_elements(float*image,int width,int height){
    int count =0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if(image[i * width + j] == 0.0f) count++;
        }
    }
    printf("potential of compression: %d\n",count);
}


void compress_image_rle(float* image, int width, int height, CompressedSegment** compressed_data, int* compressed_size) {
    *compressed_size = 0;

    // Determine the maximum possible size of the compressed data
    for (int y = 0; y < height; y++) {
        int count = 1;
        for (int x = 1; x < width; x++) {
            if (image[y * width + x] == image[y * width + x - 1]) {
                count++;
            } else {
                (*compressed_size)++;  // Increment for count and value
                count = 1;
            }
        }
        (*compressed_size)++;  // Increment for the last run-length and value
    }

    *compressed_data = (CompressedSegment*)malloc(*compressed_size * sizeof(CompressedSegment));
    if (*compressed_data == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    int index = 0;

    // Compress the data
    for (int y = 0; y < height; y++) {
        int count = 1;
        for (int x = 1; x < width; x++) {
            if (image[y * width + x] == image[y * width + x - 1]) {
                count++;
            } else {
                // Write the run-length and value to the compressed data
                (*compressed_data)[index].run_length = count;
                (*compressed_data)[index].value = image[y * width + x - 1];
                index++;
                count = 1;
            }
        }

        // Write the last run-length and value to the compressed data
        (*compressed_data)[index].run_length = count;
        (*compressed_data)[index].value = image[y * width + width - 1];
        index++;
    }

}



size_t get_array_size_bytes(float* array, int size) {
    return size * sizeof(float);
}