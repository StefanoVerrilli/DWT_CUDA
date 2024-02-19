#include <stdio.h>
#include "../Shared/utils.c"
#include "../Shared/compression_utils.c"
#include "DWT_gpu.cu"

int main(int argn,const char *argv[]) {
    float QuantizationStep,Threshold;

    if (argn < 2 || argn > 5) {
        printf("Usage: %s <file_path> <NeedsCompression> [<QuantizationStep> <ThresholdValue>]\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];
    int flag = atoi(argv[2]);
    
    if (flag != 0 && flag != 1) {
        printf("Invalid flag. Flag must be either 0 or 1.\n");
        return 1;
    }

    if (flag == 1 && argn != 5) {
        printf("Flag is 1, but expected additional arguments are missing.\n");
        return 1;
    }

    // If flag is 1, we expect two additional arguments
    if (flag == 1) {
        QuantizationStep = atoi(argv[3]);
        Threshold = atoi(argv[4]);
    }

    int width, height;

    // Load grayscale image
    unsigned char* image = loadImage(filename, &width, &height);

    if (!image) {
        return 1; // Error loading image
    }

    // Convert the image to float for the transform
    float *floatImage = (float *)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) {
        floatImage[i] = (float)image[i];
    }
     
    size_t original_size_bytes = get_array_size_bytes(floatImage, width*height);

    DWTForward97(floatImage,width,height);
    
    saveImage("Results/direct.bmp",floatImage,width,height);

    if(flag){
        quantize_image(floatImage,width,height,QuantizationStep);
        threshold_image(floatImage,width,height,Threshold);
        count_non_zero_elements(floatImage,width,height);
        CompressedSegment* compressed_data;
        int compressed_size;
        compress_image_rle(floatImage,width,height,&compressed_data,&compressed_size);

        printf("Original Size:%zu\n",sizeof(float)*width*height);
        printf("Compressed size:%zu\n",sizeof(compressed_data)*compressed_size);
        printf("Memory saved:%zu\n",(sizeof(float)*width*height) - (sizeof(compressed_data)*compressed_size));
    }

    DWTInverse97(floatImage,width,height);

    saveImage("Results/inverse.bmp",floatImage,width,height);

    free(floatImage);
    free(image);

    return 0;
}
