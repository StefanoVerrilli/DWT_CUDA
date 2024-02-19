## Guide for compilation

The compilation and actual running of this parallel implementation is completely managed by the files that you can find in this folder:

- compile.sh
- my_run.sh

To compile the **main.cu** file you need to run the following command while in this folder:

``` ./compile.sh ```

The file should be able to successfully link the SDL2 libraries contained in the lib folder.

## Guide for execution

To execute the actual CUDA version of lifting you need to specify some paramentes, let's break them down:

- FilePath: First parameter for execution is the filepath of the BMP file to compress, some sample images are contained in Images folder (../Images/)
- Compression: This is a boolean parameters used to define if the image has to be compressed after the forward pass of DWT. This parameter also indicate if the last other 2 parameters have to be inserted, 1 stands for compression and 0 for no need of compresssion.
- Quantization_Step: **In case you have set the Compression parameter as 1**, this paramenter needs to be included to define the step size between each quantization level.
- Threshold Value: **In case you have set the Compression parameter as 1**, this parameter is utilized to set the limit value for which each pixel below will be equal to 0 in the output image.

Once parameters have been carefully selected and defined we can run the program with one of the following forms:

```./my_run.sh ../Images/barbara.bmp 0```

This last case doesn't include any compression in the final output.

```./my_run.sh ../Images/barbara.bmp 1 100 100```

In this case we included Quantization and Thresholding.

## Reviewing the Result

To visualize the results produced by the program you just need to move into the **Results** folder in which you will find the result of the forward transform and the reconstructed image (compressed or not).