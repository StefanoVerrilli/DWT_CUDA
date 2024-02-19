#include <cuda.h>
#include <stdio.h>

#define BLOCK_SIZE 16
#define ALPHA -1.586134342
#define BETA -0.05298011854
#define GAMMA 0.8829110762
#define DELTA 0.4435068522
#define K  1.230174105
#define IK  0.812893066

struct GeneralInfo{
    int width;
    int height;
    dim3 nBlocks;
    dim3 nThreadPerBlock;
};

__global__ void dwt97Rows(float *data, int width, int height,float coeff1, float coeff2);
__global__ void Normalization(float* data,int width,int height);
__global__ void INormalization(float *data,int width,int height);
__global__ void InverseDwt97(float *data, int width, int height,float coeff1,float coeff2);
__global__ void ISaveData(float*input,float* temp,int width,int height);
__global__ void SaveData(float*input,float*temp, int width,int height);
__global__ void transposeMatrix(float *input, float *output, int rows, int cols);

void DWTForward97(float *h_image, int width, int height);
void DWTInverse97(float *h_image,int width,int height);
void StandardProcedure(float *data,struct GeneralInfo *general_info, bool needsTranspose,bool inverse);

/* 
   Normalize the values previusly elaborated by predict and update 1->2.
   
   This function implements the last step of lifting approach using K and IK normalization constants for respectively
   even and odd elements. The entire computation is conducted in place using a thread setting besed on blocks where
   each thread is responsable for a single element in the block
   
   @param data: Vectorized version of the original image expressed in float elements vector.
   @param width: Width of the original bidimensional matrix.
   @param height: Height of the original bidimensional matrix.
   @return None: The entire operation is conducted in place.
   @see dwt97Rows, StandardProcedure, DWTForward97
*/
__global__ void Normalization(float* data,int width,int height){
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row_idx * width + col_idx;
    __syncthreads();
    // Check if within the valid range
    if (row_idx < height && col_idx < width) {

        if (col_idx % 2 == 0) {
            data[idx] *= K;
        }

        if (col_idx % 2 == 1) {
            data[idx] *= IK;
        }
    }
}

/* 
   Fundamental step to perform Forward pass of lifting by implementing predict and update step.
   
   Through the parametrizatiozation of this procedure is possible to perform both predict1 and update1 with ALPHA
   and BETA constant and predict2 and update2 with GAMMA and DELTA parameters. The smoothing process is performed
   again inplace and number of thread is evenly distributed between even and odd elements.

   This procedure is conducted separately alongside columns and rows and the final result can be obtained by applying
   this operation first on rows and then, onto the result, on cols or viceversa. 

   Complete prodecure:
    - DWt97Rows(Predict1,Update1) 
    - DWT97Rows(Predict2,Update2) (after transpose)
    - Normalization
    - SaveData

   ------
   IMPORTANT: To ensure proper utilization of updated values a __threadfance() function is used between prediction
   and updating step. This function call is mandatory since we need to update the data vector with the new computed
   values in a sequential style, without it will result in conflicts throughout the image.
   ------
   
   @param data: Vectorized version of the original image expressed in float elements vector.
   @param width: Width of the original bidimensional matrix.
   @param height: Height of the original bidimensional matrix.
   @param coeff1: ALPHA/GAMMA parameter used in lifting procedure
   @param coeff2: BETA/DELTA parameter used in lifting procedure
   @return None: The entire operation is conducted in place.
   @see: InverseDw797,StandardProcedure,SaveData
*/
__global__ void dwt97Rows(float *data, int width, int height,float coeff1,float coeff2) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row_idx * width + col_idx;

    __syncthreads();
    // Check if within the valid range
    if (row_idx < height && col_idx < width) {
        // Prediction Step

        if (col_idx % 2 == 1 && col_idx < (width -2)) {
                data[idx] += coeff1 * (data[(idx-1)] + data[(idx + 1)]);
            }
            
        if(col_idx %2 == 1 && col_idx == (width -1)){
                data[idx] += (coeff1 * (data[idx-1]));
            }

        __threadfence();
        // Update Step

        if (col_idx %2 == 0 && col_idx < (width - 1) && col_idx > 0){
                data[idx] += (coeff2 * (data[(idx-1)] + data[(idx+1)]));
        }
        
        if(col_idx % 2 == 0 && col_idx == 0){
                data[idx] += (coeff2 *(data[(idx+1)]));
        }
        __threadfence();}}

/* 
   Fundamental step to perform Inverse pass of lifting, can be utilized to preform Inverse Update1/2 and Inverse Predict 1/2
   
   As for what happend with the forward pass, we can parametrize the function to perform different predict and upadte
   steps by inverting the coefficients. In fact the inverse procedure is based on the reproposition of the forward steps
   in reverse order to achive a final reconstruction: 
   ISaveData -> INormalization -> InverseDWT97(Update2,Predict2) -> InverseDWT97(Update1,Predict1)

   ------
   IMPORTANT: To ensure proper utilization of updated values a __threadfance() function is used between prediction
   and updating step. This function call is mandatory since we need to update the data vector with the new computed
   values in a sequential style, without it will result in conflicts throughout the image.
   ------
   
   @param data,width,height: @see dwt97Rows for details 
   @param coeff1: BETA/DELTA parameter used in lifting procedure
   @param coeff2: ALPHA/GAMMA parameter used in lifting procedure
   @return None: The entire operation is conducted in place.
   @see: DWT97Rows,StandardProcedure,ISaveData
*/
__global__ void InverseDwt97(float *data, int width, int height,float coeff1,float coeff2){
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row_idx * width + col_idx;


    __syncthreads();
    if(col_idx < width && row_idx < height){

        //Undo Update
        if (col_idx % 2 == 0 && col_idx < (width - 1) && col_idx > 0) {
            data[idx] += (-coeff2 * (data[(idx-1)] + data[(idx+1)]));
        }

        if (col_idx %2 == 0 && col_idx == 0) {
            data[idx] += (-coeff2 *(data[(idx+1)]));
        }

        __threadfence();

        //Undo predict
        if (col_idx % 2 == 1 && col_idx < (width - 2)) {
            data[idx] += (-coeff1 * (data[(idx-1)] + data[(idx + 1)]));
        }

        if (col_idx %2 == 1 && col_idx == (width-1)) {
            data[idx] += (-coeff1 * (data[idx-1]));
        }

        __threadfence();}}

/* 
   This function has the purpose to invert the normalization step
   
   In constrast with the forward Normalization step, which is performed as the last step of direct procedure,
   the INormalization function is performed as first step to achive the inversion through a reverse ordering of the
   operations. This is achived by dividing back the even and odd elements by the K and IK normalization coefficients.
   
   @param data: Vectorized version of the original image expressed in float elements vector.
   @param width: Width of the original bidimensional matrix.
   @param height: Height of the original bidimensional matrix.
   @return None: The entire operation is conducted in place.
   @see Inversedwt97, StandardProcedure, Normalization
*/
__global__ void INormalization(float *data,int width,int height){
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row_idx * width + col_idx;
    __syncthreads();
    if(col_idx < width && row_idx < height){

        if (col_idx % 2 == 0) {
            data[idx] /= K;
        }

        if (col_idx % 2 == 1) {
            data[idx] /= IK;
        }
    }}


/* 
   Saving elaborated data in a quadrant style result.
   
   This step is not directly related with the lifting approch but is fundamental to ensure a meaningful rappresentation
   of the results produced. After the elaboration of alongside rows/cols though this procedure we are able to split
   even and odd elements in two different regions of the output and after repeating the process on the other axis to
   obtain a four quarters decomposition of the image.

   To observe the decomposition produces is possible to run the executable and visualizing it into ./Results folder.
   
   @param data: Vectorized version of the original image expressed in float elements vector.
   @param temp: Used to store the result produced by the splitting, need to be copied back to the original data vector.
   @param width: Width of the original bidimensional matrix.
   @param height: Height of the original bidimensional matrix.
   @return The entire operation is conducted using values of data and storing them into temp vector.
   @see ISaveData,DWT97Rows,StandardProcedure
*/
__global__ void SaveData(float *data,float*temp,int width,int height){
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row_idx * width + col_idx;
    if(col_idx<width && row_idx < height){
        if(col_idx% 2 == 0){
            temp[((row_idx*width) + (col_idx/2))] = data[idx];
        }else{
            temp[(((width)/2) + ((row_idx*width) + (col_idx/2)))] = data[idx];
        }}}


/* 
   Recompacting the decomposition produced by SaveData into the final image
   
   Since the elaboration done by the forward pass and the decomposition of SaveData produced a quadrants image
   is now necessary to perform the inverse procedure. This function performs the first step of the inverse routine
   and the effect that it produces is the actual compacting of odd and even elements into adjacent locations. By
   repeating this step BEFORE each InverseDWT97 function call and INormalization we ensure a correct working of the
   algorithm.
   
   @param data: Vectorized version of the original image expressed in float elements vector.
   @param temp: Used to store the result produced by the compacting step, need to be copied back to the original data vector.
   @param width: Width of the original bidimensional matrix.
   @param height: Height of the original bidimensional matrix.
   @return The entire operation is conducted using values of data and storing them into temp vector.
   @see SaveData,InverseDWT97,StandardProcedure
*/
__global__ void ISaveData(float* data, float* temp, int width, int height) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < width / 2 && row_idx < height) {
        int data_idx = col_idx + row_idx * (width / 2);

        temp[data_idx * 2] = data[col_idx + row_idx * width];
        temp[data_idx * 2 + 1] = data[col_idx + row_idx * width + width / 2];
        }}

__global__ void transposeMatrix(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Transpose colled\n");

    // Check if within the valid range
    if (row < rows && col < cols) {
        int input_index = row * cols + col;
        int output_index = col * rows + row;
        output[output_index] = input[input_index];
    }}

/*
    Wrapper function to perform direct DWT97 and taking the times of the execution

    This function is in charge to instantiate the blocks of threads used for the computation of the dwt97 in an
    adaptive manner. By using the StandardProcedure function call first alongside rows (0,0) then the cols (1,0)
    through a transposition of the matrix we can perform the steps of predict1,update1,predict2,update2,normalization
    and take the time of their execution.

    @param data: Vectorized version of the original image expressed in float elements vector.
    @param width: Width of the original bidimensional matrix.
    @param height: Height of the original bidimensional matrix.
    @see: StandardProcedure

*/
void DWTForward97(float *h_image, int width, int height) {

    struct GeneralInfo INFO;

    dim3 nThreadPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    dim3 nBlocks((width/nThreadPerBlock.x)+((width%nThreadPerBlock.x) == 0 ? 0:1),
                 (height/nThreadPerBlock.y)+((height%nThreadPerBlock.y) == 0 ? 0:1));

    cudaEvent_t start_direct, stop_direct;
    cudaEventCreate(&start_direct);
    cudaEventCreate(&stop_direct);

    INFO.nThreadPerBlock = nThreadPerBlock;
    INFO.nBlocks = nBlocks;
    INFO.width = width;
    INFO.height = height;
    size_t size = width*height*sizeof(float);

    float *d_image;

    cudaMalloc((void **)&d_image, size);

    cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start_direct);
    StandardProcedure(d_image,&INFO,0,0);
    StandardProcedure(d_image,&INFO,1,0);
    cudaEventRecord(stop_direct);

    memset(h_image,0,size);
    cudaMemcpy(h_image, d_image, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop_direct);
    cudaDeviceSynchronize();


    float time_direct = 0;
    cudaEventElapsedTime(&time_direct,start_direct,stop_direct);

    printf("time for direct: %8.2f s\n",time_direct);

    cudaFree(d_image);
}

void DWTInverse97(float *h_image,int width,int height){
    struct GeneralInfo INFO;

    dim3 nThreadPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    dim3 nBlocks((width/nThreadPerBlock.x)+((width%nThreadPerBlock.x) == 0 ? 0:1),
                 (height/nThreadPerBlock.y)+((height%nThreadPerBlock.y) == 0 ? 0:1));

    cudaEvent_t start_inverse,stop_inverse;
    cudaEventCreate(&start_inverse);
    cudaEventCreate(&stop_inverse);



    INFO.nThreadPerBlock = nThreadPerBlock;
    INFO.nBlocks = nBlocks;
    INFO.width = width;
    INFO.height = height;
    size_t size = width*height*sizeof(float);

    float *d_image;

    cudaMalloc((void **)&d_image, size);

    cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start_inverse);
    StandardProcedure(d_image,&INFO,0,1);
    StandardProcedure(d_image,&INFO,1,1);
    cudaEventRecord(stop_inverse);

    memset(h_image,0,size);
    cudaMemcpy(h_image, d_image, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop_inverse);

    float time_inverse = 0;
    cudaEventElapsedTime(&time_inverse,start_inverse,stop_inverse);

    printf("time elapsed for inverse: %8.2f s\n",time_inverse);

    cudaFree(d_image);
}

/*
    Parametrized routine to perform forward and inverse DWT97 transform on vectorized image.

    Since the steps conducted by the direct and inverse procedure of DWT97 share much similarities we implemented
    a common routine to perform them with different setting for each one of them.

    ---
    IMPORTANT: Since the image that we are working with is saved in vectorized form by rows to simplify the computation
    we opted for a trasposition of the matrix to perform the operation alongside the columns. The made choice is supported
    by an actual parallel implementation of the trasposition operation to avoid performance degradation at execution time.
    ----

    Cases of use:
        - Direct alongside rows: StandardProcedure(...,....,...,0,0)
        - Direct alongside cols: StandardProcedure (...,...,...,1,0) -> Transpose then dwt97Rows
        - Inverse alognside rows: StandardProcedure(...,...,...,0,1)
        - Inverse alongside cols: StandardProcedure(...,...,...,1,1) -> Transpose then Inversedwt97

    @param data: Vectorized version of the original image expressed in float elements vector.
    @param general_info: Struct made to include case-based variable such as width,height,nThredPerBlockm,nBlocks.
    @param needsTranspose: Boolean variable to indicate if we need to traspose the matrix before performing the steps of lifting
                - 0: no transpose
                - 1: transpose
    @param inverse: Boolean variable used to switch between direct and inverse procedure.
                - 0: direct procedure
                - 1: Inverse procedure
    @return The entire operation is conducted using values of data and storing them into temp vector.
    @see DWTInverse97,DWTForward97

*/
void StandardProcedure(float *data,struct GeneralInfo *general_info, bool needsTranspose,bool inverse){
    dim3 nBlocks = general_info->nBlocks;
    dim3 nThreadPerBlock = general_info->nThreadPerBlock;
    size_t size = general_info->width*general_info->height*sizeof(float);

    float * temp;
    cudaMalloc((void **) &temp,size);
    cudaMemcpy(temp,data,size,cudaMemcpyDeviceToDevice);

    if(needsTranspose){
        transposeMatrix<<<nBlocks,nThreadPerBlock>>>(data,temp,general_info->width,general_info->height);
        cudaDeviceSynchronize();
        cudaMemcpy(data,temp,size,cudaMemcpyDeviceToDevice);
        cudaMemset(temp,0,size);
    }
    if(inverse){
        ISaveData<<<nBlocks,nThreadPerBlock>>>(data,temp,general_info->width,general_info->height);
        cudaDeviceSynchronize();
        cudaMemset(data,0,size);
        cudaMemcpy(data,temp,size,cudaMemcpyDeviceToDevice);
        cudaMemset(temp,0,size);

        INormalization<<<nBlocks,nThreadPerBlock>>>(data,general_info->width,general_info->height);
        cudaDeviceSynchronize();
        InverseDwt97<<<nBlocks,nThreadPerBlock>>>(data,general_info->width,general_info->height,GAMMA,DELTA);
        cudaDeviceSynchronize();
        InverseDwt97<<<nBlocks,nThreadPerBlock>>>(data,general_info->width,general_info->height,ALPHA,BETA);
        cudaDeviceSynchronize();
    }else{
        dwt97Rows<<<nBlocks,nThreadPerBlock>>>(data,general_info->width,general_info->height,ALPHA,BETA);
        cudaDeviceSynchronize();
        dwt97Rows<<<nBlocks,nThreadPerBlock>>>(data,general_info->width,general_info->height,GAMMA,DELTA);
        cudaDeviceSynchronize();
        Normalization<<<nBlocks,nThreadPerBlock>>>(data,general_info->width,general_info->height);
        cudaDeviceSynchronize();

        SaveData<<<nBlocks,nThreadPerBlock>>>(data,temp,general_info->width,general_info->height);
        cudaDeviceSynchronize();
        cudaMemcpy(data,temp,size,cudaMemcpyDeviceToDevice);
        cudaMemset(temp,0,size);
    }
    if(needsTranspose){
        transposeMatrix<<<nBlocks,nThreadPerBlock>>>(data,temp,general_info->width,general_info->height);
        cudaDeviceSynchronize();
        cudaMemcpy(data,temp,size,cudaMemcpyDeviceToDevice);
        cudaMemset(temp,0,size);
    }

    cudaFree(temp);
}


