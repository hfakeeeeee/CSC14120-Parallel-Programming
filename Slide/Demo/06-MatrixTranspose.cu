#include <stdio.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}
#define BLOCK_SIZE 32
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

//Read GMEM uncoalesce, Write is OK
__global__ void transpose1(float *iMatrix, float *oMatrix, int w)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;	
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (r < w && c < w)
		oMatrix[r * w + c] = iMatrix[c * w + r];	
}
//Read Write GMEM is uncoalesce, Read is OK
__global__ void transpose2(float *iMatrix, float *oMatrix, int w)
{
	int c = blockIdx.y * blockDim.y + threadIdx.y;	
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (r < w && c < w)
		oMatrix[r * w + c] = iMatrix[c * w + r];	
}

//Use SHARE MEMORY, BANK conflict
__global__ void transpose3(float *iMatrix, float *oMatrix, int w)
{
	__shared__ float s_blkData[BLOCK_SIZE][BLOCK_SIZE];

	int iR = blockIdx.x * blockDim.x + threadIdx.y;
	int iC = blockIdx.y * blockDim.y + threadIdx.x;
	s_blkData[threadIdx.y][threadIdx.x] = iMatrix[iR * w + iC];
	__syncthreads();
	
	// Each block write data efficiently from SMEM to GMEM
	int oR = blockIdx.y * blockDim.y + threadIdx.y;
	int oC = blockIdx.x * blockDim.x + threadIdx.x;
	oMatrix[oR * w + oC] = s_blkData[threadIdx.x][threadIdx.y];
}
//Use SHARE MEMORY, NO BANK conflict
__global__ void transpose4(float *iMatrix, float *oMatrix, int w)
{
	__shared__ float s_blkData[BLOCK_SIZE][BLOCK_SIZE+1];

	// Each block load data efficiently from GMEM to SMEM
	int iR = blockIdx.x * blockDim.x + threadIdx.y;
	int iC = blockIdx.y * blockDim.y + threadIdx.x;
	s_blkData[threadIdx.y][threadIdx.x] = iMatrix[iR * w + iC];
	__syncthreads();

	// Each block write data efficiently from SMEM to GMEM
	int oR = blockIdx.y * blockDim.y + threadIdx.y;
	int oC = blockIdx.x * blockDim.x + threadIdx.x;
	oMatrix[oR * w + oC] = s_blkData[threadIdx.x][threadIdx.y];
}


void matrix_transpose(float* in, float* out, int w,
    bool useDevice = false, dim3 blockSize = dim3(1),int kernelType=1)
{
    GpuTimer timer;
    timer.Start();
    if (useDevice == false)
    {
		timer.Start();
        for (int row = 0; row < w; row++)
        {
            for (int col = 0; col < w; col++)
				out[row * w + col] = in[col*w+row];            
        }
		timer.Stop();
    }
    else // Use device
    {
        // Allocate device memories
        float* d_in, *d_out;
        size_t nBytes = w * w * sizeof(float);
		CHECK(cudaMalloc(&d_in, nBytes));
        CHECK(cudaMalloc(&d_out, nBytes));

        // Copy data to device memories
        CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));

        // TODO: Set grid size and call kernel
        dim3 gridSize((w - 1) / blockSize.x + 1,
            (w - 1) / blockSize.y + 1);
        
		timer.Start();
		if (kernelType == 1)
			transpose1<<<gridSize, blockSize>>>(d_in, d_out, w);
		else if (kernelType == 2)
			transpose2<<<gridSize, blockSize>>>(d_in, d_out, w);				
		else if (kernelType == 3)
			transpose3<<<gridSize, blockSize>>>(d_in, d_out, w);
		else if (kernelType == 4)
			transpose4<<<gridSize, blockSize>>>(d_in, d_out, w);

		CHECK(cudaDeviceSynchronize();)
		timer.Stop();
        // Copy result from device memory
        CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

        // Free device memories
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
		
		printf("Grid size: %d * %d, block size: %d * %d\n", 
					gridSize.x,gridSize.y, blockSize.x,blockSize.y);
    }
    float time = timer.Elapsed();
    printf("Processing time (%s): %f ms\n",
        useDevice == true ? "use device" : "use host", time);
}

float checkCorrectness(float * a1, float* a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)	
		err += abs(a1[i] - a2[i]);
	err /= n;
	return err;
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");

}
int main(int argc, char** argv)
{
	printDeviceInfo();
	
	//Declare variables
    float* h_in; // The Input matrix
    float* h_out; // The Output  matrix
    float* correct_out; // The output C matrix

    int w = (1 << 12);  // Square matrix w x w
	
	size_t nBytes = w * w * sizeof(float);   

    // Set up input data
    h_in = (float*)malloc(nBytes);
    h_out = (float*)malloc(nBytes);
    correct_out = (float*)malloc(nBytes);

    for (int i = 0; i < w; i++)
        for (int j = 0;j < w;j++)
            h_in[i*w+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		
    // Add vectors (on host)
    matrix_transpose(h_in,correct_out,w);
	printf("\n");

	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // Default
	if (argc == 3)
	{
		blockSize.x = atoi(argv[1]);
		blockSize.y = atoi(argv[2]);
	} 
    // Add in1 & in2 on device
	printf("Read GMEM uncoalesce, Write is OK:\n");
    matrix_transpose(h_in, h_out, w, true,blockSize,1);
	float err = checkCorrectness(h_out, correct_out,w*w);
	printf("Error between device result and host result: %f\n\n", err);

	printf("Read Write GMEM is uncoalesce, Read is OK:\n");
    matrix_transpose(h_in, h_out, w, true,blockSize,2);
	err = checkCorrectness(h_out, correct_out,w*w);
	printf("Error between device result and host result: %f\n\n", err);	
	
	printf("Shared memory Matrix Transpose (Bank conflict):\n");
    matrix_transpose(h_in, h_out, w, true,blockSize,3);
	err = checkCorrectness(h_out, correct_out,w*w);
	printf("Error between device result and host result: %f\n\n", err);	

	printf("Shared memory Matrix Transpose (NO Bank conflict):\n");
    matrix_transpose(h_in, h_out, w, true,blockSize,4);
	err = checkCorrectness(h_out, correct_out,w*w);
	printf("Error between device result and host result: %f", err);	
	
    free(h_in);
    free(h_out);
    free(correct_out);

    return 0;
}
