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

__global__ void divergenceKernel0(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        if(i%2)
          C[i] = A[i] + B[i];
        else
          C[i] = A[i] / B[i];
    }
}

__global__ void divergenceKernel1(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        if(i%2)
          C[i] = A[i] * B[i];
        else
          C[i] = A[i] / B[i];
    }
}

__global__ void divergenceKernel2(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        if(i%4)
          C[i] = A[i] * B[i];
        else
          C[i] = A[i] / B[i];
    }
}

__global__ void divergenceKernel3(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
		if( (i/32) % 2 == 0)
          C[i] = A[i] * B[i];
        else
          C[i] = A[i] / B[i];
    }
}

__global__ void divergenceKernel4(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        if(blockIdx.x%2)
          C[i] = A[i] * B[i];
        else
          C[i] = A[i] / B[i];
    }
}

void divergenceTest(const float *A, const float *B, float *C, int numElements, , dim3 blockSize=dim3(1))
{
	// Allocate the device input vector A,B, C
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, size));
    CHECK(cudaMalloc((void **)&d_C, size));
	CHECK(cudaMalloc((void **)&d_C, size));

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	dim3 gridSize((numElements - 1)/blockSize.x + 1);
	
    // Ready timer
	GpuTimer timer;
	float kernelTime = 0;
    CHECK(cudaDeviceSynchronize());

	// run kernel 1
    timer.Start();
    divergenceKernel0<<<gridSize, blockSize>>>(d_A, d_B, d_C, numElements);
    CHECK(cudaDeviceSynchronize());	
	timer.Stop();
	kernelTime = timer.Elapsed();
    printf("mathKernel1 <<< %4d %4d >>> elapsed %f sec \n", gridSize.x, blockSize.x,  kernelTime);
	CHECK(cudaGetLastError());

	// run kernel 2
    timer.Start();
    divergenceKernel2<<<gridSize, blockSize>>>(d_A, d_B, d_C, numElements);
    CHECK(cudaDeviceSynchronize());	
	timer.Stop();
	kernelTime = timer.Elapsed();
    printf("mathKernel2 <<< %4d %4d >>> elapsed %f sec \n", gridSize.x, blockSize.x,  kernelTime);
	CHECK(cudaGetLastError());

	// run kernel 3
    timer.Start();
    divergenceKernel3<<<gridSize, blockSize>>>(d_A, d_B, d_C, numElements);
    CHECK(cudaDeviceSynchronize());	
	timer.Stop();
	kernelTime = timer.Elapsed();
    printf("mathKernel3 <<< %4d %4d >>> elapsed %f sec \n", gridSize.x, blockSize.x,  kernelTime);
	CHECK(cudaGetLastError());

	// run kernel 4
    timer.Start();
    divergenceKernel4<<<gridSize, blockSize>>>(d_A, d_B, d_C, numElements);
    CHECK(cudaDeviceSynchronize());	
	timer.Stop();
	kernelTime = timer.Elapsed();
    printf("mathKernel4 <<< %4d %4d >>> elapsed %f sec \n", gridSize.x, blockSize.x,  kernelTime);
	CHECK(cudaGetLastError());

	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));
}

int main(int argc, char ** argv)
{
	// Print the vector length to be used, and compute its size
    int numElements = 33554432;

    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A,B,C
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }
	
	dim3 blockSize(1024); // Default
	if (argc == 2)	
		blockSize.x = atoi(argv[1]);
	
	divergenceTest(h_A,h_B,h_C, numElements, blockSize);	
}
