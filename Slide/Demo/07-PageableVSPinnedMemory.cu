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

void InitMatrix(int *in1, int nRows, int nCols)
{
	for (int i = 0; i < nRows * nCols; i++)
	{
		in1[i] = rand() & 0xff; // Random int in [0, 255]
	}
}

void TransferMatrix(int *in1, int nRows, int nCols, int kernelType=1)
{
	GpuTimer timer;
	size_t nBytes = nRows * nCols * sizeof(int);
	printf("Total data need to copy: %f GB\n", float(nBytes)/(1<<30));

	int * d_in1;
	if (kernelType == 1)
	{
		in1 = (int *)malloc(nBytes);
		InitMatrix(in1,nRows,nCols);

		CHECK(cudaMalloc(&d_in1, nBytes));
		// Copy data to device memories
		timer.Start();
		CHECK(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));
		timer.Stop();
		
		CHECK(cudaFree(d_in1));
		free(in1);		
	}
	else
	{	
		CHECK(cudaMallocHost(&in1,nBytes));
		InitMatrix(in1,nRows,nCols);
		
		CHECK(cudaMalloc(&d_in1, nBytes));

		// Copy data to device memories
		timer.Start();
		CHECK(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));
		timer.Stop();
		
		CHECK(cudaFree(d_in1));
		CHECK(cudaFreeHost(in1));
	}
	
	float time = timer.Elapsed();
	float bandwidth = nBytes / (time/1000 * (1<<30));
	printf("Copying time: %f ms \nBandwidth: %f GBps\n", time, bandwidth);
}

int main(int argc, char ** argv)
{
	printDeviceInfo();
	
    int nRows, nCols ; // Matrix size
    int *in1; // Input matrixes

    // Input data into nRows and nCols
    nRows = 1 << 13;
    nCols = 1 << 13;
    printf("# rows = %d, # cols = %d\n", nRows, nCols);
 
	printf("Copy with Pageable memory\n");
	TransferMatrix(in1,nRows,nCols,1);
	printf("\n");
	printf("Copy with Pinned memory\n");
	TransferMatrix(in1,nRows,nCols,2);
}
