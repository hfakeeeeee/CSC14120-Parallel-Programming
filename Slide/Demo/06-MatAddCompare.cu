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

//Coalesces
__global__ void addMatKernel1(int *in1, int *in2, int nRows, int nCols, 
        int *out)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < nRows && c < nCols)
    {
        int i = r * nCols + c;
        out[i] = in1[i] + in2[i];
    }
}

//Non Coalesces
__global__ void addMatKernel2(int *in1, int *in2, int nRows, int nCols, 
        int *out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < nRows && c < nCols)
    {
        int i = r * nCols + c;
        out[i] = in1[i] + in2[i];
    }
}

void addMat(int *in1, int *in2, int nRows, int nCols, 
        int *out, 
        bool useDevice=false, dim3 blockSize=dim3(1), int kernelType=1)
{
	GpuTimer timer;
	
	if (useDevice == false)
	{
        timer.Start();
        for (int r = 0; r < nRows; r++)
        {
            for (int c = 0; c < nCols; c++)
            {
                int i = r * nCols + c;
                out[i] = in1[i] + in2[i];
            }
        }
		timer.Stop();
	}
	else // Use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// TODO: Allocate device memories
        int * d_in1, * d_in2, * d_out;
        size_t nBytes = nRows * nCols * sizeof(int);
        CHECK(cudaMalloc(&d_in1, nBytes));
        CHECK(cudaMalloc(&d_in2, nBytes));
        CHECK(cudaMalloc(&d_out, nBytes));

		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_in2, in2, nBytes, cudaMemcpyHostToDevice));

		// TODO: Set grid size and call kernel
        dim3 gridSize((nCols - 1) / blockSize.x + 1, 
                      (nRows - 1) / blockSize.y + 1);
					  
		timer.Start();
		if (kernelType == 1)
			addMatKernel1<<<gridSize, blockSize >>>(d_in1, d_in2, nRows, nCols, d_out);
		else
			addMatKernel2<<<gridSize, blockSize >>>(d_in1, d_in2, nRows, nCols, d_out);

		CHECK(cudaDeviceSynchronize());
		timer.St	op();
		
		// TODO: Copy result from device memory
        CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

		// TODO: Free device memories
        CHECK(cudaFree(d_in1));
        CHECK(cudaFree(d_in2));
        CHECK(cudaFree(d_out));
	}
	
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}

int main(int argc, char ** argv)
{
    int nRows, nCols ; // Matrix size
    int *in1, *in2; // Input matrixes
    int *out, *correctOut; // Output matrix

    // Input data into nRows and nCols
    nRows = 1 << 11 + 1;
    nCols = 1 << 12 + 1;
    printf("# rows = %d, # cols = %d\n\n", nRows, nCols);

    // Allocate memories for in1, in2, out
    size_t nBytes = nRows * nCols * sizeof(int);
    in1 = (int *)malloc(nBytes);
    in2 = (int *)malloc(nBytes);
    out = (int *)malloc(nBytes);
    correctOut = (int *)malloc(nBytes);

    // Input data into in1, in2
    for (int i = 0; i < nRows * nCols; i++)
    {
    	in1[i] = rand() & 0xff; // Random int in [0, 255]
    	in2[i] = rand() & 0xff;
    }

    // Add in1 & in2 on host
    addMat(in1, in2, nRows, nCols, correctOut);

    // Add in1 & in2 on device
	dim3 blockSize(32, 32); // Default
	if (argc == 3)
	{
		blockSize.x = atoi(argv[1]);
		blockSize.y = atoi(argv[2]);
	} 
    addMat(in1, in2, nRows, nCols, out, true, blockSize,1);
	addMat(in1, in2, nRows, nCols, out, true, blockSize,2);

    // Check correctness
    for (int i = 0; i < nRows * nCols; i++)
    {
    	if (out[i] != correctOut[i])
    	{
    		printf("INCORRECT :(\n");
    		return 1;
    	}
    }
    printf("CORRECT :)\n");
}
