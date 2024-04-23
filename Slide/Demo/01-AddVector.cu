#include <stdio.h>
#define N 4194304
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

void addVecOnHost(float* in1, float* in2, float* out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = in1[i] + in2[i];    
}

__global__ void addVecOnDevice(float* in1, float* in2, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = in1[i] + in2[i];
}

void addVec(float* in1, float* in2, float* out, int n, 
        bool useDevice=false)
{
	GpuTimer timer;
	
	if (useDevice == false)
	{
		timer.Start();
		addVecOnHost(in1, in2, out, n);
		timer.Stop();        
	}
	else // Use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// Host allocates memories on device
		float *d_in1, *d_in2, *d_out;
        size_t nBytes = n * sizeof(float);
        CHECK(cudaMalloc(&d_in1, nBytes));
        CHECK(cudaMalloc(&d_in2, nBytes));
        CHECK(cudaMalloc(&d_out, nBytes));

		// Host copies data to device memories
        CHECK(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_in2, in2, nBytes, cudaMemcpyHostToDevice));

		// Host invokes kernel function to add vectors on device
		dim3 blockSize(512); // For simplicity, you can temporarily view blockSize as a number
		dim3 gridSize((n - 1) / blockSize.x + 1); // Similarity, view gridSize as a number
		
		timer.Start();
		addVecOnDevice<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, n); 
		
		cudaDeviceSynchronize(); 
		timer.Stop();
		// Host copies result from device memory
        CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

		// Free device memories
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
    float *in1, *in2; // Input vectors
    float *out,*correctOut;  // Output vector

    
    // Allocate memories for in1, in2, out
    size_t nBytes = N * sizeof(float);
    in1 = (float *)malloc(nBytes);
    in2 = (float *)malloc(nBytes);
    out = (float *)malloc(nBytes);
    correctOut = (float *)malloc(nBytes);
    
    // Input data into in1, in2
    for (int i = 0; i < N; i++)
    {
    	in1[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    	in2[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    
    // Add vectors (on host)
    addVec(in1, in2, correctOut, N);
	
    // Add in1 & in2 on device
    addVec(in1, in2, out, N, true);

    // Check correctness
    for (int i = 0; i < N; i++)
    {
    	if (out[i] != correctOut[i])
    	{
    		printf("INCORRECT :(\n");
    		return 1;
    	}
    }
    printf("CORRECT :)\n");
}
