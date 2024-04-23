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

#define NF 100
#define NI (1<<24)
#define NO (NI - NF + 1)
#define BLOCKSIZE 512
#define SMEMSIZE (BLOCKSIZE+NF-1)

__constant__ float const_flt[NF];

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

void convOnHost(float *h_in, float *h_out, float* h_flt)
{
	for (int i = 0;i<NO;i++)
	{
		float s = 0;
		for (int j = 0; j < NF; j++)
		{
			s += h_flt[j] * h_in[i+j];
		}
		h_out[i] = s;
	}
}

__global__ void conv1DKernel0(float *d_in, float *d_out,float* d_flt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NO)
	{
		d_out[i] = 0;
		for (int j = 0; j < NF; j++)
		{
			d_out[i]+= d_flt[j] * d_in[i + j];
		}
	}
}

//All use global memory
__global__ void conv1DKernel1(float *d_in, float *d_out,float* d_flt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NO)
	{
		float s = 0;
		for (int j = 0; j < NF; j++)
		{
			s += d_flt[j] * d_in[i + j];
		}
		d_out[i] = s;					
	}
}

// Filter use constant memory
__global__ void conv1DKernel2(float *d_in, float *d_out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NO)
	{
		float s = 0;
		for (int j = 0; j < NF; j++)
		{
			s += const_flt[j] * d_in[i + j];
		}
		d_out[i] = s;
	}
}

// use shared memory
__global__ void conv1DKernel3(float *d_in, float *d_out)
{
	__shared__ float s_Data[SMEMSIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	s_Data[threadIdx.x] = (i < NI) ? d_in[i]: 0;
	
	if (threadIdx.x < (NF - 1))
		s_Data[threadIdx.x+blockDim.x] = (i+blockDim.x < NI) ? d_in[i+blockDim.x]: 0;
	
	__syncthreads();
	
	if (i < NO)
	{
		float s = 0;
		for (int j = 0; j < NF; j++)
		{
			s += const_flt[j] * s_Data[threadIdx.x + j];
		}
		d_out[i] = s;
	}	
}

void conv1D(float* in, float* out, float* flt,
		bool useDevice=false, dim3 blockSize=dim3(1), int kernelType=1)
{

	GpuTimer timer;	
	if (useDevice == false)
	{
		convOnHost(in,out,flt);
	}
	else // Use device
	{
		// Allocate device memories
		float * d_in, * d_out,*d_flt;
		CHECK(cudaMalloc(&d_in, NI * sizeof(float)));
		CHECK(cudaMalloc(&d_out, NO * sizeof(float)));
		CHECK(cudaMalloc(&d_flt, NF * sizeof(float)));
		
		// Copy data to device memories
		CHECK(cudaMemcpy(d_in,in,NI * sizeof(float),cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_flt,flt,NF * sizeof(float),cudaMemcpyHostToDevice));
		cudaMemcpyToSymbol(const_flt, flt, NF * sizeof(float));
		
		dim3 gridSize((NO - 1) / blockSize.x + 1); // Similarity, view gridSize as a number

		timer.Start();
		// Call kernel
		if (kernelType == 0)
		{
			conv1DKernel0<<<gridSize, blockSize>>>(d_in, d_out,d_flt);
		}
		else if (kernelType == 1)
		{
			conv1DKernel1<<<gridSize, blockSize>>>(d_in, d_out,d_flt);
		}
		else if (kernelType == 2)
		{
			
			conv1DKernel2<<<gridSize, blockSize>>>(d_in, d_out);
		}
		else 
		{
			conv1DKernel3<<<gridSize, blockSize>>>(d_in, d_out);
		}
		CHECK(cudaDeviceSynchronize();)
		timer.Stop();
		float time = timer.Elapsed();
		CHECK(cudaGetLastError());
		
		// Copy result from device memories
		CHECK(cudaMemcpy(out, d_out, NO * sizeof(float), cudaMemcpyDeviceToHost));
		
		// Free device memories
		CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_out));
		CHECK(cudaFree(d_flt));		
			
		printf("Processing time: %f ms\n", time);
	}
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

int main(int argc, char ** argv)
{
	printDeviceInfo();

	// Allocate memories for input, filter, output; set up data for input, filter
	float *in, *flt, *out,*correctOut;
	in = (float*)malloc(NI * sizeof(float));
	out = (float*)malloc(NO * sizeof(float));
	correctOut = (float*)malloc(NO * sizeof(float));
	flt = (float*)malloc(NF * sizeof(float));

	for (int i = 0; i < NI; i++)
	{        
		in[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}	
	for (int i = 0; i < NF; i++)
	{        
		flt[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}	
	
	 // Conv1D NOT using device
	printf("Convolution 1D using host\n");
	conv1D(in, correctOut,flt);
	printf("\n");

	dim3 blockSize(BLOCKSIZE);
	float err = 0;

	//Conv1D using device, kernel0
	printf("Convolution 1D, with global memory & local memory\n");
	conv1D(in, out, flt, true, blockSize,0);
	err = checkCorrectness(out, correctOut,NO);
	printf("Error between device result and host result: %f\n\n", err);	
	
	// Conv1D using device, kernel1
	printf("Convolution 1D, with global memory, more register\n");
	conv1D(in, out, flt, true, blockSize,1);
	err = checkCorrectness(out, correctOut,NO);
	printf("Error between device result and host result: %f\n\n", err);	

	//Conv1D using device, kernel2
	printf("Convolution 1D, with constant memory for filter\n");
	conv1D(in, out, flt,true, blockSize, 2);
	err = checkCorrectness(out, correctOut,NO);
	printf("Error between device result and host result: %f\n\n", err);	
	
	//Conv1D using device, kernel3
	printf("Convolution 1D, with constant memory and shared memory\n");
	conv1D(in, out, flt,true, blockSize, 3);
	err = checkCorrectness(out, correctOut,NO);
	printf("Error between device result and host result: %f\n\n", err);	
	
	// Free memories
	free(in);
	free(out);
	free(flt);
	
	CHECK(cudaDeviceReset());
}
