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

void squareVecOnHost(float *in,float* out, int n)
{
	for (int i = 0;i<n;i++)
		out[i] = in[i] * in[i];
}

__global__ void square_kernel(float *in,float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n)    
        out[i] = in[i] * in[i];    
}

void squareVec(float* in, float* out, int n, bool useDevice=false)
{
	if (useDevice == false)
	{
		squareVecOnHost(in,out,n);
	}
	else
	{
		float *d_in, *d_out;
		size_t nBytes = n * sizeof(float);

		CHECK(cudaMalloc(&d_in, nBytes));
		CHECK(cudaMalloc(&d_out, nBytes));

		// Declare and create stream
		cudaStream_t stream1, stream2;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
	
		// Setup kernel configuration
		dim3 blockSize(512); // For simplicity, you can temporarily view blockSize as a number
		dim3 gridSize((n - 1) / blockSize.x + 1); // Similarity, view gridSize as a number

		//Process first part
		CHECK(cudaMemcpyAsync(d_in,in,nBytes/2,cudaMemcpyHostToDevice, stream1));
		square_kernel<<<gridSize,blockSize,0,stream1>>>(d_in,d_out,n/2);
		CHECK(cudaMemcpyAsync(out,d_out,nBytes/2,cudaMemcpyDeviceToHost, stream1));
	
		//Process second part
		int start = n/2;
		int len = n - n/2;
		CHECK(cudaMemcpyAsync(&d_in[start],&in[start],len * sizeof(float),cudaMemcpyHostToDevice, stream2));
		square_kernel<<<gridSize,blockSize,0,stream2>>>(&d_in[start],&d_out[start],len);
		CHECK(cudaMemcpyAsync(&out[start],&d_out[start],len * sizeof(float),cudaMemcpyDeviceToHost, stream2));		
		
		// Destroy device streams
		CHECK(cudaStreamSynchronize(stream1));
		cudaStreamDestroy(stream1);
		CHECK(cudaStreamSynchronize(stream2));
		cudaStreamDestroy(stream2);		
	}
}
		
float checkCorrectness(float * a1, float* a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)	
	{
		// printf("%f: %f\n",a1[i],a2[i]);
		err += abs(a1[i] - a2[i]);
	}
	err /= n;
	return err;
}
	
int main()
{	
	float *in; // Input vectors
    float *out,*correctOut;  // Output vector
	int N = (1<<20);

    // Allocate memories for in, out, and correctOut
    size_t nBytes = N * sizeof(float);
    CHECK(cudaHostAlloc(&in, nBytes,cudaHostAllocDefault));
    CHECK(cudaHostAlloc(&out,nBytes,cudaHostAllocDefault));
    correctOut = (float*)malloc(nBytes);
    
    // Input data into in1, in2
    for (int i = 0; i < N; i++)    
    	in[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        
	squareVec(in,correctOut,N,false);
	squareVec(in,out,N,true);
	printf("Error: %f",checkCorrectness(out,correctOut,N));	
	
	CHECK(cudaFreeHost(in));
	CHECK(cudaFreeHost(out));
	free(correctOut);	
}