#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
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

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

__device__ int blockCount = 0;
volatile __device__ int blockCount1 = 0;

__global__ void eScan(int * in, int * out, int n, volatile int * blockSums) {
    __shared__ int blockIndex; 

    if (threadIdx.x == 0) {
        blockIndex = atomicAdd(&blockCount, 1);
    }
    __syncthreads();

    extern __shared__ int sharedData[];
    int index1 = blockIndex * 2 * blockDim.x + threadIdx.x;
    int index2 = index1 + blockDim.x;

    if (index1 < n && index1 >= 1) {
        sharedData[threadIdx.x] = in[index1 - 1];
    }
    else {
        sharedData[threadIdx.x] = 0;
    }
    if (index2 < n && index2 >= 1) {

        sharedData[threadIdx.x + blockDim.x] = in[index2 - 1];
    }
    else {
        sharedData[threadIdx.x + blockDim.x] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2) {
        int sharedDataIndex = (threadIdx.x + 1) * 2 * stride - 1;
        if (sharedDataIndex < 2 * blockDim.x) {
            sharedData[sharedDataIndex] += sharedData[sharedDataIndex - stride];
        }
        __syncthreads();
    }

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int sharedDataIndex = (threadIdx.x + 1) * 2 * stride - 1 + stride;
        if (sharedDataIndex < 2 * blockDim.x) {
            sharedData[sharedDataIndex] += sharedData[sharedDataIndex - stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        blockSums[blockIndex] = sharedData[2 * blockDim.x - 1];
        if (blockIndex > 0) {
            while (blockCount1 < blockIndex) {} 
            blockSums[blockIndex] += blockSums[blockIndex-1];
            __threadfence(); 
        }

        blockCount1 += 1;    
    }
    __syncthreads();

    if (index1 < n) {
        out[index1] = sharedData[threadIdx.x] + blockSums[blockIndex - 1];
    }
    if (index2 < n) {
        out[index2] = sharedData[threadIdx.x + blockDim.x] + blockSums[blockIndex - 1];
    }
}

__global__ void computeBits(uint32_t * src, int * bits, int bitIdx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        bits[i] = (src[i] >> bitIdx) & 1;
    }
}

__global__ void computeRank(int * bits, int * nOnesBefore, int n, uint32_t * src, uint32_t * dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nZeros = n - nOnesBefore[n-1] - bits[n-1];

    if (i < n) {
        int rank;
        if (bits[i] == 0) {
            rank = i - nOnesBefore[i];
        }
        else {
            rank = nZeros + nOnesBefore[i];
        }
        dst[rank] = src[i];
    }
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize) {
    int gridDimensions = (n - 1) / blockSize + 1;

    // Allocate device memories
    uint32_t * deviceSrc, * deviceDst;

    CHECK(cudaMalloc(&deviceSrc, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&deviceDst, n * sizeof(uint32_t)));
    CHECK(cudaMemcpy(deviceSrc, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    int * deviceBits, * deviceOnesBefore;
    CHECK(cudaMalloc(&deviceBits, n * sizeof(int)));
    CHECK(cudaMalloc(&deviceOnesBefore, n * sizeof(int)));

    int bytesBlockSums = gridDimensions * sizeof(int);
    int * deviceBlockSums;
    CHECK(cudaMalloc(&deviceBlockSums, bytesBlockSums));
    
    for (int bitIndex = 0; bitIndex < sizeof(uint32_t) * 8; bitIndex++)
    {
        int reset = 0;
        CHECK(cudaMemcpyToSymbol(blockCount, &reset, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(blockCount1, &reset, sizeof(int)));

        computeBits<<<gridDimensions, blockSize>>>(deviceSrc, deviceBits, bitIndex, n);
        CHECK(cudaDeviceSynchronize());

        if (gridDimensions == 1)
        {
            deviceBlockSums = NULL;
        }

        size_t sharedMemoryBytes = 2 * blockSize * sizeof(int);
        eScan<<<gridDimensions, blockSize, sharedMemoryBytes>>>(deviceBits, deviceOnesBefore, n, deviceBlockSums);
        CHECK(cudaDeviceSynchronize());

        computeRank<<<gridDimensions, blockSize>>>(deviceBits, deviceOnesBefore, n, deviceSrc, deviceDst);
        CHECK(cudaDeviceSynchronize());

        uint32_t * temporary = deviceSrc;
        deviceSrc = deviceDst;
        deviceDst = temporary;
    }

    CHECK(cudaMemcpy(out, deviceSrc, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(deviceDst));
    CHECK(cudaFree(deviceBits));
    CHECK(cudaFree(deviceOnesBefore));
    CHECK(cudaFree(deviceBlockSums));
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1) {
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
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
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    //int n = 50; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        //in[i] = rand() % 255; // For test by eye
        in[i] = rand();
    }
    //printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    //printArray(correctOut, n); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    //printArray(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
