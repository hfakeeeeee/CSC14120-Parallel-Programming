#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4194304

// Initialize input arrays
void initializeInputArrays(float* in1, float* in2, int n) {
    for (int i = 0; i < n; i++) {
        in1[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        in2[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

// Calc Host Time
float calculateHostTime(float* in1, float* in2, float* out, int n) {
    clock_t start = clock();
    
    for (int i = 0; i < n; i += 2) {
        out[i] = in1[i] + in2[i];
        out[i + 1] = in1[i + 1] + in2[i + 1];
    }
    
    clock_t end = clock();
    
    return ((float)(end - start) / CLOCKS_PER_SEC) * 1000.0;
}

// Add vector (Ver1)
__global__ void addVecOnDeviceVersion1(float* in1, float* in2, float* out, int n) {
    int threadID = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    if (threadID < n) {
        out[threadID] = in1[threadID] + in2[threadID];
        int threadID_2 = threadID + blockDim.x;
        
        if (threadID_2 < n) {
            out[threadID_2] = in1[threadID_2] + in2[threadID_2];
        }
    }
}

// Add vector (Ver2)
__global__ void addVecOnDeviceVersion2(float* in1, float* in2, float* out, int n) {
    int threadID = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    
    if (threadID < n) {
        out[threadID] = in1[threadID] + in2[threadID];
        int threadID_2 = threadID + 1;
        
        if (threadID_2 < n) {
            out[threadID_2] = in1[threadID_2] + in2[threadID_2];
        }
    }
}

// Calculate device time (Ver1)
float calculateDeviceTimeVersion1(float* in1, float* in2, float* out, int n) {
    float *d_in1, *d_in2, *d_out;
    size_t nBytes = n * sizeof(float);
    cudaEvent_t start, stop;
    
    // Allocate memory
    cudaMalloc((void**)&d_in1, nBytes);
    cudaMalloc((void**)&d_in2, nBytes);
    cudaMalloc((void**)&d_out, nBytes);
    
    cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, in2, nBytes, cudaMemcpyHostToDevice);
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch the kernel
    dim3 blockSize(256);
    dim3 gridSize((n - 1) / (2 * blockSize.x) + 1);
    
    cudaEventRecord(start, 0);
    addVecOnDeviceVersion1<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, n);
    cudaEventRecord(stop, 0);
    
    // Synchronize device and calc time
    cudaDeviceSynchronize();
    float deviceTime;
    cudaEventElapsedTime(&deviceTime, start, stop);
    cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
    
    return deviceTime;
}

// Calculate device time (Ver2)
float calculateDeviceTimeVersion2(float* in1, float* in2, float* out, int n) {
    float *d_in1, *d_in2, *d_out;
    size_t nBytes = n * sizeof(float);
    cudaEvent_t start, stop;
    
    // Allocate memory
    cudaMalloc((void**)&d_in1, nBytes);
    cudaMalloc((void**)&d_in2, nBytes);
    cudaMalloc((void**)&d_out, nBytes);
    
    cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, in2, nBytes, cudaMemcpyHostToDevice);
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch the kernel
    dim3 blockSize(256);
    dim3 gridSize((n - 1) / (2 * blockSize.x) + 1);
    
    cudaEventRecord(start, 0);
    addVecOnDeviceVersion2<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, n);
    cudaEventRecord(stop, 0);
    
    // Synchronize device and calc time
    cudaDeviceSynchronize();
    float deviceTime;
    cudaEventElapsedTime(&deviceTime, start, stop);
    cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
    
    return deviceTime;
}

int main() {
    srand(static_cast<unsigned>(time(NULL)));
    
    int arraySizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    
    for (int i = 0; i < sizeof(arraySizes) / sizeof(arraySizes[0]); i++) {
        int n = arraySizes[i];
        
        float *in1, *in2, *out;
        
        // Allocate memory on the host
        in1 = (float*)malloc(sizeof(float) * n);
        in2 = (float*)malloc(sizeof(float) * n);
        out = (float*)malloc(sizeof(float) * n);
        
        // Initialize input arrays 
        initializeInputArrays(in1, in2, n);
        
        // Calc host time
        float hostTime = calculateHostTime(in1, in2, out, n);
        
        // Calc device time for Ver1
        float deviceTimeVersion1 = calculateDeviceTimeVersion1(in1, in2, out, n);
        
        // Calc device time for Ver2
        float deviceTimeVersion2 = calculateDeviceTimeVersion2(in1, in2, out, n);
        
        printf("Vector size: %d\n", n);
        printf("Host time: %f\n", hostTime);
        printf("Device time (Version 1): %f\n", deviceTimeVersion1);
        printf("Device time (Version 2): %f\n", deviceTimeVersion2);
        
        // Free memory
        free(in1);
        free(in2);
        free(out);
    }
    
    return 0;
}