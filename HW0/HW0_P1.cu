#include <stdio.h>

void printDeviceInfo() {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("GPU card name: %s\n", devProp.name);
    printf("GPU computation capabilities %d.%d\n", devProp.major, devProp.minor);
    printf("Maximum Number of Block Dimensions: %d\n", devProp.maxThreadsDim[0]);
    printf("Maximum Number of Grid Dimensions: %d\n", devProp.maxGridSize[0]);
    printf("Maximum Size of GPU Memory: %zu bytes\n", devProp.totalGlobalMem);
    printf("Amount of Constant Memory: %zu bytes\n", devProp.totalConstMem);
    printf("Warp Size: %d\n", devProp.warpSize);
}

int main(int argc, char **argv) {
    printDeviceInfo();
    cudaDeviceReset(); 
    return 0;
}
