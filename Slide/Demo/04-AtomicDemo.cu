#include <stdio.h>

__global__ void naive_incr(int *value) {
	int temp = *value;
	temp = temp + 1;
	*value = temp;
}

__global__ void atomic_incr(int *value) {
	atomicAdd(value,1);
}

int main(int argc, char **argv)
{
	int* h_a,*d_a;
	h_a = (int*)malloc(sizeof(int));
	*h_a = 0;
	printf("Before: %d\n",*h_a);
	cudaMalloc((void**)&d_a,sizeof(int));
	cudaMemcpy(d_a,h_a,sizeof(int),cudaMemcpyHostToDevice);

    naive_incr<<<1, 64>>>(d_a); // 1 group of 64 threads do this function in parallel
    // atomic_incr<<<1, 64>>>(d_a); // 1 group of 64 threads do this function in parallel
	
	cudaMemcpy(h_a,d_a,sizeof(int),cudaMemcpyDeviceToHost);
	printf("After: %d\n",*h_a);
	
    cudaDeviceReset(); // Force to print
    return 0;
}

