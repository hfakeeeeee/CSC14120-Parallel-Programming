#include<iostream>
class Fraction {
	int numerator;
	int denominator;
public:
	Fraction() = default;
	__host__ __device__ Fraction(int a, int b) : x(a), y(b) {}
	__host__ __device__ Fraction square() {
		return Fraction(x * x, y * y);
	}
	__host__ void print() { std::cout << x << "/" << y << "\n"; }
};

__global__ void square(Fraction* array, int n) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		array[tid] = array[tid].square();
}

int main(void) {
	Fraction* h_arr, * d_arr;
	h_arr = new Fraction[10];

	for (int i = 0; i < 10; ++i)
		h_arr[i] = Fraction(i, i - 1);

	for (int i = 0; i < 10; ++i)
		h_arr[i].print();
	std::cout<<std::endl;
	// Sends data to device
	cudaMalloc(&d_arr, 10 * sizeof(Fraction));
	cudaMemcpy(d_arr, h_arr, 10 * sizeof(Fraction), cudaMemcpyHostToDevice);

	// Runs kernel on device
	square <<< 2, 5 >>> (d_arr, 10);

	// Retrieves data from device
	cudaMemcpy(h_arr, d_arr, 10 * sizeof(Fraction), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; ++i)
		h_arr[i].print();

	cudaFree(d_arr);
	delete[] h_arr;
	return 0;
}