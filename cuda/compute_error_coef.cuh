#include "compute_kernel.cuh"

/*
 * Device code
 */

__forceinline__ void compute_error_coef(float *err, float *d_a, float *d_b, uint rows, uint cols, cublasHandle_t& handle)
{
	float *d_diff = NULL;
	cudaMalloc((void**)&d_diff, sizeof(float)*rows*cols);

	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 blocksPerGrid( (rows+threadsPerBlock.x-1)/threadsPerBlock.x, (cols+threadsPerBlock.y-1)/threadsPerBlock.y);
	kernel_subtract<<<blocksPerGrid, threadsPerBlock>>>(d_diff, d_a, d_b, rows, cols);

	cublasStatus_t status = cublasSasum(handle, rows*cols, d_diff, 1, err);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	*err = (float)(*err)/((float)rows*cols);
	cudaFree(d_diff);
}