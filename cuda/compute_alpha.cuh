#include "compute_kernel.cuh"

/*
 * Device code
 */

template< typename T>
__global__ void kernel_alpha(T* d_alpha, const T* d_df,  T lambda, T c, uint nA, uint n)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < nA && j < n)
	{
		uint idx = j*nA+i;

		//compute alpha_proximal 
		//alpha_proximal = alpha - 1/c*df;

		T temp = d_alpha[idx] = d_alpha[idx] - 1.0/c*d_df[idx];

		//alpha(alpha.^2 < 2*lambda/c) = 0;

		if ((temp*temp <= 2.0*lambda/c) || (i==j))
			d_alpha[idx] = 0.0f;
	}
}

// d_adjmat: size nn x n
__forceinline__ void compute_alpha(float *d_alpha, float *d_AtA, float *d_AtX, float lambda, float c, uint nA, uint n, cublasHandle_t& handle)
{
	float* d_df = NULL;
	cudaMalloc((void**)&d_df, sizeof(float)*nA*n);

	cudaMemcpy(d_df,d_AtX,sizeof(float)*nA*n,cudaMemcpyDeviceToDevice);	
	
	float ta=2.0f; float *alpha = &ta; float tb=-2.0f; float *beta = &tb;
	cublasStatus_t  status = cublasSgemm (
        handle,            // blas handle 
        CUBLAS_OP_N,    //  op A
        CUBLAS_OP_N,    // op B
        nA,                // op A rows 
        n,                // op B cols
        nA,                // op A cols op B rows
        alpha,                // alpha
        d_AtA,            // A address
        nA,                // lda
        d_alpha,            // B address
        nA,                // ldb
        beta,                // beta
        d_df,            // C address
        nA                // ldc
    );

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	//compute_df(d_df, d_AtA, d_alphai, nA, handle);

	
	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 blocksPerGrid((nA+threadsPerBlock.x-1)/threadsPerBlock.x, (n+threadsPerBlock.y-1)/threadsPerBlock.y);

	kernel_alpha<<<blocksPerGrid, threadsPerBlock>>>(d_alpha, d_df, lambda, c, nA, n);

	cudaFree(d_df);
}