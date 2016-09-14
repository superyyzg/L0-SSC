#include "compute_kernel.cuh"

/*
 * Device code
 */

template< typename T>
__global__ void kernel_l0_spar_err( T* d_l0_spar_err_mat, const T* d_alpha, T lambda, uint nA, uint n)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < nA && j < n)
	{
		uint idx = j*nA+i;
		d_l0_spar_err_mat[idx]  = lambda*(T)(abs(d_alpha[idx]) > eps);
	}
}


/*
function [obj,l2err,l1_spar_err,l0_spar_err] = compute_obj_robust(X,i,A,alphai,alpha_neighbor,W0_neighbor,lambda_l1,lambda_l0)
    n = size(alphai,1);
    l2err = norm(X(:,i)-A*alphai,'fro')^2;
    l1_spar_err = lambda_l1*sum(abs(alphai));
    
    nn = size(alpha_neighbor,2);
    l0_spar_err = lambda_l0*(abs(repmat(alphai,1,nn) - alpha_neighbor)>eps).*repmat(W0_neighbor,n,1);
    l0_spar_err = sum(l0_spar_err(:));
    obj = l2err + l1_spar_err + l0_spar_err;
end
*/

__forceinline__ void compute_obj_robust(float& obj, float& l2err, float& l0_spar_err, float *d_X, float *d_A, float *d_alpha, float lambda, uint nA, uint n, uint d, cublasHandle_t& handle)
{
	float* d_X_A_alpha = NULL, *d_l0_spar_err_mat = NULL;
	cudaMalloc((void**)&d_X_A_alpha, sizeof(float)*d*n);
	cudaMalloc((void**)&d_l0_spar_err_mat, sizeof(float)*nA*n);

	cudaMemcpy(d_X_A_alpha,d_X,sizeof(float)*d*n,cudaMemcpyDeviceToHost);

	
	float ta=-1.0f; float *alpha = &ta; float tb=1.0f; float *beta = &tb;
	cublasStatus_t  status = cublasSgemm (
        handle,            // blas handle 
        CUBLAS_OP_N,    //  op A
        CUBLAS_OP_N,    // op B
        d,                // op A rows 
        n,                // op B cols
        nA,                // op A cols op B rows
        alpha,                // alpha
        d_A,            // A address
        d,                // lda
        d_alpha,            // B address
        nA,                // ldb
        beta,                // beta
        d_X_A_alpha,            // C address
        d                // ldc
    );

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	

	status = cublasSnrm2(handle, d*n, d_X_A_alpha, 1, &l2err);
	l2err = l2err*l2err;
	//debug
	//printf("l2err1 = %.5f \n", l2err);

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 blocksPerGrid( (nA+threadsPerBlock.x-1)/threadsPerBlock.x, (n+threadsPerBlock.y-1)/threadsPerBlock.y);

	kernel_l0_spar_err<<<blocksPerGrid, threadsPerBlock>>>(d_l0_spar_err_mat, d_alpha, lambda, nA, n);

	status = cublasSasum(handle, nA*n, d_l0_spar_err_mat, 1, &l0_spar_err);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	obj = l2err + l0_spar_err;

	cudaFree(d_X_A_alpha);
	cudaFree(d_l0_spar_err_mat);

}