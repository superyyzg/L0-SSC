//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include "compute_kernel.cuh"
#include "compute_alpha.cuh"
#include "compute_obj_robust.cuh"
#include "compute_error_coef.cuh"
#include "utility.h"






int main(int argc, char *argv[])
{
	cudaSetDevice(0);
	
	float lambda = static_cast<float>(atof(argv[1]));
	int maxIter = atoi(argv[2]);

	bool verbose = false;
	if (argc == 4)
		verbose = static_cast<bool>(atoi(argv[3]));		

	MATFile *proximal_l0graph_input = matOpen("proximal_l0graph_input.mat","r");
	mxArray *XArray = matGetVariable(proximal_l0graph_input, "X");
	mxArray *l1graph_alphaArray = matGetVariable(proximal_l0graph_input, "alpha");
	mxArray *AArray = matGetVariable(proximal_l0graph_input, "A");
	mxArray *AtAArray = matGetVariable(proximal_l0graph_input, "AtA");
	mxArray *AtXArray = matGetVariable(proximal_l0graph_input, "AtX");
	mxArray *S1Array = matGetVariable(proximal_l0graph_input, "S1");
	mxArray *thrArray = matGetVariable(proximal_l0graph_input, "thr");

	float *h_X = static_cast<float*>(mxGetData(XArray));
	float *h_l1graph_alpha = static_cast<float*>( mxGetData(l1graph_alphaArray));
	float *h_A = static_cast<float*>( mxGetData(AArray));
	float *h_AtA = static_cast<float*>( mxGetData(AtAArray));
	float *h_AtX = static_cast<float*>( mxGetData(AtXArray));
	float S1 = *(static_cast<float*>( mxGetData(S1Array)));
	float thr = *(static_cast<float*>( mxGetData(thrArray)));
	
	

	const mwSize *Xsize = mxGetDimensions(XArray);
	uint d = static_cast<uint>(Xsize[0]);
	uint n = static_cast<uint>(Xsize[1]);
	const mwSize *Asize = mxGetDimensions(AArray);
	uint nA = static_cast<uint>(Asize[1]);	

	float *d_X = NULL, *d_alpha = NULL, *d_A = NULL, *d_AtA = NULL, *d_AtX = NULL;
	float *d_alpha0 = NULL;

	cudaMalloc((void**)&d_X,				sizeof(float)*d*n);
	cudaMalloc((void**)&d_alpha,			sizeof(float)*nA*n);
	cudaMalloc((void**)&d_A,				sizeof(float)*d*nA);
	cudaMalloc((void**)&d_AtA,				sizeof(float)*nA*nA);
	cudaMalloc((void**)&d_AtX,				sizeof(float)*nA*n);
	cudaMalloc((void**)&d_alpha0,			sizeof(float)*nA*n);
	
	
	//debug
	//float *h_alpha = (float*)malloc(sizeof(float)*(nA*n));
	
	
	cudaMemcpy(d_X,h_X,sizeof(float)*d*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_alpha,h_l1graph_alpha,sizeof(float)*nA*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_A,h_A,sizeof(float)*d*nA,cudaMemcpyHostToDevice);
	cudaMemcpy(d_AtA,h_AtA,sizeof(float)*nA*nA,cudaMemcpyHostToDevice);
	cudaMemcpy(d_AtX,h_AtX,sizeof(float)*nA*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_alpha0,h_l1graph_alpha,sizeof(float)*nA*n,cudaMemcpyHostToDevice);

	
	
	matClose(proximal_l0graph_input);


	// create and initialize CUBLAS library object 
	cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            std::cout << "CUBLAS object instantialization error" << std::endl;
        }
        getchar ();
        return 0;
    }

	float elapsedTime = 0;
	
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);	

	float err = 0.0f;

	for (int iter = 0; iter < maxIter; iter++)
	{
		float c = 2.0f;
		compute_alpha(d_alpha, d_AtA, d_AtX, lambda, c, nA, n, handle);

		//debug
		//cudaMemcpy(h_alpha,d_alpha,sizeof(float)*nA*n,cudaMemcpyDeviceToHost);
		
		compute_error_coef(&err, d_alpha0, d_alpha, nA, n, handle);

		cudaMemcpy(d_alpha0,d_alpha,sizeof(float)*nA*n,cudaMemcpyDeviceToHost);


		float obj = 0.0f, l2err = 0.0f, l0_spar_err = 0.0f;
		compute_obj_robust(obj, l2err, l0_spar_err, d_X, d_A, d_alpha, lambda, nA, n, d, handle);

		printf("proximal_manifold: errors = %.5f, iter: %d \n",err,iter);

		if (verbose)
		{
			printf("obj is %.5f, l2err is %.5f, spar_err is %.5f \n", obj,l2err,l0_spar_err);
		}

		/*if (err < thr)
		{
			if (verbose)
			{
				printf("proximal_l0graph converges at iter %d \n", iter);
			}
			break;
		}*/		
		
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
	//cudaDeviceSynchronize();
	//finishTime=clock();
	//elapsedTime =(float)(finishTime - startTime);

	// Clean up:
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	printf("time to compute on gpu is %.10f second\n",elapsedTime/(CLOCKS_PER_SEC));




	MATFile *rFile = matOpen("proximal_l0graph_result.mat","w");	
	
	mxArray* alphaoutArray = mxCreateNumericMatrix(nA,n, mxSINGLE_CLASS, mxREAL);
	
	float *h_alphaout   = (float*)mxGetData(alphaoutArray);
	
	//transfer the data from gpu to cpu
	//set Z as the output for its sparsity
	cudaMemcpy(h_alphaout,d_alpha,sizeof(float)*nA*n,cudaMemcpyDeviceToHost);

	matPutVariable(rFile, "l0graph_alpha", alphaoutArray);
	matClose(rFile);

	mxDestroyArray(alphaoutArray);

	//destroy the input matlab Arrays
	mxDestroyArray(XArray);
	mxDestroyArray(l1graph_alphaArray);
	mxDestroyArray(AArray);
	mxDestroyArray(AtAArray);
	mxDestroyArray(AtXArray);
	mxDestroyArray(S1Array);
	mxDestroyArray(thrArray);


	//deallocation

	//debug
	//free(h_alphai_proximal);

	cudaFree(d_X);
	cudaFree(d_alpha);
	cudaFree(d_A);
	cudaFree(d_AtA);
	cudaFree(d_AtX);
	cudaFree(d_alpha0);
	
	return 0;
}