#pragma once
#include "cuda_runtime.h"
#include <cuda.h>
#include "cublas_v2.h"
#include <ctime>
#include <iostream>

#define BLOCK_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
//Diagonal coordinates for avoiding the partition camping in kernel_transpose
//#define DIAGONAL_COORDINATE_TRANSPOSE
typedef unsigned int uint;


#define eps 2.2204e-16

/*
 * Device code
 */

template< typename T>
__global__ void kernel_transpose_2d(T *dst, const T *src, uint Rows, uint Cols) {
    __shared__ T block[BLOCK_SIZE][BLOCK_SIZE+1];

	uint blockIdx_x, blockIdx_y;
	
#ifdef DIAGONAL_COORDINATE_TRANSPOSE
	if (Rows == Cols)
	{
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
	} else
	{
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid%gridDim.y;
		blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
	}
#else
	blockIdx_x = blockIdx.x; blockIdx_y = blockIdx.y;
#endif

    // read the matrix tile into shared memory
    uint xIndex = blockIdx_x * blockDim.x + threadIdx.x;
    uint yIndex = blockIdx_y * blockDim.y + threadIdx.y;
	

    if((xIndex < Rows) && (yIndex < Cols)) {
        uint index_in = yIndex * Rows + xIndex;

        block[threadIdx.x][threadIdx.y] = src[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx_y * blockDim.y + threadIdx.x;
    yIndex = blockIdx_x * blockDim.x + threadIdx.y;
	

    if((xIndex < Cols) && (yIndex < Rows)) {
        uint index_out = yIndex * Cols + xIndex;

        dst[index_out] = block[threadIdx.y][threadIdx.x];
    }
}


template< typename T>
__global__ void kernel_abs( T* dst, T* src, uint Rows, uint Cols)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < Rows && j < Cols)
	{
		uint idx = j*Rows+i;
		dst[idx]  = abs(src[idx]);
	}
}

template< typename T>
__global__ void kernel_add( T* dst, T* src, T added_value, uint Rows, uint Cols)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < Rows && j < Cols)
	{
		uint idx = j*Rows+i;
		dst[idx]  = src[idx] + added_value;
	}
}

template< typename T>
__global__ void kernel_add( T* dst, T added_value, uint Rows, uint Cols)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < Rows && j < Cols)
	{
		uint idx = j*Rows+i;
		dst[idx]  = dst[idx] + added_value;
	}
}

// dist = op1 - op2
template< typename T>
__global__ void kernel_subtract( T* dst, const T* op1, const T* op2, uint Rows, uint Cols)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < Rows && j < Cols)
	{
		uint idx = j*Rows+i;
		dst[idx]  = op1[idx] - op2[idx];
	}
}

// dist = op1 - op2
template< typename T>
__global__ void kernel_subtract( T* dst, const T* op1, const T* op2, uint Rows)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < Rows)
	{
		dst[i]  = op1[i] - op2[i];
	}
}

template< typename T>
__global__ void kernel_filter_zero( T *dst, uint Rows, uint Cols)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < Rows && j < Cols)
	{
		uint idx = j*Rows+i;
		if (dst[idx] < (T)0) 
			dst[idx] = (T)0;
	}
}



