#include <stdio.h>

//Compute multiplication of 2 matrices
__global__ void kmatbymat(const int nThreads, const float *m1, const float *m2, float *output) {
    
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = m1[i] * m2[i];
	  }
}

__device__ float* dmatbymat(const float *m1, const float *m2, float *output, const int width, const int height){

	kmatbymat <<< width, height >>> ( width * height, m1, m2, output );
    cudaDeviceSynchronize();
    return output;
}

//Compute subtraction of 2 matrices
__global__ void kmatsub(const int nThreads, const float *m1, const float *m2, float *output) {
    

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = m1[i] - m2[i];
	  }
}

__device__ float* dmatsub(const float *m1, const float *m2, float *output, const int width, const int height){

	kmatsub <<< width, height >>> ( width * height, m1, m2, output );
    cudaDeviceSynchronize();
    return output;
}

//Sigmoid function calculation: f(x) = 1/(1 + e^-x)
__global__ void ksigmoid(const int nThreads, float const *input, float *output){
    

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = 1.0 / (1.0 + std::exp(-input[i]));
	  }
}

__device__ void dsigmoid(float const *input, float *output, const int height, const int width){

	ksigmoid <<< height, width >>> (height * width, input, output);
	cudaDeviceSynchronize();
}

//Sigmoid function derivative calculation: f'(x) = f(x)(1 - f(x))
__global__ void kdersigmoid(const int nThreads, float const *input, float *output) {
	

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = input[i] * (1 - input[i]);
	  }
}

__device__ float* ddersigmoid(float const *input, float *output, const int rows, const int columns){
	kdersigmoid <<< rows, columns >>> (rows*columns, input, output);
	cudaDeviceSynchronize();
	return output;
}

//Dot product computation
__global__ void kdotprod(const int nThreads, const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_columns ){
	

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	{
	    int r = (int)i / m2_columns;
	    int c = i % m2_columns;
	    float t_output = 0.f;

	    for( int k = 0; k < m1_columns; ++k ) {
	        t_output += m1[ r * m1_columns + k ] * m2[ k * m2_columns + c ];
	    }

	    output[i] = t_output;
	}
}

__device__ float* ddotprod(const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_columns ){

	kdotprod <<< m1_rows, m2_columns >>> (m1_rows * m2_columns, m1, m2, output, m1_rows , m1_columns, m2_columns );
	cudaDeviceSynchronize();
	return output;
}

__global__ void km1dotm2T(const int nThreads, const float *m1, const float *m2, float *output, const int m1_columns, const int m2_rows ){
	

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	{
		int r = (int)i / m2_rows;
		int c = i % m2_rows;
		float t_output = 0.0;
		int id_T;

		for( int k = 0; k < m1_columns; ++k ) {
			id_T = c * m1_columns + k;
			t_output += m1[ r * m1_columns + k ] * m2[ id_T ];
		}

		output[i] = t_output;
	}
}

__device__ float* dm1dotm2T(const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_rows )
{
	km1dotm2T <<< m1_rows, m2_rows >>> ( m1_rows * m2_rows, m1, m2, output, m1_columns, m2_rows );
	cudaDeviceSynchronize();
	return output;
}

__global__ void km1Tdotm2(const int nThreads, const float *m1, const float *m2, float *output, const int m1_rows,
							const int m1_columns, const int m2_columns ){
	

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	{
	    int r = (int)i / m2_columns;
	    int c = i % m2_columns;
	    int id_T;
	    float t_output = 0.0;

	    for( int k = 0; k < m1_rows; ++k ) {
	    	id_T = k * m1_columns + r;
	        t_output += m1[ id_T ] * m2[ k * m2_columns + c ];
	    }

	    output[i] += t_output;
	}
}

__device__ void dm1Tdotm2(const float *m1, const float *m2, float *output, const int m1_height , const int m1_width, const int m2_width )
{
	km1Tdotm22 <<< m1_width, m2_width >>> (m1_width * m2_width, m1, m2, output, m1_height, m1_width, m2_width );
	cudaDeviceSynchronize();
}

//Printing input array of dimension (h,w)
__device__ void kdispmat (const float* M, int h, int w) {
    
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			printf("%f  ", M[i*w+j]);
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void kfit(	const float* X, const int X_w, const int X_h,
						const float* y, const int y_w,
						float* l1, const int l1_w, float* l_1_d,
						float* pred, float* pred_d,
						float* W0,
						float* W1,
						float* buffer
						)
{
	for (unsigned i = 0; i < 50; ++i) {

        dsigmoid(ddot(X, W0, l1, X_h, X_w, l1_w), l1, X_h, l1_w);
        dsigmoid(ddot(l1, W1, pred, X_h, l1_w, y_w), pred, X_h, y_w);
        dMartixByMatrixElementwise(dmatsub(y, pred, pred_d, X_h, y_w), ddersigmoid(pred, buffer, X_h, y_w), pred_d, X_h, y_w );
        dMartixByMatrixElementwise(dm1dotm2T(pred_d, W1, l_1_d, X_h, y_w, l1_w), ddersigmoid(l1, buffer, X_h, l1_w), l_1_d, X_h, l1_w);
        dm1Tdotm2( l1, pred_d, W1, X_h, l1_w, y_w );
        dm1Tdotm2( X, l_1_d, W0, X_h, X_w, l1_w );
    }
}
