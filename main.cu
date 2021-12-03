#include <stdio.h>
#include <stdint.h>
#include "nn.cu"

int main(void){

	const int TRAINING_SIZE = 101;
	const int TRAINING_DIM = 4;
	const int L1_SIZE = 8;

	// X, the first 101 lines from Iris dataset
	float h_X[TRAINING_SIZE*TRAINING_DIM] = {5.1,3.5,1.4,0.2,
4.9,3,1.4,0.2,
4.7,3.2,1.3,0.2,
4.6,3.1,1.5,0.2,
5,3.6,1.4,0.2,
5.4,3.9,1.7,0.4,
4.6,3.4,1.4,0.3,
5,3.4,1.5,0.2,
4.4,2.9,1.4,0.2,
4.9,3.1,1.5,0.1,
5.4,3.7,1.5,0.2,
4.8,3.4,1.6,0.2,
4.8,3,1.4,0.1,
4.3,3,1.1,0.1,
5.8,4,1.2,0.2,
5.7,4.4,1.5,0.4,
5.4,3.9,1.3,0.4,
5.1,3.5,1.4,0.3,
5.7,3.8,1.7,0.3,
5.1,3.8,1.5,0.3,
5.4,3.4,1.7,0.2,
5.1,3.7,1.5,0.4,
4.6,3.6,1,0.2,
5.1,3.3,1.7,0.5,
4.8,3.4,1.9,0.2,
5,3,1.6,0.2,
5,3.4,1.6,0.4,
5.2,3.5,1.5,0.2,
5.2,3.4,1.4,0.2,
4.7,3.2,1.6,0.2,
4.8,3.1,1.6,0.2,
5.4,3.4,1.5,0.4,
5.2,4.1,1.5,0.1,
5.5,4.2,1.4,0.2,
4.9,3.1,1.5,0.1,
5,3.2,1.2,0.2,
5.5,3.5,1.3,0.2,
4.9,3.1,1.5,0.1,
4.4,3,1.3,0.2,
5.1,3.4,1.5,0.2,
5,3.5,1.3,0.3,
4.5,2.3,1.3,0.3,
4.4,3.2,1.3,0.2,
5,3.5,1.6,0.6,
5.1,3.8,1.9,0.4,
4.8,3,1.4,0.3,
5.1,3.8,1.6,0.2,
4.6,3.2,1.4,0.2,
5.3,3.7,1.5,0.2,
5,3.3,1.4,0.2,
7,3.2,4.7,1.4,
6.4,3.2,4.5,1.5,
6.9,3.1,4.9,1.5,
5.5,2.3,4,1.3,
6.5,2.8,4.6,1.5,
5.7,2.8,4.5,1.3,
6.3,3.3,4.7,1.6,
4.9,2.4,3.3,1,
6.6,2.9,4.6,1.3,
5.2,2.7,3.9,1.4,
5,2,3.5,1,
5.9,3,4.2,1.5,
6,2.2,4,1,
6.1,2.9,4.7,1.4,
5.6,2.9,3.6,1.3,
6.7,3.1,4.4,1.4,
5.6,3,4.5,1.5,
5.8,2.7,4.1,1,
6.2,2.2,4.5,1.5,
5.6,2.5,3.9,1.1,
5.9,3.2,4.8,1.8,
6.1,2.8,4,1.3,
6.3,2.5,4.9,1.5,
6.1,2.8,4.7,1.2,
6.4,2.9,4.3,1.3,
6.6,3,4.4,1.4,
6.8,2.8,4.8,1.4,
6.7,3,5,1.7,
6,2.9,4.5,1.5,
5.7,2.6,3.5,1,
5.5,2.4,3.8,1.1,
5.5,2.4,3.7,1,
5.8,2.7,3.9,1.2,
6,2.7,5.1,1.6,
5.4,3,4.5,1.5,
6,3.4,4.5,1.6,
6.7,3.1,4.7,1.5,
6.3,2.3,4.4,1.3,
5.6,3,4.1,1.3,
5.5,2.5,4,1.3,
5.5,2.6,4.4,1.2,
6.1,3,4.6,1.4,
5.8,2.6,4,1.2,
5,2.3,3.3,1,
5.6,2.7,4.2,1.3,
5.7,3,4.2,1.2,
5.7,2.9,4.2,1.3,
6.2,2.9,4.3,1.3,
5.1,2.5,3,1.1,
5.7,2.8,4.1,1.3};
	const signed int X_size = sizeof(h_X);

	float *d_X;
	cudaMalloc(&d_X, X_size);
	cudaMemcpy(d_X, h_X, X_size, cudaMemcpyHostToDevice);

	//WEIGHTS_0
	const long signed int W0_size = L1_SIZE*TRAINING_DIM*sizeof(float);
	float *h_W0 = (float*)malloc(W0_size);
	for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++){
	    h_W0[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	}

	float *d_W0;
	cudaMalloc(&d_W0, W0_size);
	cudaMemcpy(d_W0, h_W0, W0_size, cudaMemcpyHostToDevice);

	//LAYER 1, LAYER 1 DELTA AND BUFFER OF LAYER 1 SIZE
	const long signed int L1_size = L1_SIZE*TRAINING_SIZE*sizeof(float);

	float* h_layer_1 = (float*)malloc(L1_size);
	float* h_layer_1_delta = (float*)malloc(L1_size);
	float* h_buffer = (float*)malloc(L1_size);

	for (int i = 0; i < L1_SIZE*TRAINING_SIZE; i++){
	    h_layer_1[i] = 0.0;
	    h_buffer[i] = 0.0;
	    h_layer_1_delta[i] = 0.0;
	}

	float *d_layer_1;
	cudaMalloc(&d_layer_1, L1_size);
	cudaMemcpy(d_layer_1, h_layer_1, L1_size, cudaMemcpyHostToDevice);

	float *d_buffer;
	cudaMalloc(&d_buffer, L1_size);
	cudaMemcpy(d_buffer, h_buffer, L1_size, cudaMemcpyHostToDevice);

	float *d_layer_1_delta;
	cudaMalloc(&d_layer_1_delta, L1_size);
	cudaMemcpy(d_layer_1_delta, h_layer_1_delta, L1_size, cudaMemcpyHostToDevice);

	//WEIGHTS 1
	const long signed int W1_size = L1_SIZE*sizeof(float);
	float *h_W1 = (float*)malloc(W1_size);
	for (int i = 0; i < L1_SIZE; i++){
	    h_W1[i] = 0.1* (2.0*rand()/RAND_MAX-1.0);
	}

	float *d_W1;
	cudaMalloc(&d_W1, W1_size);
	cudaMemcpy(d_W1, h_W1, W1_size, cudaMemcpyHostToDevice);

	//Y
	float h_y[101] = {0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1};
	const signed int y_size = sizeof(h_y);
	float *d_y;
	cudaMalloc(&d_y, y_size);
	cudaMemcpy(d_y, h_y, y_size, cudaMemcpyHostToDevice);

	//PRED AND PRED_DELTA
	float* h_pred = (float*)malloc(y_size);
	float* h_pred_delta = (float*)malloc(y_size);
	for (int i = 0; i < TRAINING_SIZE; i++){
	    h_pred[i] = 0.0;
	    h_pred_delta[i] = 0.0;
	}

	float *d_pred;
	cudaMalloc(&d_pred, y_size);
	cudaMemcpy(d_pred, h_pred, y_size, cudaMemcpyHostToDevice);

	float *d_pred_delta;
	cudaMalloc(&d_pred_delta, y_size);
	cudaMemcpy(d_pred_delta, h_pred_delta, y_size, cudaMemcpyHostToDevice);

	kfit <<< 1, 1 >>> (	d_X, TRAINING_DIM, TRAINING_SIZE,
						d_y, 1,
						d_layer_1, L1_SIZE, d_layer_1_delta,
						d_pred,
						d_pred_delta,
						d_W0,
						d_W1,
						d_buffer);

	cudaMemcpy(h_pred, d_pred, y_size, cudaMemcpyDeviceToHost);

	cudaFree(d_pred);
	cudaFree(d_X);
	cudaFree(d_y);
	cudaFree(d_layer_1_delta);
	cudaFree(d_pred_delta);
	cudaFree(d_W0);
	cudaFree(d_W1);
	cudaFree(d_buffer);

	free(h_layer_1_delta);
	free(h_pred_delta);
	free(h_W0);
	free(h_W1);
	free(h_buffer);

	for (int i = 0; i < TRAINING_SIZE; i++){
		printf("Prediction[%i] : %f True Value[%i] : %f Error[%i] : %f\n", i, h_pred[i], i, h_y[i], i, h_pred[i] - h_y[i]);
	}

	free(h_pred);
}