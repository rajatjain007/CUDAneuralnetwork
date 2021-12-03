#include <stdio.h>
#include <stdint.h>
#include "nn.cu"

int main(void){

	const int TRAINING_SIZE = 5;
	const int TRAINING_DIM = 8;
	const int L1_SIZE = 8;

	// X, the first 4 lines from Iris dataset
	float h_X[TRAINING_SIZE*TRAINING_DIM] = {
6,148,72,35,0,33.6,0.627,50,
1,85,66,29,0,26.6,0.351,31,
8,183,64,0,0,23.3,0.672,32,
1,89,66,23,94,28.1,0.167,21,
0,137,40,35,168,43.1,2.288,33};

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
	float h_y[4] = {	1,
						0,
						1,
						0,
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