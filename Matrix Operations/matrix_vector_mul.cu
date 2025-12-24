//Given a matrix A and a vector b, compute the product Ab.
#include<iostream>
#include<cuda_runtime.h>
#include<chrono>

#define cudaErrorCheck(ans){gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
    if (code!=cudaSuccess){
        fprintf(stderr,"GPUAssert: %s %s %d\n", cudaGetErrorString(code),file,line);
        if(abort){
            exit(code);
        }

    }
}
__global__ void matrix_vector_multiplication_kernel(const float *A, const float *B,
    float *C, int M, int N) {

        int row  = blockIdx.y*blockDim.y + threadIdx.y;
        float cval = 0;
        if (row < M){
        for(int i=0;i<N;++i){
            cval += A[row*N+i]*B[i]; 
        }
    }
        C[row]= cval;
}


extern "C" void solve(const float *A, const float *B, float *C, int M, int N){
    dim3 threadsPerBlock(1,16);
    dim3 blocksPerGrid(1,(M+threadsPerBlock.y-1)/threadsPerBlock.y);
    matrix_vector_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
        N);

}

int main(){
    int M = 4;
    int N = 4;
    printf("Performing Matrix multiplication Matrix A (%d,%d) with Vector B (%d,)",M,N,N);

    //Setup the Data on the host (CPU)
    size_t size_A = (size_t)M*N*sizeof(float);
    size_t size_B = (size_t)N*sizeof(float);
    size_t size_C = (size_t)M*sizeof(float);


    // Create empty Matrices with some Data(e.g, random numbers)
    float *h_A = new float[M*N];
    float *h_B = new float[N];
    float *h_C = new float[M];

    // Initialize and print Matrix A
    printf("\nMatrix A:\n");
    for(int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            h_A[i*N + j] = (float)rand() / RAND_MAX;
            printf("%.3f\t", h_A[i*N + j]);
        }
        printf("\n");
    }

    // Initialize and print Vector B
    printf("\nVector B:\n");
    for(int i=0; i<N; ++i){
        h_B[i] = (float)rand() / RAND_MAX;
        printf("%.3f\n", h_B[i]);
    }
    printf("\n");

    //Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaErrorCheck(cudaMalloc(&d_A, size_A));
    cudaErrorCheck(cudaMalloc(&d_B, size_B));
    cudaErrorCheck(cudaMalloc(&d_C, size_C));

    //Copy data from host to device
    printf("Copying Data from Host to Device");
    cudaErrorCheck(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Call solve
    solve(d_A,d_B,d_C,M,N);

    cudaErrorCheck(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    for(int i =0; i<M; ++i){
        printf("%f\t", h_C[i]);
    }

    //Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaErrorCheck(cudaFree(d_A));
    cudaErrorCheck(cudaFree(d_B));
    cudaErrorCheck(cudaFree(d_C));


}