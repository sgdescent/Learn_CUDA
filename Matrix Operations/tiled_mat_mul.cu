#include <iostream>
#include <cuda_runtime.h>
#include <cmath> // For fabs
#include <chrono> // For CPU timing

// Error Checking Macro
#define cudaErrorCheck(ans){gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

// --- 1. The GPU Kernel (Your Logic) ---
__global__ void tiled_matrix_mul_kernel(const float *A, const float *B, float *C, const int M, const int N, const int K, const int stride){

    extern __shared__ float fused_mem[];

    float *Mds = &fused_mem[0];
    float *Nds = &fused_mem[stride*stride];

    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * stride+ ty;
    int col = bx * stride+ tx;

    float pvalue = 0;

    // Loop over the matrix in tiles
    // Ceiling division ensures we hit the last partial tile
    for(int ph = 0; ph < (N + stride- 1) / stride; ++ph){

        int mcol = ph * stride + tx;
        int mrow = row;
        int ncol = col;
        int nrow = ph * stride + ty;

        // Load A (with boundary check)
        if(mcol < N && mrow < M)
            Mds[ty*stride + tx] = A[mrow * N + mcol]; 
        else 
            Mds[ty*stride + tx] = 0.0f;

        // Load B (with boundary check)
        if(ncol < K && nrow < N)
            Nds[ty*stride + tx] = B[nrow * K + ncol];
        else 
            Nds[ty*stride + tx] = 0.0f;

        __syncthreads(); // Wait for load

        for(int i = 0; i < stride; ++i){
            pvalue += Mds[ty*stride + i] * Nds[i*stride + tx];
        }

        __syncthreads(); // Wait for compute
    }

    if(row < M && col < K){
        C[row * K + col] = pvalue;
    }
}

// --- 2. CPU Reference Implementation (Sanity Check) ---
void cpu_matrix_mul(const float *A, const float *B, float *C, int M, int N, int K) {
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for(int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

// --- 3. Verification Function ---
bool check_correctness(const float *cpu_C, const float *gpu_C, int M, int K) {
    float epsilon = 1e-3; // Floating point tolerance
    for(int i = 0; i < M * K; ++i) {
        if(fabs(cpu_C[i] - gpu_C[i]) > epsilon) {
            printf("MISMATCH at index %d! CPU: %f, GPU: %f\n", i, cpu_C[i], gpu_C[i]);
            return false;
        }
    }
    return true;
}

// --- 4. The Solver (With Timing) ---
void solve(const float *A, const float *B, float *C, const int M, const int N, const int K){
    
    // Create CUDA Events for Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record Start
    cudaEventRecord(start);
    
    //Compute Shared memory size
    int stride = 32;
    size_t shared_mem_size = 2*stride*stride*sizeof(float);

    dim3 threadsPerBlock(stride, stride);
    dim3 blocksPerGrid((K + stride - 1) / stride, (M + stride - 1) / stride);

    // Launch this Kernel
    tiled_matrix_mul_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(A, B, C, M, N, K, stride);

    // Record Stop
    cudaEventRecord(stop);
    
    // Wait for the GPU to finish so we get accurate time
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Kernel Execution Time: %.3f ms\n", milliseconds);

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaErrorCheck(cudaGetLastError());
}

int main(){
    int M = 1024; 
    int N = 2048;
    int K = 1024;
    printf("Matrix A (%d,%d) x Matrix B (%d,%d) = Matrix C (%d,%d)\n", M, N, N, K, M, K);
    
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    float *h_A = new float[M*N];
    float *h_B = new float[N*K];
    float *h_C_gpu = new float[M*K]; // Result from GPU
    float *h_C_cpu = new float[M*K]; // Result from CPU (Reference)

    // Initialize
    srand(time(NULL));
    for(int i=0; i<M*N; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for(int i=0; i<N*K; ++i) h_B[i] = (float)rand() / RAND_MAX;

    // Device Memory
    float *d_A, *d_B, *d_C;
    cudaErrorCheck(cudaMalloc(&d_A, size_A));
    cudaErrorCheck(cudaMalloc(&d_B, size_B));
    cudaErrorCheck(cudaMalloc(&d_C, size_C));

    cudaErrorCheck(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // --- RUN GPU ---
    printf("\n--- Running GPU Kernel ---\n");
    solve(d_A, d_B, d_C, M, N, K);

    // Copy back
    cudaErrorCheck(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));

    // --- RUN CPU (Sanity Check) ---
    printf("\n--- Running CPU Reference ---\n");
    
    // Start timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    cpu_matrix_mul(h_A, h_B, h_C_cpu, M, N, K);
    
    // Stop timing
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start);
    double cpu_milliseconds = cpu_duration.count() / 1000.0;
    
    // Display in microseconds if less than 0.001 ms, otherwise in milliseconds
    if (cpu_milliseconds < 0.001) {
        printf("CPU Execution Time: %.3f microseconds\n", (double)cpu_duration.count());
    } else {
        printf("CPU Execution Time: %.3f ms\n", cpu_milliseconds);
    }

    // --- VERIFY ---
    if (check_correctness(h_C_cpu, h_C_gpu, M, K)) {
        printf("\n✅ SUCCESS: GPU result matches CPU result!\n");
        
        // Print result only if successful
        printf("\nGPU Result (First Row):\n");
        for(int j=0; j<K; ++j) printf("%.3f\t", h_C_gpu[j]);
        printf("\n");
    } else {
        printf("\n❌ FAILURE: Results do not match.\n");
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C_gpu; delete[] h_C_cpu;

    return 0;
}