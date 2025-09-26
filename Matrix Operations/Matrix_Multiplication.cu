#include <cuda_runtime.h>
#include <iostream>

// It's a good practice to have an error-checking macro
#define cudaErrorCheck(ans)                                                    \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

// ===================================================================
// YOUR LEETGPU CODE STARTS HERE
// ===================================================================

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < K) {
    float cval = 0;
    for (int k = 0; k < N; k++) {
      cval += A[row * N + k] * B[k * K + col];
    }
    C[row * K + col] = cval;
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize(); // Wait for the kernel to finish
}


int main() {
  // 1. Define Matrix Dimensions
  int M = 256;
  int N = 512;
  int K = 128;

  std::cout << "Performing C(" << M << "x" << K << ") = A(" << M << "x" << N
            << ") * B(" << N << "x" << K << ")" << std::endl;

  // 2. Setup Data on the Host (CPU)
  size_t size_A = (size_t)M * N * sizeof(float);
  size_t size_B = (size_t)N * K * sizeof(float);
  size_t size_C = (size_t)M * K * sizeof(float);

  float *h_A = new float[M * N];
  float *h_B = new float[N * K];
  float *h_C_from_gpu = new float[M * K];

  // Initialize host matrices with some data (e.g., random numbers)
  for (int i = 0; i < M * N; ++i)
    h_A[i] = (float)rand() / RAND_MAX;
  for (int i = 0; i < N * K; ++i)
    h_B[i] = (float)rand() / RAND_MAX;

  // 3. Allocate Memory on the Device (GPU)
  float *d_A, *d_B, *d_C;
  cudaErrorCheck(cudaMalloc(&d_A, size_A));
  cudaErrorCheck(cudaMalloc(&d_B, size_B));
  cudaErrorCheck(cudaMalloc(&d_C, size_C));

  // 4. Copy Data from Host to Device
  std::cout << "Copying data from Host to Device..." << std::endl;
  cudaErrorCheck(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

  // 5. Call your `solve` function to launch the CUDA kernel
  std::cout << "Launching kernel..." << std::endl;
  solve(d_A, d_B, d_C, M, N, K);

  // 6. Copy Result from Device to Host
  std::cout << "Copying result from Device to Host..." << std::endl;
  cudaErrorCheck(cudaMemcpy(h_C_from_gpu, d_C, size_C, cudaMemcpyDeviceToHost));

  // 7. Verify the result (optional, but a very good idea)
  std::cout << "Verifying result on the CPU..." << std::endl;
  bool correct = true;
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < K; ++c) {
      float cpu_val = 0;
      for (int k = 0; k < N; ++k) {
        cpu_val += h_A[r * N + k] * h_B[k * K + c];
      }
      // Compare GPU result with CPU result, allowing for small floating point
      // error
      if (fabs(cpu_val - h_C_from_gpu[r * K + c]) > 1e-4) {
        std::cout << "Verification FAILED at (" << r << "," << c << ")!"
                  << std::endl;
        std::cout << "CPU val: " << cpu_val
                  << " vs GPU val: " << h_C_from_gpu[r * K + c] << std::endl;
        correct = false;
        break;
      }
    }
    if (!correct)
      break;
  }

  if (correct) {
    std::cout << "SUCCESS! The GPU result is correct." << std::endl;
  }

  // 8. Cleanup: Free all allocated memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C_from_gpu;

  cudaErrorCheck(cudaFree(d_A));
  cudaErrorCheck(cudaFree(d_B));
  cudaErrorCheck(cudaFree(d_C));

  return 0;
}