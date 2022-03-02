#include <stdio.h>
#include "utils.cuh"
#include "kernel.cuh"

float get_sec() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) {
    return 1.0e-6 * (end - beg);
}

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    /*
   * There should be no need to modify the output string below.
   */

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           deviceId,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
};

void randomize_matrix(float *mat, int N) {
    // NOTICE: 使用gettimeofdays替代srand((unsigned)time(NULL));time精度过低，产生相同随机数
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float) (rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void copy_matrix(float *src, float *dest, int N) {
    int i;
    for (i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if (i != N)
        printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N) {
    int i;
    printf("[");
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
                printf(";\n");
        }
    }
    printf("]\n");
}

bool verify_matrix(float *mat1, float *mat2, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i], mat2[i], i);
            return false;
        }
    }
    return true;
}

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

void test_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    //cublas列主序计算：https://www.cnblogs.com/cuancuancuanhao/p/7763256.html
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

void test_mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    mysgemm_v1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(1024);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    mysgemm_v2<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(512);
    dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
    mysgemm_v3<64, 64, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    mysgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v5(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    mysgemm_v5<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

//void test_mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
//    dim3 blockDim(4);
//    dim3 gridDim(CEIL_DIV(M, 8), CEIL_DIV(N, 8));
//    mysgemm_v6<8, 8, 4, 4, 4><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//}

void test_mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    mysgemm_v6<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    mysgemm_v7<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}



void test_kernel(int kernel_num, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C,
                 cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            test_cublas(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            test_mysgemm_v1(M, N, K, alpha, A, B, beta, C);
            break;
        case 2:
            test_mysgemm_v2(M, N, K, alpha, A, B, beta, C);
            break;
        case 3:
            test_mysgemm_v3(M, N, K, alpha, A, B, beta, C);
            break;
        case 4:
            test_mysgemm_v4(M, N, K, alpha, A, B, beta, C);
            break;
        case 5:
            test_mysgemm_v5(M, N, K, alpha, A, B, beta, C);
            break;
        case 6:
            test_mysgemm_v6(M, N, K, alpha, A, B, beta, C);
            break;
        case 7:
            test_mysgemm_v7(M, N, K, alpha, A, B, beta, C);
            break;
        default:
            break;
    }
}