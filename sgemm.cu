#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <utils.cuh>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Please select a kernel (range 0 - 11, here 0 is for NVIDIA cuBLAS).\n");
        exit(EXIT_FAILURE);
    }

    // cuda kernel num
    int kernel_num = atoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 11) {
        printf("Please enter a valid kernel number (0-11).\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Select kernel %d.\n", kernel_num);
    };

    // 申明句柄，创建句柄, cublasCreate会返回一个cublasStatus_t类型的值，用来判断句柄是否创建成功(值为0)
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        printf("Create cublas handle error.\n");
        exit(EXIT_FAILURE);
    };

    // 采用cudaEvent进行gpu流计时，cudaEvent相当于在目标流中发布事件任务
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // matrix size
    int size_len = 24;
    int SIZE[size_len];
    for (int i = 0; i < size_len; i++)
        SIZE[i] = 256 * (i + 1);

    int m, n, k, max_size;
    max_size = SIZE[size_len - 1];
    printf("max_size=%d\n", max_size);

    float alpha = 1.0, beta = 0.; //two arbitary input parameters，C=α*AB+β*C

    float *A = NULL, *B = NULL, *C = NULL, *C_ref = NULL;     //host matrices
    float *dA = NULL, *dB = NULL, *dC = NULL, *dC_ref = NULL; //device matrices

    A = (float *) malloc(sizeof(float) * max_size * max_size);
    B = (float *) malloc(sizeof(float) * max_size * max_size);
    C = (float *) malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *) malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);
    copy_matrix(C, C_ref, max_size * max_size);

    cudaCheck(cudaMalloc((void **) &dA, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dC_ref, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C_ref, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    int repeat_times = 10;
    for (int i = 0; i < size_len; i++) {
        m = n = k = SIZE[i];

        printf("m=n=k=%d\n", m);
        // 验证计算正确性，同时在核函数计时前预先执行一次，避免冷启动误差
        if (kernel_num != 0) {
            test_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);      // cuBLAS
            test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle); // user define
            cudaDeviceSynchronize();
            cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            if (!verify_matrix(C_ref, C, m * n)) {
                printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
                exit(EXIT_FAILURE);
            }
        }
        cudaDeviceSynchronize();

        cudaEventRecord(beg);
        for (int j = 0; j < repeat_times; j++) {
            test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.; //换算成秒

        printf("Average elasped time: (%f) second, performance: (%f) GFLOPS. size: (%d).\n",
               elapsed_time / repeat_times, 2. * 1e-9 * repeat_times * m * n * k / elapsed_time, m);
        fflush(stdout);
        copy_matrix(C_ref, C, m * n); //sync C with cuBLAS to prepare for the next run
    }

    // 释放CPU和GPU空间
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);

    return 0;
};
