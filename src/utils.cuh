#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*
=====================================
CUDA操作
=====================================
*/
void cudaCheck(cudaError_t error, const char *file, int line); //CUDA错误检查
void CudaDeviceInfo();                                         // 打印CUDA信息

/*
=====================================
矩阵操作
=====================================
*/
void randomize_matrix(float *mat, int N);            // 随机初始化矩阵
void copy_matrix(float *src, float *dest, int N);    // 复制矩阵
void print_matrix(const float *A, int M, int N);     // 打印矩阵
bool verify_matrix(float *mat1, float *mat2, int N); // 验证矩阵

/*
=====================================
计时操作
=====================================
*/
float get_current_sec();                        // 获取当前时刻
float cpu_elapsed_time(float &beg, float &end); // 计算时间差

/*
=====================================
kernel操作
=====================================
*/
//调用指定核函数计算矩阵乘法
void test_kernel(int kernel_num, int m, int n, int k, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle);