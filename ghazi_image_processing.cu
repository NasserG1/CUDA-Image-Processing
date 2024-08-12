#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

__constant__ float d_kernel[9]; // Sobel filter (3x3)

__global__ void convolutionKernel(const unsigned char* d_input, unsigned char* d_output, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int half_kernel = kernel_size / 2;

    for (int ky = -half_kernel; ky <= half_kernel; ky++) {
        for (int kx = -half_kernel; kx <= half_kernel; kx++) {
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);
            sum += d_input[iy * width + ix] * d_kernel[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
        }
    }

    d_output[y * width + x] = min(max(int(sum), 0), 255);
}

int main() {
    std::string directoryPath = "C:\\Users\\Nasser\\OneDrive\\Desktop\\Programming\\CUDA\\GPU project 2\\CUDAatScaleForTheEnterpriseCourseProjectTemplate\\misc\\misc";

    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        std::string imagePath = entry.path().string();
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }

        int width = image.cols;
        int height = image.rows;
        int image_size = width * height * sizeof(unsigned char);

        unsigned char* h_input = image.data;
        unsigned char* h_output = (unsigned char*)malloc(image_size);

        unsigned char *d_input, *d_output;
        cudaMalloc((void**)&d_input, image_size);
        cudaMalloc((void**)&d_output, image_size);

        cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);

        // Define the Sobel filter kernel
        float h_kernel[9] = {-1, 0, 1,
                             -2, 0, 2,
                             -1, 0, 1};

        cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(float) * 9);

        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, 3);

        cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

        cv::Mat output_image(height, width, CV_8UC1, h_output);
        std::string outputImagePath = imagePath + "_sobel.png";
        cv::imwrite(outputImagePath, output_image);

        cudaFree(d_input);
        cudaFree(d_output);
        free(h_output);

        std::cout << "Processed and saved: " << outputImagePath << std::endl;
    }

    return 0;
}
