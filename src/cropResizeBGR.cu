
#include <iostream>
#include <stdio.h>

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cropResizeBGR.h"
#include "datatype.hpp"

#include <cstdint>
#include <memory>
#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void kernel_cudaCrop(unsigned char *input, int inputWidth,
                                int inputChannel, float *output, int xMin,
                                int yMin, int outputWidth, int outputHeight) {
  assert(input && output);
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x > outputWidth - 1 || y > outputHeight - 1)
    return;

  int x_pos = xMin + x;
  int y_pos = yMin + y;

  int outwcSize = outputWidth * inputChannel;
  int inwcSize = inputWidth * inputChannel;
  for (int c = 0; c < inputChannel; ++c)
    output[y * outwcSize + x * inputChannel + c] =
        (float)input[y_pos * inwcSize + x_pos * inputChannel + c];
}

__global__ void kernel_cudaResize_nn(float *input, int inputWidth,
                                     int inputChannel, float *output,
                                     int outputWidth, int outputHeight,
                                     float2 scale) {
  assert(input && output);
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x > outputWidth - 1 || y > outputHeight - 1)
    return;

  const int dx = ((float)x * scale.x);
  const int dy = ((float)y * scale.y);

  int outwcSize = outputWidth * inputChannel;
  int inwcSize = inputWidth * inputChannel;

  for (int c = 0; c < inputChannel; ++c)
    output[y * outwcSize + x * inputChannel + c] =
        input[dy * inwcSize + dx * inputChannel + c];
}

__global__ void kernel_cudaResize_bilinear(float *output, const int inputWidth,
                                           const int inputHeight,
                                           const int inputChannel,
                                           const int outputWidth,
                                           const int outputHeight,
                                           float2 scale) {
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x > outputWidth - 1) && (y > outputHeight - 1))
    return;

  const int ind_x = floor(scale.x * x);
  const float a = scale.x * x - ind_x;

  const int ind_y = floor(scale.y * y);
  const float b = scale.y * y - ind_y;

  const int d0_xpos = ind_x * inputChannel;
  const int d1_xpos = (ind_x + 1) * inputChannel;
  const float3 d00 = {tex2D(texRef, d0_xpos, ind_y),
                      tex2D(texRef, d0_xpos + 1, ind_y),
                      tex2D(texRef, d0_xpos + 2, ind_y)};
  const float3 d10 = {tex2D(texRef, d1_xpos, ind_y),
                      tex2D(texRef, d1_xpos + 1, ind_y),
                      tex2D(texRef, d1_xpos + 2, ind_y)};
  const float3 d11 = {tex2D(texRef, d1_xpos, ind_y + 1),
                      tex2D(texRef, d1_xpos + 1, ind_y + 1),
                      tex2D(texRef, d1_xpos + 2, ind_y + 1)};
  const float3 d01 = {tex2D(texRef, d0_xpos, ind_y + 1),
                      tex2D(texRef, d0_xpos + 1, ind_y + 1),
                      tex2D(texRef, d0_xpos + 2, ind_y + 1)};

  float3 result_tmp1, result_tmp2, result_target;
  result_tmp1 = d10 * a + (-a * d00 + d00);
  result_tmp2 = d11 * a + (-a * d01 + d01);
  result_target = result_tmp2 * b + (-b * result_tmp1 + result_tmp1);

  int wcSize = outputWidth * inputChannel;
  output[y * wcSize + x * inputChannel] = result_target.x;
  output[y * wcSize + x * inputChannel + 1] = result_target.y;
  output[y * wcSize + x * inputChannel + 2] = result_target.z;
}

__global__ void kernel_packed_to_planar(float *input, int inputWidth,
                                        int inputChannel, float *output,
                                        int outputWidth, int outputHeight) {
  assert(input && output);
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x > outputWidth - 1 || y > outputHeight - 1)
    return;

  int imgSize = outputWidth * outputHeight;
  int wcSize = inputWidth * inputChannel;

  for (int c = 0; c < inputChannel; ++c)
    output[c * imgSize + y * outputWidth + x] =
        input[y * wcSize + x * inputChannel + c];
}

__global__ void kernel_merge(float *input, int inputWidth, int inputHeight,
                             int inputChannel, float *output, int batchIdx) {
  assert(input && output);
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x > inputWidth - 1 || y > inputHeight - 1)
    return;

  int imgSize = inputWidth * inputHeight;
  int pos = y * inputWidth + x;
  int totalSize = inputWidth * inputHeight * inputChannel;

  for (int c = 0; c < inputChannel; ++c)
    output[c * imgSize + pos + totalSize * batchIdx] = input[c * imgSize + pos];
}

__global__ void kernel_zeros(float *input, int inputWidth, int inputHeight,
                             int inputChannel, int inputN, int init_pos) {
  assert(input);
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x > inputWidth - 1 || y > inputHeight - 1)
    return;

  int imgSize = inputWidth * inputHeight;

  for (int n = 0; n < inputN; ++n) {
    int batch_unit = n * imgSize * inputChannel;
    for (int c = 0; c < inputChannel; ++c)
      input[batch_unit + c * imgSize + y * inputWidth + x + init_pos] = 0;
  }
}

// crop planar_bgr_gpu
bool cudaCrop(gpu_shared_ptr<uint8_t> input, int inputWidth, int inputHeight,
              int inputChannel, gpu_shared_ptr<float> output, int xMin,
              int yMin, int outputWidth, int outputHeight) {
  if (!input || !output)
    return false;

  const dim3 blockDim(16, 16);
  const dim3 gridDim(iDivUp(inputWidth, blockDim.x),
                     iDivUp(inputHeight, blockDim.y));

  kernel_cudaCrop<<<gridDim, blockDim>>>(input.get(), inputWidth, inputChannel,
                                         output.get(), xMin, yMin, outputWidth,
                                         outputHeight);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return true;
}

bool cudaResize(gpu_shared_ptr<float> input, int inputWidth, int inputHeight,
                int inputChannel, gpu_shared_ptr<float> output, int outputWidth,
                int outputHeight, int interpolation) {
  if (!input || !output)
    return false;

  if (inputWidth * inputHeight * outputWidth * outputHeight == 0)
    false;

  if (interpolation == 0) {
    const float2 scale = make_float2(float(inputWidth) / float(outputWidth),
                                     float(inputHeight) / float(outputHeight));
    const dim3 blockDim(16, 16);
    const dim3 gridDim(iDivUp(outputWidth, blockDim.x),
                       iDivUp(outputHeight, blockDim.y));

    kernel_cudaResize_nn<<<gridDim, blockDim>>>(
        input.get(), inputWidth, inputChannel, output.get(), outputWidth,
        outputHeight, scale);
    cudaGetLastError();
    cudaDeviceSynchronize();
  } else {
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *src_arr;
    cudaMallocArray(&src_arr, &channelDesc, inputWidth * inputChannel,
                    inputHeight);
    cudaMemcpyToArray(src_arr, 0, 0, input.get(),
                      inputWidth * inputHeight * inputChannel * sizeof(float),
                      cudaMemcpyHostToDevice);
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;

    cudaBindTextureToArray(texRef, src_arr, channelDesc);

    float2 scale = make_float2(inputWidth * 1.0f / outputWidth,
                               inputHeight * 1.0f / outputHeight);

    dim3 blockDim(16, 16);
    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x,
                 (outputHeight + blockDim.y - 1) / blockDim.y);
    kernel_cudaResize_bilinear<<<gridDim, blockDim>>>(
        output.get(), inputWidth, inputHeight, inputChannel, outputWidth,
        outputHeight, scale);
    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaFreeArray(src_arr);
  }

  return true;
}

// convert from packed_bgr_gpu to planar_bgr_gpu
bool cudaPackedToPlanar(gpu_shared_ptr<float> input, int inputWidth,
                        int inputHeight, int intputChannel,
                        gpu_shared_ptr<float> output, int outputWidth,
                        int outputHeight) {
  if (!input || !output)
    return false;

  if (inputWidth * inputHeight * outputWidth * outputHeight * intputChannel ==
      0)
    false;

  const dim3 blockDim(16, 16);
  const dim3 gridDim(iDivUp(outputWidth, blockDim.x),
                     iDivUp(outputHeight, blockDim.y));

  kernel_packed_to_planar<<<gridDim, blockDim>>>(input.get(), inputWidth,
                                                 intputChannel, output.get(),
                                                 outputWidth, outputHeight);
  cudaGetLastError();
  cudaDeviceSynchronize();
  return true;
}

// merge planar_bgr_gpu images
bool cudaMerge(std::vector<gpu_shared_ptr<float>> inputs, int inputWidth,
               int inputHeight, int inputChannel, int initInputIdx,
               gpu_shared_ptr<float> output, int batch) {
  if (inputs.empty() || !output)
    return false;

  if (inputWidth * inputHeight * inputChannel * batch == 0)
    false;

  const dim3 blockDim(16, 16);
  const dim3 gridDim(iDivUp(inputWidth, blockDim.x),
                     iDivUp(inputHeight, blockDim.y));

  // //make all black image
  kernel_zeros<<<gridDim, blockDim>>>(output.get(), inputWidth, inputHeight,
                                      inputChannel, batch, 0);
  cudaGetLastError();
  cudaDeviceSynchronize();
  int inputSize = inputs.size();
  int maxInputIdx =
      batch + initInputIdx > inputSize ? inputSize : batch + initInputIdx;
  for (int i = initInputIdx; i < maxInputIdx; ++i) {
    int batchIdx = (i % batch);
    kernel_merge<<<gridDim, blockDim>>>(inputs[i].get(), inputWidth,
                                        inputHeight, inputChannel, output.get(),
                                        batchIdx);
    cudaGetLastError();
    cudaDeviceSynchronize();
  }
}

bool cropResizeBGR(std::shared_ptr<cv::Mat> bgr_packed_cpu,
                   std::shared_ptr<SimpleBBoxes> boxes,
                   std::vector<gpu_shared_ptr<float>> bgr_planar_gpu_array,
                   int outputWidth, int outputHeight) {
  // null check
  if (bgr_packed_cpu == nullptr || boxes == nullptr ||
      outputWidth * outputHeight == 0)
    return false;

  int batch_num = 4;
  int inputWidth = bgr_packed_cpu->cols;
  int inputHeight = bgr_packed_cpu->rows;
  int inputChannel = bgr_packed_cpu->channels();
  auto box_size = static_cast<int>(boxes->size());

  // crop in gpu
  std::vector<gpu_shared_ptr<float>> d_cropImgs(box_size);
  int src_buffer = sizeof(uint8_t) * inputWidth * inputHeight * inputChannel;
  auto d_src = new_gpu_uint8(src_buffer);
  cudaMemcpy(d_src.get(), bgr_packed_cpu->data, src_buffer,
             cudaMemcpyHostToDevice);

  int i = 0;
  for (auto b : *boxes) {
    auto dst_buffer = sizeof(float) * b.w * b.h * inputChannel;
    d_cropImgs[i] = new_gpu_float(dst_buffer);
    cudaCrop(d_src, inputWidth, inputHeight, inputChannel, d_cropImgs[i], b.x,
             b.y, b.w, b.h);
    ++i;
  }

  // resize in GPU
  int cropImg_size = d_cropImgs.size();
  std::vector<gpu_shared_ptr<float>> d_resizeImgs(cropImg_size);
  for (int i = 0; i < cropImg_size; ++i) {
    auto dst_buffer = outputWidth * outputHeight * inputChannel;
    d_resizeImgs[i] = new_gpu_float(dst_buffer);
    cudaResize(d_cropImgs[i], (*boxes)[i].w, (*boxes)[i].h, inputChannel,
               d_resizeImgs[i], outputWidth, outputHeight, 1);
  }

  // packedToPlanar
  int resizeImg_size = d_resizeImgs.size();
  std::vector<gpu_shared_ptr<float>> d_planarImgs(resizeImg_size);
  for (int i = 0; i < resizeImg_size; ++i) {
    auto dst_buffer = outputWidth * outputHeight * inputChannel;
    d_planarImgs[i] = new_gpu_float(dst_buffer);
    cudaPackedToPlanar(d_resizeImgs[i], outputWidth, outputHeight, inputChannel,
                       d_planarImgs[i], outputWidth, outputHeight);
  }

  // make output array
  int outputSize_buffer = batch_num * outputWidth * outputHeight * 3;
  int mergeImgSize = (box_size - 1) / batch_num + 1;
  for (int j = 0; j < mergeImgSize; j++) {
    gpu_shared_ptr<float> bgr_planar_gpu;
    bgr_planar_gpu = new_gpu_float(outputSize_buffer);
    cudaMerge(d_planarImgs, outputWidth, outputHeight, inputChannel,
              j * batch_num, bgr_planar_gpu, batch_num);
    bgr_planar_gpu_array.push_back(bgr_planar_gpu);
  }

  return true;
}
