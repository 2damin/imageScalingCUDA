#pragma once
#include "gpu_shared.h"
#include <vector>

namespace cv {
class Mat;
}

class SimpleBBox;
typedef std::vector<SimpleBBox> SimpleBBoxes;

inline __device__ __host__ int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline __device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ __host__ float3 operator-(const float &a, const float3 &b) {
  return make_float3(b.x - a, b.y - a, b.z - a);
}

inline __device__ __host__ float3 operator*(const float3 &a, const float3 &b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ __host__ float3 operator*(const float3 &a, const float &b) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __device__ __host__ float3 operator*(const float &a, const float3 &b) {
  return make_float3(b.x * a, b.y * a, b.z * a);
}

bool cropResizeBGR(std::shared_ptr<cv::Mat> bgr_packed_cpu,
                   std::shared_ptr<SimpleBBoxes> boxes,
                   std::vector<gpu_shared_ptr<float>> bgr_planar_gpu_array,
                   int outputWidth, int outputHeight);
