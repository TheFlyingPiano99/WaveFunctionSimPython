#ifndef CUDA_COMMON_INCLUDE
#define CUDA_COMMON_INCLUDE

#include <cupy/complex.cuh>

constexpr float M_PI_f = 3.14159265358979323846264338327950288419716939937510;
constexpr double M_PI_d = 3.14159265358979323846264338327950288419716939937510;

__device__ float3 abs(const float3& v)
{
    return {fabsf(v.x), fabsf(v.y), fabsf(v.z)};
}

__device__ double3 abs(const double3& v)
{
    return {abs(v.x), abs(v.y), abs(v.z)};
}

template<typename T>
__device__ complex<T> mul(const complex<T>& a, const complex<T>& b)
{
    return complex<T>(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real() );
}

__device__ constexpr float3 scalarVectorMul(const float s, const float3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

__device__ constexpr float3 operator*(float s, const float3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

__device__ constexpr float3 operator*(const float3& v, float s)
{
    return {s * v.x, s * v.y, s * v.z};
}

__device__ constexpr double3 scalarVectorMul(const double s, const double3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

__device__ constexpr double3 operator*(double s, const double3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

__device__ constexpr double3 operator*(const double3& v, double s)
{
    return {s * v.x, s * v.y, s * v.z};
}

__device__ constexpr float3 operator-(const float3& v)
{
    return {-v.x, -v.y, -v.z};
}

__device__ constexpr double3 operator-(const double3& v)
{
    return {-v.x, -v.y, -v.z};
}

constexpr __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

constexpr __device__ float3 cross(const float3& a, const float3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

constexpr __device__ double dot(const double3& a, const double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

constexpr __device__ double3 cross(const double3& a, const double3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__device__ const complex<float> exp_i(float angle)
{
    return complex<float>(cosf(angle), sinf(angle));
}

__device__ const complex<float> cexp_i(const complex<float>& cangle)
{
    return complex<float>(cosf(cangle.real()), sinf(cangle.real())) * expf(-cangle.imag());
}

__device__ const complex<double> exp_i(double angle)
{
    return complex<double>(cos(angle), sin(angle));
}

__device__ const complex<double> cexp_i(const complex<double>& cangle)
{
    return complex<double>(cos(cangle.real()), sin(cangle.real())) * exp(-cangle.imag());
}

__device__ constexpr float3 add(const float3& a, const float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ constexpr float3 operator+(const float3& a, const float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ float3& operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__device__ constexpr double3 add(const double3& a, const double3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ constexpr double3 operator+(const double3& a, const double3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ double3& operator+=(double3& a, const double3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__device__ constexpr float3 diff(const float3& a, const float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ constexpr float3 operator-(const float3& a, const float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ constexpr double3 diff(const double3& a, const double3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ constexpr double3 operator-(const double3& a, const double3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ float3 mul(const float3& a, const float3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ float3 operator*(const float3& a, const float3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ double3 mul(const double3& a, const double3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ double3 operator*(const double3& a, const double3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ constexpr float3 div(const float3& a, const float3& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ constexpr float3 operator/(const float3& a, const float3& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ constexpr float3 operator/(const float3& v, float s)
{
    return {v.x / s, v.y / s, v.z / s};
}

__device__ constexpr double3 div(const double3& a, const double3& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ constexpr double3 operator/(const double3& a, const double3& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ constexpr double3 operator/(const double3& v, double s)
{
    return {v.x / s, v.y / s, v.z / s};
}

__device__ float length(const float3& a)
{
    return sqrtf(dot(a, a));
}

__device__ float3 normalize(const float3& a)
{
    return a / sqrtf(dot(a, a));
}

__device__ double length(const double3& a)
{
    return sqrt(dot(a, a));
}

__device__ double3 normalize(const double3& a)
{
    return a / sqrt(dot(a, a));
}

__device__ constexpr float3 transform_corner_origin_to_center_origin_system(const float3& pos)
{
    return pos - 0.5f *  float3{(float)(gridDim.x * blockDim.x), (float)(gridDim.y * blockDim.y), (float)(gridDim.z * blockDim.z)};
}

__device__ constexpr double3 transform_corner_origin_to_center_origin_system(const double3& pos)
{
    return pos - 0.5 * double3{(double)(gridDim.x * blockDim.x), (double)(gridDim.y * blockDim.y), (double)(gridDim.z * blockDim.z)};
}

__device__ constexpr uint3 get_voxel_count_3d()
{
    return {
        gridDim.x * blockDim.x,
        gridDim.y * blockDim.y,
        gridDim.z * blockDim.z
    };
}

__device__ constexpr uint2 get_voxel_count_2d()
{
    return {
        gridDim.x * blockDim.x,
        gridDim.y * blockDim.y
    };
}

__device__ uint3 get_voxel_coords()
{
    return {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    };
}

__device__ uint3 get_voxel_coords_inverted()
{
    unsigned int x = gridDim.x * blockDim.x - (blockIdx.x * blockDim.x + threadIdx.x) - 1;
    unsigned int y = gridDim.y * blockDim.y - (blockIdx.y * blockDim.y + threadIdx.y) - 1;
    unsigned int z = gridDim.z * blockDim.z - (blockIdx.z * blockDim.z + threadIdx.z) - 1;
    return {x, y, z};
}

__device__ unsigned int get_array_index()
{
    uint3 voxel = get_voxel_coords();
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}

__device__ unsigned int get_array_index(const uint3& voxel)
{
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}

__device__ unsigned int get_array_index(const uint3& voxel, const uint3& N)
{
    return voxel.x * N.y * N.z
            + voxel.y * N.z
            + voxel.z;
}

__device__ unsigned int get_array_index_inverted()
{
    uint3 voxel = get_voxel_coords_inverted();
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}


__device__ float3 operator*(const float (&m)[3][3], const float3& v)
{
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
    };
}

__device__ double3 operator*(const double (&m)[3][3], const double3& v)
{
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
    };
}

__device__ float4 operator*(const float (&m)[4][4], const float4& v)
{
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
        m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w,
    };
}

__device__ double4 operator*(const double (&m)[4][4], const double4& v)
{
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
        m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w,
    };
}

__device__ float3 rotate_vector(const float3& v, const float3& axis, float rad)
{
    float q0 = cosf(rad / 2.0f);
    float q1 = sinf(rad / 2.0f) * axis.x;
    float q2 = sinf(rad / 2.0f) * axis.y;
    float q3 = sinf(rad / 2.0f) * axis.z;
    float Q[3][3] = { { 0.0f } }; // 3x3 rotation matrix

    Q[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
    Q[0][1] = 2.0f * (q1*q2 - q0*q3);
    Q[0][2] = 2.0f * (q1*q3 + q0*q2);

    Q[1][0] = 2.0f * (q2*q1 + q0*q3);
    Q[1][1] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
    Q[1][2] = 2.0f * (q2*q3 - q0*q1);

    Q[2][0] = 2.0f * (q3*q1 - q0*q2);
    Q[2][1] = 2.0f * (q3*q2 + q0*q1);
    Q[2][2] = q0*q0 - q1*q1 - q2*q2 + q3*q3;

    return Q * v;
}

__device__ double3 rotate_vector(const double3& v, const double3& axis, double rad)
{
    double q0 = cosf(rad / 2.0);
    double q1 = sinf(rad / 2.0) * axis.x;
    double q2 = sinf(rad / 2.0) * axis.y;
    double q3 = sinf(rad / 2.0) * axis.z;
    double Q[3][3] = { { 0.0 } }; // 3x3 rotation matrix

    Q[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
    Q[0][1] = 2.0 * (q1*q2 - q0*q3);
    Q[0][2] = 2.0 * (q1*q3 + q0*q2);

    Q[1][0] = 2.0 * (q2*q1 + q0*q3);
    Q[1][1] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
    Q[1][2] = 2.0 * (q2*q3 - q0*q1);

    Q[2][0] = 2.0 * (q3*q1 - q0*q2);
    Q[2][1] = 2.0 * (q3*q2 + q0*q1);
    Q[2][2] = q0*q0 - q1*q1 - q2*q2 + q3*q3;

    return Q * v;
}

__device__ float mix(const float3& xyz, float u, float v)
{
    return xyz.z * v + (1.0f - v) * (xyz.y * u + xyz.x * (1.0f - u));
}

__device__ double mix(const double3& xyz, double u, double v)
{
    return xyz.z * v + (1.0 - v) * (xyz.y * u + xyz.x * (1.0 - u));
}

__device__ void warpReduce(volatile int * sdata, unsigned int tid, unsigned int blockSize) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6( int* g_idata, int* g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32)
        warpReduce(sdata, tid, blockSize);
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

__device__ unsigned int get_block_local_idx_3d()
{
    return threadIdx.x * blockDim.y * blockDim.z
            + threadIdx.y * blockDim.z
            + threadIdx.z;
}

__device__ unsigned int get_block_local_idx_2d()
{
    return threadIdx.x * blockDim.y
            + threadIdx.y;
}

template<typename T>
__device__ T get_simpson_coefficient_3d(const uint3& voxel, const uint3& sample_count);

template<>
__device__ float get_simpson_coefficient_3d<float>(const uint3& voxel, const uint3& sample_count)
{
    uint3 n = get_voxel_count_3d();    // In the integrated volume
    float sX = 1.0f;
    if (voxel.x > 0 && voxel.x < sample_count.x - 1) {
        if (voxel.x % 2 == 0)
            sX = 2.0f;
        else
            sX = 4.0f;
    }
    float sY = 1.0f;
    if (voxel.y > 0 && voxel.y < sample_count.y - 1) {
        if (voxel.y % 2 == 0)
            sY = 2.0f;
        else
            sY = 4.0f;
    }
    float sZ = 1.0f;
    if (voxel.z > 0 && voxel.z < sample_count.z - 1) {
        if (voxel.z % 2 == 0)
            sZ = 2.0f;
        else
            sZ = 4.0f;
    }
    return sX * sY * sZ / 27.0f;    // ... / 3^3
}

template<>
__device__ double get_simpson_coefficient_3d<double>(const uint3& voxel, const uint3& sample_count)
{
    uint3 n = get_voxel_count_3d();    // In the integrated volume
    double sX = 1.0;
    if (voxel.x > 0 && voxel.x < sample_count.x - 1) {
        if (voxel.x % 2 == 0)
            sX = 2.0;
        else
            sX = 4.0;
    }
    double sY = 1.0;
    if (voxel.y > 0 && voxel.y < sample_count.y - 1) {
        if (voxel.y % 2 == 0)
            sY = 2.0;
        else
            sY = 4.0;
    }
    double sZ = 1.0;
    if (voxel.z > 0 && voxel.z < sample_count.z - 1) {
        if (voxel.z % 2 == 0)
            sZ = 2.0;
        else
            sZ = 4.0;
    }
    return sX * sY * sZ / 27.0;    // ... / 3^3
}

template<typename T>
__device__ T get_simpson_coefficient_2d(const uint2& pixel, const uint2& sample_count);

template<>
__device__ float get_simpson_coefficient_2d<float>(const uint2& pixel, const uint2& sample_count)
{
    float sX = 1.0f;
    if (pixel.x > 0 && pixel.x < sample_count.x - 1) {
        if (pixel.x % 2 == 0)
            sX = 2.0f;
        else
            sX = 4.0f;
    }
    float sY = 1.0f;
    if (pixel.y > 0 && pixel.y < sample_count.y - 1) {
        if (pixel.y % 2 == 0)
            sY = 2.0f;
        else
            sY = 4.0f;
    }
    return sX * sY / 9.0f;    // ... / 3^2
}

template<>
__device__ double get_simpson_coefficient_2d<double>(const uint2& pixel, const uint2& sample_count)
{
    double sX = 1.0;
    if (pixel.x > 0 && pixel.x < sample_count.x - 1) {
        if (pixel.x % 2 == 0)
            sX = 2.0;
        else
            sX = 4.0;
    }
    double sY = 1.0;
    if (pixel.y > 0 && pixel.y < sample_count.y - 1) {
        if (pixel.y % 2 == 0)
            sY = 2.0;
        else
            sY = 4.0;
    }
    return sX * sY / 9.0;    // ... / 3^2
}

template<typename T, typename T3>
__device__ complex<T> gaussian_wave_packet(const T3& sigma, const T3& r, const T3& r_0, const T3& k_0);


template<>
__device__ complex<float> gaussian_wave_packet<float, float3>(const float3& sigma, const float3& r, const float3& r_0, const float3& k_0)
{
    float3 a = 2.0f * sigma;
    float g_x = powf(2.0f / M_PI_f / a.x / a.x, 1.0f / 4.0f)
        * expf(
            -(r.x - r_0.x) * (r.x - r_0.x) / a.x / a.x
        );
    float g_y = powf(2.0f / M_PI_f / a.y / a.y, 1.0f / 4.0f)
        * expf(
            -(r.y - r_0.y) * (r.y - r_0.y) / a.y / a.y
        );
    float g_z = powf(2.0f / M_PI_f / a.z / a.z, 1.0f / 4.0f)
        * expf(
            -(r.z - r_0.z) * (r.z - r_0.z) / a.z / a.z
        );
    return g_x * g_y * g_z * exp_i(dot(k_0, r));
}

template<>
__device__ complex<double> gaussian_wave_packet<double, double3>(const double3& sigma, const double3& r, const double3& r_0, const double3& k_0)
{
    double3 a = 2.0 * sigma;
    double g_x = pow(2.0 / M_PI_d / a.x / a.x, 1.0 / 4.0)
        * exp(
            -(r.x - r_0.x) * (r.x - r_0.x) / a.x / a.x
        );
    double g_y = pow(2.0 / M_PI_d / a.y / a.y, 1.0 / 4.0)
        * exp(
            -(r.y - r_0.y) * (r.y - r_0.y) / a.y / a.y
        );
    double g_z = pow(2.0 / M_PI_d / a.z / a.z, 1.0 / 4.0)
        * exp(
            -(r.z - r_0.z) * (r.z - r_0.z) / a.z / a.z
        );
    return g_x * g_y * g_z * exp_i(dot(k_0, r));
}

/*
Parallel Reduction with Sequential Addressing
Based on the "Reduction #3: Sequential Addressing" code published in Mark Harris's Optimizing Parallel Reduction in CUDA presentation slides
URL: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/
template<typename T>
__device__ void parallel_reduction_sequential(unsigned int threadId, T sdata[])
{
    __syncthreads();
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
    for (unsigned int s=1; s < blockSize; s *= 2) { // Reduction #3: Sequential Addressing from Optimizing Parallel Reduction in CUDA by Mark Harris NVIDIA
        int index = 2 * s * threadId;
        if (index + s < blockSize) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
}

#endif  // CUDA_COMMON
