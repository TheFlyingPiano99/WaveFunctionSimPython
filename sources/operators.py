from numba import jit, vectorize
import numpy as np
import cupy as cp
import sources.math_utils as math_utils

kinetic_operator_kernel_source = '''
    #include <cupy/complex.cuh>

    extern "C" float M_PI = 3.14159265359;

    extern "C" __device__ float3 scalarVectorMul(float s, const float3& v)
    {
        return {s * v.x, s * v.y, s * v.z}; 
    }

    extern "C" __device__ float dot(const float3& a, const float3& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z; 
    }

    extern "C" __device__ complex<float> exp_i(float angle)
    {
        return complex<float>(cosf(angle), sinf(angle)); 
    }

    extern "C" __device__ float3 diff(float3 a, float3 b)
    {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }
    
    extern "C" __device__ float3 div(float3 a, float3 b)
    {
        return {a.x / b.x, a.y / b.y, a.z / b.z};
    }

    extern "C" __global__
    void kinetic_operator_kernel(
        complex<float>* kinetic_operator,
        
        float delta_x,
        float delta_y,
        float delta_z,
        
        float delta_t
    )
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int idx = i * gridDim.x * blockDim.x * gridDim.y * blockDim.y
                + j * gridDim.x * blockDim.x
                + k;
        
        float3 f = div(
            {(float)k, (float)j, (float)i},
            {(float)(gridDim.x * blockDim.x - 1), (float)(gridDim.y * blockDim.y - 1), (float)(gridDim.z * blockDim.z - 1)}
        );
        float3 delta_r = {delta_x, delta_y, delta_z};
        
        // Account for numpy fftn's "negative frequency in second half" pattern
        if (f.x > 0.5f)
            f.x = 1.0f - f.x;
        if (f.y > 0.5f)
            f.y = 1.0f - f.y;
        if (f.z > 0.5f)
            f.z = 1.0f - f.z;

        float3 momentum = scalarVectorMul(2.0f * M_PI, div(f, delta_r));
        float angle = -dot(momentum, momentum) * delta_t / 4.0f;
        kinetic_operator[idx] = exp_i(angle);
    }
'''


def init_kinetic_operator(delta_x_3: np.array, delta_time: float, shape: np.shape):
    kinetic_operator_kernel = cp.RawKernel(kinetic_operator_kernel_source,
                                 'kinetic_operator_kernel',
                                 enable_cooperative_groups=False)
    grid_size = (64, 64, 64)
    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
    P_kinetic = cp.zeros(shape=shape, dtype=cp.csingle)
    kinetic_operator_kernel(
        grid_size,
        block_size,
        (
            P_kinetic,

            cp.float32(delta_x_3[0]),
            cp.float32(delta_x_3[1]),
            cp.float32(delta_x_3[2]),

            cp.float32(delta_time)
        )
    )
    return P_kinetic

# P_potential: cp.ndarray, V: cp.ndarray, delta_time:float
potential_operator_kernel_source = '''
    #include <cupy/complex.cuh>

    extern "C" float M_PI = 3.14159265359;

    extern "C" __device__ float3 scalarVectorMul(float s, const float3& v)
    {
        return {s * v.x, s * v.y, s * v.z}; 
    }

    extern "C" __device__ float dot(const float3& a, const float3& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z; 
    }

    extern "C" __device__ complex<float> exp_i(float angle)
    {
        return complex<float>(cosf(angle), sinf(angle)); 
    }

    extern "C" __device__ complex<float> cexp_i(complex<float> cangle)
    {
        return complex<float>(cosf(cangle.real()), sinf(cangle.real())) * expf(-cangle.imag()); 
    }

    extern "C" __device__ float3 diff(float3 a, float3 b)
    {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    extern "C" __device__ float3 div(float3 a, float3 b)
    {
        return {a.x / b.x, a.y / b.y, a.z / b.z};
    }

    extern "C" __global__
    void potential_operator_kernel(
        complex<float>* potential_operator,
        complex<float>* V,
        float delta_t
    )
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int i = blockIdx.z * blockDim.z + threadIdx.z;

        int vK = gridDim.x * blockDim.x - k - 1;
        int vJ = gridDim.y * blockDim.y - j - 1;
        int vI = gridDim.z * blockDim.z - i - 1;
        int vIdx = vI * gridDim.x * blockDim.x * gridDim.y * blockDim.y
                + vJ * gridDim.x * blockDim.x
                + vK;
        complex<float> angle = -delta_t * V[vIdx];

        int idx = i * gridDim.x * blockDim.x * gridDim.y * blockDim.y
                + j * gridDim.x * blockDim.x
                + k;
        //potential_operator[idx] = cexp_i(angle);
        potential_operator[idx] = 1.0f;
    }
'''


def init_potential_operator(P_potential: cp.ndarray, V: cp.ndarray, delta_time:float):
    potential_operator_kernel = cp.RawKernel(potential_operator_kernel_source,
                                 'potential_operator_kernel',
                                 enable_cooperative_groups=False)
    shape = P_potential.shape
    grid_size = (64, 64, 64)
    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
    potential_operator_kernel(
        grid_size,
        block_size,
        (
            P_potential,
            V,
            cp.float32(delta_time)
        )
    )
    return P_potential
