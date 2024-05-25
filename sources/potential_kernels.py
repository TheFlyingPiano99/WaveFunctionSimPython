
draining_potential_kernel_source = """
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
    void draining_potential_kernel(
        complex<float>* V,
        float delta_x,
        float inner_radius_bohr_radii,
        float outer_radius_bohr_radii,
        float max_potential_hartree,
        float exponent
    )
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int i = blockIdx.z * blockDim.z + threadIdx.z;

        int idx = i * gridDim.x * blockDim.x * gridDim.y * blockDim.y
                + j * gridDim.x * blockDim.x
                + k;
                
        float3 pos = diff(
            scalarVectorMul(delta_x, {k, j, i}),
            scalarVectorMul(gridDim.x * blockDim.x * delta_x * 0.5f, {1.0, 1.0, 1.0})
        ); 
        float t = fminf(
            fmaxf(
                0.0f,
                (sqrtf(dot(pos, pos)) - inner_radius_bohr_radii)
                / (outer_radius_bohr_radii - inner_radius_bohr_radii)
            ),
            1.0f
        );
        V[idx] += complex<float>(0.0f, powf(t, exponent) * max_potential_hartree);
    }
"""

potential_wall_kernel_source = """
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
    
    __device__ float3 transform_corner_origin_to_center_origin_system(float3 pos)
    {
        return diff(
            pos,
            scalarVectorMul(0.5f, {(float)(gridDim.x * blockDim.x), (float)(gridDim.y * blockDim.y), (float)(gridDim.z * blockDim.z)})
        );
    }


    extern "C" __global__
    void potential_wall_kernel(
        complex<float>* V,
        float delta_x,
        float center_x,
        float center_y,
        float center_z,
        float normal_x,
        float normal_y,
        float normal_z,
        float thickness_bohr_radius,
        float potential_hartree
    )
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int i = blockIdx.z * blockDim.z + threadIdx.z;

        int idx = i * gridDim.x * blockDim.x * gridDim.y * blockDim.y
                + j * gridDim.x * blockDim.x
                + k;
                
        float3 center = {center_x, center_y, center_z};
        float3 normal = {normal_x, normal_y, normal_z};
        
        float3 r = scalarVectorMul(delta_x, transform_corner_origin_to_center_origin_system({(float)i, (float)j, (float)k}));
        float d = dot(normal, diff(center, r));
        if (d <= thickness_bohr_radius * 0.5f && d >= -thickness_bohr_radius * 0.5f)
        {
            V[idx] += potential_hartree * (1.0f - fmaxf(2.0f * fabsf(d / thickness_bohr_radius * 2.0f) - 1.0, 0.0f));
        }
    }
"""
