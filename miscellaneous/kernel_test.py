import cupy as cp
from cupyx.scipy import ndimage

def test_func():
    print("Test")
    kernel = cp.RawKernel(
        r"""
            #include <cupy/complex.cuh>
            
            extern "C" __global__
            void test_kernel(
                complex<double>* in1,
                complex<double>* in2,
                complex<double>* out,
                float fVal,
                int size
            )
            {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                int j = blockIdx.y * blockDim.y + threadIdx.y;
                int k = blockIdx.z * blockDim.z + threadIdx.z;
                int idx = i * size * size + j * size + k;
                for (int n = 1; n < ) {
                    out[idx] = complex<double>((double)i, (double)j); 
                } 
            }
        """,
        "test_kernel"
    )

    n = 512
    g = 64
    array_in1 = cp.ones(shape=(n, n, n), dtype=cp.complex128)
    array_in2 = cp.ones(shape=(n, n, n), dtype=cp.complex128)
    array_out = cp.zeros(shape=(n, n, n), dtype=cp.complex128)
    kernel((g, g, g), (n // g, n // g, n // g),
           (array_in1, array_in2, array_out, cp.double(12.0), cp.int32(n)))
    print(array_out)

if __name__ == "__main__":
    test_func()