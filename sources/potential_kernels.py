from pathlib import Path

draining_potential_kernel_source = Path("sources/cuda_kernels/draining_potential.cu").read_text()

potential_wall_kernel_source = Path("sources/cuda_kernels/potential_wall.cu").read_text()

double_slit_kernel_source = Path("sources/cuda_kernels/double_slit.cu").read_text()
