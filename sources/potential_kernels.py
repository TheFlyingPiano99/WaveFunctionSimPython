from pathlib import Path
import os

absorbing_potential_kernel_source = Path("sources/cuda_kernels/absorbing_potential.cu").read_text().replace("PATH_TO_SOURCES", os.path.abspath("sources"))
potential_wall_kernel_source = Path("sources/cuda_kernels/potential_wall.cu").read_text().replace("PATH_TO_SOURCES", os.path.abspath("sources"))
double_slit_kernel_source = Path("sources/cuda_kernels/double_slit.cu").read_text().replace("PATH_TO_SOURCES", os.path.abspath("sources"))

