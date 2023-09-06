import numpy as np
import math
import matplotlib.pyplot as plt

N = 32
array = np.zeros(shape=[N], dtype=np.complex_)

f = 5

for i in range(N):
    array[i] = math.cos(f * i / float(N) * 2.0 * math.pi) + 1j * math.sin(f * i / float(N) * 2.0 * math.pi)

print(array)

transformed_array = np.fft.fft(array, norm="forward")

print(transformed_array)

array = np.fft.fft(transformed_array, norm="backward")

scale = np.arange(0, N, 1)

fig, axs = plt.subplots(2, 1)
axs[0].plot(scale , array.real, scale , array.imag)
axs[0].set_xlim(0, N)
axs[0].set_xlabel('x')
axs[0].set_ylabel('Real / Imag')
axs[0].grid(True)

axs[1].plot(scale , transformed_array.real, scale , transformed_array.imag)
axs[1].set_xlim(0, N)
axs[1].set_xlabel('x')
axs[1].set_ylabel('Transformed')
axs[1].grid(True)

fig.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------------

array = np.zeros(shape=[N, N], dtype=np.complex_)
f = 0.25 / 2.0 * 2.0 * math.pi
print(f"Freq: {f}")
for x in range(N):
    for y in range(N):
        array[x, y] = math.cos(f * x) + 1j * math.sin(f * x)

magnitude = np.abs(array)

transformed_array = np.fft.fftn(array, norm="forward")

fft_magnitude = np.abs(transformed_array)

plt.imshow(magnitude, cmap='gray')
plt.show()

plt.imshow(fft_magnitude, cmap='gray')
plt.show()
