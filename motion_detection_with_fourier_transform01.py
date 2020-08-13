import cv2
import numpy as np
import matplotlib.pyplot as plt
for i in range(10): plt.close()



image1_filename = "../test_images/consecutive_frames_before.jpg"
image2_filename = "../test_images/consecutive_frames_after.jpg"
image1 = cv2.imread(image1_filename)[:,:,0]
image2 = cv2.imread(image2_filename)[:,:,0]

crop_size = 1800
image1_cp = image1[:crop_size, :crop_size]
image2_cp = image2[:crop_size, :crop_size]



f1 = cv2.dft(image1_cp.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
f2 = cv2.dft(image2_cp.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)

f1_shf = np.fft.fftshift(f1)
f2_shf = np.fft.fftshift(f2)

f1_shf_cplx = f1_shf[:,:,0]*1j + 1*f1_shf[:,:,1]
f2_shf_cplx = f2_shf[:,:,0]*1j + 1*f2_shf[:,:,1]

f1_shf_abs = np.abs(f1_shf_cplx)
f2_shf_abs = np.abs(f2_shf_cplx)
total_abs = f1_shf_abs * f2_shf_abs

P_real = (np.real(f1_shf_cplx)*np.real(f2_shf_cplx) +
          np.imag(f1_shf_cplx)*np.imag(f2_shf_cplx))/total_abs
P_imag = (np.imag(f1_shf_cplx)*np.real(f2_shf_cplx) +
          np.real(f1_shf_cplx)*np.imag(f2_shf_cplx))/total_abs
P_complex = P_real + 1j*P_imag

P_inverse = np.abs(np.fft.ifft2(P_complex))

max_id = [0, 0]
max_val = 0
for idy in range(crop_size):
    for idx in range(crop_size):
        if P_inverse[idy,idx] > max_val:
            max_val = P_inverse[idy,idx]
            max_id = [idy, idx]
shift_x = crop_size - max_id[0]
shift_y = crop_size - max_id[1]
print(shift_x, shift_y)

image1_rgb = cv2.imread(image1_filename)
image2_rgb = cv2.imread(image2_filename)
cv2.rectangle(image1_rgb, (0, 0), (image1_rgb.shape[0] - shift_x, image1_rgb.shape[1] - shift_y), (0, 255, 0), 10)
cv2.rectangle(image2_rgb, (shift_x, shift_y), (image2_rgb.shape[0], image2_rgb.shape[1]), (0, 255, 0), 10)

cv2.imwrite("image1_rectangle.jpg", image1_rgb)
cv2.imwrite("image2_rectangle.jpg", image2_rgb)
