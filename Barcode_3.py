import cv2
import matplotlib.pyplot as plt
import numpy as np

# Gaussian Filtering 구현 (3x3 커널)
def gaussian_filtering(image):

    #gaussian filtering에서 사용되는 gaussian kernel (3X3) 생성
    gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

    H, W= image.shape
    filtered_img = np.zeros((H, W), dtype=np.uint8)

    # 인덱싱 할 수 없는 부분은 for문에서 제외
    for h in range(1, H - 1):
        for w in range(1, W - 1):
                sliced_img = image[h-1:h+2, w-1:w+2]
                filtered_img[h][w] = np.sum(sliced_img * gaussian_kernel)
    return filtered_img.astype(np.uint8)

# Morphology (erosion or dilation)
def morphology(img, method, kernel, inverse=False):
    h, w = img.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    if not inverse:
        pad_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    else:
        pad_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)

    result = img.copy()

    for i in range(h):
        for j in range(w):
            if method == 1:
                result[i, j] = erosion(pad_img[i:i + kh, j:j + kw], kernel)
            elif method == 2:
                result[i, j] = dilation(pad_img[i:i + kh, j:j + kw], kernel)

    return result

# Erosion 연산
def erosion(boundary, kernel):
    # 선택된 영역 내 kernel과 일치하지 않는 부분있는지 체크
    out = np.where((kernel > 0) & (boundary > 0), 1, 0)
    if (out == kernel).all():
        return 255
    else:
        return 0

# Dilation 연산
def dilation(boundary, kernel):
    boundary = boundary * kernel
    # 선택된 영역 내 kernel과 일치하는 부분있는지 체크
    if np.max(boundary) != 0:
        return 255
    else:
        return 0

# Closing (Dilation -> Erosion)
def closing(img, kernel):
    dilation_img = morphology(img, 2, kernel)
    closing_img = morphology(dilation_img, 1, kernel)
    return closing_img

# Opening (Erosion -> Dilation)
def opening(img, kernel):
    erosion_img = morphology(img, 1, kernel)
    opening_img = morphology(erosion_img, 2, kernel)
    return opening_img

# Global thresholding
def global_thresholding(image, T, T_prev):
    flat_img = image.ravel()

    # global threshold T가 수렴할때까지 반복
    while T != T_prev:
        T_prev = T
        G1_th = np.where(flat_img > T)
        G2_th = np.where(flat_img <= T)

        m1 = np.delete(image, G1_th).mean()
        m2 = np.delete(image, G2_th).mean()

        T = int((m1 + m2) / 2)

    segment = np.where(image > T, 1, 0)
    segment = segment.astype(np.uint8)

    return segment

# Vertical edge kernel
vertical_kernel = np.array([[-1, 2, -1]])

# Dilation kernel
dilation_kernel = np.ones((1, 50), np.uint8)

# Opening kernel
opening_kernel = np.ones((90, 90), np.uint8)


if __name__ == "__main__":
    # 1. image load 및 grayscale 변환
    image = cv2.imread("barcode3.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Zero padding
    M, N = gray.shape
    P, Q = 2 * M, 2 * N
    padded_image = np.zeros((P, Q))
    padded_image[:M, :N] = gray

    # 3. Centered Spectrum (저주파를 center로 이동)
    padded_image_new = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            padded_image_new[x, y] = padded_image[x, y] * ((-1) ** (x + y))

    # 4. DFT 적용 (주파수 변환)
    dft2d = np.fft.fft2(padded_image_new)
    dft2d_ = np.log(np.abs(dft2d))

    # 5. Selective Filter 마스크 생성 (전체 1로 초기화)
    rows, cols = dft2d.shape
    mask = np.ones((rows, cols), dtype=np.complex128)
    crow, ccol = rows // 2, cols // 2

    for dy, dx in [(30, 30), (-30, -30), (30, -30), (-30, 30),
                   (70, 70), (-70, -70), (70, -70), (-70, 70),
                   (140, 140), (-140, -140), (140, -140), (-140, 140)]:
        y, x = crow + dy, ccol + dx
        mask[y - 20:y + 20, x - 20:x + 20] = 0

    # 6. Filtering (pattern noise 제거)
    G = np.multiply(dft2d, mask)
    G_ = np.log(np.abs(G) + 1)

    plt.subplot(1, 3, 1)
    plt.imshow(dft2d_.real, cmap='gray')
    plt.title('DFT')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask.real, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(G_.real, cmap='gray')
    plt.title('Masked DFT')
    plt.axis('off')
    plt.show()

    # 7. Inverse DFT 수행
    idft2d = np.fft.ifft2(G)

    # 8. De-Centering
    for x in range(P):
        for y in range(Q):
            idft2d[x, y] = idft2d[x, y] * ((-1) ** (x + y))

    # 9. Remove Zero-padding
    idft2d = idft2d[:M, :N]
    idft_min, idft_max = np.min(idft2d), np.max(idft2d)
    idft2d = (idft2d - idft_min) / (idft_max - idft_min) * 255.0


    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(idft2d.real, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    plt.show()

    # 10. Frequency Domain -> Spatial Domain
    norm_img = idft2d.astype(np.uint8)

    # 11. Gaussian filtering (Smoothing)
    gaussian_img = gaussian_filtering(norm_img)

    # 12. 수직 방향 edge detection
    vertical = cv2.filter2D(gaussian_img, -1, vertical_kernel)

    # 13. global thresholding으로 이진화
    binary = global_thresholding(vertical, 0, 127)

    # 14. 바코드 연결 (수평 방향) 위해 dilation 적용
    dilated = morphology(binary, 2, dilation_kernel)

    # 15. 바코드보다 작은 segment 삭제하기 위해 opening 적용
    opened = opening(dilated, opening_kernel)

    plt.subplot(1, 3, 1)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(dilated, cmap='gray')
    plt.title('Dilated Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(opened, cmap='gray')
    plt.title('Opened Image')
    plt.axis('off')

    plt.show()

    # 16. Contour를 찾고 가장 큰 영역을 bounding box로 시각화
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounded = image.copy()

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(bounded, (x, y), (x + w, y + h), (0, 255, 0), 5)

    plt.imshow(cv2.cvtColor(bounded, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Image")
    plt.show()

