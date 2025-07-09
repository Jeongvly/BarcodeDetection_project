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
vertical_kernel = np.array([[-1, 2, -1]] * 10)

# Dilation kernel
dilation_kernel = np.ones((1, 50), np.uint8)

# Opening kernel
opening_kernel = np.ones((90, 90), np.uint8)

if __name__ == "__main__":
    # 1. image load 및 grayscale 변환 및 반전
    image = cv2.imread("barcode1.jpg")
    gray = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian filtering (Smoothing)
    gaussian_img = gaussian_filtering(gray)

    # 3. 수직 방향 edge detection
    vertical = cv2.filter2D(gaussian_img, -1, vertical_kernel)

    # 4. global thresholding으로 이진화
    binary = global_thresholding(vertical, 127, 0)

    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')
    plt.show()

    # 6. 바코드 연결 (수평 방향) 위해 dilation 적용
    dilated = morphology(binary, 2, dilation_kernel)

    # 7. 바코드보다 작은 segment 삭제하기 위해 opening 적용
    opened = opening(dilated, opening_kernel)

    plt.subplot(1, 2, 1)
    plt.imshow(dilated, cmap='gray')
    plt.title('Dilated Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(opened, cmap='gray')
    plt.title('Opened Image')
    plt.axis('off')
    plt.show()

    # 8. Contour를 찾고 가장 큰 영역을 bounding box로 시각화
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounded = image.copy()

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(bounded, (x, y), (x + w, y + h), (0, 255, 0), 10)

    plt.imshow(cv2.cvtColor(bounded, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Image")
    plt.show()
