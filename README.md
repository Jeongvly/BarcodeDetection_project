# 📚 Barcode Detection using Image Processing Techniques

## 🔧 프로젝트 목적

본 프로젝트는 다양한 영상처리 기법을 활용하여 이미지 내 바코드를 검출하는 것을 목표로 합니다. 각 이미지의 특성에 맞는 전처리와 Morphology 연산을 조합하여 바코드 영역을 정확히 분리합니다.

---

## 📌 사용 기법 및 목적

| 기법 | 목적 |
|------|------|
| **1. Gaussian Filtering** | Edge 검출 전 smoothing을 통해 영상 내 noise 완화 |
| **2. Erosion** | 작은 object 제거 및 객체 테두리 축소 (Opening의 첫 단계) |
| **3. Dilation** | 객체의 테두리 확장, 바코드 선 연결 및 Opening 연산에 사용 |
| **4. Opening** | Erosion 후 Dilation 적용. 불필요한 object 제거 및 바코드 유지 |
| **5. Median Filtering** | Salt & Pepper noise 제거 (kernel 내 중간값 선택) |
| **6. DCT + Selective Filtering** | Frequency domain에서 pattern noise 제거 |
| **7. Global Thresholding** | 픽셀값을 이진화하여 바코드와 배경 분리 |

---

## 📌 개발 환경

- OpenCV (v4.x 이상 권장)
- C++17 또는 Python (해당 구현 환경에 따라)

---

## 📌 바코드별 적용 순서 및 결과 분석

### 📄 1번 바코드

**적용 순서**
1. Grayscale 변환 및 반전
2. Gaussian Filtering (3x3)
3. Vertical Edge Detection (kernel: 10x1)
4. Global Thresholding
5. Dilation (1x50)
6. Opening (90x90)

**결과 분석**
- 노이즈가 거의 없는 이미지로, edge 검출이 매우 안정적으로 수행됨
- Dilation으로 바코드 선 연결, Opening으로 불필요한 object 제거
- 바코드 영역만 정확히 검출되었고, bounding box도 정밀하게 생성됨
- 단, 고해상도 이미지로 인해 Opening 연산에 **약 10분** 소요 → 개선 필요

<img width="689" height="432" alt="Image" src="https://github.com/user-attachments/assets/ae2aae5f-bded-4631-abfa-edfb79816df1" />

---

### 📄 2번 바코드

**적용 순서**
1. Median Filtering (5x5) ×2
2. Gaussian Filtering
3. Vertical Edge Detection
4. Global Thresholding
5. Dilation (1x50)
6. Opening (90x90)

**결과 분석**
- Salt & Pepper noise가 심하여 Median Filtering 2회 적용
- 이후 Gaussian Filtering과 Vertical Edge Detection으로 안정화
- 최종적으로 정확한 바코드 분리 및 검출 성공

<img width="705" height="594" alt="Image" src="https://github.com/user-attachments/assets/38fae395-2a94-4ea8-8650-925da90ae9c8" />

---

### 📄 3번 바코드

**적용 순서**
1. DCT 변환
2. Selective Filtering (mask 적용)
3. Inverse DCT
4. Gaussian Filtering
5. Vertical Edge Detection
6. Global Thresholding
7. Dilation (1x50)
8. Opening (90x90)

**결과 분석**
- 이미지에 얇은 천 같은 **pattern noise** 존재
- DCT 후 frequency domain에서 해당 noise 제거
- Inverse DCT로 복원된 이미지에서 noise 대부분 제거됨
- 이후 전처리 및 Morphology 연산을 통해 바코드만 정확히 검출

<img width="796" height="631" alt="Image" src="https://github.com/user-attachments/assets/5ad5c9eb-fb0f-4274-b345-7d88efcc3e4b" />

---

## 🚀 개선 사항

1. **Morphology 연산 속도 문제**
   - 현재 Opening 연산이 이중 for-loop 기반으로 구현되어 고해상도 이미지 처리 시 **연산 시간이 과도함**
   - → **OpenCV 내장 함수(cv::morphologyEx 등)로 대체**하거나 CUDA/OpenCL 가속 등 개선 필요

2. **Edge Detection 방향성 한계**
   - Vertical Edge만 고려 → 수직이 아닌 바코드는 검출 불가
   - → Sobel, Laplacian 등의 **방향성과 무관한 필터** 사용 또는 **Adaptive edge 방향 탐색** 기법 도입 필요

---


