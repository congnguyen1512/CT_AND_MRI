# Hệ Thống Nhận Diện & Phân Loại Ảnh Y Tế (CT & MRI) - DACN3

![CI/CD Pipeline](https://github.com/congnguyen1512/CT_AND_MRI/actions/workflows/ci.yml/badge.svg)
![Node Version](https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen.svg)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)
![Deployment](https://img.shields.io/badge/deployed_with-Vercel-black.svg)

Hệ thống bảng điều khiển (Dashboard) thông minh phục vụ trực quan hóa quá trình huấn luyện và ứng dụng mô hình Deep Learning trong phân loại hình ảnh y tế:
1. **🔍 Tự động nhận diện Modal (CT vs MRI)**: Sử dụng MobileNetV2 (Keras) để tự động phân định ảnh chụp CT hay MRI, sau đó chuyển tiếp đến mô hình chuyên khoa tương ứng.
2. **🧠 MRI - U não**: Sử dụng AlexNet (PyTorch) phân loại 4 trạng thái: Glioma, Meningioma, Tuyến yên (Pituitary), và Không có u (Notumor). Độ chính xác đạt **97.22%**.
3. **🩻 CT - Đột quỵ não**: Sử dụng ResNet18 (PyTorch) phát hiện Xuất huyết não (Bleeding), Nhồi máu não (Ischemia), và Bình thường (Normal). Độ chính xác đạt **97.74%**.

---

## 📁 Cấu trúc dự án

```text
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD Pipeline (Lint, Test, Validate, Build)
├── api/
│   └── index.js                 # Serverless Function Adapter cho Vercel
├── artifacts/                   # Checkpoints, Classification reports, Confusion matrix, History
│   ├── brain_mri_*.json
│   ├── brain_stroke_ct_*.json
│   └── *.pth (ignored in git)
├── backend/
│   ├── server.js                # Express API & Static file server
│   └── inference_server.py      # Python Flask Inference API (PyTorch + TensorFlow)
├── frontend/
│   └── public/
│       ├── app.js               # Frontend Client Logic & Chart.js Visualizer
│       ├── index.html           # Dashboard HTML
│       └── styles.css           # Modern Medical Glassmorphism UI
├── model/                       # Model MobileNetV2 nhận diện CT/MRI (.keras)
├── test/
│   └── api.test.js              # Automated Test Suite (Node:test)
├── vercel.json                  # Vercel Deployment Configuration
├── package.json
└── requirements.txt
```

---

## 🚀 Hướng dẫn chạy Local

### 1. Cài đặt Node dependencies
```bash
npm install
```

### 2. Chạy Dashboard Server
```bash
npm start
```
Truy cập giao diện tại: `http://localhost:3000`

### 3. Chạy Inference API (để dự đoán ảnh thực tế)
Yêu cầu Python 3.10 - 3.12 (khuyên dùng Python 3.12):
```bash
pip install -r requirements.txt
npm run inference
```
Inference API sẽ lắng nghe tại: `http://127.0.0.1:5050`.

### 4. Chạy kiểm thử tự động (Test Suite)
```bash
npm test
```

---

## ☁️ Triển khai lên Vercel (Deployment)

Dự án đã được cấu hình tối ưu sẵn sàng cho Vercel (`vercel.json`, `api/index.js`, `.vercelignore`):

### Triển khai tự động qua Vercel CLI:
```bash
npx vercel --prod
```

### Triển khai qua GitHub (CI/CD):
1. Truy cập [Vercel Dashboard](https://vercel.com/dashboard)
2. Nhấn **Add New Project** -> Chọn repository `congnguyen1512/CT_AND_MRI`.
3. Vercel sẽ tự động build và deploy mỗi khi bạn `git push` lên branch `main`.
4. Nếu kết nối với máy chủ AI inference riêng, cấu hình biến môi trường `INFERENCE_URL` trong mục **Settings > Environment Variables** trên Vercel.

---

## 🧪 CI/CD Pipeline

Pipeline GitHub Actions được thiết lập tại `.github/workflows/ci.yml`:
- Tự động kích hoạt khi có `push` hoặc `pull_request` vào nhánh `main`.
- Kiểm tra tính toàn vẹn của các file Artifacts JSON.
- Chạy test suite trên môi trường Node 18 & 20.
- Đảm bảo mã nguồn đạt chuẩn trước khi merge và deploy.
