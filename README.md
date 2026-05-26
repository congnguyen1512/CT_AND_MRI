# DACN3 Medical Dashboard

## Cau truc thu muc

- `frontend/public/`: giao dien web tinh (`index.html`, `styles.css`, `app.js`).
- `backend/`: API Node.js va API inference Python.
- `artifacts/`: checkpoint `.pth`, label, report, history va hinh artifact.
- `model/`: model Keras nhan dien anh CT hay MRI (`ct_mri_classifier.keras`).
- `notebooks/`: notebook Colab huan luyen mo hinh.
- `requirements.txt`: thu vien Python cho inference.
- `package.json`: script chay dashboard va inference.

## Lenh chay

```bash
npm install
pip install -r requirements.txt
npm run inference
npm start
```

Dashboard mac dinh chay o `http://localhost:3000`. Inference API mac dinh chay o `http://127.0.0.1:5050`.

Tab `Tu dong` dung model Keras trong `model/` de nhan dien anh CT hay MRI truoc, sau do tu goi model CT hoac MRI tuong ung de phan loai benh.
May hien dung Python qua moi co the khong cai duoc TensorFlow bang pip. De dung tab `Tu dong`, hay tao virtualenv Python 3.12:

```powershell
winget install -e --id Python.Python.3.12
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-auto.txt
npm run inference
```

Mo terminal khac va chay:

```powershell
npm start
```
