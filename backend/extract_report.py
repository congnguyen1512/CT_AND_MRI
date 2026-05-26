from docx import Document
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

doc = Document(PROJECT_ROOT / 'BaoCaoDoAnChuyenNganh3.docx')
for i, para in enumerate(doc.paragraphs):
    text = para.text.strip()
    if text:
        print(f'{i}: {text[:200]}')
