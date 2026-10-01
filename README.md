# ⚖️ Indonesian Legal RAG System

Aplikasi **Retrieval-Augmented Generation (RAG)** berbasis lokal untuk menganalisis dan menjawab pertanyaan berdasarkan dokumen hukum dan perundang-undangan Indonesia.

Sistem ini menggabungkan **Hybrid Retrieval**, yaitu pencarian berbasis semantic similarity menggunakan **FAISS** dan pencarian berbasis keyword menggunakan **BM25**, kemudian menggunakan Large Language Model (LLM) untuk menghasilkan jawaban berdasarkan konteks dokumen yang ditemukan.

## 🚀 Fitur Utama

- **Hybrid Retrieval**
  - FAISS untuk semantic/vector search.
  - BM25 untuk keyword-based search.
  - Menggabungkan hasil retrieval untuk meningkatkan relevansi konteks.

- **LLM Engine**
  - `Qwen/Qwen2.5-1.5B-Instruct`
  - Mendukung quantization 4-bit untuk mengurangi penggunaan VRAM.
  - Dioptimalkan untuk inference menggunakan NVIDIA CUDA.

- **Embedding Model**
  - `BAAI/bge-m3`
  - Digunakan untuk menghasilkan embedding dari dokumen hukum dan query pengguna.

- **Backend API**
  - FastAPI.
  - Endpoint utama: `/api/v1/ask`.

- **Frontend**
  - Streamlit.
  - Menyediakan antarmuka chat untuk berinteraksi dengan sistem RAG.

- **Hardware Optimization**
  - Dirancang agar dapat dijalankan pada GPU NVIDIA dengan VRAM terbatas.
  - Pengujian utama dilakukan pada NVIDIA RTX 2050 4GB VRAM.

## 🏗️ Arsitektur Sistem

```text
                    ┌──────────────────────┐
                    │      User Query      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Query Processing   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Hybrid Retrieval   │
                    │                      │
                    │  ┌────────┐ ┌─────┐ │
                    │  │ FAISS  │ │BM25 │ │
                    │  └───┬────┘ └──┬──┘ │
                    └──────┼─────────┼────┘
                           │         │
                           └────┬────┘
                                ▼
                    ┌──────────────────────┐
                    │ Context / Documents  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Qwen LLM Engine    │
                    │ Qwen2.5-1.5B-Instruct│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Generated Answer  │
                    └──────────────────────┘
```

## 📁 Struktur Project

```text
legal-rag-mlops/
│
├── src/
│   ├── api.py
│   ├── ...
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
└── ...
```

> Folder seperti model weights, Hugging Face cache, virtual environment, dan database/vectorstore lokal tidak disimpan di repository dan sudah dikecualikan melalui `.gitignore`.

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/legal-rag-mlops.git
cd legal-rag-mlops
```

### 2. Buat Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Jika PowerShell menolak aktivasi karena execution policy, kamu dapat menjalankan Python dan executable dari virtual environment secara langsung, misalnya:

```powershell
.\venv\Scripts\python.exe --version
```

### 3. Install Dependencies

Install seluruh dependency:

```powershell
pip install -r requirements.txt
```

Pastikan versi PyTorch yang digunakan sesuai dengan environment CUDA pada komputer.

Untuk memeriksa apakah PyTorch dapat mendeteksi GPU:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output jika GPU berhasil terdeteksi kurang lebih:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 2050
```

## ▶️ Running the Application

### 1. Jalankan FastAPI Backend

Buka terminal PowerShell pertama:

```powershell
.\venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
```

Backend akan berjalan pada:

```text
http://localhost:8000
```

API documentation tersedia di:

```text
http://localhost:8000/docs
```

### 2. Jalankan Streamlit Frontend

Buka terminal PowerShell kedua:

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

Kemudian buka alamat yang diberikan oleh Streamlit, biasanya:

```text
http://localhost:8501
```

## 📦 Dependencies

Library utama yang digunakan dalam project:

- PyTorch
- FastAPI
- Uvicorn
- Streamlit
- LangChain
- Transformers
- Sentence Transformers
- FAISS
- BM25
- PyPDF
- Pydantic
- Requests

Daftar versi dependency yang digunakan dalam environment dapat dilihat pada:

```text
requirements.txt
```

## 🔐 Environment Variables

Jika project membutuhkan API key atau konfigurasi rahasia, simpan di file `.env`.

Contoh:

```env
API_KEY=your_api_key_here
```

Jangan pernah melakukan commit terhadap file `.env`.

## 📌 Git Workflow

Setelah melakukan perubahan pada project:

```powershell
git status
git add .
git commit -m "feat: update RAG pipeline"
git push
```

Untuk melihat repository yang terhubung:

```powershell
git remote -v
```

## 📄 License

This project is intended for educational and research purposes.

---

## ⚠️ Disclaimer

Sistem ini dibuat untuk tujuan **edukasi, penelitian, dan eksperimen teknologi RAG/ML Engineering**.

Jawaban yang dihasilkan oleh sistem AI tidak dapat dianggap sebagai nasihat atau pendapat hukum. Untuk keperluan hukum yang sebenarnya, pengguna tetap harus melakukan verifikasi terhadap sumber hukum resmi dan berkonsultasi dengan profesional yang kompeten.
