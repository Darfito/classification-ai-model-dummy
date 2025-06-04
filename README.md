# 🧠 Chest Model Classification App

This repository contains a Python application for chest image classification using a YOLOv8 model (`yolov8n.pt`). It is powered by FastAPI, Ultralytics, and other deep learning libraries.

---

## ⚙️ Requirements

- **Python 3.12.7**
- Git
- pip (Python package manager)

You can verify your Python version with:

```bash
python --version
```

---

## 🐍 Setup Virtual Environment

To keep dependencies isolated, use a Python virtual environment (`venv`).

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

#### ✅ On **Windows PowerShell**:

```powershell
.env\Scripts\Activate.ps1
```

> If you encounter a policy restriction error, run PowerShell as Administrator and set policy:

```powershell
Set-ExecutionPolicy RemoteSigned
```

#### ✅ On **Windows CMD**:

```cmd
venv\Scriptsctivate.bat
```

#### ✅ On **macOS / Linux**:

```bash
source venv/bin/activate
```

> When activated, your terminal prompt should show: `(venv)`.

---

## 📦 Install Dependencies

After activating the environment, install all required packages with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the App

To start the application:

```bash
python main.py
```

Make sure the model file `yolov8n.pt` is available in the root directory.

---

## 📁 Project Structure

```
dataset-model-chest/
├── main.py                  # Main application script
├── yolov8n.pt               # YOLOv8 model weights (file hasil pelatihan (training) dari model deep learning YOLOv8)
├── static/                  # Static assets (if any)
├── venv/                    # Virtual environment (ignored by git)
├── __pycache__/             # Python cache (ignored)
├── requirements.txt         # List of dependencies
├── .gitignore               # Files to ignore in git
└── README.md                # Project documentation
```

---

## 🛑 .gitignore Info

Git will ignore the following:

- `venv/` – your virtual environment
- `__pycache__/`, `*.pyc`, `*.pyo` – compiled Python files
- `*.pt` – model weight files
- `.env` – secret environment configs
- `.vscode/`, `.idea/` – editor configs

---

## ✅ Notes

- This app is ideal for local model testing or integration into a larger ML pipeline.
- You can serve the model with an API using `FastAPI` or deploy to a service like Hugging Face, Streamlit Cloud, etc.

---

## 📬 Contributions

Feel free to fork this repo and open pull requests for any enhancements or bug fixes.

---

## 📜 License

This project is licensed under the MIT License.