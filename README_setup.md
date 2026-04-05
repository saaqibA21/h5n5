# 🚀 H5N5 & Pima QSVM Project: Fresh Setup Guide

If you are starting on a **new laptop** with **nothing installed**, follow these exact steps to get the environment running.

## 1. Prerequisites (The "One-Time" Installs)

Before you can run any code, you need two basic tools installed on your computer:

1.  **Python 3.11 or 12**:
    - Download and install from [python.org](https://www.python.org/downloads/).
    - **IMPORTANT**: During installation, check the box that says **"Add Python to PATH"**.
2.  **Git**:
    - Download and install from [git-scm.com](https://git-scm.com/downloads).
    - Use the default settings during installation.

## 2. Get the Code

Open your **Terminal** (CMD or PowerShell on Windows) and run these commands:

```powershell
# Go to your documents folder (or wherever you want the project)
cd Documents

# Download the repository
git clone https://github.com/saaqibA21/h5n5.git

# Enter the project folder
cd h5n5
```

## 3. Setup the Virtual Environment

This keeps the project's libraries separate from your system so they don't cause conflicts.

```powershell
# Create the environment
python -m venv .venv

# Activate it
.venv\Scripts\activate
```

## 4. Install Libraries

Install all necessary tools like libraries for Quantum Computing (Qiskit), Data Science (pandas, scikit-learn), and the Dashboard (Streamlit).

```powershell
# Update pip first
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install dashboard tools (if not in requirements)
pip install streamlit plotly
```

## 5. Configure Your Credentials

Create a file named `.env` in the root folder (`h5n5/`) to allow the script to download data from NCBI.

1.  Create a file named `.env`.
2.  Paste this inside:
    ```
    NCBI_EMAIL=your_email@example.com
    NCBI_API_KEY=your_ncbi_api_key_if_you_have_one
    ```

## 6. Run the Analysis Pipeline

This will download sequences, train 5 models (including Quantum SVM), and generate a detailed report.

```powershell
python run_all.py
```
*(Note: Training the Quantum SVM may take 15-30 minutes depending on your CPU).*

## 7. Launch the Dashboard

Once the analysis is done, you can explore the results visually:

```powershell
streamlit run app.py
```

---

### 📂 Key Folders After Completion:
- `data/processed`: Contains your trained models (`.joblib`) and the Word report (`.docx`).
- `src/`: Where the underlying model logic lives (Classical SVM, QSVC, Random Forest, etc.)
