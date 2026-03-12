# 🫁 Pneumonia Detection — Computer Vision Project

## Overview

In this project you will build a **pneumonia detection system** from chest X-ray images.  
You will implement three deep learning architectures, expose them through a **FastAPI** backend,  
and interact with them through a **Streamlit** frontend.

At the end, you generate a CSV of predictions on the test set and submit it to the class leaderboard.

---

## Project Structure

```
pneumonia_project/
│
├── models/
│   ├── unet.py              ← U-Net architecture        (to complete)
│   ├── resnet.py            ← ResNet architecture       (to complete)
│   └── inception.py        ← Inception architecture    (to complete)
│
├── api/
│   └── main.py              ← FastAPI backend            (to complete)
│
├── app/
│   └── streamlit_app.py     ← Streamlit frontend        (to complete)
│
├── data/
│   ├── train/
│   │   ├── PNEUMONIA/
│   │   └── NORMAL/
│   ├── val/
│   │   ├── PNEUMONIA/
│   │   └── NORMAL/
│   └── test_for_students/   ← unlabelled test images (for submission)
│
├── sample_submission.csv    ← format reference for leaderboard submission
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd pneumonia_project
```

### 2. Create a virtual environment

A **virtual environment** is an isolated Python installation that lives inside your project folder. It keeps all the packages you install for this project (PyTorch, FastAPI, Streamlit, …) completely separate from your system Python and from other projects on your machine.

**Why does this matter?**
- Different projects often need different versions of the same library. Without isolation they clash and break each other.
- Your global Python stays clean — uninstalling the project is as simple as deleting the `venv/` folder.
- Everyone on the team works with the exact same dependency versions, so "it works on my machine" stops being an excuse.

```bash
# Create the virtual environment (only needed once)
python -m venv venv
```

This creates a `venv/` folder in the project root. It contains a private copy of the Python interpreter and a `site-packages/` directory where libraries will be installed.

### 3. Activate the virtual environment

You must activate the environment **every time you open a new terminal** before running any project command.

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

Once activated, your prompt will show `(venv)` at the beginning, confirming you are inside the environment.

To leave the environment at any time:
```bash
deactivate
```

### 4. Install dependencies

With the environment active, install all required packages in one command:

```bash
pip install -r requirements.txt
```

This reads `requirements.txt` and installs the exact libraries needed — PyTorch, torchvision, FastAPI, Streamlit, scikit-learn, matplotlib, and more — only inside your virtual environment.

---

## How to run

> Make sure your virtual environment is **activated** (`source venv/bin/activate`) before running any of the commands below.

**1. Start the FastAPI backend**
```bash
cd api
uvicorn main:app --reload --port 8000
```

**2. Start the Streamlit frontend** (in a second terminal)
```bash
cd app
streamlit run streamlit_app.py
```

Then open your browser at `http://localhost:8501`.

---

## Workflow

1. **Choose a model** (U-Net, ResNet, or Inception) in the sidebar
2. **Set hyperparameters** (learning rate, epochs, batch size, …)
3. Click **Train** — the frontend calls the FastAPI `/train` endpoint
4. View **training & validation curves** and metrics live
5. Click **Generate predictions** — calls `/predict` on the test set
6. **Download** the generated `submission.csv`
7. Upload it to the class leaderboard 🏆

---

## Models to implement

| File | Architecture | Key idea |
|---|---|---|
| `models/unet.py` | U-Net | Encoder-decoder with skip connections |
| `models/resnet.py` | ResNet | Residual blocks adapted for classification |
| `models/inception.py` | Inception | Multi-scale convolutions |

Each model file contains the class skeleton and the expected interface.  
**Do not change the class names or the `forward()` signature.**

---

## Leaderboard submission format

Your `submission.csv` must have exactly two columns:

```
id,prediction
img_0001,0.91
img_0002,0.07
img_0003,0.83
...
```

- `id` — image filename without extension (e.g. `img_0001`)
- `prediction` — probability of PNEUMONIA between 0 and 1 (not a hard label)

---

## Data

- `PNEUMONIA` images → label **1**
- `NORMAL` images → label **0**


-- Mehyar MLAWEH