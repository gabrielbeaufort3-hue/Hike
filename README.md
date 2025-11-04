# HikeSafe Advisor

HikeSafe Advisor is an AI-powered hiking safety assistant. Give it objective trail conditions (distance, elevation gain, altitude, temperature, exposure, surface, duration) and it predicts a categorical risk level (GREEN / YELLOW / RED) plus matching gear advice, just like a seasoned mountain guide warning city visitors before they head out.

---
## Quick Start (Beginner Friendly)

1. **Install Python** (3.9–3.11 recommended). Download from [python.org/downloads](https://www.python.org/downloads/) and make sure `python`/`pip` are on your PATH.
2. **Open a terminal** and move into the project folder:
   ```bash
   cd /path/to/hikesafe
   ```
3. **Create an isolated environment (optional but recommended):**
   - macOS / Linux:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .venv\\Scripts\\Activate.ps1
     ```
4. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. **Use the bundled dataset** or regenerate it (see below). The first run works out of the box with `data/trails_sample.csv`.
6. **Train the model** by running the notebook (instructions below). This exports `models/risk_model.pkl`.
7. **Launch the Streamlit app** and try a few scenarios.

You can stop here for a demo version. The following sections break down each step in more detail.

---
## Project Layout
```
hikesafe/
├─ app/
│  └─ safety_app.py
├─ data/
│  ├─ prepare_kaggle_dataset.py
│  └─ trails_sample.csv        
├─ models/
│  └─ risk_model.pkl
├─ notebooks/
│  └─ trail_risk_train.ipynb
├─ README.md
└─ requirements.txt

```

---
## Dataset Options

> **Note on Dataset Size:**  
> The original Kaggle source file (`kaggle_trails.csv`, ≈351 MB) is excluded from this submission to keep the repository lightweight.  
> It can be downloaded from Kaggle’s [GPX Hike Tracks dataset](https://www.kaggle.com/datasets/roccoli/gpx-hike-tracks) if reproduction from raw data is required.  
> Running `python data/prepare_kaggle_dataset.py` will rebuild the processed `data/trails_sample.csv` used for training.

### Option A – Use the packaged dataset
`data/trails_sample.csv` already contains 750 balanced samples (green/yellow/red). This is enough to run the notebook and Streamlit app without any extra work.

### Option B – Regenerate from Kaggle data
1. Download the Kaggle Swiss Alpine Club trail CSV to `data/kaggle_trails.csv` (already provided in this repo).
2. Run the cleaner:
   ```bash
   python data/prepare_kaggle_dataset.py
   ```
   The script removes outliers, maps SAC grades to risk tiers, flags exposure/slipperiness, balances the classes, and overwrites `data/trails_sample.csv`.
3. (Optional) Open the CSV in a spreadsheet to inspect the results.

---
## Train the Decision Tree Model

### Option 1 – Jupyter Notebook (recommended for beginners)
1. Launch Jupyter from the project root:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/trail_risk_train.ipynb`.
3. Run `Cell > Run All` (or execute each cell in order). The notebook will:
   - Load `data/trails_sample.csv`.
   - Split train/test with stratification.
   - Train a `DecisionTreeClassifier` (`max_depth=4`).
   - Print the classification report and confusion matrix.
   - Plot feature importance.
   - Export the model to `models/risk_model.pkl` via `joblib.dump`.
4. Confirm that `models/risk_model.pkl` exists. The Streamlit app will refuse to start if this file is missing.

### Option 2 – Run the cells manually (for automation)
If you prefer scripting, copy the training code into a `.py` file or run it in an interactive shell. The notebook is still the single source of truth, so beginners should stick with it.

---
## Launch the Streamlit App
1. Stay in the project root and activate your environment (if you created one).
2. Start Streamlit:
   ```bash
   cd app
   streamlit run safety_app.py
   ```
3. Your browser will open automatically (or visit `http://localhost:8501`).
4. Enter trail parameters, click **Assess risk**, and read the predicted label and gear guidance.

**Tip:** Try contrasting inputs, e.g. a mellow town trail vs. an alpine ridge, to see the GREEN vs. RED advice change.

---
## Troubleshooting
- **`ModuleNotFoundError`**: make sure the virtual environment is activated and `pip install -r requirements.txt` completed without errors.
- **`Model artifact missing` message in Streamlit**: rerun the notebook so `models/risk_model.pkl` is created.
- **Port already in use**: pass a new port (e.g. `streamlit run safety_app.py --server.port 8502`).
- **Command not found (`jupyter` or `streamlit`)**: they install with the requirements. If the shell still cannot find them, close and reopen the terminal so PATH updates apply.

---
## Business Impact Snapshot
- **Decision support**: automates the judgement of an experienced mountain guide or insurance risk officer.
- **Cost avoidance**: helps tourism boards, insurers, and hiking platforms reduce rescue missions and medical claims.
- **Explainable AI**: the shallow decision tree plus feature-importance plot highlights intuitive drivers—elevation gain, exposure, temperature—so stakeholders trust the result.

---
## Where to Go Next
- Add weather forecasts or user fitness profiles to personalize the risk output.
- Experiment with ensemble models (Random Forest, Gradient Boosting) and compare accuracy vs. interpretability.
- Expand the Streamlit UI with file uploads (GPX), multi-language support, or printable trip reports.

Enjoy hiking safely!

---

## AI Tools Used

This project was developed with the help of **ChatGPT (GPT-5)** and **OpenAI Codex** as coding and documentation assistants.  
They were used for:

- Generating the initial project skeleton (folders, notebooks, and Streamlit boilerplate).  
- Writing the dataset-cleaning and preprocessing logic to map Kaggle trail data to model-ready features.  
- Drafting explanatory Markdown sections for the README and refining English phrasing for clarity.  
- Troubleshooting minor runtime issues (e.g., relative paths, dependency mismatches, Streamlit config).  

All model training, feature design, and evaluation decisions were ultimately made by the student team.

### Example Prompts

> **Prompt #1:**  
> “Create a Python script that reads a hiking dataset (distance, elevation, temperature) and predicts trail risk level using a decision tree.  
> Include gear advice for each risk category.”

> **Prompt #2:**  
> “Generate a Streamlit interface that lets a user input trail parameters and shows risk prediction results in real time.”

> **Prompt #3:**  
> “Explain how to convert Kaggle GPX hike track data (length_3d, uphill, max_elevation, moving_time) into training features compatible with our model schema.”


These examples demonstrate how AI tools accelerated implementation and documentation while keeping human oversight on data, modeling, and interpretation.

---

## Authors and Acknowledgment

**Author:** Ziyue Wang  

**Acknowledgments:**  
We thank the open-source contributors of Kaggle’s *GPX Hike Tracks* dataset and the Python / Streamlit / scikit-learn communities for making reproducible AI projects accessible to students.

---

## License

MIT License © 2025 Ziyue Wang  
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:  
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

---
