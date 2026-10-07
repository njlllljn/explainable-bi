# Explainable BI — Machine Learning Decision Support Studio

An interactive academic project for exploring business and customer data with machine learning. The dashboard connects model outputs to readable explanations, validation metrics, scenario comparisons, and an Excel decision history.

## Project objective

Build an explainable business decision-support tool that uses historical business and customer data to estimate a numeric business outcome and customer risk, discover groups and unusual records, and compare potential actions. For each model-based result, the dashboard shows how it was produced and what validation evidence is available so a user can judge its limitations.

The built-in dataset is synthetic and exists to make the project runnable. It is not evidence about a real business and its recommendations should not be used for operational decisions.

## Requirements

- Python 3.11 or newer
- macOS, Windows, or Linux
- An internet connection for the first installation of Python packages
- Brave, Chrome, Safari, or another modern browser

## Install and run

Open Terminal, move into the project directory, create a virtual environment, install dependencies, and start Streamlit.

### macOS or Linux

```bash
cd "/Users/kinjaljadhav/Documents/ChatGPT/ML_project"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
cd "C:\path\to\ML_project"
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in Brave. To stop the app, return to Terminal and press **Ctrl+C**.

## Dashboard pages

Use **Go to page** in the left sidebar to move between focused views. The layout uses compact responsive charts and a selectable overview so the dashboard stays usable in a laptop browser window.

| Page | What it does | Main outputs |
| --- | --- | --- |
| **Overview** | Explore the loaded data and filter records by risk group. | KPI cards, outcome distribution, risk mix, selectable scatterplot, correlation heatmap, data quality, record preview. |
| **Project guide** | Read why methods were selected and how the backend works. | Data flow, assumptions, metric definitions, confidence guidance, implementation source viewer. |
| **Revenue analysis** | Compare models for a numeric outcome. | Linear, polynomial, Ridge, Lasso, and random forest regression; R², RMSE, MAE; permutation importance; optional PCA projection. |
| **Risk decisions** | Train classifiers and inspect one record. | Logistic regression, SVM, Random Forest; precision, recall, F1, ROC AUC, accuracy; selected-record probability; local sensitivity explanation; expandable validation scorecard. |
| **Customer segments** | Group observations or mark density outliers. | K-Means, DBSCAN, 2D PCA projection, group profiles. |
| **Decision simulator** | Demonstrate an ensemble and a small Q-learning policy. | Random Forest feature importance and simulated action values with an explanation of the selected action. |
| **Ask your data** | Ask supported descriptive questions in everyday language. | Local answers for averages, totals, medians, categories, missing cells, group summaries, and numeric correlations. |
| **What-if & history** | Change numeric inputs for one record and compare model estimates. | Baseline vs. scenario outcome, per-input estimated shifts, optional risk comparison, validation scorecards, Excel change log. |

## Use your own dataset

1. In the sidebar, choose **Upload a personal dataset**.
2. Upload a `.csv` or `.xlsx` file. For Excel workbooks, the first worksheet is loaded.
3. Use one row per observation. The machine-learning pages currently require at least two usable numeric feature columns and at least 12 rows; a much larger dataset is recommended for meaningful validation.
4. Choose numeric predictor columns and a numeric outcome in the model pages. Classification targets must contain both classes coded as `0` and `1`.
5. Review missing values, duplicates, units, class balance, dates, and possible target leakage before trusting a result. Do not use identifiers as model features.

Text columns can be explored by the data assistant, but they are not automatically encoded as ML predictors in this version. Uploaded data is held by the running Streamlit app; it is not copied into the project automatically. The local rule-based assistant does not send the data to an external AI service.

## How the machine-learning workflow works

1. The user chooses an outcome and predictor columns.
2. Predictor columns are converted to numeric values. Missing numeric values are imputed with medians learned from the training split.
3. The data is split into 75% training and 25% holdout records using a fixed seed. The models fit on the training portion; the holdout portion estimates performance on unseen records.
4. Each selected algorithm is fitted and scored. Metrics are shown in the dashboard and can be expanded for detail.
5. For a selected risk record, the app reports the model's positive-class probability and changes one feature at a time to the training median to show how the model estimate shifts.
6. The what-if page keeps a fitted model fixed, changes user-selected input values, and compares the resulting model estimates.

### Coursework objectives

| Outcome | Page | Algorithms and methods |
| --- | --- | --- |
| **CO1** | Revenue analysis | Feature selection, median imputation, standardization, PCA for visualization, multiple linear regression, degree-2 polynomial regression, Ridge, Lasso, Random Forest regression, held-out metrics, permutation importance. |
| **CO2** | Risk decisions | L2-regularized Logistic Regression, RBF SVM, Random Forest classification, accuracy, precision, recall, F1, ROC AUC, confusion matrix, local sensitivity explanation. |
| **CO3** | Customer segments | Standardization, K-Means, DBSCAN, PCA visualization, segment profiles and noise detection. |
| **CO4** | Decision simulator | Random Forest ensemble and tabular Q-learning in a small, explicitly synthetic state/action/reward environment. |

## Interpreting scores and explanations

- **Accuracy** is the share of holdout records classified correctly.
- **Precision** is the share of records flagged positive that were actually positive.
- **Recall** is the share of actual positive records the model found.
- **F1** balances precision and recall.
- **ROC AUC** measures how well the model ranks positive cases above negative cases across possible thresholds.
- **R²** describes how much target variation is explained on the holdout set. **RMSE** and **MAE** summarize regression error in the target's units.
- A selected record's predicted probability is a model estimate, not a guarantee or automatically calibrated certainty. Validation metrics summarize overall performance and do not quantify confidence for one record.
- The local one-feature-at-a-time chart shows model sensitivity around one record. It is not a causal effect or a SHAP value. Correlations and feature importance also do not prove causation.
- Q-learning's rewards and state transitions are teaching assumptions, not learned from a company's operational history. The action recommendation is illustrative.

## Excel decision history

On **What-if & history**, choose **Save this scenario to the Excel change log** to append a row to `outputs/decision_log.xlsx`. The workbook records the timestamp, dataset filename, selected record, decision and notes, changed inputs, baseline and scenario estimates, estimated changes, and model validation metrics. The page can display the history and download the workbook.

The workbook is local and may contain business-sensitive values. It is excluded from Git by `.gitignore`; do not force-add it to a public repository.

## Project files

```text
app.py               Streamlit interface and dashboard pages
data.py              Fixed-seed synthetic business dataset
ml.py                scikit-learn workflows, evaluation, explanations, Q-learning
data_assistant.py    Local descriptive dataset Q&A
decision_log.py      Append/read Excel scenario history
requirements.txt     Python dependencies
README.md            Setup, project guide, and GitHub instructions
```

## GitHub readiness and upload

### Check whether this folder is connected to GitHub

Run these commands from the project folder:

```bash
git --version
git status --short
git branch --show-current
git remote -v
git log -1 --oneline
```

What to look for:

- `git remote -v` should show a GitHub URL named `origin`. **If it prints nothing, this local project is not connected to a GitHub repository yet.**
- `git branch --show-current` shows the branch you will push, usually `main`.
- `git log -1 --oneline` should show a commit. If it says there are no commits yet, make the first commit before pushing.
- `git status --short` lists local changes. Review this list before staging anything.

If you use GitHub CLI, also check its sign-in status:

```bash
gh --version
gh auth status
```

`gh auth status` should say you are logged into the GitHub account where you plan to create the repository. If `gh` is not installed on macOS and Homebrew is available, install it with `brew install gh`, then run `gh auth login` and follow the prompts.

If Git does not know the commit author on this computer, set it before committing:

```bash
git config --global user.name "Your Name"
git config --global user.email "YOUR-GITHUB-EMAIL"
```

### First upload when no GitHub repository is connected

If this is the first upload, review the status and add only the project files. These explicit paths avoid staging local assistant configuration folders or personal data by accident.

```bash
git add .gitignore README.md app.py data.py data_assistant.py decision_log.py ml.py requirements.txt
git diff --cached --check
git diff --cached --stat
git commit -m "Add explainable BI machine learning dashboard"
```

Then create a **private** GitHub repository and push it with GitHub CLI:

```bash
gh repo create explainable-bi --private --source=. --remote=origin --push
```

To create a public repository instead, replace `--private` with `--public`. Only choose public if you have reviewed every staged file and confirmed it contains no private dataset, decision log, credentials, or personal information.

If you prefer to create the repository on GitHub.com yourself, create an empty repository (do not initialize it with a README), then replace `YOUR-USERNAME` and `YOUR-REPOSITORY` below with its URL:

```bash
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

If your branch is not named `main`, either push the branch shown by `git branch --show-current`, or rename it before pushing:

```bash
git branch -M main
git push -u origin main
```

### Push later changes

```bash
git status --short
git add README.md app.py data.py data_assistant.py decision_log.py ml.py requirements.txt .gitignore
git diff --cached --check
git diff --cached --stat
git commit -m "Describe the change"
git push
```

### Check whether a push should work

After `origin` is configured and you have a commit, run:

```bash
gh auth status
git remote -v
git status --short
git push --dry-run origin main
```

The dry run checks whether Git can contact the remote and prepare the push without uploading the branch. A successful dry run is a good readiness signal; the actual `git push -u origin main` performs the upload. If you use SSH instead of HTTPS, make sure your GitHub SSH key is configured and test the connection with `ssh -T git@github.com`.
