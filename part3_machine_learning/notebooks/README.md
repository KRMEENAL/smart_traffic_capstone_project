# Metro Interstate Traffic Volume — Capstone Notebooks

How to set up and run the six task notebooks in this project. All of them read the same
engineered features file, so one setup covers all of them.

**Note:** this README covers Tasks 1,2, 3, 4, 5, and 6 — the notebooks currently in this folder.


## 1. Prerequisites

- Python 3.10+
- Jupyter (Notebook or Lab)
- Packages:

  ```bash
  pip install pandas numpy scikit-learn matplotlib scipy joblib
  pip install mlflow          # Task 4, Task 6
  pip install tensorflow shap # Task 3
  pip install fastapi uvicorn # Task 6
  ```

## 2. Folder layout

Put everything in **one working directory** and launch Jupyter from there — every notebook
uses relative paths, so this matters:

```
your-project-folder/
├── Features_Metro_Interstate_Traffic_Volume.csv
├── Task1_Supervised_Models.ipynb
├── Task2_Unsupervised_Models.ipynb
├── Task3_Deep_Learning_Explainability.ipynb
├── Task4_MLflow_Experiment_Tracking.ipynb
├── Task5_Traffic_Recommendation_System.ipynb
├── Task6_MLOps_Deployment_Simulation.ipynb
└── deployment_api.py
```

`Features_Metro_Interstate_Traffic_Volume.csv` is the feature-engineered CSV from the
earlier data pipeline stage (`feature_engineering.py` / the `report.md` pipeline) — every
notebook expects it at that path and will raise a clear `KeyError`/`FileNotFoundError` naming
what's missing if it isn't there or an expected column is absent.

## 3. Running each notebook

The notebooks don't depend on each other's *output* — each one reloads the raw features CSV
and rebuilds whatever it needs from scratch — so they can technically run in any order. Reading
them in this order will make the most sense, since each one builds on the last conceptually:

| Order | Notebook | What it does |
|---|---|---|
| 1 | `Task1_Supervised_Models.ipynb` | Trains and compares regression models (traffic volume) and classification models (a proxy accident-risk label) |
| 2 | `Task2_Unsupervised_Models.ipynb` | K-means clustering and assocation rule mining | |
| 3 | `Task3_Deep_Learning_Explainability.ipynb` | An LSTM for sequential traffic prediction, plus SHAP explainability | |
| 4 | `Task4_MLflow_Experiment_Tracking.ipynb` | Logs Task 1's models to MLflow (params, metrics, model artifacts) |
| 5 | `Task5_Traffic_Recommendation_System.ipynb` | Builds a travel-timing recommender on top of the trained models |
| 6 | `Task6_MLOps_Deployment_Simulation.ipynb` |Model versioning, MLflow registry, a FastAPI deployment mock-up, drift monitoring, and an alerting dashboard |

Run every cell top to bottom within a notebook — later cells depend on variables defined
earlier in the *same* notebook.

### Task 6 specifically

- `Task6_MLOps_Deployment_Simulation.ipynb` and `deployment_api.py` must sit in the **same
  folder** and be run from there. The notebook trains the "production" model and writes three
  files (`production_model.joblib`, `model_metadata.json`, `reference_stats.json`) that
  `deployment_api.py` loads — run the notebook's sections 6.1/6.2/the bridge cell *before*
  importing `deployment_api` in section 6.3, and before running `deployment_api.py` standalone.
- To serve the API for real (outside the notebook's in-process demo):
  ```bash
  uvicorn deployment_api:app --reload
  ```
  then see `http://127.0.0.1:8000/docs` for the interactive schema.

## 4. If you're running on Databricks

Databricks blocks a local `file:` MLflow tracking store by default. Add this **before**
`import mlflow` in Task 4 and Task 6 (already included in Task 6; add it to Task 4 too if
needed), or set it as a cluster-level environment variable instead (Compute → your cluster →
Edit → Advanced options → Environment variables):

```python
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
```

If MLflow was already imported once in the current kernel session, restart the kernel first —
the setting only takes effect before that first import.

## 5. Common issues

- **`MlflowException` on `set_tracking_uri`/`set_experiment`:** delete any existing `mlruns/`
  folder in the working directory and restart the kernel before re-running — a partially
  written store from an earlier failed run is the usual cause.
- **`ModuleNotFoundError: No module named 'deployment_api'`:** `deployment_api.py` isn't in
  Jupyter's current working directory. Run `import os; os.listdir(".")` in a cell to check, and
  move the file (or the notebook) so both are together.
- **Missing feature columns / `KeyError`:** usually means `data/Features_Metro_Interstate_Traffic_Volume.csv`
  is an older version that predates a feature used downstream — regenerate it from the feature
  engineering pipeline first.
