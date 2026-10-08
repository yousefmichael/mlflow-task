# ML Experiment Tracking with MLflow

## 1. Project Description

This project demonstrates **machine learning experiment tracking using MLflow** for a House Price Prediction problem.

The project uses the **California Housing dataset** and trains the same regression model three times with different hyperparameter configurations. Each experiment is tracked using MLflow.

For every run, MLflow records:

* Model parameters
* RMSE
* MAE
* R²
* The trained model as an artifact

The three experiments use different values for `max_depth` and `learning_rate`.

The main goal is to **train, track, compare, and select** the best-performing model.

---

## 2. Experiment Configurations

The following three configurations are evaluated:

| Run   | Max Depth | Learning Rate |
| ----- | --------: | ------------: |
| Run 1 |         3 |           0.1 |
| Run 2 |         5 |          0.05 |
| Run 3 |         7 |          0.01 |

The models are evaluated on the validation dataset using:

* **RMSE** — Root Mean Squared Error
* **MAE** — Mean Absolute Error
* **R²** — R-squared

For model selection, **RMSE is used as the primary metric**. The model with the lowest RMSE is selected as the best model.

---

## 3. Project Structure

```text
mlflow-task/
│
├── train.py
├── requirements.txt
└── README.md
```

### `train.py`

Contains the complete machine-learning workflow:

* Loads the California Housing dataset
* Splits the dataset into training and validation sets
* Trains the three model configurations
* Calculates validation metrics
* Logs parameters and metrics to MLflow
* Logs the trained models as MLflow artifacts

### `requirements.txt`

Contains the Python dependencies required to run the project.

### `README.md`

Contains the project description, instructions, experiment results, comparison, and selected model.

---

## 4. How to Run the Project

### Step 1 — Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd mlflow-task
```

### Step 2 — Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install the dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the training script

```bash
python train.py
```

The script trains all three experiments and logs their parameters, metrics, and trained models to MLflow.

### Step 5 — Start the MLflow UI

After running the training script, start the MLflow interface:

```bash
mlflow ui
```

Then open the MLflow UI in your browser using the address displayed in the terminal, commonly:

```text
http://127.0.0.1:5000
```

From the MLflow UI, the three experiment runs can be inspected and compared.

---

## 5. MLflow Experiment Results

The following screenshot shows the three MLflow experiment runs and their recorded parameters and metrics.

![MLflow Experiment Results](mlflow-results.png)

> **Note:** Replace `mlflow-results.png` with the actual screenshot filename that you add to this repository.

---

## 6. Comparison of the Three Runs

The experiments were compared using RMSE, MAE, and R².

| Run   | Max Depth | Learning Rate |         RMSE |          MAE |           R² |
| ----- | --------: | ------------: | -----------: | -----------: | -----------: |
| Run 1 |         3 |           0.1 | `ADD RESULT` | `ADD RESULT` | `ADD RESULT` |
| Run 2 |         5 |          0.05 | `ADD RESULT` | `ADD RESULT` | `ADD RESULT` |
| Run 3 |         7 |          0.01 | `ADD RESULT` | `ADD RESULT` | `ADD RESULT` |

### Model Selection

RMSE is the primary metric used to select the best model.

Therefore, the run with the **lowest RMSE** is selected as the final model.

---

## 7. Selected Best Model

**Best Model: `ADD WINNING RUN`**

Configuration:

```text
Max Depth: ADD VALUE
Learning Rate: ADD VALUE
RMSE: ADD VALUE
MAE: ADD VALUE
R²: ADD VALUE
```

---

## 8. Why Was This Model Selected?

The selected model was chosen because it achieved the **lowest RMSE** among the three experiments.

RMSE was used as the primary model-selection metric because this is the criterion specified for the experiment. The other metrics, MAE and R², were also recorded to provide additional information about model performance.

After comparing the three MLflow runs, the selected configuration provided the best validation performance according to RMSE.

---

## 9. Experiment Workflow

The project follows this workflow:

```text
California Housing Dataset
          |
          v
     80/20 Split
          |
          v
   +------+------+
   |      |      |
   v      v      v
 Run 1  Run 2  Run 3
   |      |      |
   +------+------+
          |
          v
    Calculate Metrics
          |
          v
       MLflow
          |
          v
      Compare Runs
          |
          v
   Lowest RMSE
          |
          v
     Best Model
```

---

## 10. Conclusion

This project demonstrates how MLflow can be used to track and compare machine-learning experiments.

Instead of training several models without keeping track of their configurations and results, MLflow provides a central place to record:

* Hyperparameters
* Evaluation metrics
* Trained model artifacts
* Experiment results

The final model is selected by comparing the three experiments and choosing the configuration with the **lowest RMSE**.
