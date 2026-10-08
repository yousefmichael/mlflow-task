from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from xgboost import XGBRegressor
import mlflow
import pandas as pd

print("Loading California Housing dataset...")
data = fetch_california_housing(as_frame=True)

X = data.data
y = data.target

print("\nSplitting data into 80% training and 20% validation...")
X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2,
    random_state=42
)

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")


print("\nSetting up MLflow experiment...")
# 1.Experiment 1
mlflow.set_experiment("California_Housing_XGBoost")

# 2. Define the three exact configurations 
configs = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]

print(f"Successfully defined {len(configs)} configurations to test.")


print("\nStarting MLflow experiments...")

for config in configs:
    # 1. Open a new MLflow Run for this configuration
    with mlflow.start_run(run_name=config["run_name"]):
        print(f"Training {config['run_name']} (Depth: {config['max_depth']}, LR: {config['learning_rate']})...")
        
        # 2. Initialize and Train the Model
        model = XGBRegressor(
            max_depth=config["max_depth"], 
            learning_rate=config["learning_rate"], 
            random_state=42 # Fixed seed for the model so results are reproducible
        )
        model.fit(X_train, y_train)
        
        # 3. Make predictions on the validation set
        predictions = model.predict(X_val)
        
        # 4. Calculate metrics
        # RMSE is simply the square root of the Mean Squared Error
        rmse = mean_squared_error(y_val, predictions) ** 0.5
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        
        # 5. Log parameters to MLflow
        mlflow.log_param("max_depth", config["max_depth"])
        mlflow.log_param("learning_rate", config["learning_rate"])
        
        # 6. Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # 7. Log the trained model as an artifact
        mlflow.xgboost.log_model(model, "model")
        
        # Print a quick summary to the terminal
        print(f"  -> Results: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")

print("\nAll experiments completed and logged to MLflow successfully!")