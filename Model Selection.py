import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# LOADING AND SPLITTING THE COMBINED CYCLE POWER PLANT DATA

df = pd.read_csv('CCPP_data.csv')

# Display first few rows
# print(df.head())

# Define features and target
X = df[['AT', 'V', 'AP', 'RH']]  # Feature columns
y = df['PE']  # Target column

from sklearn.model_selection import train_test_split

# 80/20 split for train and test. Setting random state for reproducability
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.model_selection import KFold

# DEFINE 5 FOLD CROSS VALIDATION
kf = KFold(n_splits=5, shuffle=True, random_state=42)



# DEFINING THE LINEAR REGRESSION AND RANDOM FOREST MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Dictionary to store performance
model_performance = {}

# EVALUATE MODELS USING CROSS-VALIDATION
for name, model in models.items():
    # 5-fold RMSE
    neg_mse_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(-neg_mse_scores)
    
    # 5-fold R^2
    r2_scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=kf)
    
    avg_rmse = rmse_scores.mean()
    avg_r2 = r2_scores.mean()
    
    # Save scores
    model_performance[name] = {
        'model': model,
        'avg_rmse': avg_rmse,
        'avg_r2': avg_r2
    }

    # Print model performance
    print(f"\nModel: {name}")
    print(f"Average RMSE: {avg_rmse:.2f}")
    print(f"Average R²: {avg_r2:.4f}")

# FINDING BEST MODEL BASED ON LOWEST RMSE
best_model_name = min(model_performance, key=lambda name: model_performance[name]['avg_rmse'])
best_model = model_performance[best_model_name]['model']

print(f"\n Best Model Based on Cross-Validation: {best_model_name}")

# TRAIN THE BEST MODEL ON FULL TRAINING SET
best_model.fit(X_train, y_train)

# EVALUATE THE BEST MODEL WITH THE TEST DATA
y_pred = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

print(f"\n Final Evaluation on Test Set using {best_model_name}:")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test R²: {test_r2:.4f}")

# VISUALISATION - PREDICTED VS ACTUAL ENERGY OUTPUT
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--')
plt.title(f'{best_model_name} — Predicted vs Actual Energy Output', fontsize=14)
plt.xlabel('Actual PE (MW)', fontsize=12)
plt.ylabel('Predicted PE (MW)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()