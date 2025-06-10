# ⚡ Predicting Power Output from Environmental Variables: A Machine Learning Approach

This project builds and evaluates regression models to predict the net hourly electrical energy output (PE) of a Combined Cycle Power Plant. The plant operates using gas and steam turbines with heat recovery, and the predictions are based on environmental sensor data.

## 📂 Project Overview

- **Objective:** Predict electrical power output (PE) using ambient environmental variables.
- **Data:** 9,568 hourly readings from a Combined Cycle Power Plant.
- **Source:**  
  Pınar Tüfekci, *Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods*, IJEPES, 2014.

---

## 🔧 Features Used

The dataset includes the following features:

- `Temperature (T)` in °C  
- `Ambient Pressure (AP)` in millibar  
- `Relative Humidity (RH)` in %  
- `Exhaust Vacuum (V)` in cm Hg  
- `PE (target)` — Net hourly electrical energy output in MW

---

## 🧠 Modeling Approach

- **Type of Task:** Supervised regression
- **Algorithms Used:**
  - Linear Regression
  - Random Forest Regressor
- **Validation Strategy:** 5-Fold Cross-Validation
- **Evaluation Metric:** Root Mean Squared Error (RMSE), R² Score

We compared models using average RMSE and R² across folds and selected the best-performing model based on validation results.

---

## 📊 Model Evaluation

- **Best Model:** Random Forest Regressor
- **Test RMSE:** _<your value>_  
- **Test R² Score:** _<your value>_

The final model was retrained on the full training set and evaluated on a hold-out test set. A scatter plot of predicted vs. actual PE values demonstrates the model's accuracy.

---

## 📈 Visualization

The following plot compares predicted and actual energy output:

![Predicted vs Actual]
> Points close to the red dashed line indicate high model accuracy.

---

