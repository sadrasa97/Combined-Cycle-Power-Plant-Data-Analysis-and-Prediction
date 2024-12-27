
---

# Combined Cycle Power Plant Data Analysis and Prediction

This project demonstrates the end-to-end analysis of the **Combined Cycle Power Plant Dataset** from the UCI Machine Learning Repository. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning modeling to predict power output based on environmental and operational features.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Features of the Notebook](#features-of-the-notebook)
4. [Prerequisites](#prerequisites)
5. [Installation and Usage](#installation-and-usage)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Modeling and Results](#modeling-and-results)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

The goal of this project is to predict the power output of a combined cycle power plant under various environmental and operational conditions. The notebook contains a structured pipeline for:

- Loading and preprocessing data.
- Performing exploratory data analysis to understand the dataset.
- Selecting features using importance metrics.
- Evaluating machine learning models for regression tasks.

## Dataset Information

- **Source**: [UCI Machine Learning Repository - Combined Cycle Power Plant Dataset](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
- **Features**:
  - **Ambient Temperature (AT)**: Temperature in Celsius.
  - **Ambient Pressure (AP)**: Pressure in millibar.
  - **Relative Humidity (RH)**: Percentage.
  - **Exhaust Vacuum (V)**: Vacuum pressure in cm Hg.
  - **Power Output (Target)**: Power output of the plant in MW.
- **Size**: 9,568 instances and 5 columns (4 features + 1 target).
- **Task**: Regression - Predict the power output of the plant (`Target`).

---

## Features of the Notebook

The notebook is divided into several sections, providing a streamlined workflow for analysis:

1. **Data Loading**:
   - Automated dataset retrieval using the `ucimlrepo` library.
   - Combining metadata and raw data into a single pandas DataFrame.

2. **Data Preprocessing**:
   - Removing duplicates and handling missing values.
   - Detecting and addressing outliers using IQR and Z-scores.
   - Encoding categorical variables (if present) and scaling numerical features.

3. **Exploratory Data Analysis (EDA)**:
   - Visualizing feature distributions using histograms, box plots, and KDE plots.
   - Analyzing correlations with heatmaps.
   - Understanding relationships between features and target using scatter plots and pair plots.

4. **Feature Engineering**:
   - Feature importance analysis with Random Forest.
   - Selecting relevant features based on correlation and importance scores.

5. **Modeling**:
   - Comparing multiple regression models:
     - **SGD Regressor**
     - **Lasso Regression**
     - **Ridge Regression**
     - **Random Forest Regressor**
     - **XGBoost Regressor**
   - Training and evaluation with metrics:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - R-squared (R²)

6. **Results Visualization**:
   - Plotting model performance metrics.
   - Comparing true vs. predicted values.

---

## Prerequisites

### Required Libraries

Make sure you have the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost ucimlrepo
```

---

## Installation and Usage

### Clone the Repository

```bash
git clone https://github.com/sadrasa97/Combined-Cycle-Power-Plant-Data-Analysis-and-Prediction.git
cd Combined-Cycle-Power-Plant-Data-Analysis-and-Prediction
```

### Run the Notebook

1. Launch Jupyter Notebook in your terminal:
   ```bash
   jupyter notebook
   ```
2. Open the `CCPP_Analysis_and_Prediction.ipynb` file.
3. Execute the cells in sequence to replicate the analysis.

---

## Exploratory Data Analysis (EDA)

### Examples of Insights

#### Correlation Heatmap:
![Correlation Matrix](path/to/correlation_matrix.png)

#### Feature Distributions:
![Feature Distributions](path/to/feature_distributions.png)

#### Target vs. Feature Relationships:
![Scatter Plots](path/to/scatter_plots.png)

---

## Modeling and Results

### Models Evaluated

| Model                | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | R² Score |
|----------------------|---------------------------|---------------------------|----------|
| SGD Regressor        | 0.0501                   | 0.0041                   | 0.92     |
| Lasso Regression     | 0.0483                   | 0.0039                   | 0.93     |
| Ridge Regression     | 0.0467                   | 0.0038                   | 0.94     |
| Random Forest        | 0.0378                   | 0.0028                   | 0.95     |
| XGBoost              | 0.0356                   | 0.0025                   | 0.96     |

#### Predicted vs. Actual Target:
![Predicted vs Actual](path/to/predicted_vs_actual.png)

---

## Contributing

Contributions are welcome! If you'd like to improve this notebook, follow these steps:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request.

---

