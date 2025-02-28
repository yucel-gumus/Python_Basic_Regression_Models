#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multiple Linear Regression Model

This script demonstrates a multiple linear regression model that predicts healthcare expenses
based on age and BMI (Body Mass Index).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def load_data(file_path):
    """
    Load and prepare data for the multiple linear regression model.
    
    Args:
        file_path (str): Path to the CSV file containing insurance data.
        
    Returns:
        tuple: A tuple containing (X_features, y_target)
            - X_features (numpy.ndarray): Features (age and BMI).
            - y_target (numpy.ndarray): Target variable (expenses).
    """
    try:
        data = pd.read_csv(file_path)
        print("Data columns:", data.columns)
        
        # Extract target variable (expenses)
        expenses = data.expenses.values.reshape(-1, 1)
        
        # Extract features (age and BMI)
        age_bmis = data.iloc[:, [0, 2]].values
        
        return age_bmis, expenses
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def train_model(X, y):
    """
    Train a multiple linear regression model.
    
    Args:
        X (numpy.ndarray): Features (age and BMI).
        y (numpy.ndarray): Target variable (expenses).
        
    Returns:
        sklearn.linear_model.LinearRegression: Trained regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    """
    Evaluate the model performance.
    
    Args:
        model (sklearn.linear_model.LinearRegression): Trained regression model.
        X (numpy.ndarray): Features (age and BMI).
        y (numpy.ndarray): Target variable (expenses).
        
    Returns:
        dict: Dictionary containing R2 score and RMSE.
    """
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return {"r2": r2, "rmse": rmse}


def make_predictions(model, test_data):
    """
    Make predictions using the trained model.
    
    Args:
        model (sklearn.linear_model.LinearRegression): Trained regression model.
        test_data (numpy.ndarray): Test data containing age and BMI values.
        
    Returns:
        numpy.ndarray: Predicted expense values.
    """
    return model.predict(test_data)


def plot_results(X, y, y_pred):
    """
    Plot the actual vs predicted values.
    
    Args:
        X (numpy.ndarray): Features (age and BMI).
        y (numpy.ndarray): Actual target values (expenses).
        y_pred (numpy.ndarray): Predicted target values (expenses).
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, color="red", alpha=0.5, label="Actual Expenses")
    plt.scatter(X[:, 0], y_pred, color="blue", alpha=0.5, label="Predicted Expenses")
    plt.xlabel("Age")
    plt.ylabel("Healthcare Expenses")
    plt.title("Multiple Linear Regression: Healthcare Expenses Prediction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the multiple linear regression example."""
    # Load and prepare data
    file_path = "insurance.csv"
    X, y = load_data(file_path)
    if X is None or y is None:
        return
    
    # Train the model
    model = train_model(X, y)
    
    # Evaluate the model
    metrics = evaluate_model(model, X, y)
    print(f"\nModel Performance:")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Coefficients: Age = {model.coef_[0][0]:.4f}, BMI = {model.coef_[0][1]:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    
    # Make predictions for specific test cases
    test_data = np.array([[30, 20], [30, 21], [20, 22], [20, 23], [20, 24]])
    predictions = make_predictions(model, test_data)
    
    # Display predictions
    print("\nPredictions:")
    for i, (age, bmi) in enumerate(test_data):
        print(f"Age: {age}, BMI: {bmi} → Predicted Expense: ${predictions[i][0]:.2f}")
    
    # Plot the results
    y_pred = model.predict(X)
    plot_results(X, y, y_pred)


if __name__ == "__main__":
    main()
