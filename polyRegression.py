#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Polynomial Regression Model

This script demonstrates both linear and polynomial regression models to predict
salary based on position level, showing how polynomial regression can capture
non-linear relationships in the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error


def load_data(file_path):
    """
    Load and prepare data for the regression models.
    
    Args:
        file_path (str): Path to the CSV file containing position data.
        
    Returns:
        tuple: A tuple containing (X_features, y_target)
            - X_features (numpy.ndarray): Features (position level).
            - y_target (numpy.ndarray): Target variable (salary).
    """
    try:
        data = pd.read_csv(file_path)
        
        # Extract features (position level) and target (salary)
        level = data.iloc[:, 1].values.reshape(-1, 1)
        salary = data.iloc[:, 2].values.reshape(-1, 1)
        
        return level, salary
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def train_linear_model(X, y):
    """
    Train a linear regression model.
    
    Args:
        X (numpy.ndarray): Features (position level).
        y (numpy.ndarray): Target variable (salary).
        
    Returns:
        sklearn.linear_model.LinearRegression: Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_polynomial_model(X, y, degree=4):
    """
    Train a polynomial regression model.
    
    Args:
        X (numpy.ndarray): Features (position level).
        y (numpy.ndarray): Target variable (salary).
        degree (int): Degree of the polynomial features.
        
    Returns:
        tuple: A tuple containing (model, poly_features)
            - model (sklearn.linear_model.LinearRegression): Trained polynomial regression model.
            - poly_features (sklearn.preprocessing.PolynomialFeatures): Polynomial features transformer.
    """
    # Transform features to polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # Train the model on polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)
    
    return model, poly_features


def evaluate_models(linear_model, poly_model, poly_features, X, y):
    """
    Evaluate both linear and polynomial regression models.
    
    Args:
        linear_model (sklearn.linear_model.LinearRegression): Trained linear regression model.
        poly_model (sklearn.linear_model.LinearRegression): Trained polynomial regression model.
        poly_features (sklearn.preprocessing.PolynomialFeatures): Polynomial features transformer.
        X (numpy.ndarray): Original features (position level).
        y (numpy.ndarray): Target variable (salary).
        
    Returns:
        dict: Dictionary containing R2 scores and RMSEs for both models.
    """
    # Linear model predictions and metrics
    y_pred_linear = linear_model.predict(X)
    r2_linear = r2_score(y, y_pred_linear)
    rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
    
    # Polynomial model predictions and metrics
    X_poly = poly_features.transform(X)
    y_pred_poly = poly_model.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
    
    return {
        "linear": {"r2": r2_linear, "rmse": rmse_linear},
        "polynomial": {"r2": r2_poly, "rmse": rmse_poly}
    }


def make_predictions(linear_model, poly_model, poly_features, test_value):
    """
    Make predictions using both linear and polynomial regression models.
    
    Args:
        linear_model (sklearn.linear_model.LinearRegression): Trained linear regression model.
        poly_model (sklearn.linear_model.LinearRegression): Trained polynomial regression model.
        poly_features (sklearn.preprocessing.PolynomialFeatures): Polynomial features transformer.
        test_value (float): Test value to predict salary for.
        
    Returns:
        tuple: A tuple containing (linear_prediction, poly_prediction)
            - linear_prediction (float): Prediction from linear model.
            - poly_prediction (float): Prediction from polynomial model.
    """
    # Reshape test value for prediction
    test_array = np.array([[test_value]])
    
    # Linear model prediction
    linear_pred = linear_model.predict(test_array)[0][0]
    
    # Polynomial model prediction
    poly_pred = poly_model.predict(poly_features.transform(test_array))[0][0]
    
    return linear_pred, poly_pred


def plot_results(X, y, linear_model, poly_model, poly_features):
    """
    Plot the data points, linear regression line, and polynomial regression curve.
    
    Args:
        X (numpy.ndarray): Features (position level).
        y (numpy.ndarray): Target variable (salary).
        linear_model (sklearn.linear_model.LinearRegression): Trained linear regression model.
        poly_model (sklearn.linear_model.LinearRegression): Trained polynomial regression model.
        poly_features (sklearn.preprocessing.PolynomialFeatures): Polynomial features transformer.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot original data points
    plt.scatter(X, y, color="red", alpha=0.6, label="Data Points")
    
    # Create a range of X values for smooth curves
    X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    
    # Plot linear regression line
    plt.plot(X_range, linear_model.predict(X_range), 
             color="blue", linewidth=2, label="Linear Regression")
    
    # Plot polynomial regression curve
    X_poly_range = poly_features.transform(X_range)
    plt.plot(X_range, poly_model.predict(X_poly_range), 
             color="green", linewidth=2, label="Polynomial Regression")
    
    plt.title("Linear vs Polynomial Regression")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the polynomial regression example."""
    # Load and prepare data
    file_path = "positions.csv"
    X, y = load_data(file_path)
    if X is None or y is None:
        return
    
    # Train linear regression model
    linear_model = train_linear_model(X, y)
    
    # Train polynomial regression model
    poly_model, poly_features = train_polynomial_model(X, y, degree=4)
    
    # Evaluate both models
    metrics = evaluate_models(linear_model, poly_model, poly_features, X, y)
    
    # Print model performance metrics
    print("Model Performance:")
    print("\nLinear Regression:")
    print(f"R² Score: {metrics['linear']['r2']:.4f}")
    print(f"RMSE: {metrics['linear']['rmse']:.4f}")
    
    print("\nPolynomial Regression:")
    print(f"R² Score: {metrics['polynomial']['r2']:.4f}")
    print(f"RMSE: {metrics['polynomial']['rmse']:.4f}")
    
    # Make predictions for a test value
    test_value = 8.3
    linear_pred, poly_pred = make_predictions(linear_model, poly_model, poly_features, test_value)
    
    print(f"\nPredictions for position level {test_value}:")
    print(f"Linear Regression: ${linear_pred:.2f}")
    print(f"Polynomial Regression: ${poly_pred:.2f}")
    
    # Plot the results
    plot_results(X, y, linear_model, poly_model, poly_features)


if __name__ == "__main__":
    main()
