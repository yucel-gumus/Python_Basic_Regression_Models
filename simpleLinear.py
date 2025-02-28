#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Linear Regression Model

This script demonstrates a simple linear regression model that predicts weight based on height.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def load_data(file_path):
    """
    Load and prepare data for the regression model.
    
    Args:
        file_path (str): Path to the CSV file containing height and weight data.
        
    Returns:
        pandas.DataFrame: DataFrame containing Height and Weight columns.
    """
    try:
        data = pd.read_csv(file_path)
        return data[['Height', 'Weight']]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def train_model(X, y):
    """
    Train a linear regression model.
    
    Args:
        X (pandas.DataFrame): Features (Height).
        y (pandas.Series): Target variable (Weight).
        
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
        X (pandas.DataFrame): Features (Height).
        y (pandas.Series): Target variable (Weight).
        
    Returns:
        dict: Dictionary containing R2 score and RMSE.
    """
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return {"r2": r2, "rmse": rmse}


def make_predictions(model, heights):
    """
    Make predictions using the trained model.
    
    Args:
        model (sklearn.linear_model.LinearRegression): Trained regression model.
        heights (numpy.ndarray): Array of height values to predict weights for.
        
    Returns:
        numpy.ndarray: Predicted weight values.
    """
    return model.predict(heights.reshape(-1, 1))


def plot_regression(X, y, model):
    """
    Plot the data points and regression line.
    
    Args:
        X (pandas.DataFrame): Features (Height).
        y (pandas.Series): Target variable (Weight).
        model (sklearn.linear_model.LinearRegression): Trained regression model.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Data Points')
    
    x_range = np.linspace(min(X.values), max(X.values), 100).reshape(-1, 1)
    plt.plot(x_range, model.predict(x_range), color="red", linewidth=2, label='Regression Line')
    
    plt.xlabel("Height (inches)")
    plt.ylabel("Weight (pounds)")
    plt.title("Simple Linear Regression: Weight vs Height")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the simple linear regression example."""
    # Load and prepare data
    file_path = "hw_25000.csv"
    data = load_data(file_path)
    if data is None:
        return
    
    # Train the model
    X = data[['Height']]
    y = data['Weight']
    model = train_model(X, y)
    
    # Evaluate the model
    metrics = evaluate_model(model, X, y)
    print(f"Model Performance:")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    # Make predictions for specific heights
    test_heights = np.array([60, 62, 64, 66, 68, 70])
    predictions = make_predictions(model, test_heights)
    
    # Display predictions
    print("\nPredictions:")
    for height, weight in zip(test_heights, predictions):
        print(f"Height: {height} inches, Predicted Weight: {weight:.2f} pounds")
    
    # Plot the regression line and data points
    plot_regression(X, y, model)


if __name__ == "__main__":
    main()
