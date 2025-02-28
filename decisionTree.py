#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decision Tree Regression Model

This script demonstrates a decision tree regression model to predict
salary based on position level, showing how decision trees can capture
non-linear relationships in the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error


def load_data(file_path):
    """
    Load and prepare data for the decision tree regression model.
    
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


def train_model(X, y, max_depth=None, random_state=None):
    """
    Train a decision tree regression model.
    
    Args:
        X (numpy.ndarray): Features (position level).
        y (numpy.ndarray): Target variable (salary).
        max_depth (int, optional): Maximum depth of the decision tree.
        random_state (int, optional): Random state for reproducibility.
        
    Returns:
        sklearn.tree.DecisionTreeRegressor: Trained decision tree regression model.
    """
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    """
    Evaluate the model performance.
    
    Args:
        model (sklearn.tree.DecisionTreeRegressor): Trained decision tree regression model.
        X (numpy.ndarray): Features (position level).
        y (numpy.ndarray): Target variable (salary).
        
    Returns:
        dict: Dictionary containing R2 score and RMSE.
    """
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return {"r2": r2, "rmse": rmse}


def make_prediction(model, value):
    """
    Make a prediction using the trained model.
    
    Args:
        model (sklearn.tree.DecisionTreeRegressor): Trained decision tree regression model.
        value (float): Value to predict salary for.
        
    Returns:
        float: Predicted salary.
    """
    return model.predict(np.array([[value]]))[0]


def plot_results(X, y, model):
    """
    Plot the data points and decision tree regression predictions.
    
    Args:
        X (numpy.ndarray): Features (position level).
        y (numpy.ndarray): Target variable (salary).
        model (sklearn.tree.DecisionTreeRegressor): Trained decision tree regression model.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot original data points
    plt.scatter(X, y, color="red", alpha=0.6, label="Data Points")
    
    # Create a range of X values for smooth curve
    X_range = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
    
    # Plot decision tree predictions
    plt.plot(X_range, model.predict(X_range), color="orange", 
             linewidth=2, label="Decision Tree Prediction")
    
    plt.title("Decision Tree Regression")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the decision tree regression example."""
    # Load and prepare data
    file_path = "positions.csv"
    X, y = load_data(file_path)
    if X is None or y is None:
        return
    
    # Train the model
    model = train_model(X, y)
    
    # Evaluate the model
    metrics = evaluate_model(model, X, y)
    print("Model Performance:")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    # Make a prediction for a specific value
    test_value = 8.9
    prediction = make_prediction(model, test_value)
    print(f"\nPrediction for position level {test_value}: ${prediction:.2f}")
    
    # Plot the results
    plot_results(X, y, model)


if __name__ == "__main__":
    main()
