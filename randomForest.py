#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Forest Regression Model

This script demonstrates a random forest regression model to predict
salary based on position level, showing how ensemble methods can improve
prediction accuracy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def load_data(file_path):
    """
    Load and prepare data for the random forest regression model.
    
    Args:
        file_path (str): Path to the CSV file containing position data.
        
    Returns:
        tuple: A tuple containing (X_features, y_target)
            - X_features (numpy.ndarray): Features (position level).
            - y_target (numpy.ndarray): Target variable (salary).
    """
    try:
        data = pd.read_csv(file_path)
        level = data.iloc[:, 1].values.reshape(-1, 1)
        salary = data.iloc[:, 2].values
        return level, salary
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def train_random_forest(X, y, n_estimators=10, max_depth=None, random_state=None):
    """
    Train a random forest regression model.
    
    Args:
        X (numpy.ndarray): Features (position level).
        y (numpy.ndarray): Target variable (salary).
        n_estimators (int): Number of trees in the forest.
        max_depth (int, optional): Maximum depth of the trees.
        random_state (int, optional): Random state for reproducibility.
        
    Returns:
        sklearn.ensemble.RandomForestRegressor: Trained random forest regression model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    """
    Evaluate the model performance.
    
    Args:
        model (sklearn.ensemble.RandomForestRegressor): Trained random forest regression model.
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
        model (sklearn.ensemble.RandomForestRegressor): Trained random forest regression model.
        value (float): Value to predict salary for.
        
    Returns:
        float: Predicted salary.
    """
    return model.predict(np.array([[value]]))[0]


def plot_results(X, y, model):
    """
    Plot the data points and random forest regression predictions.
    
    Args:
        X (numpy.ndarray): Features (position level).
        y (numpy.ndarray): Target variable (salary).
        model (sklearn.ensemble.RandomForestRegressor): Trained random forest regression model.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot original data points
    plt.scatter(X, y, color="red", alpha=0.6, label="Data Points")
    
    # Create a range of X values for smooth curve
    X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    
    # Plot random forest predictions
    plt.plot(X_range, model.predict(X_range), color="blue", 
             linewidth=2, label="Random Forest Prediction")
    
    # Add feature importance as text annotation
    importances = model.feature_importances_
    plt.annotate(f"Feature Importance: {importances[0]:.4f}", 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.title("Random Forest Regression")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the random forest regression example."""
    # Load and prepare data
    file_path = "positions.csv"
    X, y = load_data(file_path)
    if X is None or y is None:
        return
    
    # Train the model
    n_estimators = 100  # Increased from 10 for better performance
    random_state = 42   # Fixed random state for reproducibility
    model = train_random_forest(X, y, n_estimators=n_estimators, random_state=random_state)
    
    # Evaluate the model
    metrics = evaluate_model(model, X, y)
    print("Model Performance:")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    # Make a prediction for a specific value
    test_value = 8.3
    prediction = make_prediction(model, test_value)
    print(f"\nPrediction for position level {test_value}: ${prediction:.2f}")
    
    # Display feature importance
    importances = model.feature_importances_
    print(f"Feature Importance: {importances[0]:.4f}")
    
    # Plot the results
    plot_results(X, y, model)


if __name__ == "__main__":
    main()
