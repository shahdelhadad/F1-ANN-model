import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import matplotlib.pyplot as plt

def build_model_a(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_model_b(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_model_c(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Loading preprocessed data...")
    if not os.path.exists('data/processed_f1_data.csv'):
        print("Error: Processed data not found. Please run src/data_preprocessing.py first.")
        return

    df = pd.read_csv('data/processed_f1_data.csv')
    X = df.drop('Podium', axis=1)
    y = df['Podium']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_dim = X_train.shape[1]
    
    models_config = {
        'Model_A': build_model_a(input_dim),
        'Model_B': build_model_b(input_dim),
        'Model_C': build_model_c(input_dim)
    }
    
    results = {}
    
    for name, model in models_config.items():
        print(f"\nTraining {name}...")
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
        
        y_prob = model.predict(X_test).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob)
        }
        
        results[name] = {
            'metrics': metrics,
            'loss_curve': history.history['loss']
        }
        
        os.makedirs('models', exist_ok=True)
        model.save(f'models/{name}.keras')
        print(f"{name} Results: {metrics}")
        
    joblib.dump(results, 'models/training_results.pkl')

    print("\n" + "=" * 68)
    print("  MODEL PERFORMANCE COMPARISON")
    print("=" * 68)
    print(f"  {'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("  " + "-" * 64)
    for name, r in results.items():
        m = r['metrics']
        print(
            f"  {name:<12} {m['accuracy']:>9.1%} {m['precision']:>10.1%} "
            f"{m['recall']:>10.1%} {m['f1']:>10.1%} {m['auc']:>10.4f}"
        )
    best = max(results, key=lambda n: results[n]['metrics']['f1'])
    print("=" * 68)
    print(f"  Best by F1: {best}")
    print("=" * 68)
    print("\nTraining complete. Models and results saved.")

if __name__ == "__main__":
    main()
