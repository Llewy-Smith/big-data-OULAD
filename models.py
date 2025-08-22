### models.py
"""
Train and evaluate ML models to predict course failure.
Includes:
- Baseline (7-day rule)
- Logistic Regression
- Random Forest
- MLP (shallow NN)
- CNN (TensorFlow)
- Evaluation, comparison, and summary
"""
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, precision_recall_curve, precision_recall_fscore_support, average_precision_score, auc
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from pipelines import prepare_data_for_modeling, scale_features

# =============================================================================
# CONFIG
# =============================================================================

class Config:
    RANDOM_STATE = 42
    CV_FOLDS = 5
    SCORING = 'f1'
    N_JOBS = -1
    MAX_CNN_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 8
    LR_PATIENCE = 4
    VERBOSE = False

np.random.seed(Config.RANDOM_STATE)
tf.random.set_seed(Config.RANDOM_STATE)

# =============================================================================
# EVALUATION UTIL
# =============================================================================

def evaluate_model_performance(y_true, y_pred, y_prob=None, model_name="Model", verbose=True):
    results = {}
    if y_prob is not None:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        results['PR_AUC'] = auc(recall_curve, precision_curve)
    results['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    results['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    results['F1'] = f1_score(y_true, y_pred, zero_division=0)
    if verbose:
        print(f"\n{model_name} Results:")
        print(f" Recall:    {results['Recall']:.2f}")
        print(f" Precision: {results['Precision']:.2f}")
        print(f" F1-Score:  {results['F1']:.2f}")
        if 'PR_AUC' in results:
            print(f" PR-AUC:    {results['PR_AUC']:.2f}")
    return results
 
# =============================================================================
# BASELINE MODEL (No Login Rules)
# =============================================================================
 
def baseline_predict_days_since_last_login(X, threshold_days, feature_name="days_since_last_login"):
    ''' 
    Predict FAIL (1) if inactive >= threshold_days, else PASS (0)
    '''    
    if feature_name not in X.columns:
        raise ValueError(f"Feature '{feature_name}' not found in X.")
    return (X[feature_name].values >= threshold_days).astype(int)
 
def run_baseline_sweep(X, y, thresholds, feature_name="days_since_last_login", verbose=False):
    rows = []
    for t in thresholds:
        y_pred = baseline_predict_days_since_last_login(X, t, feature_name)
        support = y_pred.sum()  
        if verbose:
            print(f"Threshold {t:2d}: predicted {support} at-risk students")

        y_prob = y_pred.astype(float)

        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
        pr_auc = average_precision_score(y, y_prob)

        rows.append({
            "threshold_days": t,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "PR_AUC": pr_auc,
            "Flagged": support
        })

    return pd.DataFrame(rows).sort_values("threshold_days").reset_index(drop=True)

def run_baseline_model(X, y, threshold_days=10, feature_name="days_since_last_login"):
    start = time.time()
    model_name=f"{threshold_days}-Day Rule"
    y_pred = baseline_predict_days_since_last_login(X, threshold_days, feature_name)
    y_prob = y_pred.astype(float)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    pr_auc = average_precision_score(y, y_prob)
    print(f"✅ Baseline model '{model_name}' run in {time.time() - start:.1f}s")
    return {
        "name": model_name,
        "metrics": {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "PR_AUC": pr_auc
        },
        "y_true": y,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

# =============================================================================
# TRADITIONAL MODELS
# =============================================================================

def gridsearch_logistic_model(X_train, y_train):
    start = time.time()
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000]
    }
    clf = GridSearchCV(LogisticRegression(random_state=Config.RANDOM_STATE), param_grid,
                       cv=Config.CV_FOLDS, scoring=Config.SCORING, n_jobs=Config.N_JOBS, verbose=0)
    clf.fit(X_train, y_train)
    print(f"✅ Logistic Regression trained in {time.time() - start:.1f}s")
    return clf.best_estimator_

def gridsearch_random_forest_model(X_train, y_train):
    start = time.time()
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    clf = GridSearchCV(RandomForestClassifier(random_state=Config.RANDOM_STATE), param_grid,
                       cv=Config.CV_FOLDS, scoring=Config.SCORING, n_jobs=Config.N_JOBS, verbose=0)
    clf.fit(X_train, y_train)
    print(f"✅ Random Forest trained in {time.time() - start:.1f}s")
    return clf.best_estimator_

def gridsearch_mlp_model(X_train, y_train):
    start = time.time()
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'alpha': [0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500]
    }
    clf = GridSearchCV(MLPClassifier(random_state=Config.RANDOM_STATE, early_stopping=True), param_grid,
                       cv=Config.CV_FOLDS, scoring=Config.SCORING, n_jobs=Config.N_JOBS, verbose=0)
    clf.fit(X_train, y_train)
    print(f"✅ MLP trained in {time.time() - start:.1f}s")
    return clf.best_estimator_

# =============================================================================
# CNN CLASS
# =============================================================================

class CNNModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.best_params = None
        self.best_model = None

    def build_model(self, hidden_units, dropout_rate, learning_rate):
        model = Sequential([
            Dense(hidden_units[0], activation='relu', input_shape=(self.input_dim,)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(hidden_units[1], activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy')
        return model

    def gridsearch(self, X_train, y_train):
        param_grid = {
            'hidden_units': [(64, 32), (128, 64)],
            'dropout_rate': [0.2, 0.4],
            'learning_rate': [0.1, 0.01]
        }
        best_f1 = 0
        skf = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        for hu in param_grid['hidden_units']:
            for dr in param_grid['dropout_rate']:
                for lr in param_grid['learning_rate']:
                    f1_scores = []
                    print(f"CNN configuration: hidden_units={hu}, dropout_rate={dr}, learning_rate={lr}\n")
                    for train_idx, val_idx in skf.split(X_train, y_train):
                        model = self.build_model(hu, dr, lr)
                        early_stop = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True)
                        reduce_lr = ReduceLROnPlateau(patience=Config.LR_PATIENCE, factor=0.5, verbose=0)
                        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                                  validation_data=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                                  epochs=Config.MAX_CNN_EPOCHS, batch_size=32,
                                  callbacks=[early_stop, reduce_lr], verbose=0)
                        preds = (model.predict(X_train.iloc[val_idx]) > 0.5).astype(int)
                        f1_scores.append(f1_score(y_train.iloc[val_idx], preds))
                    avg_f1 = np.mean(f1_scores)
                    print(f"   Mean F1 = {avg_f1:.2f}"+"\n")
                    if avg_f1 > best_f1:
                        self.best_params = {'hidden_units': hu, 'dropout_rate': dr, 'learning_rate': lr}
                        self.best_model = self.build_model(hu, dr, lr)
                        best_f1 = avg_f1
        print(f"🎯 Best CNN config: {self.best_params}, F1: {best_f1:.2f}")
        early_stop = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=Config.LR_PATIENCE, factor=0.5, verbose=0)
        self.best_model.fit(X_train, y_train, validation_split=0.2,
                            epochs=Config.MAX_CNN_EPOCHS, batch_size=32,
                            callbacks=[early_stop, reduce_lr], verbose=0)
        return self.best_model

# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        smin, smax = scores.min(), scores.max() # normalise scores to [0, 1] range
        y_prob = (scores - smin) / (smax - smin + 1e-12)
    else:
        preds = model.predict(X_test)
        y_prob = preds.astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    pr_auc = average_precision_score(y_test, y_prob)
    return {
        "name": model_name,
        "metrics": {"Precision": precision, "Recall": recall, "F1": f1, "PR_AUC": pr_auc},
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# =============================================================================
# RESULTS SUMMARY TABLE
# =============================================================================

def create_results_summary_table(results_list):
    rows = []
    for r in results_list:
        m = r['metrics']
        rows.append({
            'Model': r['name'],
            'Recall': f"{m.get('Recall', 0):.2f}",
            'Precision': f"{m.get('Precision', 0):.2f}",
            'F1-Score': f"{m.get('F1', 0):.2f}",
            'PR-AUC': f"{m.get('PR_AUC', 0):.2f}" if m.get('PR_AUC') else 'N/A'
        })
    df = pd.DataFrame(rows)
    print("\nFINAL RESULTS SUMMARY\n" + "="*60)
    print(df.to_string(index=False))
    return df