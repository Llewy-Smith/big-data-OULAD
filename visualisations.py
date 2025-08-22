### visualisations.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from models import baseline_predict_days_since_last_login

def plot_model_comparison(results_list, save_path="model_comparison.png"):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Course Failure Prediction - Model Comparison', fontsize=16, fontweight='bold')

    # Bar Plot: Precision, Recall, F1
    models = [r['name'] for r in results_list]
    metrics = ['Recall', 'Precision', 'F1']
    metric_values = {
        metric: [r['metrics'].get(metric, 0) for r in results_list]
        for metric in metrics
    }
    x = np.arange(len(models))
    width = 0.25
    
    ax1.bar(x - width, metric_values['Recall'], width, label='Recall')
    ax1.bar(x, metric_values['Precision'], width, label='Precision')
    ax1.bar(x + width, metric_values['F1'], width, label='F1-Score')
    ax1.set_title('Key Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision-Recall Curve
    for r in results_list:
        y_true = r["y_true"]
        y_prob = r["y_prob"]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, label=f'{r["name"]} (AP={pr_auc:.2f})')
    ax2.set_title('Precision-Recall Curves')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Placeholder: Feature importance panel
    ax3.set_title('Feature Importance (TBD)')
    ax3.text(0.5, 0.5, 'Add Feature Importance Here', ha='center', va='center')
    ax3.axis('off')

    # Complexity vs F1
    complexity_map = {
        '7-Day Rule': 1,
        'Logistic Regression': 2,
        'Random Forest': 3,
        'MLP': 4,
        'Deep CNN': 5
    }
    f1_scores = [r['metrics'].get('F1', 0) for r in results_list]
    complexities = [complexity_map.get(r['name'], 0) for r in results_list]
    ax4.scatter(complexities, f1_scores, s=100)
    for i, r in enumerate(results_list):
        ax4.annotate(r['name'], (complexities[i], f1_scores[i]), textcoords="offset points", xytext=(5,5), ha='left')
    ax4.set_title('Complexity vs Performance')
    ax4.set_xlabel('Model Complexity')
    ax4.set_ylabel('F1 Score')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_baseline_sweep(df, save_path="baseline_sweep.png"):
    '''
    Plot the baseline sweep results.
    DF columns: threshold_days, Precision, Recall, F1, PR_AUC
    '''
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Precision, Recall, F1
    axes[0].plot(df["threshold_days"], df["Precision"], label="Precision", marker="o")
    axes[0].plot(df["threshold_days"], df["Recall"], label="Recall", marker="o")
    axes[0].plot(df["threshold_days"], df["F1"], label="F1 Score", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Threshold (Days Since Last Login)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision, Recall, and F1 by Threshold")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Number of students flagged
    if "Flagged" in df.columns:
        axes[1].plot(df["threshold_days"], df["Flagged"], label="Students Flagged", color="darkorange", marker="s")
        axes[1].set_xlabel("Threshold (Days Since Last Login)")
        axes[1].set_ylabel("Number of Students")
        axes[1].set_title("Number of Students Flagged as At-Risk")
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, "No 'Flagged' column found in DataFrame", 
                     ha="center", va="center", fontsize=12)
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def plot_confusion_for_baseline(X, y, threshold=7, feature_name="days_since_last_login"):
    #print(f"Confusion matrix for threshold = {threshold} days (feature: {feature_name})")
    y_pred = baseline_predict_days_since_last_login(X, threshold, feature_name)
    cm = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pass", "Fail"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (threshold: {threshold} days without activity)")
    plt.show()

def print_brief_metrics(model_name, result_dict):
    m = result_dict['metrics']
    print(f"{model_name:20} | Precision: {m['Precision']:.3f} | Recall: {m['Recall']:.3f} | F1: {m['F1']:.3f} | PR-AUC: {m.get('PR_AUC', 0):.3f}")

def print_model_report(model_name, result_dict, show_confusion=True):
    """
    Prints a brief summary and shows a confusion matrix for a given model result.
    Expects result_dict from evaluate_model().
    """
    m = result_dict['metrics']
    y_true = result_dict['y_true']
    y_pred = result_dict['y_pred']

    print(f"\n{model_name} Performance Summary")
    print("-" * 40)
    print(f" Precision : {m['Precision']:.2f}")
    print(f" Recall    : {m['Recall']:.2f}")
    print(f" F1 Score  : {m['F1']:.2f}")
    if m.get("PR_AUC") is not None:
        print(f" PR-AUC    : {m['PR_AUC']:.2f}")

    if show_confusion:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pass", "Fail"])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{model_name} - Confusion Matrix")
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()