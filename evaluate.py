
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from data_pipeline import build_label_dataframe, split_data, make_dataset
from config import CLASS_NAMES


def evaluate(model, csv_path, image_dir):
    df = build_label_dataframe(csv_path, image_dir)
    _, _, test_df = split_data(df)
    test_ds = make_dataset(test_df, augment=False, shuffle=False)

    y_pred = model.predict(test_ds)
    y_true = test_df[CLASS_NAMES].values.astype("float32")

    results = {}
    print("\n=== Per-Class AUC-ROC on Test Set ===")
    for i, label in enumerate(CLASS_NAMES):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            results[label] = auc
            print(f"{label:<22} AUC: {auc:.4f}")
        except ValueError:
            results[label] = float("nan")
            print(f"{label:<22} AUC: N/A")

    macro = np.nanmean(list(results.values()))
    print(f"\nMacro Average AUC: {macro:.4f}")
    return results, y_pred, y_true


def plot_roc_curves(y_true, y_pred, results, save_path="results/roc_curves.png"):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    for i, label in enumerate(CLASS_NAMES):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            axes[i].plot(fpr, tpr, color="steelblue", lw=2,
                         label=f"AUC={results[label]:.3f}")
            axes[i].plot([0,1],[0,1],"k--", alpha=0.3)
            axes[i].set_title(label, fontsize=11, fontweight="bold")
            axes[i].legend(loc="lower right", fontsize=9)
            axes[i].grid(True, alpha=0.3)
        except:
            axes[i].axis("off")
    for j in range(len(CLASS_NAMES), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Per-Class ROC Curves — NIH ChestX-ray14 (MobileNetV2)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"ROC curves saved to {save_path}")
