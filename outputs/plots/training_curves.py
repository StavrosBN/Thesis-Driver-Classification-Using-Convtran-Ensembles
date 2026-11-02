"""
This generates Figure 5.1 of the thesis: training/validation loss curves
from the OFFICIAL, stabilized (seed=42) training run of ConvTran
(the 'cleaned_1' group) — the very same run from which all the final
results in the work are derived (Table 5.4, Chapter 7).

The best epoch (determined by early stopping—i.e., the point of minimum validation loss)
is epoch 22; this corresponds to the epoch cited in Table 5.4 and to the
'epoch' field within the saved checkpoint file (outputs/models/convtran_cleaned_1.pt),
as verified using the `check_checkpoints.py` script.
"""

import matplotlib.pyplot as plt

epochs = list(range(1, 48))

train_loss = [0.8274, 0.5081, 0.3874, 0.3117, 0.3041, 0.2651, 0.2104, 0.2276, 0.2088, 0.2558,
              0.1955, 0.2046, 0.1468, 0.1919, 0.2103, 0.1571, 0.1688, 0.1149, 0.1004, 0.1752,
              0.1115, 0.0824, 0.1286, 0.1726, 0.1262, 0.1238, 0.1257, 0.1032, 0.0683, 0.0761,
              0.1057, 0.0996, 0.0558, 0.0721, 0.0749, 0.0994, 0.0766, 0.0609, 0.1042, 0.0851,
              0.0515, 0.0691, 0.0566, 0.0601, 0.0506, 0.0561, 0.0381]

val_loss = [1.0147, 0.2188, 0.2916, 0.6601, 0.1021, 0.2367, 0.5769, 0.7307, 1.3646, 0.8419,
            0.4979, 1.4812, 0.2622, 0.5390, 0.3940, 0.9666, 0.0730, 0.1012, 0.0865, 0.1599,
            1.1226, 0.0250, 0.5675, 0.3806, 0.2598, 0.0649, 0.4808, 0.3285, 0.7927, 0.3608,
            1.5697, 0.6169, 0.4275, 0.5434, 0.1318, 0.6492, 0.2743, 0.3038, 0.1872, 0.3637,
            0.5349, 0.8880, 0.8267, 0.0771, 0.4755, 0.2163, 0.0714]

BEST_EPOCH = 22
OUTPUT_PATH = "outputs/plots/figure_5_1_training_curves.png"

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(epochs, train_loss, color="#2f6f4f", linewidth=1.8, label="Training loss")
    ax.plot(epochs, val_loss, color="#c96a2e", linewidth=1.2, alpha=0.85, label="Validation loss")
    ax.axvline(BEST_EPOCH, color="#555555", linestyle="--", linewidth=1.2)
    ax.text(BEST_EPOCH + 0.6, 1.35, f"Καλύτερο epoch ({BEST_EPOCH})",
            fontsize=9, color="#333333")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Καμπύλες εκπαίδευσης ConvTran (cleaned_1, seed=42)")
    ax.legend(loc="upper right", frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.65)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches='tight', facecolor='white')
    print(f"Αποθηκεύτηκε: {OUTPUT_PATH}")