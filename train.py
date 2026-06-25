"""
train.py — Face Mask Detection: Model Training Script
======================================================
Trains a CNN on the Kaggle 'omkargurav/face-mask-dataset' dataset.

Expected folder structure:
    data/
    ├── with_mask/
    │   └── *.jpg
    └── without_mask/
        └── *.jpg

Usage:
    python train.py
"""

import os
import sys
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
#    All tunable hyper-parameters and paths in one place so they're easy to
#    change without hunting through the file.
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR       = "data"          # Root folder that holds with_mask / without_mask
IMG_SIZE       = (128, 128)      # Height × Width fed into the network
BATCH_SIZE     = 32
EPOCHS         = 5
VALIDATION_SPLIT = 0.2          # 20 % of images reserved for validation
SEED           = 123            # Reproducible train/val split
SAVE_PATH      = "mask_detector.keras"   # Output model file


# ─────────────────────────────────────────────────────────────────────────────
# 2. VERIFY DATA DIRECTORY
#    Give a clear error message rather than a cryptic TensorFlow crash if the
#    user forgot to download / extract the dataset.
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset(data_dir: str) -> None:
    """Raise a helpful SystemExit if the dataset structure looks wrong."""
    required_classes = {"with_mask", "without_mask"}
    if not os.path.isdir(data_dir):
        sys.exit(
            f"[ERROR] Dataset directory '{data_dir}' not found.\n"
            "Download it from Kaggle (omkargurav/face-mask-dataset) and "
            f"extract so that '{data_dir}/with_mask/' and "
            f"'{data_dir}/without_mask/' exist."
        )
    found = {d for d in os.listdir(data_dir)
             if os.path.isdir(os.path.join(data_dir, d))}
    missing = required_classes - found
    if missing:
        sys.exit(
            f"[ERROR] Missing sub-folders in '{data_dir}': {missing}\n"
            "Expected both 'with_mask' and 'without_mask'."
        )

    total = sum(
        len(files)
        for cls in required_classes
        for _, _, files in os.walk(os.path.join(data_dir, cls))
    )
    print(f"[INFO] Dataset OK — {total} images found in '{data_dir}'.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD DATASETS
#    image_dataset_from_directory handles:
#      • Recursive file discovery
#      • Label assignment based on sub-folder names (alphabetical → 0-indexed)
#        "with_mask"    → 0
#        "without_mask" → 1
#      • Deterministic train / validation split via 'seed'
#      • Automatic batching and basic decoding
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets(data_dir: str):
    """Return (train_ds, val_ds, class_names)."""

    common_kwargs = dict(
        directory=data_dir,
        image_size=IMG_SIZE,       # Resize every image on load
        batch_size=BATCH_SIZE,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training",
        **common_kwargs,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation",
        **common_kwargs,
    )

    class_names = train_ds.class_names
    print(f"[INFO] Classes (label order): {class_names}")
    # Typical output: ['with_mask', 'without_mask']

    return train_ds, val_ds, class_names


# ─────────────────────────────────────────────────────────────────────────────
# 4. PERFORMANCE OPTIMISATION
#    • AUTOTUNE  — let TensorFlow decide prefetch / parallelism counts at
#                  runtime based on available hardware.
#    • cache()   — keep decoded images in RAM after the first epoch so
#                  subsequent epochs read from memory, not disk.
#    • prefetch  — overlap GPU training of batch N with CPU loading of batch
#                  N+1, eliminating idle GPU time.
# ─────────────────────────────────────────────────────────────────────────────

def optimise(ds):
    AUTOTUNE = tf.data.AUTOTUNE
    return ds.cache().prefetch(buffer_size=AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD THE CNN MODEL
#    Architecture rationale:
#      • Rescaling(1./255) — normalise pixel values [0, 255] → [0.0, 1.0]
#        inside the model so the saved .keras file handles preprocessing
#        automatically at inference time (no external normalisation step).
#      • Three Conv2D → MaxPooling blocks — progressively extract and
#        downsample spatial features.  Filter counts double (16 → 32 → 64)
#        so deeper layers capture more abstract patterns.
#      • Dense(128) — compact learned representation before classification.
#      • Dense(2, no activation) — raw logits for two classes.
#        SparseCategoricalCrossentropy(from_logits=True) handles the softmax
#        internally for numerical stability.
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(128, 128, 3)),

            # ── Normalisation ──────────────────────────────────────────────
            tf.keras.layers.Rescaling(1.0 / 255),

            # ── Feature extraction block 1 ─────────────────────────────────
            tf.keras.layers.Conv2D(16, 3, activation="relu",
                                   padding="same"),
            tf.keras.layers.MaxPooling2D(),          # 128→64

            # ── Feature extraction block 2 ─────────────────────────────────
            tf.keras.layers.Conv2D(32, 3, activation="relu",
                                   padding="same"),
            tf.keras.layers.MaxPooling2D(),          # 64→32

            # ── Feature extraction block 3 ─────────────────────────────────
            tf.keras.layers.Conv2D(64, 3, activation="relu",
                                   padding="same"),
            tf.keras.layers.MaxPooling2D(),          # 32→16

            # ── Classifier head ────────────────────────────────────────────
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes),     # Raw logits
        ],
        name="face_mask_cnn",
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(model, train_ds, val_ds) -> tf.keras.callbacks.History:
    """Fit the model and return the history object."""
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATE & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_save(model, val_ds, save_path: str) -> None:
    """Print final metrics and persist the model."""
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"\n[RESULT] Validation loss     : {loss:.4f}")
    print(f"[RESULT] Validation accuracy : {accuracy * 100:.2f} %")

    model.save(save_path)
    print(f"[INFO] Model saved → '{save_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Guard: ensure TF can see a GPU (optional; falls back to CPU silently)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[INFO] GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("[INFO] No GPU detected — training on CPU (slower).")

    check_dataset(DATA_DIR)
    train_ds, val_ds, class_names = load_datasets(DATA_DIR)

    train_ds = optimise(train_ds)
    val_ds   = optimise(val_ds)

    model = build_model(num_classes=len(class_names))
    train(model, train_ds, val_ds)
    evaluate_and_save(model, val_ds, SAVE_PATH)
