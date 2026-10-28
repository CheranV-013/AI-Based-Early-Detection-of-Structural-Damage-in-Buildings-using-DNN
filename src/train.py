from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)

def load_dataset(split, shuffle):
    folder = DATASET / split
    if not folder.exists():
        raise SystemExit(f"Missing {folder}. Run python src/prepare_dataset.py first.")
    return tf.keras.utils.image_dataset_from_directory(
        folder, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="int", shuffle=shuffle,
        seed=SEED if shuffle else None,
    )

train_ds = load_dataset("train", True)
val_ds = load_dataset("validation", False)
test_ds = load_dataset("test", False)
class_names = train_ds.class_names

if class_names != ["crack", "no_crack"]:
    raise SystemExit(
        f"Expected class folders ['crack', 'no_crack'], got {class_names}. "
        "Keep folder names consistent."
    )
for split, ds in [("validation", val_ds), ("test", test_ds)]:
    if ds.class_names != class_names:
        raise SystemExit(f"Class names differ in {split} split.")

autotune = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(autotune)
val_ds = val_ds.prefetch(autotune)
test_ds = test_ds.prefetch(autotune)

base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.04)(x)
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.35)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(2, activation="softmax", name="class_probabilities")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

model_path = MODEL_DIR / "structural_damage_dnn.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(model_path), monitor="val_loss", save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=2, factor=0.3, min_lr=1e-6
    ),
]
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
)
with open(OUTPUT_DIR / "training_history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].legend()
axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
plt.close()

best_model = tf.keras.models.load_model(model_path)
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
probs = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(probs, axis=1)
report = classification_report(
    y_true, y_pred, labels=[0, 1], target_names=class_names,
    output_dict=True, zero_division=0
)
metrics = {
    "class_names": class_names,
    "test_accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    "classification_report": report,
}
with open(OUTPUT_DIR / "evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
plt.close()
print(json.dumps({
    "test_accuracy": metrics["test_accuracy"],
    "macro_f1": metrics["macro_f1"],
    "classification_report": report,
}, indent=2))
print(f"Saved model: {model_path}")
