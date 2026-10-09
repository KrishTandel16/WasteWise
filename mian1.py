import os
import json
import cv2
import imghdr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# WasteWise - 10-Class Waste Classification
# TensorFlow + RTX 3050 GPU + CNN
# ============================================================

# -----------------------------
# 1. GPU SETUP
# -----------------------------
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nGPU detected:")
    for gpu in gpus:
        print(" ", gpu)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
else:
    print("\nWARNING: No GPU detected. Training will use CPU.")

print("TensorFlow version:", tf.__version__)


# -----------------------------
# 2. PATHS
# -----------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        "\nDataset folder 'data' was not found.\n"
        "Expected: data/battery, data/biological, etc."
    )


# -----------------------------
# 3. DATASET CHECK
# -----------------------------
print("\nChecking dataset...")

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

class_dirs = sorted(
    [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]
)

if not class_dirs:
    raise ValueError(
        "\nNo class folders found inside 'data'."
    )

print(f"\nFound {len(class_dirs)} class folders:")

total_images = 0

for class_name in class_dirs:
    class_path = os.path.join(DATA_DIR, class_name)

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(valid_extensions)
    ]

    total_images += len(images)
    print(f"  {class_name}: {len(images)} images")

    if len(images) == 0:
        print(f"WARNING: {class_name} contains no supported images.")

print(f"\nTotal images found: {total_images}")

if total_images == 0:
    raise ValueError("No images were found in the dataset.")


# -----------------------------
# 4. LOAD DATASET
# -----------------------------
print("\nLoading dataset...")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

class_names = data.class_names
num_classes = len(class_names)

print("\nDetected classes:")
for i, class_name in enumerate(class_names):
    print(f"  {i}: {class_name}")

print("\nNumber of classes:", num_classes)

if num_classes != 10:
    print(
        f"\nWARNING: Expected 10 classes, but found {num_classes}."
    )


# -----------------------------
# 5. SHOW SAMPLE IMAGES
# -----------------------------
for images, labels in data.take(1):
    plt.figure(figsize=(12, 8))

    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "sample_images.png"),
        dpi=150
    )
    plt.show()


# -----------------------------
# 6. TRAIN / VALIDATION / TEST
# -----------------------------
dataset_size = len(data)

train_size = int(dataset_size * 0.70)
val_size = int(dataset_size * 0.20)
test_size = dataset_size - train_size - val_size

if train_size < 1 or val_size < 1 or test_size < 1:
    raise ValueError(
        "Dataset is too small to create train/validation/test sets."
    )

train = data.take(train_size)

remaining = data.skip(train_size)

val = remaining.take(val_size)

test = remaining.skip(val_size)

print("\nDataset batches:")
print("  Training:", len(train))
print("  Validation:", len(val))
print("  Testing:", len(test))


# -----------------------------
# 7. DATA AUGMENTATION
# -----------------------------
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)

AUTOTUNE = tf.data.AUTOTUNE


def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


train = train.map(
    lambda x, y: (
        data_augmentation(
            tf.cast(x, tf.float32) / 255.0,
            training=True
        ),
        y
    ),
    num_parallel_calls=AUTOTUNE
)

val = val.map(
    preprocess,
    num_parallel_calls=AUTOTUNE
)

test = test.map(
    preprocess,
    num_parallel_calls=AUTOTUNE
)

train = train.prefetch(AUTOTUNE)
val = val.prefetch(AUTOTUNE)
test = test.prefetch(AUTOTUNE)


# -----------------------------
# 8. BUILD 10-CLASS CNN
# -----------------------------
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(224, 224, 3)
        ),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            256,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            256,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.5),

        # Automatically creates 10 outputs for your 10 classes.
        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ],
    name="WasteWise_CNN"
)


# -----------------------------
# 9. COMPILE
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()


# -----------------------------
# 10. CALLBACKS
# -----------------------------
checkpoint_path = os.path.join(
    MODEL_DIR,
    "wastewise_best.keras"
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.TensorBoard(
        log_dir="logs"
    )
]


# -----------------------------
# 11. TRAIN
# -----------------------------
print("\n========================================")
print("        STARTING WASTEWISE TRAINING")
print("========================================")
print("Classes:", class_names)
print("Number of classes:", num_classes)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("GPU:", tf.config.list_physical_devices("GPU"))
print("========================================\n")

history = model.fit(
    train,
    validation_data=val,
    epochs=20,
    callbacks=callbacks
)


# -----------------------------
# 12. SAVE MODEL
# -----------------------------
final_model_path = os.path.join(
    MODEL_DIR,
    "wastewise_final.keras"
)

model.save(final_model_path)

with open(
    os.path.join(MODEL_DIR, "class_names.json"),
    "w"
) as f:
    json.dump(class_names, f, indent=4)

print("\nModel saved:")
print(final_model_path)


# -----------------------------
# 13. LOSS GRAPH
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("WasteWise Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(RESULTS_DIR, "loss.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# 14. ACCURACY GRAPH
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("WasteWise Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(RESULTS_DIR, "accuracy.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# 15. TEST EVALUATION
# -----------------------------
print("\nEvaluating on test dataset...")

test_loss, test_accuracy = model.evaluate(
    test,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# -----------------------------
# 16. PREDICTIONS
# -----------------------------
y_true = []
y_pred = []

for images, labels in test:
    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


# -----------------------------
# 17. CLASSIFICATION REPORT
# -----------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\n========================================")
print("        CLASSIFICATION REPORT")
print("========================================")
print(report)

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:
    f.write(report)


# -----------------------------
# 18. CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.xticks(
    range(num_classes),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(num_classes),
    class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("WasteWise Confusion Matrix")

plt.colorbar()

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# 19. FINAL RESULT
# -----------------------------
print("\n========================================")
print("          WASTEWISE COMPLETE")
print("========================================")
print(f"Classes:       {num_classes}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Model:         {final_model_path}")
print(f"Results:       {RESULTS_DIR}/")
print("========================================")
