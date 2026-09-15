# -*- coding: utf-8 -*-
"""
🤟 تدريب مودل التعرف على لغة الإشارة — محلياً على الجهاز
نفس الخطوات والشبكة المستخدمة في notebook/sign_language_training.ipynb
(النسخة المحلية بديلة — للتسليم شغّل الـ Notebook على Google Colab)

الاستخدام:  python train_local.py
الناتج:     app/sign_language_model.keras + رسوم التقييم في مجلد training_results/
"""

import os

import matplotlib

matplotlib.use("Agg")  # بدون واجهة رسومية — حفظ الرسوم كملفات فقط
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_repo", "Dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "training_results")
MODEL_PATH = os.path.join(BASE_DIR, "app", "sign_language_model.keras")

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ---------------------------------------------------------------------------
# 1) تحميل الـ Dataset — صور من مجلدات الفئات (0-9)
# ---------------------------------------------------------------------------
print("=" * 55)
print("1) تحميل الـ Dataset...")
X_list, labels_list = [], []
for digit in range(10):
    folder = os.path.join(DATA_DIR, str(digit))
    for fname in os.listdir(folder):
        img = Image.open(os.path.join(folder, fname)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        X_list.append(np.array(img))
        labels_list.append(digit)

X = np.array(X_list)
Y = keras.utils.to_categorical(labels_list, 10)
labels = np.array(labels_list)

print(f"   عدد الصور: {X.shape[0]} — الأبعاد: {X.shape[1:]}")
print(f"   الصور في كل فئة: {dict(enumerate(np.bincount(labels)))}")

# ---------------------------------------------------------------------------
# 2) المعالجة المسبقة والتقسيم: 70% تدريب / 15% تحقق / 15% اختبار
# ---------------------------------------------------------------------------
print("=" * 55)
print("2) المعالجة المسبقة والتقسيم...")
X = X.astype("float32") / 255.0

X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y, test_size=0.30, random_state=42, stratify=labels)

labels_temp = np.argmax(y_temp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=labels_temp)

print(f"   تدريب: {len(X_train)} | تحقق: {len(X_val)} | اختبار: {len(X_test)}")

# Data Augmentation لبيانات التدريب فقط
AUTOTUNE = tf.data.AUTOTUNE


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(1024)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ---------------------------------------------------------------------------
# 3) بناء الشبكة العصبية CNN من الصفر (بدون Transfer Learning)
# ---------------------------------------------------------------------------
print("=" * 55)
print("3) بناء الشبكة العصبية (CNN من الصفر)...")
model = keras.Sequential(
    [
        layers.Input(shape=(64, 64, 3)),
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        # Block 2
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        # Block 3
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        # Block 4
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        # طبقة التصنيف
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ],
    name="SignLanguageCNN",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------------------------------------------------------------------------
# 4) التدريب
# ---------------------------------------------------------------------------
print("=" * 55)
print("4) التدريب...")
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# ---------------------------------------------------------------------------
# 5) التقييم
# ---------------------------------------------------------------------------
print("=" * 55)
print("5) التقييم...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
print(f"   دقة النموذج على بيانات الاختبار: {test_accuracy * 100:.2f}%")
print(f"   الخسارة على بيانات الاختبار: {test_loss:.4f}")

# منحنيات التعلم
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"], label="Training Loss", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"), dpi=100, bbox_inches="tight")
plt.close()

# Confusion Matrix
y_pred_probs = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=100, bbox_inches="tight")
plt.close()

report = classification_report(y_true, y_pred, digits=4)
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print(report)

# ---------------------------------------------------------------------------
# 6) حفظ المودل
# ---------------------------------------------------------------------------
print("=" * 55)
model.save(MODEL_PATH)
print(f"6) تم حفظ المودل: {MODEL_PATH}")
print(f"   حجم الملف: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.1f} MB")
print(f"   رسوم التقييم محفوظة في: {RESULTS_DIR}")

if test_accuracy >= 0.70:
    print("✅ النتيجة أعلى من الحد المطلوب (70%) — المشروع ناجح!")
else:
    print("⚠️ النتيجة أقل من 70% — أعد التدريب")
