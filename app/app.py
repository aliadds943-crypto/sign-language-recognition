# -*- coding: utf-8 -*-
"""
🤟 نظام التعرف على لغة الإشارة — واجهة العرض
المشروع النهائي لمادة تمييز الأنماط

خادم Flask يستقبل صورة يد من المستخدم،
يمررها عبر المودل المدرب (CNN) ويعيد الرقم المتوقع مع نسبة الثقة.
"""

import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_language_model.keras")

# حجم الصور التي دُرّب عليها المودل
IMG_SIZE = 64
NUM_CLASSES = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # حد أقصى 10MB للصورة


# ---------------------------------------------------------------------------
# تحميل المودل مرة واحدة عند تشغيل الخادم
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"لم يتم العثور على ملف المودل!\n"
        f"المتوقع في: {MODEL_PATH}\n"
        f"قم بتدريب المودل من الـ Notebook في Google Colab ثم انسخ ملف "
        f"'sign_language_model.keras' إلى مجلد app."
    )

print("⏳ جارٍ تحميل المودل...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ تم تحميل المودل بنجاح!")


def preprocess_image(image_file):
    """تحضير الصورة بنفس طريقة المعالجة أثناء التدريب:
    تحويل إلى RGB → تغيير الحجم إلى 64×64 → Normalization إلى [0, 1]"""
    img = Image.open(image_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    return arr[np.newaxis, ...]  # إضافة بُعد الـ batch


def predict(image_file):
    """التنبؤ بالرقم من صورة اليد — يعيد الرقم ونسبة الثقة وأفضل 3 توقعات"""
    img_array = preprocess_image(image_file)
    probs = model.predict(img_array, verbose=0)[0]

    class_id = int(np.argmax(probs))
    top3_idx = np.argsort(probs)[::-1][:3]

    return {
        "digit": class_id,
        "confidence": round(float(probs[class_id]) * 100, 2),
        "top3": [
            {"digit": int(i), "confidence": round(float(probs[i]) * 100, 2)}
            for i in top3_idx
        ],
    }


# ---------------------------------------------------------------------------
# المسارات (Routes)
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """الصفحة الرئيسية — واجهة رفع الصور"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """استقبال الصورة وتنفيذ التنبؤ"""
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "الرجاء اختيار صورة أولاً"}), 400

    try:
        result = predict(request.files["image"])
        return jsonify(result)
    except Exception:
        return jsonify({"error": "تعذر معالجة الصورة — تأكد من أنها ملف صورة صالح (PNG/JPG)"}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
