# app.py
# ---------------------------------------------------------
# เว็บแอป Streamlit สำหรับทำนายผลด้วยโมเดลที่ฝึกไว้แล้ว (.pkcls)
# หมายเหตุสำคัญ: โค้ดนี้เป็น "เทมเพลตทั่วไป" สำหรับโมเดลที่รับ
# ฟีเจอร์แบบตาราง (tabular features) เช่น อาการ/ค่าตรวจต่าง ๆ
# ไม่ใช่โมเดลที่รับภาพ X-ray โดยตรง (โมเดลรับภาพจะต้องใช้ CNN/PyTorch/TensorFlow
# และมีวิธีโหลด-ทำนายต่างออกไปทั้งหมด) — ดูคำอธิบายท้ายไฟล์
# ---------------------------------------------------------

import streamlit as st
import joblib
import pandas as pd
import glob
import os

# -----------------------------
# 1) ตั้งค่าหน้าเว็บและหัวข้อแอป
# -----------------------------
st.set_page_config(page_title="จำแนกโรค Covid จากข้อมูล", page_icon="🩺")
st.title("โปรแกรมจำแนกโรค Covid จากภาพ X-ray")

st.write(
    "กรอกข้อมูลด้านล่าง แล้วกดปุ่ม **ทำนายผล** "
    "เพื่อให้โมเดลที่เลือกไว้ทำการจำแนกผล"
)

# -----------------------------------------------------
# 2) ส่วนเลือกโมเดล (.pkcls) ให้ผู้ใช้เลือกเองได้
#    - วิธีที่ 1: สแกนหาไฟล์ .pkcls ในโฟลเดอร์ "models"
#      (ให้ดาวน์โหลดไฟล์จาก Google Drive โฟลเดอร์ ModelW10
#       มาวางไว้ในโฟลเดอร์ models/ ข้าง ๆ ไฟล์ app.py นี้ก่อนรันแอป
#       เนื่องจาก Streamlit ไม่สามารถอ่านไฟล์จาก Google Drive
#       โดยตรงได้ ต้องดาวน์โหลดมาเก็บไว้ในเครื่อง/เซิร์ฟเวอร์ก่อน)
#    - วิธีที่ 2: อัปโหลดไฟล์ .pkcls เองผ่านหน้าเว็บ (สำรอง)
# -----------------------------------------------------
st.sidebar.header("⚙️ เลือกโมเดล")

MODEL_DIR = "models"  # โฟลเดอร์เก็บไฟล์โมเดลที่ดาวน์โหลดมาจาก Google Drive
os.makedirs(MODEL_DIR, exist_ok=True)

# ค้นหาไฟล์นามสกุล .pkcls ทั้งหมดในโฟลเดอร์ models
model_files = glob.glob(os.path.join(MODEL_DIR, "*.pkcls"))
model_names = [os.path.basename(f) for f in model_files]

selected_model_path = None

if model_names:
    selected_name = st.sidebar.selectbox("เลือกไฟล์โมเดลจากโฟลเดอร์ models/", model_names)
    selected_model_path = os.path.join(MODEL_DIR, selected_name)
else:
    st.sidebar.info("ไม่พบไฟล์ .pkcls ในโฟลเดอร์ models/ กรุณาอัปโหลดไฟล์โมเดลด้านล่าง")

# ทางเลือกสำรอง: อัปโหลดไฟล์โมเดลเอง (เผื่อไม่มีไฟล์ในโฟลเดอร์ models)
uploaded_model = st.sidebar.file_uploader("หรืออัปโหลดไฟล์โมเดล (.pkcls)", type=["pkcls"])
if uploaded_model is not None:
    # บันทึกไฟล์ที่อัปโหลดลงในโฟลเดอร์ models เพื่อใช้งาน
    selected_model_path = os.path.join(MODEL_DIR, uploaded_model.name)
    with open(selected_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    st.sidebar.success(f"อัปโหลดไฟล์ {uploaded_model.name} สำเร็จ")

# -----------------------------------------------------
# 3) โหลดโมเดลด้วย joblib (ใช้ st.cache_resource กันโหลดซ้ำทุกครั้งที่รีเฟรช)
# -----------------------------------------------------
@st.cache_resource
def load_model(path):
    """โหลดโมเดลจากไฟล์ .pkcls ด้วย joblib"""
    return joblib.load(path)

model = None
if selected_model_path is not None and os.path.exists(selected_model_path):
    try:
        model = load_model(selected_model_path)
        st.sidebar.success(f"โหลดโมเดล '{os.path.basename(selected_model_path)}' สำเร็จ")
    except Exception as e:
        st.sidebar.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
else:
    st.warning("กรุณาเลือกหรืออัปโหลดไฟล์โมเดล (.pkcls) ก่อนใช้งาน")

# -----------------------------------------------------
# 4) ส่วนกรอกค่าตัวแปรต้น (features)
#    *** จุดนี้เป็นตัวอย่างเท่านั้น ***
#    ต้องแก้ไขชื่อฟีเจอร์ ชนิดข้อมูล และตัวเลือกให้ตรงกับ
#    คอลัมน์จริงที่ใช้ตอนฝึกโมเดลของคุณ (ดูคำอธิบายท้ายไฟล์)
# -----------------------------------------------------
st.subheader("กรอกข้อมูลผู้ป่วย / ค่าตัวแปรต้น")

col1, col2 = st.columns(2)

with col1:
    # ตัวอย่างตัวแปรตัวเลข (numeric) -> ใช้ st.number_input
    age = st.number_input("อายุ (ปี)", min_value=0, max_value=120, value=30, step=1)
    temperature = st.number_input("อุณหภูมิร่างกาย (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
    oxygen_level = st.number_input("ระดับออกซิเจนในเลือด (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)

with col2:
    # ตัวอย่างตัวแปรหมวดหมู่ (categorical) -> ใช้ st.selectbox
    gender = st.selectbox("เพศ", ["ชาย", "หญิง"])
    fever = st.selectbox("มีไข้หรือไม่", ["มี", "ไม่มี"])
    cough = st.selectbox("มีอาการไอหรือไม่", ["มี", "ไม่มี"])
    breathing_difficulty = st.selectbox("หายใจลำบากหรือไม่", ["มี", "ไม่มี"])

# -----------------------------------------------------
# 5) ปุ่มทำนายผล
# -----------------------------------------------------
if st.button("ทำนายผล"):
    if model is None:
        st.error("ยังไม่ได้โหลดโมเดล กรุณาเลือกหรืออัปโหลดไฟล์โมเดลก่อน")
    else:
        # -------------------------------------------------
        # 5.1 รวมค่าที่ผู้ใช้กรอกเป็น DataFrame แถวเดียว
        #     ชื่อคอลัมน์ต้อง "ตรงกับชื่อคอลัมน์ตอนฝึกโมเดล" ทุกตัว
        # -------------------------------------------------
        input_dict = {
            "age": [age],
            "temperature": [temperature],
            "oxygen_level": [oxygen_level],
            "gender": [gender],
            "fever": [fever],
            "cough": [cough],
            "breathing_difficulty": [breathing_difficulty],
        }
        input_df = pd.DataFrame(input_dict)

        # -------------------------------------------------
        # 5.2 One-hot encode คอลัมน์ข้อความ (categorical)
        #     ให้เหมือนกับตอนฝึกโมเดล (เช่น ใช้ pd.get_dummies)
        # -------------------------------------------------
        categorical_cols = ["gender", "fever", "cough", "breathing_difficulty"]
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

        # -------------------------------------------------
        # 5.3 จัดคอลัมน์ให้ตรงกับตอนฝึกโมเดล
        #     - ถ้าโมเดล (เช่น scikit-learn) มี attribute feature_names_in_
        #       จะใช้ค่านี้จัดเรียง/เติมคอลัมน์ที่ขาดให้อัตโนมัติ
        #     - ถ้าไม่มี ให้ระบุ "training_columns" เองแบบ hard-code
        #       (คัดลอกรายชื่อคอลัมน์จากตอนฝึกโมเดลมาใส่ตรงนี้)
        # -------------------------------------------------
        if hasattr(model, "feature_names_in_"):
            training_columns = list(model.feature_names_in_)
        else:
            # *** ตัวอย่าง: ต้องแก้ไขให้ตรงกับคอลัมน์จริงตอนฝึกโมเดล ***
            training_columns = [
                "age", "temperature", "oxygen_level",
                "gender_ชาย", "gender_หญิง",
                "fever_มี", "fever_ไม่มี",
                "cough_มี", "cough_ไม่มี",
                "breathing_difficulty_มี", "breathing_difficulty_ไม่มี",
            ]

        # เติมคอลัมน์ที่ขาดด้วยค่า 0 และเรียงลำดับคอลัมน์ให้ตรงกับตอนฝึก
        input_final = input_encoded.reindex(columns=training_columns, fill_value=0)

        # -------------------------------------------------
        # 5.4 ส่งเข้าโมเดลเพื่อทำนายผล
        # -------------------------------------------------
        try:
            prediction = model.predict(input_final)[0]

            # ถ้าโมเดลรองรับ predict_proba จะแสดงความมั่นใจ (%) ด้วย
            proba_text = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_final)[0]
                max_proba = max(proba) * 100
                proba_text = f" (ความมั่นใจประมาณ {max_proba:.2f}%)"

            # -------------------------------------------------
            # 6) แสดงผลการทำนายให้อ่านง่าย
            # -------------------------------------------------
            if str(prediction).lower() in ["1", "covid", "positive", "yes", "มี"]:
                st.error(f"⚠️ ผลการทำนาย: มีความเสี่ยงเป็น Covid{proba_text}")
            else:
                st.success(f"✅ ผลการทำนาย: ไม่พบความเสี่ยงเป็น Covid{proba_text}")

            st.write("ค่าดิบที่โมเดลทำนายได้:", prediction)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")
