# app.py
# ---------------------------------------------------------
# เว็บแอป Streamlit สำหรับทำนายผลด้วยโมเดล Orange (.pkcls)
# หมายเหตุสำคัญ: ไฟล์ .pkcls เป็นฟอร์แมตของโปรแกรม Orange Data Mining
# (ไม่ใช่ scikit-learn เพียว ๆ) จึงต้องติดตั้งไลบรารี "Orange3"
# และใช้วิธีทำนายผลแบบเฉพาะของ Orange (ผ่าน Domain/Table)
# จุดดีคือ: โมเดล Orange เก็บข้อมูล "Domain" (ชื่อฟีเจอร์ + ชนิด +
# ตัวเลือก) ไว้ในตัวเอง ทำให้เราสร้างฟอร์มกรอกข้อมูลแบบไดนามิก
# ที่ตรงกับฟีเจอร์จริงของโมเดลได้อัตโนมัติ ไม่ต้อง hardcode เอง
# ---------------------------------------------------------

import streamlit as st
import joblib
import glob
import os
from Orange.data import Domain, Table

# -----------------------------
# 1) ตั้งค่าหน้าเว็บและหัวข้อแอป
# -----------------------------
st.set_page_config(page_title="จำแนกโรค Covid", page_icon="🩺")
st.title("โปรแกรมจำแนกโรค Covid จากภาพ X-ray")

st.write(
    "กรอกข้อมูลด้านล่าง แล้วกดปุ่ม **ทำนายผล** "
    "เพื่อให้โมเดลที่เลือกไว้ทำการจำแนกผล"
)

# -----------------------------------------------------
# 2) ส่วนเลือกโมเดล (.pkcls) ให้ผู้ใช้เลือกเองได้
#    - สแกนหาไฟล์ .pkcls ในโฟลเดอร์ "models" (ต้องดาวน์โหลดจาก
#      Google Drive มาวางไว้ใน repo ก่อน ตามที่ทำไปแล้ว)
#    - หรืออัปโหลดไฟล์ .pkcls เองผ่านหน้าเว็บ (สำรอง)
# -----------------------------------------------------
st.sidebar.header("⚙️ เลือกโมเดล")

MODEL_DIR = "models"  # ต้องตรงกับชื่อโฟลเดอร์ใน repo GitHub
os.makedirs(MODEL_DIR, exist_ok=True)

model_files = glob.glob(os.path.join(MODEL_DIR, "*.pkcls"))
model_names = [os.path.basename(f) for f in model_files]

selected_model_path = None

if model_names:
    selected_name = st.sidebar.selectbox(f"เลือกไฟล์โมเดลจากโฟลเดอร์ {MODEL_DIR}/", model_names)
    selected_model_path = os.path.join(MODEL_DIR, selected_name)
else:
    st.sidebar.info(f"ไม่พบไฟล์ .pkcls ในโฟลเดอร์ {MODEL_DIR}/ กรุณาอัปโหลดไฟล์โมเดลด้านล่าง")

uploaded_model = st.sidebar.file_uploader("หรืออัปโหลดไฟล์โมเดล (.pkcls)", type=["pkcls"])
if uploaded_model is not None:
    selected_model_path = os.path.join(MODEL_DIR, uploaded_model.name)
    with open(selected_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    st.sidebar.success(f"อัปโหลดไฟล์ {uploaded_model.name} สำเร็จ")

# -----------------------------------------------------
# 3) โหลดโมเดลด้วย joblib (ใช้ st.cache_resource กันโหลดซ้ำทุกครั้งที่รีเฟรช)
#    หมายเหตุ: การ unpickle โมเดล Orange ต้อง import Orange ไว้ก่อน
#    (import ไว้ด้านบนของไฟล์แล้ว) มิฉะนั้นจะขึ้น "No module named 'Orange'"
# -----------------------------------------------------
@st.cache_resource
def load_model(path):
    """โหลดโมเดล Orange จากไฟล์ .pkcls ด้วย joblib"""
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
# 4) สร้างฟอร์มกรอกข้อมูลแบบไดนามิก ตาม Domain ของโมเดล Orange
#    - model.domain.attributes คือลิสต์ตัวแปรต้น (features) ที่โมเดลใช้ฝึก
#    - ตัวแปรแบบ Discrete (หมวดหมู่) -> ใช้ st.selectbox พร้อมตัวเลือกจริง
#      จาก var.values ที่เก็บไว้ในโมเดล
#    - ตัวแปรแบบ Continuous (ตัวเลข) -> ใช้ st.number_input
# -----------------------------------------------------
input_values = []
attrs = []

if model is not None:
    domain = model.domain
    attrs = list(domain.attributes)

    st.subheader("กรอกข้อมูลผู้ป่วย / ค่าตัวแปรต้น")
    st.caption(f"โมเดลนี้ใช้ทั้งหมด {len(attrs)} ฟีเจอร์ (ดึงชื่อและชนิดข้อมูลมาจากตัวโมเดลโดยตรง)")

    cols = st.columns(2)
    for i, var in enumerate(attrs):
        col = cols[i % 2]
        with col:
            if var.is_discrete:
                # ตัวแปรหมวดหมู่: ใช้ตัวเลือกจริงที่โมเดลรู้จัก (var.values)
                val = st.selectbox(var.name, list(var.values), key=f"feat_{var.name}")
            else:
                # ตัวแปรตัวเลข (continuous)
                val = st.number_input(var.name, value=0.0, key=f"feat_{var.name}")
            input_values.append(val)

# -----------------------------------------------------
# 5) ปุ่มทำนายผล
# -----------------------------------------------------
if st.button("ทำนายผล"):
    if model is None:
        st.error("ยังไม่ได้โหลดโมเดล กรุณาเลือกหรืออัปโหลดไฟล์โมเดลก่อน")
    else:
        try:
            # -------------------------------------------------
            # 5.1 สร้าง Orange Table จากค่าที่ผู้ใช้กรอก
            #     ใช้ Domain เดียวกับตอนฝึกโมเดล (เฉพาะฝั่ง attributes
            #     ไม่รวม class_var เพราะเรายังไม่รู้คำตอบ)
            #     Orange จะแปลงค่า string/number ให้ตรงกับตัวแปรแต่ละ
            #     ตัวให้อัตโนมัติ (เหมือนการ one-hot/encode ภายในตัว)
            # -------------------------------------------------
            predict_domain = Domain(attrs)
            input_table = Table.from_list(predict_domain, [input_values])

            # -------------------------------------------------
            # 5.2 ส่งเข้าโมเดลเพื่อทำนายผล
            #     เรียก model(table) จะได้ค่ารหัส (float) ของคลาสที่ทำนายได้
            #     ต้องแปลงกลับเป็นชื่อจริงด้วย domain.class_var.values
            # -------------------------------------------------
            prediction_code = model(input_table)
            predicted_label = model.domain.class_var.values[int(prediction_code[0])]

            # ถ้าต้องการความน่าจะเป็น (probability) ด้วย ใช้ ret=Model.ValueProbs
            proba_text = ""
            try:
                from Orange.base import Model as OrangeModel
                _, probs = model(input_table, ret=OrangeModel.ValueProbs)
                max_proba = max(probs[0]) * 100
                proba_text = f" (ความมั่นใจประมาณ {max_proba:.2f}%)"
            except Exception:
                pass  # ถ้าโมเดลไม่รองรับ predict_proba ก็ข้ามไป ไม่ error

            # -------------------------------------------------
            # 6) แสดงผลการทำนายให้อ่านง่าย
            # -------------------------------------------------
            if str(predicted_label).lower() in ["1", "covid", "positive", "yes", "มี", "ป่วย"]:
                st.error(f"⚠️ ผลการทำนาย: {predicted_label}{proba_text}")
            else:
                st.success(f"✅ ผลการทำนาย: {predicted_label}{proba_text}")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")
