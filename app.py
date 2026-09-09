# app.py
# ---------------------------------------------------------
# เว็บแอป Streamlit สำหรับทำนายผลด้วยโมเดล Orange (.pkcls)
# หมายเหตุสำคัญ: ไฟล์ .pkcls เป็นฟอร์แมตของโปรแกรม Orange Data Mining
# (ไม่ใช่ scikit-learn เพียว ๆ) จึงต้องติดตั้งไลบรารี "Orange3"
# และใช้วิธีทำนายผลแบบเฉพาะของ Orange (ผ่าน Domain/Table)
# จุดดีคือ: โมเดล Orange เก็บข้อมูล "Domain" (ชื่อฟีเจอร์ + ชนิด +
# ตัวเลือก) ไว้ในตัวเอง ทำให้เราสร้างฟอร์มกรอกข้อมูลแบบไดนามิก
# ที่ตรงกับฟีเจอร์จริงของโมเดลได้อัตโนมัติ ไม่ต้อง hardcode เอง
#
# อัปเดตในเวอร์ชันนี้:
#   - รองรับ "อัปโหลดไฟล์ CSV" เพื่อกรอกข้อมูลทีเดียว แทนการกรอกทีละช่อง
#     (เหมาะมากเมื่อโมเดลมีฟีเจอร์จำนวนมาก เช่น 2048 ฟีเจอร์)
#   - มีปุ่ม "ดาวน์โหลด CSV แม่แบบ" ที่สร้างคอลัมน์ให้ตรงกับฟีเจอร์
#     ของโมเดลที่เลือกไว้โดยอัตโนมัติ กรอกแล้วอัปโหลดกลับมาได้เลย
#   - ฟอร์มกรอกมือทีละช่องยังใช้งานได้ตามเดิม แต่ถ้าฟีเจอร์เยอะ
#     (มากกว่า 20 ช่อง) จะถูกซ่อนไว้ใน expander เพื่อไม่ให้หน้ายาวเกินไป
#   - รองรับการทำนายพร้อมกันหลายแถว (batch) เมื่ออัปโหลด CSV ที่มี
#     มากกว่า 1 แถว จะแสดงผลลัพธ์เป็นตาราง พร้อมดาวน์โหลดผลลัพธ์ได้
#   - [ใหม่] รองรับ "อัปโหลดภาพ X-ray" โดยตรง — ระบบจะสกัด Image
#     Embedding (2048 มิติ) จากภาพให้อัตโนมัติผ่าน Orange
#     (orangecontrib.imageanalytics.ImageEmbedder) แล้วส่งเข้าโมเดล
#     ทันที ไม่ต้องแปลงเป็น CSV เอง — ต้องเลือกโมเดล embedder ให้ตรง
#     กับตอนฝึกโมเดลใน Orange ไม่งั้นค่าที่ได้จะผิด (ค่า default คือ
#     Inception v3 เพราะให้ผล 2048 มิติตรงกับที่โมเดลต้องการ)
# ---------------------------------------------------------

import streamlit as st
import joblib
import glob
import os
import io
import tempfile
import pandas as pd
from Orange.data import Domain, Table

# -----------------------------
# 1) ตั้งค่าหน้าเว็บและหัวข้อแอป
# -----------------------------
st.set_page_config(page_title="จำแนกโรค Covid", page_icon="🩺", layout="wide")
st.title("โปรแกรมจำแนกโรค Covid จากภาพ X-ray")

st.write(
    "เลือกวิธีกรอกข้อมูลด้านล่าง แล้วกดปุ่ม **ทำนายผล** "
    "เพื่อให้โมเดลที่เลือกไว้ทำการจำแนกผล"
)

# -----------------------------------------------------
# ตัวเลือกโมเดล embedder ของ Orange ("Image Embedding" widget)
# key ที่ใช้เรียก ImageEmbedder ต้องตรงกับใน orangecontrib.imageanalytics
# *** ต้องเลือกให้ตรงกับตอนฝึกโมเดลจริง ไม่งั้นค่าที่ได้จะผิด ***
# -----------------------------------------------------
EMBEDDER_OPTIONS = {
    "Inception v3 (ค่าเริ่มต้นของ Orange, 2048 มิติ, ต้องต่อเน็ต)": "inception-v3",
    "SqueezeNet (ประมวลผลในเครื่อง ไม่ต้องต่อเน็ต)": "squeezenet",
    "Painters (ต้องต่อเน็ต)": "painters",
    "VGG-16 (ต้องต่อเน็ต)": "vgg16",
    "VGG-19 (ต้องต่อเน็ต)": "vgg19",
    "DeepLoc (ต้องต่อเน็ต)": "deeploc",
    "openface (ต้องต่อเน็ต)": "openface",
}

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
# 4) ฟังก์ชันช่วยเหลือ: ทำนายผลจาก DataFrame หนึ่งหรือหลายแถว
#    คืนค่าเป็น DataFrame ที่มีคอลัมน์ผลการทำนาย + ความมั่นใจ (ถ้ามี)
# -----------------------------------------------------
def predict_dataframe(model, attrs, df: pd.DataFrame) -> pd.DataFrame:
    predict_domain = Domain(attrs)
    # เรียงคอลัมน์ให้ตรงกับลำดับฟีเจอร์ของโมเดล และแปลงเป็น float
    ordered = df[[var.name for var in attrs]].astype(float)
    input_table = Table.from_list(predict_domain, ordered.values.tolist())

    prediction_codes = model(input_table)
    class_values = model.domain.class_var.values
    predicted_labels = [class_values[int(code)] for code in prediction_codes]

    result_df = df.copy()
    result_df["ผลการทำนาย"] = predicted_labels

    # ถ้าโมเดลรองรับ predict_proba ให้แสดงความมั่นใจด้วย
    try:
        from Orange.base import Model as OrangeModel
        _, probs = model(input_table, ret=OrangeModel.ValueProbs)
        result_df["ความมั่นใจ (%)"] = [f"{max(p) * 100:.2f}" for p in probs]
    except Exception:
        pass

    return result_df


def is_positive_label(label: str) -> bool:
    return str(label).lower() in ["1", "covid", "positive", "yes", "มี", "ป่วย"]


# -----------------------------------------------------
# 5) เตรียมรายชื่อฟีเจอร์ (attrs) จาก Domain ของโมเดล
# -----------------------------------------------------
attrs = []
if model is not None:
    domain = model.domain
    attrs = list(domain.attributes)
    st.caption(f"โมเดลนี้ใช้ทั้งหมด {len(attrs)} ฟีเจอร์ (ดึงชื่อและชนิดข้อมูลมาจากตัวโมเดลโดยตรง)")

input_df = None  # DataFrame ที่จะใช้ทำนายผล (1 แถวขึ้นไป)

if model is not None:
    st.subheader("กรอกข้อมูลผู้ป่วย / ค่าตัวแปรต้น")

    input_mode = st.radio(
        "เลือกวิธีกรอกข้อมูล",
        [
            "📄 อัปโหลดไฟล์ CSV (แนะนำเมื่อมีฟีเจอร์จำนวนมาก)",
            "✍️ กรอกข้อมูลทีละช่องด้วยตนเอง",
            "🖼️ อัปโหลดภาพ X-ray (สกัด embedding อัตโนมัติ)",
        ],
        horizontal=True,
    )

    # -------------------------------------------------
    # 5.1 โหมดอัปโหลด CSV
    # -------------------------------------------------
    if input_mode.startswith("📄"):
        # สร้าง CSV แม่แบบให้ดาวน์โหลด (1 แถว ค่าเริ่มต้น 0.0 หรือค่าตัวเลือกแรกของ discrete)
        template_row = {}
        for var in attrs:
            if var.is_discrete:
                template_row[var.name] = var.values[0] if var.values else ""
            else:
                template_row[var.name] = 0.0
        template_df = pd.DataFrame([template_row])
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="⬇️ ดาวน์โหลด CSV แม่แบบ (มีชื่อคอลัมน์ตรงกับฟีเจอร์ของโมเดล)",
            data=csv_buffer.getvalue(),
            file_name="template_input.csv",
            mime="text/csv",
        )

        st.caption(
            "กรอกค่าในไฟล์แม่แบบ (แก้ไขได้หลายแถวเพื่อทำนายพร้อมกันหลายเคส) "
            "แล้วอัปโหลดกลับมาด้านล่าง"
        )

        uploaded_csv = st.file_uploader("อัปโหลดไฟล์ข้อมูล (.csv)", type=["csv"])
        if uploaded_csv is not None:
            try:
                df_raw = pd.read_csv(uploaded_csv)
                missing_cols = [var.name for var in attrs if var.name not in df_raw.columns]
                if missing_cols:
                    st.error(
                        f"ไฟล์ CSV ขาดคอลัมน์ที่จำเป็น {len(missing_cols)} คอลัมน์ เช่น: "
                        f"{', '.join(missing_cols[:10])}"
                        + (" ..." if len(missing_cols) > 10 else "")
                    )
                else:
                    input_df = df_raw
                    st.success(f"อ่านไฟล์สำเร็จ พบข้อมูล {len(input_df)} แถว")
                    st.dataframe(input_df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"อ่านไฟล์ CSV ไม่สำเร็จ: {e}")

    # -------------------------------------------------
    # 5.2 โหมดกรอกมือทีละช่อง
    #     ถ้าฟีเจอร์เยอะ (> 20) จะซ่อนไว้ใน expander เพื่อไม่ให้หน้ายาวเกินไป
    # -------------------------------------------------
    elif input_mode.startswith("✍️"):
        manual_values = {}

        def render_manual_fields():
            cols = st.columns(2)
            for i, var in enumerate(attrs):
                col = cols[i % 2]
                with col:
                    if var.is_discrete:
                        val = st.selectbox(var.name, list(var.values), key=f"feat_{var.name}")
                    else:
                        val = st.number_input(var.name, value=0.0, key=f"feat_{var.name}")
                    manual_values[var.name] = val

        if len(attrs) > 20:
            st.info(f"โมเดลนี้มี {len(attrs)} ฟีเจอร์ — แนะนำให้ใช้โหมดอัปโหลด CSV แทนเพื่อความสะดวก")
            with st.expander(f"คลิกเพื่อกรอกทีละช่อง ({len(attrs)} ช่อง)"):
                render_manual_fields()
        else:
            render_manual_fields()

        if manual_values:
            input_df = pd.DataFrame([manual_values])

    # -------------------------------------------------
    # 5.3 [ใหม่] โหมดอัปโหลดภาพ X-ray โดยตรง
    #     สกัด Image Embedding ผ่าน Orange (orangecontrib.imageanalytics)
    #     แล้วแปลงเป็น DataFrame 1 แถว ที่มีคอลัมน์ตรงกับ attrs ของโมเดล
    #     ใช้ st.session_state เก็บผลลัพธ์ embedding ไว้ เพราะการกดปุ่ม
    #     "ทำนายผล" หลักด้านล่างจะทำให้สคริปต์รันใหม่ทั้งหมด
    #     (ถ้าไม่เก็บไว้ embedding ที่สกัดไปแล้วจะหายไปทันที)
    # -------------------------------------------------
    else:  # input_mode.startswith("🖼️")
        embedder_label = st.selectbox(
            "เลือกโมเดล Image Embedding (ต้องตรงกับตอนฝึกโมเดลใน Orange)",
            options=list(EMBEDDER_OPTIONS.keys()),
            index=0,  # ค่าเริ่มต้น = Inception v3 (2048 มิติ)
        )
        embedder_key = EMBEDDER_OPTIONS[embedder_label]

        st.caption(
            "⚠️ Embedder ส่วนใหญ่ (ยกเว้น SqueezeNet) จะส่งภาพไปประมวลผลที่เซิร์ฟเวอร์ "
            "ของ Orange (biolab) ผ่านอินเทอร์เน็ต เช่นเดียวกับตอนใช้ widget "
            "'Image Embedding' ใน Orange ตอนฝึกโมเดล กรุณาเลือกโมเดลให้ตรงกับตอนฝึก "
            "ไม่งั้นค่าที่ได้จะผิดแม้จำนวนมิติจะเท่ากันก็ตาม"
        )

        uploaded_image = st.file_uploader(
            "อัปโหลดภาพ X-ray (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            st.image(uploaded_image, caption="ภาพที่อัปโหลด", width=300)

            if st.button("🔎 สกัด embedding จากภาพนี้"):
                tmp_path = None
                try:
                    suffix = os.path.splitext(uploaded_image.name)[1] or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_image.getbuffer())
                        tmp_path = tmp.name

                    with st.spinner("กำลังสกัด embedding จากภาพ (อาจใช้เวลาสักครู่)..."):
                        from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

                        with ImageEmbedder(model=embedder_key) as embedder:
                            embeddings = embedder([tmp_path])

                    if not embeddings or embeddings[0] is None:
                        st.error(
                            "ไม่สามารถสกัด embedding จากภาพนี้ได้ "
                            "(เซิร์ฟเวอร์ embedding อาจไม่ตอบสนอง หรือไฟล์ภาพเสียหาย) "
                            "กรุณาลองใหม่อีกครั้ง"
                        )
                    else:
                        vec = embeddings[0]
                        if len(vec) != len(attrs):
                            st.error(
                                f"จำนวนมิติของ embedding ที่ได้ ({len(vec)}) "
                                f"ไม่ตรงกับที่โมเดลต้องการ ({len(attrs)} มิติ) "
                                f"กรุณาเปลี่ยนตัวเลือกโมเดล Image Embedding ด้านบน "
                                f"ให้ตรงกับตอนฝึกโมเดลใน Orange แล้วลองใหม่"
                            )
                        else:
                            row = {var.name: float(v) for var, v in zip(attrs, vec)}
                            # เก็บผลลัพธ์ไว้ใน session_state กันหายตอนกดปุ่ม "ทำนายผล"
                            st.session_state["image_input_df"] = pd.DataFrame([row])
                            st.success(
                                "สกัด embedding สำเร็จ ✅ "
                                "กดปุ่ม 'ทำนายผล' ด้านล่างเพื่อดูผลลัพธ์ได้เลย"
                            )
                except ModuleNotFoundError as e:
                    st.error(
                        f"ขาดไลบรารีที่จำเป็น: {e} — กรุณาติดตั้ง Orange3-ImageAnalytics "
                        f"ตาม requirements.txt ก่อนใช้งานโหมดนี้"
                    )
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดระหว่างสกัด embedding: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

        # ถ้ามี embedding ที่สกัดสำเร็จค้างอยู่ใน session_state ให้ใช้เป็น input_df
        if "image_input_df" in st.session_state:
            input_df = st.session_state["image_input_df"]
            with st.expander("ดูค่า embedding ที่จะส่งเข้าโมเดล (2048 มิติ)"):
                st.dataframe(input_df, use_container_width=True)

# -----------------------------------------------------
# 6) ปุ่มทำนายผล
# -----------------------------------------------------
if st.button("ทำนายผล", type="primary"):
    if model is None:
        st.error("ยังไม่ได้โหลดโมเดล กรุณาเลือกหรืออัปโหลดไฟล์โมเดลก่อน")
    elif input_df is None or input_df.empty:
        st.error(
            "ยังไม่มีข้อมูลสำหรับทำนายผล กรุณากรอกข้อมูล อัปโหลดไฟล์ CSV "
            "หรืออัปโหลดภาพ X-ray แล้วกด 'สกัด embedding' ก่อน"
        )
    else:
        try:
            result_df = predict_dataframe(model, attrs, input_df)

            if len(result_df) == 1:
                # กรณีทำนายทีละเคส แสดงผลแบบเน้น ๆ
                label = result_df["ผลการทำนาย"].iloc[0]
                proba_text = ""
                if "ความมั่นใจ (%)" in result_df.columns:
                    proba_text = f" (ความมั่นใจประมาณ {result_df['ความมั่นใจ (%)'].iloc[0]}%)"

                if is_positive_label(label):
                    st.error(f"⚠️ ผลการทำนาย: {label}{proba_text}")
                else:
                    st.success(f"✅ ผลการทำนาย: {label}{proba_text}")
            else:
                # กรณีทำนายหลายเคสพร้อมกัน (batch) แสดงเป็นตาราง
                st.subheader(f"ผลการทำนาย ({len(result_df)} เคส)")
                st.dataframe(result_df, use_container_width=True)

                positive_count = result_df["ผลการทำนาย"].apply(is_positive_label).sum()
                st.caption(f"พบผลบวก {positive_count} จากทั้งหมด {len(result_df)} เคส")

                out_buffer = io.StringIO()
                result_df.to_csv(out_buffer, index=False)
                st.download_button(
                    label="⬇️ ดาวน์โหลดผลการทำนายทั้งหมด (.csv)",
                    data=out_buffer.getvalue(),
                    file_name="prediction_results.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")
