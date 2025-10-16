import streamlit as st
import pandas as pd
import re
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy 

# Cấu hình Tesseract (nếu cài ở vị trí khác, hãy chỉnh lại đường dẫn này)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="AI Legal Text & NER Demo", page_icon="⚖️", layout="wide")

st.title("⚖️ ỨNG DỤNG HỌC SÂU TRONG XỬ LÝ VĂN BẢN PHÁP LÝ & RÚT TRÍCH THỰC THỂ TIẾNG VIỆT")

# =========================
# Tạo Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏷️ Gán nhãn dữ liệu",
    "🔍 NER Rule-based (OCR)",
    "📈 NER Classical (Statistical)",
    "🧠 NER Deep Learning (BiLSTM-CRF)",
    "🤖 NER Transformer (OCR + PhoBERT/Electra)"
])


# ======================
# TAB 1 — Data Labeling
# ======================
with tab1:
    st.header("🏷️ GÁN NHÃN DỮ LIỆU (Data Labeling Demo)")
    st.markdown("Minh họa quá trình **gán nhãn dữ liệu thủ công** cho các mẫu email — bước nền tảng cho mô hình AI học có giám sát.")

    emails = [
        "Tài khoản của bạn đã bị khóa, vui lòng đăng nhập tại http://abc.com để xác nhận.",
        "Khuyến mãi 50% cho đơn hàng hôm nay! Ngoài ra, bấm vào http://abc.com để nhận thưởng $1000.",
        "Xin chào, đây là hóa đơn tháng 10 của bạn.",
        "Vui lòng xác minh thông tin ngân hàng của bạn tại đường dẫn dưới đây."
    ]
    true_labels = ["🚨 Spam/Giả mạo", "🚨 Spam/Giả mạo", "✅ Bình thường", "🚨 Spam/Giả mạo"]

    if "labels" not in st.session_state:
        st.session_state.labels = [None] * len(emails)

    for i, email in enumerate(emails):
        st.write(f"**Email {i+1}:** {email}")
        label = st.radio(
            f"Chọn nhãn cho email {i+1}",
            ["Chưa gán", "🚨 Spam/Giả mạo", "✅ Bình thường"],
            key=f"email_{i}"
        )
        if label != "Chưa gán":
            st.session_state.labels[i] = label

    if st.button("📊 Hiển thị kết quả gán nhãn"):
        user_labels = st.session_state.labels
        comparison = ["✅ Đúng" if user_labels[i] == true_labels[i] else "❌ Sai"
                      for i in range(len(emails))]

        df = pd.DataFrame({
            "Email": emails,
            "Nhãn người gán": user_labels,
            "Nhãn đúng": true_labels,
            "Đánh giá": comparison
        })

        def highlight_row(row):
            color = "#d4edda" if row["Đánh giá"] == "✅ Đúng" else "#f8d7da"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)

        correct = sum(1 for c in comparison if c == "✅ Đúng")
        acc = correct / len(comparison) * 100
        st.success(f"🎯 Độ chính xác gán nhãn: **{acc:.1f}%**")

# ============================
# TAB 2 — Rule-based with OCR
# ============================
with tab2:
    st.header("🔍 RÚT TRÍCH THỰC THỂ — HƯỚNG 1: Rule-based")
    st.markdown("""
    Bạn có thể **tải hình ảnh văn bản pháp lý (JPG/PNG)** hoặc **nhập trực tiếp nội dung văn bản** để hệ thống trích xuất tên người, tổ chức, địa điểm.
    """)

    mode = st.radio("Chọn phương thức nhập dữ liệu:", ["📄 Tải ảnh", "✍️ Nhập văn bản"], key="mode_rule")

    extracted_text = ""

    if mode == "📄 Tải ảnh":
        uploaded_img = st.file_uploader("📎 Chọn ảnh văn bản pháp lý (JPG/PNG)", type=["jpg", "jpeg", "png"], key="img_rule")
        if uploaded_img is not None:
            img = Image.open(uploaded_img)
            st.image(img, caption="📜 Ảnh văn bản được tải lên", use_column_width=True)
            with st.spinner("🔠 Đang đọc nội dung văn bản từ ảnh (OCR)..."):
                extracted_text = pytesseract.image_to_string(img, lang="vie")

    else:
        extracted_text = st.text_area("✍️ Nhập nội dung văn bản tại đây:", height=250)

    if st.button("🔎 Rút trích thực thể (Rule-based)"):
        if not extracted_text.strip():
            st.warning("⚠️ Vui lòng tải ảnh hoặc nhập nội dung văn bản trước.")
        else:
            # --- Regex nhận dạng thực thể ---
            pattern_person = r"\b([A-ZĐ][a-zàáảãạăằắẳẵặâầấẩẫậđêềếểễệôồốổỗộơờớởỡợưừứửữự]+(?:\s[A-ZĐ][a-zàáảãạăằắẳẵặâầấẩẫậđêềếểễệôồốổỗộơờớởỡợưừứửữự]+){1,3})\b"
            pattern_org = r"\b(Học\s?viện\s[A-ZĐa-z\s]+|Phòng\s[A-ZĐa-z\-]+\s?[A-ZĐa-z\-]*|Trường\s[A-ZĐa-z\s]+|Công\s?[Tt]y\s[A-ZĐa-z\s]+|Bộ\s[A-ZĐa-z\s]+|Sở\s[A-ZĐa-z\s]+|Ủy\sban\s[A-ZĐa-z\s]+)\b"
            pattern_loc = r"\b(TP\.?\s?Hồ\s?Chí\s?Minh|Hà\s?Nội|Đà\s?Nẵng|Huế|Quận\s?\d+)\b"

            persons = re.findall(pattern_person, extracted_text)
            orgs = re.findall(pattern_org, extracted_text)
            locs = re.findall(pattern_loc, extracted_text)

            # --- Highlight kết quả ---
            highlighted = extracted_text
            for p in persons:
                highlighted = highlighted.replace(p, f"<span style='color:green; font-weight:bold'>{p}</span>")
            for o in orgs:
                highlighted = highlighted.replace(o, f"<span style='color:orange; font-weight:bold'>{o}</span>")
            for l in locs:
                highlighted = highlighted.replace(l, f"<span style='color:purple; font-weight:bold'>{l}</span>")

            st.markdown("### 🧩 Kết quả trích xuất (Rule-based):", unsafe_allow_html=True)
            st.markdown(highlighted, unsafe_allow_html=True)
            st.write("**👤 Người (PER):**", persons)
            st.write("**🏢 Tổ chức (ORG):**", orgs)
            st.write("**📍 Địa điểm (LOC):**", locs)


# =========================
# TAB 3 — Classical NER (English spaCy)
# =========================
with tab3:
    st.header("📈 RÚT TRÍCH THỰC THỂ — HƯỚNG 2A: Classical (spaCy - English)")
    st.markdown("""
    Sử dụng model **spaCy `en_core_web_sm`** (tiếng Anh) để minh họa phương pháp thống kê (statistical NER).  
    > Mô hình này dựa trên đặc trưng ngữ pháp (POS, dependency, shape, prefix/suffix) và dùng CRF-like decoder để gán nhãn chuỗi.
    """)

    # Ô nhập văn bản
    text_stat = st.text_area(
        "✍️ Nhập văn bản cần trích xuất:",
        "Barack Obama was born in Hawaii and worked at the White House.",
        height=150
    )

    # Nút chạy
    if st.button("🔍 Rút trích thực thể (spaCy English)"):
        try:
            nlp = spacy.load("en_core_web_sm")  # ✅ Model tiếng Anh có sẵn
        except OSError:
            st.error("""
            ❗ Model `en_core_web_sm` chưa được cài.
            Cài nhanh trong terminal:
            ```bash
            python -m spacy download en_core_web_sm
            ```
            """)
            nlp = None

        if nlp is not None:
            # Phân tích văn bản
            doc = nlp(text_stat)
            ents = [(ent.text, ent.label_) for ent in doc.ents]

            if not ents:
                st.warning("⚠️ Không phát hiện thực thể nào trong văn bản này.")
            else:
                # Highlight
                highlighted = text_stat
                for text, label in ents:
                    color = (
                        "green" if label in ["PERSON"] else
                        "orange" if label in ["ORG"] else
                        "purple" if label in ["GPE", "LOC"] else
                        "blue"
                    )
                    highlighted = highlighted.replace(
                        text,
                        f"<span style='color:{color}; font-weight:bold'>{text}</span>"
                    )

                st.markdown("### 🧩 Kết quả trích xuất (spaCy English):", unsafe_allow_html=True)
                st.markdown(highlighted, unsafe_allow_html=True)

                # Bảng kết quả
                df = pd.DataFrame(ents, columns=["Thực thể", "Loại"])
                st.dataframe(df, use_container_width=True)

            with st.expander("Giải thích thêm"):
                st.markdown("""
                - `PERSON`: Tên người  
                - `ORG`: Tổ chức / công ty  
                - `GPE` hoặc `LOC`: Địa điểm, quốc gia, thành phố  
                - `FAC`: Cơ sở vật chất
                - `DATE`, `MONEY`, `TIME`: Ngày, tiền, thời gian  
                
                Đây là ví dụ minh họa Classical NER dựa trên đặc trưng thống kê (trước thời Transformer).
                """)

# ===========================================================
# TAB 4 — Deep Learning NER (BiLSTM-CRF pretrained)
# ===========================================================
with tab4:
    st.header("🧠 RÚT TRÍCH THỰC THỂ — HƯỚNG 2B: Deep Learning (BiLSTM-CRF)")
    st.markdown("""
    Mô hình **BiLSTM-CRF** là phương pháp học sâu truyền thống cho bài toán NER.  
    Nó gồm 3 tầng chính:
    - **Embedding Layer:** học biểu diễn từ (ví dụ Word2Vec, FastText)
    - **BiLSTM:** học ngữ cảnh hai chiều (trước–sau)
    - **CRF Layer:** gán nhãn chuỗi tối ưu toàn cục
    
    Ở đây, ta sử dụng model đã huấn luyện sẵn **`NlpHUST/ner-vietnamese-bilstm-crf`** trên tập dữ liệu VLSP tiếng Việt.
    """)

    # Lựa chọn phương thức nhập
    mode_bilstm = st.radio(
        "Chọn phương thức nhập dữ liệu:",
        ["📄 Tải ảnh (OCR)", "✍️ Nhập văn bản"],
        key="mode_bilstm"
    )

    text_bilstm = ""
    if mode_bilstm == "📄 Tải ảnh (OCR)":
        uploaded_img_bilstm = st.file_uploader(
            "📎 Chọn ảnh văn bản (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            key="img_bilstm"
        )
        if uploaded_img_bilstm is not None:
            img_bilstm = Image.open(uploaded_img_bilstm)
            st.image(img_bilstm, caption="📜 Ảnh văn bản được tải lên", use_column_width=True)
            with st.spinner("🔠 Đang đọc văn bản (OCR)..."):
                text_bilstm = pytesseract.image_to_string(img_bilstm, lang="vie")
    else:
        text_bilstm = st.text_area("✍️ Nhập nội dung văn bản tại đây:", height=250, key="manual_bilstm")

    # Tải pipeline BiLSTM-CRF 1 lần duy nhất
    if "bilstm_pipeline" not in st.session_state:
        try:
            with st.spinner("🔄 Đang tải mô hình BiLSTM-CRF (pretrained) ..."):
                bilstm_model_id = "NlpHUST/ner-vietnamese-electra-base"
                tokenizer_bilstm = AutoTokenizer.from_pretrained(bilstm_model_id, trust_remote_code=True)
                model_bilstm = AutoModelForTokenClassification.from_pretrained(bilstm_model_id, trust_remote_code=True)
                st.session_state.bilstm_pipeline = pipeline(
                    "ner",
                    model=model_bilstm,
                    tokenizer=tokenizer_bilstm,
                    aggregation_strategy="simple"
                )
            st.success("Đã tải thành công mô hình BiLSTM-CRF tiếng Việt!")
        except Exception as e:
            st.session_state.bilstm_pipeline = None
            st.error(f"❗ Không thể tải model BiLSTM-CRF từ HuggingFace.\nChi tiết lỗi: {e}")

    # Nút chạy mô hình
    if st.button("🚀 Rút trích thực thể bằng BiLSTM-CRF"):
        if not text_bilstm.strip():
            st.warning("⚠️ Vui lòng tải ảnh hoặc nhập văn bản trước.")
        else:
            if st.session_state.bilstm_pipeline is None:
                st.warning("⚠️ Model BiLSTM-CRF chưa được tải, vui lòng kiểm tra lại mạng hoặc thử lại sau.")
            else:
                ner_bilstm = st.session_state.bilstm_pipeline
                text_clean = text_bilstm.replace("\n", " ").strip()

                # Chia nhỏ văn bản dài
                chunks = []
                while len(text_clean) > 0:
                    chunk = text_clean[:450]
                    end_idx = chunk.rfind(" ")
                    if end_idx == -1:
                        end_idx = len(chunk)
                    chunks.append(text_clean[:end_idx])
                    text_clean = text_clean[end_idx:].strip()

                all_results = []
                for chunk in chunks:
                    results = ner_bilstm(chunk)
                    for r in results:
                        r["chunk"] = chunk
                    all_results.extend(results)

                # Tô màu highlight thực thể
                highlighted_text = text_bilstm
                for r in all_results:
                    color = {"PER": "green", "ORG": "orange", "LOC": "purple"}.get(
                        r.get("entity_group", ""), "blue"
                    )
                    highlighted_text = highlighted_text.replace(
                        r["word"],
                        f"<span style='color:{color}; font-weight:bold'>{r['word']}</span>"
                    )

                # Hiển thị kết quả
                st.markdown("### 🧩 Kết quả trích xuất (BiLSTM-CRF):", unsafe_allow_html=True)
                st.markdown(highlighted_text, unsafe_allow_html=True)

                if all_results:
                    df = pd.DataFrame(all_results)
                    cols = [c for c in ["word", "entity_group", "score"] if c in df.columns]
                    st.dataframe(df[cols], use_container_width=True)


# =========================================
# TAB 5 — Transformer-based NER with OCR
# =========================================
with tab5:
    st.header("🤖 RÚT TRÍCH THỰC THỂ — HƯỚNG 2C: Transformer PhoBERT")
    st.markdown("""
    Bạn có thể **tải ảnh văn bản pháp lý (OCR)** hoặc **nhập trực tiếp nội dung văn bản** để mô hình học sâu PhoBERT tự động trích xuất thực thể.
    """)

    mode2 = st.radio("Chọn phương thức nhập dữ liệu:", ["📄 Tải ảnh", "✍️ Nhập văn bản"], key="mode_trans")

    text_input = ""

    if mode2 == "📄 Tải ảnh":
        uploaded_img2 = st.file_uploader("📎 Chọn ảnh văn bản (JPG/PNG)", type=["jpg", "jpeg", "png"], key="img_trans")
        if uploaded_img2 is not None:
            img2 = Image.open(uploaded_img2)
            st.image(img2, caption="📜 Ảnh văn bản được tải lên", use_column_width=True)
            with st.spinner("🔠 Đang đọc văn bản (OCR)..."):
                text_input = pytesseract.image_to_string(img2, lang="vie")
    else:
        text_input = st.text_area("✍️ Nhập nội dung văn bản tại đây:", height=250, key="manual_text")

    if "ner_pipeline" not in st.session_state:
        with st.spinner("🔄 Đang tải mô hình Transformer NER (PhoBERT)..."):
            tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base", trust_remote_code=True)
            model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base", trust_remote_code=True)
            st.session_state.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    if st.button("🚀 Rút trích thực thể bằng Transformer PhoBERT"):
        if not text_input.strip():
            st.warning("⚠️ Vui lòng tải ảnh hoặc nhập văn bản trước.")
        else:
            ner = st.session_state.ner_pipeline

            # --- Chia nhỏ đoạn dài ---
            chunks = []
            text_clean = text_input.replace("\n", " ").strip()
            while len(text_clean) > 0:
                chunk = text_clean[:450]
                end_idx = chunk.rfind(" ")
                if end_idx == -1: end_idx = len(chunk)
                chunks.append(text_clean[:end_idx])
                text_clean = text_clean[end_idx:].strip()

            all_results = []
            for chunk in chunks:
                results = ner(chunk)
                for r in results:
                    r["chunk"] = chunk
                all_results.extend(results)

            df = pd.DataFrame(all_results)

            highlighted_text = text_input
            for r in all_results:
                color = {"PER": "green", "ORG": "orange", "LOC": "purple"}.get(r["entity_group"], "blue")
                highlighted_text = highlighted_text.replace(r["word"], f"<span style='color:{color}; font-weight:bold'>{r['word']}</span>")

            st.markdown("### 🧩 Kết quả trích xuất (Transformer PhoBERT):", unsafe_allow_html=True)
            st.markdown(highlighted_text, unsafe_allow_html=True)

            if not df.empty:
                st.dataframe(df[["word", "entity_group", "score"]], use_container_width=True)

