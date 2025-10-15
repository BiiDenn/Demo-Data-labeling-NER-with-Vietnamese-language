# ỨNG DỤNG HỌC SÂU TRONG XỬ LÝ VĂN BẢN PHÁP LÝ & RÚT TRÍCH THỰC THỂ TIẾNG VIỆT

## Giới thiệu

Phần Demo này minh họa quy trình ứng dụng **Công nghệ học sâu (Deep Learning)** trong xử lý ngôn ngữ tự nhiên (NLP) cho tiếng Việt, đặc biệt tập trung vào **văn bản pháp lý**.  
Ứng dụng được xây dựng bằng **Streamlit** để trực quan hóa toàn bộ quy trình từ:
1. Gán nhãn dữ liệu huấn luyện (Data Labeling)
2. Trích xuất thông tin (Named Entity Recognition - NER) theo hai hướng:
   - **Rule-based (dựa trên quy tắc thủ công)**
   - **Transformer (dựa trên mô hình học sâu PhoBERT)**

## Tính năng chính
### 1. Gán nhãn dữ liệu (Data Labeling Demo)
- Mô phỏng quá trình **user gán nhãn thủ công** cho các mẫu email.
- So sánh với **nhãn thực tế (ground truth)** để tính **độ chính xác (accuracy)**.
- Giúp người học hiểu vai trò của dữ liệu gán nhãn trong bài toán AI giám sát (supervised learning).

### 2. Rút trích thực thể — Hướng 1: **Rule-based (OCR)**
- Sử dụng **Tesseract OCR** để **đọc văn bản tiếng Việt từ hình ảnh pháp lý (JPG/PNG)**.
- Áp dụng **Regex (biểu thức chính quy)** để phát hiện các loại thực thể:
  - **PER (Person)** — tên người
  - **ORG (Organization)** — tổ chức, cơ quan, công ty
  - **LOC (Location)** — địa danh, thành phố
- Hiển thị kết quả trực quan bằng **highlight màu sắc** ngay trên giao diện.

### 3. Rút trích thực thể — Hướng 2: **Transformer (PhoBERT NER + OCR)**
- Dùng **mô hình học sâu tiếng Việt (NlpHUST/ner-vietnamese-electra-base)** để phát hiện thực thể tự động.
- Hệ thống hiểu **ngữ cảnh** mà không cần định nghĩa quy tắc thủ công.
- Tự động **chia nhỏ văn bản do OCR đọc được > 512 token** để đảm bảo tương thích với mô hình Transformer.
- Hiển thị bảng kết quả gồm từ, loại thực thể và độ tin cậy (confidence score).

## Kiến thức áp dụng
| Nội dung | Công nghệ / Khái niệm chính |
|----------|-----------------------------|
| Xử lý ngôn ngữ tự nhiên (NLP) | Tokenization, Named Entity Recognition (NER) |
| Machine Learning | Supervised Learning, Data Labeling |
| Deep Learning | PhoBERT Transformer models |
| Computer Vision | Optical Character Recognition (Tesseract OCR) |
| Web UI | Streamlit, Pandas, Python Regex |
"# Demo-Data-labeling-NER-with-Vietnamese-language" 


https://github.com/UB-Mannheim/tesseract/wiki
