# Phân Loại Tin Nhắn Spam

Dự án này tập trung vào việc phân loại các tin nhắn văn bản là spam hoặc không spam bằng cách sử dụng mô hình Naive Bayes. Mô hình được huấn luyện trên một tập dữ liệu có nhãn gồm các tin nhắn văn bản.

## Mục Lục
- [Cài Đặt](#cài-đặt)
- [Tập Dữ Liệu](#tập-dữ-liệu)
- [Tiền Xử Lý](#tiền-xử-lý)
- [Huấn Luyện Mô Hình](#huấn-luyện-mô-hình)
- [Đánh Giá](#đánh-giá)
- [Dự Đoán](#dự-đoán)
- [Sử Dụng](#sử-dụng)

## Cài Đặt

1. **Tải tập dữ liệu**

   ```bash
   ! gdown --id 1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R

2. **Cài Đặt Các Thư Viện Cần Thiết:**

   Dự án này yêu cầu Python và một số thư viện sau:
   
   - `nltk`
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `matplotlib`

   Cài đặt chúng bằng pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Tải Dữ Liệu NLTK:**

   Script sử dụng thư viện NLTK để xử lý văn bản, vì vậy bạn cần tải các bộ dữ liệu cần thiết:

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Tập Dữ Liệu

Tập dữ liệu sử dụng trong dự án này là một file CSV tên `2cls_spam_text_cls.csv`, chứa các tin nhắn đã được gắn nhãn với hai cột:

- `Message`: Văn bản của tin nhắn.
- `Category`: Nhãn, hoặc là `spam` hoặc là `ham` (không phải spam).

Bạn nên đặt tập dữ liệu này trong cùng thư mục với script.

## Tiền Xử Lý

Các bước tiền xử lý được áp dụng cho dữ liệu văn bản bao gồm:

- Chuyển đổi văn bản thành chữ thường.
- Loại bỏ dấu câu.
- Tách văn bản thành các từ.
- Loại bỏ các từ dừng (những từ phổ biến như "is", "and", v.v. không đóng góp nhiều vào ý nghĩa).
- Stemming (biến đổi từ thành dạng gốc).

Các bước này được thực hiện trong các hàm sau:

- `lowercase(text)`
- `punctuation_removal(text)`
- `tokenize(text)`
- `remove_stopwords(tokens)`
- `stemming(tokens)`

Mỗi tin nhắn sẽ được đưa qua các hàm tiền xử lý này để tạo ra một danh sách các từ.

## Huấn Luyện Mô Hình

Dữ liệu đã được xử lý sẽ được sử dụng để huấn luyện mô hình Naive Bayes (`GaussianNB`). Dữ liệu được chia thành các tập huấn luyện, tập xác thực và tập kiểm tra:

- **Tập Huấn Luyện:** 70% dữ liệu
- **Tập Xác Thực:** 20% dữ liệu
- **Tập Kiểm Tra:** 10% dữ liệu

Mô hình được huấn luyện sử dụng lớp `GaussianNB` từ thư viện `scikit-learn`.

## Đánh Giá

Mô hình được đánh giá trên các tập xác thực và kiểm tra sử dụng độ chính xác làm thước đo:

- **Độ Chính Xác Tập Xác Thực:** Đo lường hiệu suất của mô hình trên dữ liệu xác thực chưa từng thấy trong quá trình huấn luyện.
- **Độ Chính Xác Tập Kiểm Tra:** Đo lường hiệu suất của mô hình cuối cùng trên dữ liệu hoàn toàn chưa từng thấy.

## Dự Đoán

Bạn có thể dự đoán nhãn (spam/ham) của một tin nhắn mới bằng cách sử dụng hàm `predict`:

```python
def predict(text, model, dictionary):
    processed_text = preprocess_text(text)
    features = create_features(processed_text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]
    return prediction_cls
```

## Sử Dụng

Để chạy script và huấn luyện mô hình, chỉ cần thực thi nó trong môi trường Python của bạn. Sau khi huấn luyện, bạn có thể kiểm tra mô hình với văn bản đầu vào mới bằng cách gọi hàm `predict`.

Ví dụ sử dụng:

```python
test_input = 'I am actually thinking a way of doing something useful'
prediction_cls = predict(test_input, model, dictionary)
print(f'Prediction: {prediction_cls}')
```
