
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf

# --- KHỞI TẠO ỨNG DỤNG FLASK ---
app = Flask(__name__)

# --- CÁC THAM SỐ ---
MODEL_PATH = '/home/naoh/Documents/AI/MONEY/vietnamese_currency_model.h5'
CLASS_NAMES_PATH = 'vietnamese_currency_class_names.npy'
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

# --- TẢI MODEL VÀ CLASS NAMES ---
# Sử dụng eager execution để tương thích tốt hơn trong môi trường web
tf.config.run_functions_eagerly(True)

try:
    model = load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
    print(f"[*] Model '{MODEL_PATH}' và class names đã được tải thành công.")
    print(f"[*] Các lớp được nhận dạng: {class_names}")
except Exception as e:
    print(f"[!] Lỗi khi tải model hoặc class names: {e}")
    model = None
    class_names = []

# --- HÀM XỬ LÝ ẢNH ---
def preprocess_image(image_bytes):
    try:
        # Đọc ảnh từ bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Chuyển sang ảnh xám
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize ảnh
        resized_img = cv2.resize(gray_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # 3. Chuẩn hóa giá trị pixel
        normalized_img = resized_img.astype('float32') / 255.0
        
        input_img = normalized_img.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        
        return input_img
    except Exception as e:
        print(f"[!] Lỗi trong quá trình xử lý ảnh: {e}")
        return None

# --- CÁC ROUTE CỦA FLASK ---
@app.route('/', methods=['GET'])
def index():
    """Render trang chủ."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Nhận ảnh tải lên, xử lý, dự đoán và trả về kết quả dạng JSON.
    """
    if model is None:
        return jsonify({'error': 'Model is not loaded!'}), 500

    # Kiểm tra xem có file trong request không
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # Nếu người dùng không chọn file, trình duyệt có thể gửi một part rỗng
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            # Đọc bytes của file
            image_bytes = file.read()
            
            # Tiền xử lý ảnh
            processed_image = preprocess_image(image_bytes)
            
            if processed_image is None:
                return jsonify({'error': 'Could not process the image.'}), 400

            # Dự đoán
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            
            print(f"[*] Dự đoán: {predicted_class_name}")

            # Trả về kết quả
            return jsonify({'prediction': str(predicted_class_name)})

        except Exception as e:
            print(f"[!] Lỗi khi dự đoán: {e}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500

    return jsonify({'error': 'Unknown error'}), 500

# --- CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    # Chạy trên tất cả các interface, port 5000
    # Chế độ debug sẽ tự động reload server khi có thay đổi code
    app.run(host='0.0.0.0', port=5000, debug=True)
