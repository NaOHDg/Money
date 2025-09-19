import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

DATASET_PATH = '/home/naoh/Documents/AI/MONEY/Image' 

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
# Đối với ảnh xám, chúng ta chỉ có 1 kênh màu
CHANNELS = 1 

# Các mệnh giá tiền cần huấn luyện
TARGET_CURRENCIES = ['1000', '2000', '10000', '20000', '100000']

# --- HÀM TẢI DỮ LIỆU ---
def load_data_from_folders(dataset_path, image_size, class_list):
    images = []
    labels = []
    class_map = {name: i for i, name in enumerate(class_list)}

    print("Bắt đầu quá trình tải dữ liệu...")
    print(f"Các lớp sẽ được tải: {class_list}")

    for class_name in class_list:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"Cảnh báo: Thư mục cho lớp '{class_name}' không tồn tại. Bỏ qua.")
            continue
        
        print(f"Đang đọc ảnh từ thư mục: {class_name}")
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            
            # Đọc ảnh dưới dạng ảnh xám (grayscale)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
            
            if img is not None:
                # Resize ảnh về đúng kích thước
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(class_map[class_name])
            else:
                print(f"Cảnh báo: Không thể đọc ảnh {image_path}. Bỏ qua.")

    return np.array(images), np.array(labels), class_list

# --- BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ---
images, labels, class_names = load_data_from_folders(DATASET_PATH, IMAGE_SIZE, TARGET_CURRENCIES)

if len(images) == 0:
    print("LỖI: Không có ảnh nào được tải. Vui lòng kiểm tra lại đường dẫn DATASET_PATH.")
else:
    print(f"\nĐã tải thành công {len(images)} ảnh.")
    print(f"Kích thước của tập ảnh: {images.shape}")
    print(f"Kích thước của tập nhãn: {labels.shape}")

    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    images = images.astype('float32') / 255.0

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Reshape ảnh để phù hợp với đầu vào của ANN (làm phẳng ảnh)
    # Từ (số_ảnh, IMAGE_HEIGHT, IMAGE_WIDTH) thành (số_ảnh, IMAGE_HEIGHT * IMAGE_WIDTH)
    x_train = x_train.reshape((x_train.shape[0], IMAGE_WIDTH * IMAGE_HEIGHT))
    x_test = x_test.reshape((x_test.shape[0], IMAGE_WIDTH * IMAGE_HEIGHT))

    # One-hot encoding nhãn
    num_classes = len(class_names)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(f"Kích thước x_train sau khi flatten: {x_train.shape}")
    print(f"Kích thước y_train: {y_train.shape}")
    print(f"Kích thước x_test sau khi flatten: {x_test.shape}")
    print(f"Kích thước y_test: {y_test.shape}")

    # --- BƯỚC 2: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH ANN ---
    model = Sequential([
        Dense(512, activation='relu', input_shape=(IMAGE_WIDTH * IMAGE_HEIGHT,)),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.summary()

    # Biên dịch mô hình
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Huấn luyện mô hình
    print("\nBắt đầu huấn luyện mô hình...")
    # Tăng số lượng epoch để mô hình học tốt hơn, có thể điều chỉnh
    history = model.fit(x_train, y_train, epochs=70, batch_size=32, validation_data=(x_test, y_test)) # Tăng epoch
    print("Huấn luyện hoàn tất.")

    # --- BƯỚC 3: ĐÁNH GIÁ VÀ LƯU TRỮ ---
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nĐộ chính xác trên tập kiểm tra: {test_acc:.4f}")

    # Lưu mô hình
    model.save('vietnamese_currency_model.h5') # Đổi tên file model
    print("Đã lưu mô hình vào file 'vietnamese_currency_model.h5'")

    # Lưu danh sách tên các lớp (nhãn)
    np.save('vietnamese_currency_class_names.npy', class_names)
    print("Đã lưu danh sách tên các lớp vào file 'vietnamese_currency_class_names.npy'")

    # --- HIỂN THỊ ĐỒ THỊ KẾT QUẢ HUẤN LUYỆN ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()