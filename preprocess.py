import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from tqdm import tqdm # Để hiển thị thanh tiến trình


SOURCE_DATASET_PATH = '/home/naoh/Downloads/Money'


OUTPUT_DATASET_PATH = '/home/naoh/Documents/AI/MONEY/Image'

IMAGES_PER_CLASS_TARGET = 180 # Tăng số lượng ảnh một chút để ANN có nhiều dữ liệu hơn

# Các mệnh giá tiền cần xử lý
TARGET_CURRENCIES = ['1000', '2000', '10000', '20000', '100000']

# Kích thước ảnh chuẩn hóa (có thể thay đổi nếu cần)
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# --- KHỞI TẠO CÁC PHÉP BIẾN ĐỔI ẢNH (AUGMENTATION) ---
# Chuỗi các phép biến đổi ngẫu nhiên
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # Lật ngang 50% ảnh
    iaa.Flipud(0.5), # Lật dọc 50% ảnh
    iaa.Affine( # Các phép biến đổi affine
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # Zoom vào/ra
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # Dịch chuyển ảnh
        rotate=(-25, 25), # Xoay ảnh trong khoảng -25 đến 25 độ
        shear=(-8, 8) # Cắt xiên ảnh
    ),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))), # Thỉnh thoảng làm mờ nhẹ
    iaa.ContrastNormalization((0.75, 1.5)), # Thay đổi độ tương phản
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=False), # Thêm nhiễu ngẫu nhiên, không theo kênh
    iaa.Multiply((0.8, 1.2), per_channel=False), # Thay đổi độ sáng
], random_order=True) # Áp dụng các phép biến đổi theo thứ tự ngẫu nhiên

print(f"Bắt đầu tiền xử lý và làm đa dạng dữ liệu từ '{SOURCE_DATASET_PATH}'...")
print(f"Ảnh đầu ra (grayscale) sẽ được lưu vào '{OUTPUT_DATASET_PATH}'")

# --- HÀM TIỀN XỬ LÝ VÀ LÀM ĐA DẠNG ẢNH ---
def preprocess_and_augment_images(source_path, output_path, target_currencies, image_size, target_count):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    total_augmented_images = 0

    for currency_name in target_currencies:
        source_currency_path = os.path.join(source_path, currency_name)
        output_currency_path = os.path.join(output_path, currency_name)

        if not os.path.isdir(source_currency_path):
            print(f"Cảnh báo: Thư mục nguồn cho '{currency_name}' không tồn tại. Bỏ qua.")
            continue

        if not os.path.exists(output_currency_path):
            os.makedirs(output_currency_path)

        original_images = []
        # Đọc tất cả ảnh gốc
        for image_name in os.listdir(source_currency_path):
            image_path = os.path.join(source_currency_path, image_name)
            img = cv2.imread(image_path) # Đọc ảnh màu
            if img is not None:
                # Chuyển đổi sang RGB nếu cần (imgaug mong đợi RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, image_size)
                original_images.append(img)
            else:
                print(f"Cảnh báo: Không thể đọc ảnh {image_path}. Bỏ qua.")

        if not original_images:
            print(f"Không có ảnh gốc nào được tìm thấy trong '{source_currency_path}'. Bỏ qua.")
            continue

        print(f"\nĐang xử lý và làm đa dạng cho mệnh giá: {currency_name} (Số ảnh gốc: {len(original_images)})")

        augmented_images = []
        # Thêm tất cả ảnh gốc vào danh sách ảnh đã làm đa dạng
        augmented_images.extend(original_images)

        # Tạo thêm ảnh cho đến khi đạt số lượng mong muốn
        num_to_augment = target_count - len(original_images)
        if num_to_augment > 0:
            print(f"Đang tạo thêm {num_to_augment} ảnh mới...")
            # Sử dụng vòng lặp với tqdm để hiển thị tiến trình
            for _ in tqdm(range(num_to_augment), desc=f"Augmenting {currency_name}"):
                # Chọn ngẫu nhiên một ảnh gốc để biến đổi
                idx = np.random.randint(0, len(original_images))
                img_to_augment = original_images[idx]
                # Áp dụng một lần biến đổi
                augmented_img = seq(image=img_to_augment)
                augmented_images.append(augmented_img)
        else:
            print(f"Số lượng ảnh gốc đã đủ ({len(original_images)} >= {target_count}). Không cần tạo thêm.")
            # Cắt bớt nếu số ảnh gốc nhiều hơn target_count
            augmented_images = augmented_images[:target_count]

        # Chuyển tất cả ảnh (gốc và đã augment) sang ảnh xám và lưu
        print(f"Đang chuyển {len(augmented_images)} ảnh sang ảnh xám và lưu vào '{output_currency_path}'...")
        for i, img_rgb in enumerate(augmented_images):
            # Chuyển từ RGB (đầu ra của imgaug) sang Grayscale
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            output_image_path = os.path.join(output_currency_path, f"{currency_name}_{i:04d}.jpg")
            cv2.imwrite(output_image_path, img_gray)
        
        total_augmented_images += len(augmented_images)
        print(f"Hoàn tất mệnh giá {currency_name}. Tổng số ảnh (grayscale): {len(augmented_images)}")

    print(f"\nTổng cộng đã xử lý và làm đa dạng {total_augmented_images} ảnh.")
    print(f"Dữ liệu sẵn sàng tại: {output_path}")

# --- THỰC THI CHƯƠNG TRÌNH ---
if __name__ == '__main__':
    preprocess_and_augment_images(
        SOURCE_DATASET_PATH, 
        OUTPUT_DATASET_PATH, 
        TARGET_CURRENCIES, 
        IMAGE_SIZE, 
        IMAGES_PER_CLASS_TARGET
    )