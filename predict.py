import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 加載保存的模型
model = load_model('red_white_dot_classifier.h5')
image_size = (128, 128)

def predict_red_white_dots(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (128, 128))
    image_array = np.expand_dims(image_resized, axis=0) / 255.0

    # 使用模型進行預測
    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction, axis=1)[0]

    # 計算紅點和白點的數量
    red_count, white_count = count_dots(image)

    # 輸出結果
    if class_idx == 0:
        print("No overlap detected.")
    elif class_idx == 1:
        print("Red dots are overlapping.")
    elif class_idx == 2:
        print("White dots are overlapping.")
    elif class_idx == 3:
        print("Red and white dots are overlapping.")

    # 輸出紅點和白點數量
    print(f"Red dots: {red_count}, White dots: {white_count}")

# 點數量的計算函數
def count_dots(image):
    # 計算紅點
    red_lower = np.array([0, 0, 100])  # 紅色範圍
    red_upper = np.array([50, 50, 255])
    red_mask = cv2.inRange(image, red_lower, red_upper)
    red_count = cv2.connectedComponents(red_mask)[0] - 1  # 減去背景

    # 計算白點
    white_lower = np.array([200, 200, 200])  # 白色範圍
    white_upper = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, white_lower, white_upper)
    white_count = cv2.connectedComponents(white_mask)[0] - 1  # 減去背景

    return red_count, white_count

def predict_dots(image_path):
    image = load_img(image_path, target_size=image_size)
    image_array = img_to_array(image) / 255.0  # 正規化到 [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # 增加一個維度

    # 使用模型進行預測
    prediction = model.predict(image_array)
    red_count, white_count = prediction[0]

    print(f"Predicted Red dots: {int(round(red_count))}, White dots: {int(round(white_count))}")

# 測試預測
image_paths = [
    'dataset/images/image_0.png',
    'dataset/images/image_1.png',
    'dataset/images/image_2.png',
    'dataset/images/image_3.png',
    'dataset/images/image_4.png',
    'dataset/images/image_5.png',
    'dataset/images/image_6.png',
    'dataset/images/image_7.png',
    'dataset/images/image_8.png',
    'dataset/images/image_9.png'
]

for image_path in image_paths:
    predict_dots(image_path)