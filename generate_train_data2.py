import cv2
import numpy as np
import random
import os

# 設定參數
output_dir = 'dataset/images'
num_images = 1000  # 訓練圖片數量
image_size = (128, 128)  # 圖片大小
point_radius = 10  # 點的半徑

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)

# 生成圖片的函數
def generate_image(red_count, white_count):
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # 隨機生成紅點
    for _ in range(red_count):
        center = (random.randint(point_radius, image_size[0] - point_radius),
                  random.randint(point_radius, image_size[1] - point_radius))
        cv2.circle(image, center, point_radius, (0, 0, 255), -1)

    # 隨機生成白點
    for _ in range(white_count):
        center = (random.randint(point_radius, image_size[0] - point_radius),
                  random.randint(point_radius, image_size[1] - point_radius))
        cv2.circle(image, center, point_radius, (255, 255, 255), -1)

    return image

# 生成訓練數據
labels = []
for i in range(num_images):
    # 隨機生成點的數量
    red_count = random.randint(0, 4)
    white_count = random.randint(0, 4)
    image = generate_image(red_count, white_count)
    
    # 儲存圖片
    filename = f'{output_dir}/image_{i}.png'
    cv2.imwrite(filename, image)
    labels.append((filename, red_count, white_count))

# 儲存標籤資料
import pandas as pd
labels_df = pd.DataFrame(labels, columns=['filename', 'red_count', 'white_count'])
labels_df.to_csv('dataset/labels2.csv', index=False)