import cv2
import numpy as np
import random

def create_training_images(num_images=10, red_radius=50, white_radius=50):
    for i in range(num_images):
        # 建立黑色背景
        image = np.zeros((500, 500, 3), dtype=np.uint8)

        # 隨機生成紅點和白點的位置
        for _ in range(0):  # 生成 2 個紅點
            center = (random.randint(red_radius, 500 - red_radius), 
                      random.randint(red_radius, 500 - red_radius))
            cv2.circle(image, center, red_radius, (0, 0, 255), -1)  # 紅點

        for _ in range(4):  # 生成 2 個白點
            center = (random.randint(white_radius, 500 - white_radius),
                      random.randint(white_radius, 500 - white_radius))
            cv2.circle(image, center, white_radius, (255, 255, 255), -1)  # 白點

        # 儲存影像
        cv2.imwrite(f'./dataset/image/training_image_{i+19}.jpg', image)

# 生成訓練集影像
create_training_images()
