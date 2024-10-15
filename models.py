import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import os
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

image_dir = 'dataset/images/'
labels_file = 'dataset/labels2.csv'

# 讀取標籤
labels_df = pd.read_csv(labels_file)

# 讀取影像和標籤
X = []
Y = []
image_size = (128, 128)

for index, row in labels_df.iterrows():
    # 載入圖片並調整大小
    image = load_img(row['filename'], target_size=image_size)
    image_array = img_to_array(image) / 255.0  # 正規化到 [0, 1]
    X.append(image_array)
    Y.append([row['red_count'], row['white_count']])

# 將影像和標籤轉換為 NumPy 陣列
X = np.array(X)
Y = np.array(Y)

# 將數據集分割為訓練集和測試集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 建立 CNN 模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2)  # 輸出紅點和白點的數量
])

# 編譯模型
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 訓練模型
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2)
model.save('red_white_dot_classifier.h5')