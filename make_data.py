import os
from mtcnn import MTCNN
import cv2
import dlib
from imutils import face_utils
import numpy as np
import pickle

detector = MTCNN()
landmark_detector = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
landmark_list = []
label_list = []

# đọc thư mục face_data
raw_folder = 'face_data'

for folder in os.listdir(raw_folder):

    for file in os.listdir(os.path.join(raw_folder, folder)):
        print('process file', file)

        # phát hiện khuôn mặt(coi như 1 ảnh chỉ có 1 khuôn mặt)
        pix_file = os.path.join(raw_folder, folder, file)
        image = cv2.imread(pix_file)

        results = detector.detect_faces(image)

        if len(results) > 0:
            # có mặt thì ta lấy mặt đầu tiên
            result = results[0]

            # trích xuất tọa độ của mặt trong ảnh

            x1, y1, width, height = result['box']
            x1, y1 = abs(x1), abs(y1)
            x2 = x1 + width
            y2 = y1 + height

            face = image[y1:y2, x1:x2]

            # trích xuất landmask bằng dlib

            landmark = landmark_detector(image, dlib.rectangle(x1, y1, x2, y2))
            landmark = face_utils.shape_to_np(landmark)

            landmark = landmark.reshape(68*2)

            # thêm cái landmask vào list các landmask
            landmark_list.append(landmark)
            label_list.append(folder)

# chuyển sang numpy array
landmark_list = np.array(landmark_list)
label_list = np.array(label_list)


# write vào file landmark.pkl
file = open('landmarks.pkl', 'wb')
pickle.dump(landmark_list, file)
file.close()

# write vào file labels.pkl
file = open('labels.pkl', 'wb')
pickle.dump(label_list, file)
file.close()

