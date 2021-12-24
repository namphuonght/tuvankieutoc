import pickle
from flask import Flask, render_template, request
import dlib
import os
from mtcnn.mtcnn import MTCNN
from random import random
import cv2
from imutils import face_utils


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes


faceProto = "./model/opencv_face_detector.pbtxt"
faceModel = "./model/opencv_face_detector_uint8.pb"
ageProto = "./model/age_deploy.prototxt"
ageModel = "./model/age_net.caffemodel"
genderProto = "./model/gender_deploy.prototxt"
genderModel = "./model/gender_net.caffemodel"
padding = 20

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Nam', 'Nữ']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

filename = 'model.sav'
clf = pickle.load(open(filename, 'rb'))

# nam
desc_file = "face_desc.csv"
with open('face_desc.csv', 'r', encoding="utf8") as f:
    desc = f.readlines()
dict = {}
for line in desc:
    dict[line.split('|')[0]] = [line.split('|')[1], line.split('|')[2]]

# nữ
with open('face_desc_female.csv', 'r', encoding='utf8') as fn:
    desc_nu = fn.readlines()
dict_nu = {}
for line_nu in desc_nu:
    dict_nu[line_nu.split('|')[0]] = [line_nu.split('|')[1], line_nu.split('|')[2]]
# with open('')

detector = MTCNN()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")


# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                # print('image.filename ', image.filename)
                # print(app.config['UPLOAD_FOLDER'])
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                # print("Save = ", path_to_save)
                image.save(path_to_save)

                # Convert image to dest size tensor
                frame = cv2.imread(path_to_save)

                results = detector.detect_faces(frame)

                if len(results) != 0:

                    resultImg, faceBoxes = highlightFace(faceNet, frame)
                    if not faceBoxes:
                        print("Không phát hiện được khuôn mặt")

                    for faceBox in faceBoxes:
                        face = frame[max(0, faceBox[1] - padding):
                                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                                    :min(faceBox[2] + padding,
                                                                                         frame.shape[1] - 1)]

                        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        genderPreds = genderNet.forward()
                        gender = genderList[genderPreds[0].argmax()]

                        ageNet.setInput(blob)

                    for result in results:
                        x1, y1, width, height = result['box']

                        x1, y1 = abs(x1), abs(y1)
                        x2, y2 = x1 + width, y1 + height
                        face = frame[y1:y2, x1:x2]

                        # Extract dlib
                        landmark = predictor(frame, dlib.rectangle(x1, y1, x2, y2))
                        landmark = face_utils.shape_to_np(landmark)

                        print("O", landmark.shape)
                        landmark = landmark.reshape(68 * 2)
                        print("R", landmark.shape)

                        y_pred = clf.predict([landmark])
                        print(y_pred)

                        if gender == "Nam":
                            extra = dict[y_pred[0]][1]
                            ID = dict[y_pred[0]][0]
                        else:
                            extra = dict_nu[y_pred[0]][1]
                            ID = dict_nu[y_pred[0]][0]

                        cv2.imwrite(path_to_save, face)
                        break

                    # Trả về kết quả
                    return render_template("index.html", user_image=image.filename, rand=str(random()),
                                           msg="Tải file lên thành công", idBoolean=True, ID=ID, extra=extra,
                                           gender=gender)
                else:
                    return render_template('index.html', msg='Không nhận diện được khuôn mặt')
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được khuôn mặt 1111')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


# start server
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)