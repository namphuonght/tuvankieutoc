import pickle
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import accuracy_score
import logging


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s')
# load dữ liệu từ 2 file pkl

file = open('landmarks.pkl', 'rb')
landmark_list = pickle.load(file)
file.close()

file = open('labels.pkl', 'rb')
label_list = pickle.load(file)
file.close()

# xao tron data
sss = StratifiedShuffleSplit(test_size=0.25, random_state=0)

X_train, X_test, y_train, y_test = None, None, None, None
# Chia du lieu train, test
for train_index, test_index in sss.split(landmark_list, label_list):
    X_train, X_test = landmark_list[train_index], landmark_list[test_index]
    y_train, y_test = label_list[train_index], label_list[test_index]

logging.info('Task: training data\n')
svm = svm.SVC(kernel='linear')
svm.fit(X_train, y_train) #train

# result = svm.predict([X_test[0]])
result = []
for i in range(len(X_test)):
    result.append(svm.predict([X_test[i]]))


def convert_to_int(list):
    list_int = []
    for label in list:
        if label == 'heart':
            list_int.append(0)
        elif label == 'oblong':
            list_int.append(1)
        elif label == 'oval':
            list_int.append(2)
        elif label == 'round':
            list_int.append(3)
        else:
            list_int.append(4)

    return list_int


y_predict = np.array(convert_to_int(result))
y_true = np.array(convert_to_int(y_test))


# danh gia model
print('Accuracy =', accuracy_score(y_true, y_predict), '\n')


# lưu model vào file
model_file = 'model.sav'
file = open(model_file, 'wb')
pickle.dump(svm, file)
file.close()
logging.info('Process Done')
# load xong thì train model svm