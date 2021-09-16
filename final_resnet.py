import numpy as np
import cv2
import os
import onnx
import onnxruntime
import sys


def OppositePath():
    """相对路径"""
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    return dirname

def run(img_test):
    for folder in os.listdir(img_test):
        if os.path.isdir(img_test+'\\'+folder):
            class_folder = img_test + '\\' + folder
            for img_path in os.listdir(class_folder):
                path1 = class_folder + '\\' + img_path
                print(path1)
                #
                image = cv2.imdecode(np.fromfile(file=path1, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imdecode(np.fromfile(os.path.join(path1), dtype=np.uint8), 0)
                # img = cv2.imread(img_test + '\\' + img_path)
                img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
                img = np.divide(img, np.array([255]))
                for i in range(len(img)):
                    for j in range(len(img[i])):
                        t = img[i][j][0]
                        img[i][j][0] = img[i][j][2]
                        img[i][j][2] = t
                img = img.transpose(2, 0, 1)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                for i in range(len(img)):
                    img[i] = np.divide(np.subtract(img[i], np.array(mean[i])), np.array(std[i]))
                img = np.array([img])  # 与inputs一样

                # ***************
                # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
                ort_inputs = {ort_session.get_inputs()[0].name: img}
                ort_inputs['input'] = ort_inputs['input'].astype('float32')
                # print(ort_inputs)
                outputs = ort_session.run(None, ort_inputs)[0][0]
                for i in range(len(outputs)):
                    if outputs[i] == max(outputs):
                        predictions = i
                    else:
                        pass
                print(class_name[int(predictions)])
                # image = cv2.imdecode(np.fromfile(file=path1, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.putText(image, class_name_pinyin[int(predictions)], (40, 50), cv2.FONT_HERSHEY_PLAIN, 3.0,
                            (0, 0, 255), 2)
                cv2.imshow(path1, image)
                cv2.waitKey(0)

        else:#里面是图片
            for img_path in os.listdir(img_test):
                path1 = img_test + '\\' + img_path#
                image = cv2.imdecode(np.fromfile(file=path1, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imdecode(np.fromfile(os.path.join(path1), dtype=np.uint8), 0)
                # img = cv2.imread(img_test + '\\' + img_path)
                img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
                img = np.divide(img, np.array([255]))
                for i in range(len(img)):
                    for j in range(len(img[i])):
                        t = img[i][j][0]
                        img[i][j][0] = img[i][j][2]
                        img[i][j][2] = t
                img = img.transpose(2, 0, 1)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                for i in range(len(img)):
                    img[i] = np.divide(np.subtract(img[i], np.array(mean[i])), np.array(std[i]))
                img = np.array([img])  # 与inputs一样

                # ***************
                # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
                ort_inputs = {ort_session.get_inputs()[0].name: img}
                ort_inputs['input'] = ort_inputs['input'].astype('float32')
                # print(ort_inputs)
                outputs = ort_session.run(None, ort_inputs)[0][0]
                for i in range(len(outputs)):
                    if outputs[i] == max(outputs):
                        predictions = i
                    else:
                        pass
                print(class_name[int(predictions)])
                # image = cv2.imdecode(np.fromfile(file=path1, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.putText(image, class_name_pinyin[int(predictions)], (40, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255),
                            2)
                cv2.imshow(path1, image)
                cv2.waitKey(0)

if __name__ == '__main__':
    class_name = ['背景', '港口', '机场', '雷达阵地', '战场场景']
    class_name_pinyin = ['background', 'port', 'airport', 'radar positioons', 'battlefield scenes']
    BATCH_SIZE = 1
    num_classes = 5



    onnx_model = onnx.load('history_best_resnet50.onnx')
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("history_best_resnet50.onnx")

    img_test = OppositePath() + '\\pictures'#文件夹或者文件夹的文件夹

    run(img_test)#测试




