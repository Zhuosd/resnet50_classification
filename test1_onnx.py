import time
import numpy as np
import  transforms
from datasets import ImageFolder#torch package
import cv2
import matplotlib.pyplot as plt
import os
import onnx
import onnxruntime
import utils.data.dataloader#torch package

import sys
def OppositePath():
    """相对路径"""
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    return dirname

if __name__ == '__main__':
    class_name=['背景','港口','机场','雷达阵地','战场场景']
    class_name_pinyin=['background','port','airport','radar positioons','battlefield scenes']
    BATCH_SIZE = 1
    num_classes=5
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize鍒?24x224澶у皬
        transforms.ToTensor(),  # 杞寲鎴怲ensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 姝ｅ垯鍖?
    ])
    img_test=OppositePath()+'\\pictures'

    test_dataset = ImageFolder(img_test, transform=test_transform)
    test_dataset_original=ImageFolder(img_test)
    print(test_dataset_original.imgs[0][0])
    test_loader = utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=2)

    onnx_model = onnx.load('history_best_resnet50.onnx')
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("history_best_resnet50.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    for j, (inputs, labels) in enumerate(test_loader):
        img_path=test_dataset_original.imgs[j][0].split('\\')
        add_path=''
        for i in range(len(img_path)):
            add_path+=img_path[i]+'\\\\'
        add_path=add_path[:-2]
        print('图片序号：', add_path)
        # ***************
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
        outputs = ort_session.run(None, ort_inputs)[0][0]
        for i in range(len(outputs)):
            if outputs[i]==max(outputs):
                predictions=i
            else:
                pass
        print(class_name[int(predictions)])
        image = cv2.imdecode(np.fromfile(file=add_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.putText(image,class_name_pinyin[int(predictions)], (40, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), 2)
        cv2.imshow(add_path, image)
        cv2.waitKey(0)





