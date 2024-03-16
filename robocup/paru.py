# 纯纯的帕鲁代码


import PIL
import cv2
import numpy as np
import torch
# from tracker.sort import Sort
from . import tools
from ultralytics import YOLO



class Paru(object):
    def __init__(self, weights,dataset):
        self.device = torch.device('cuda:0')
        self.model=YOLO(model=weights)
        self.class_names = tools.load_yaml(dataset)['names']
    def detect_image(self, source, draw_img=True):
        """
        args:
        
            Args:
                source: type:Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] 
                表示想要预测的帧或图片

        returns:

            resp: (List[ultralytics.engine.results.Results])表示预测出来的结果


        """

        # 强制全部转化为list，为了适配以前代码
        if isinstance(source,list):
            pass
        else:
            image_list=[source]



        
        results=self.model.track(image_list,conf=0.5,tracker='botsort.yaml') # conf 设置置信度下限

        # desk_index = 17
        detected_imgs=[]
        # desk_box = []
        # index = []

        for result in results:

            # for box in result.boxes:
            #     if box.cls == desk_index:
            #         desk_box = box
            #         break
            #
            # if desk_box == []:
            #     continue
            #
            # for i, box in result.boxes:
            #     x = (box.xyxy[0] + box.xyxy[2])/2
            #     y = (box.xyxy[1] + box.xyxy[3])/2
            #     if (x < desk_box.xyxy[0] or x > desk_box.xyxy[2] or y > desk_box.xyxy[1] or y < desk_box.xyxy[3]):
            #         index.append(i)
            #
            # boxes = result.boxes  # Boxes object for bounding box outputs
            # masks = result.masks  # Masks object for segmentation masks outputs
            # keypoints = result.keypoints  # Keypoints object for pose outputs
            # probs = result.probs  # Probs object for classification outputs
            # result.show()  # display to screen
            detected_imgs.append(result.index_plot())


            # print("boxs:{}".format(boxes))
            # print("masks:{}".format(masks))
            # print("keypoints:{}".format(keypoints))
            # print("probs:{}".format(probs))
            # cv2.imshow("test",result.plot())
            # cv2.waitKey(0)

            # result.save(filename=f'result_{str(idx)}.jpg')  # save to disk
        return results,detected_imgs


# just for testing purposes
if __name__ == '__main__':

    myParu=Paru("../weights/yolov8s.pt","../formats/coco.yaml")
    myParu.detect_image("./test_paru.jpg")
    
    pass