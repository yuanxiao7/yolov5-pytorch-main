import time
import torch
import cv2
import numpy as np
from PIL import Image
import random
from yolo import YOLO
import numpy as np
import os
import random


@torch.no_grad()
def detect(img_path):
    yolo = YOLO()
    img = img_path  # 终端输入
    print(img)
    image = Image.open(img)

    r_image = yolo.detect_image(image, crop=False, count=False)


    bbb = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    c = range(len(bbb))
    indexs = random.sample(c, 1)

    a = np.asarray(bbb)[indexs][0]
    a = str(a)
    r_image.save("uploads\\" + a + ".jpg")
    img_result = os.path.join(a)

    print(img_result)

    return img_result



# if __name__ == '__main__':
#     resut = detect(img_path='img/street.jpg')
#     print(resut)

























