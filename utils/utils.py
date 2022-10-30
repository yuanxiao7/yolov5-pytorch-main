import numpy as np
from PIL import Image
import cv2
import torch
from torchvision.transforms import *

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size  # 符合网络的图片size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))  # 新建一张灰度图
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))  # 把缩放后的图像贴到灰度图上
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


#---------------------------------------------------#
def resize_image_1(image):
    new_image = image.resize((800, 800), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类  放在一个列表里
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框，[13, 16], ... add up 9
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        i_list = [i for i in optimizer.param_groups[0].keys()]
        return param_group['lr']

 # 图片归一化处理
def preprocess_input(image):
    image /= 255.0
    return image

# 打印出config
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

# 权重下载
def download_weights(backbone, phi, model_dir="./model_data"):  # 这里是下载主干网络的权重
    import os
    from torch.hub import load_state_dict_from_url
    if backbone == "cspdarknet":
        backbone = backbone + "_" + phi
    
    download_urls = {
        "convnext_tiny"         : "https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/convnext_tiny_1k_224_ema_no_jit.pth",
        "convnext_small"        : "https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/convnext_small_1k_224_ema_no_jit.pth",
        "cspdarknet_s"          : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_s_backbone.pth',
        'cspdarknet_m'          : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_m_backbone.pth',
        'cspdarknet_l'          : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_l_backbone.pth',
        'cspdarknet_x'          : 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_x_backbone.pth',
        'swin_transfomer_tiny'  : "https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/swin_tiny_patch4_window7.pth",
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)





class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length, fill_value):
        self.n_holes = n_holes
        self.length = length
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = self.fill_value


        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# if __name__ == "__main__":
#     # # 2、Cutout
#     img = cv2.imread("img/1_2.jpg")
#     save_path = "1_21.jpg"
#     transform = Compose([
#         transforms.ToTensor(),
#         Cutout(n_holes=30, length=10, fill_value=0.)  # length 控制擦除范围，fill_value控制用什么值进行擦除
#     ])
#     # transform暂且把他看作一个已经实例化的函数，将多步骤给合在一起的函数
#     img2 = transform(img=img)
#     img2 = img2.numpy().transpose([1, 2, 0])
#     cv2.imwrite(save_path, img2)
#     cv2.imshow("test", img2)
#     cv2.waitKey(0)