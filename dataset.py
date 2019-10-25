from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2
from libtiff import TIFF

class dataAugmentation(object):
    '''
    用于数据增强的类
    首先读取用于训练的原图及对应标签，并将二者合并为之后的增强处理做准备
    之后用ImageDataGenerator增强图像
    最后将增强过的合并图像重新分成原图与标签

    PS: 之所以在增强前需要先将原图与对应标签合二为一是因为Keras中的数据增强是对原图像进行随机变换的,
        如果只是单纯的分别对原图和标签做增强，那么增强后二者就不会对应了。只能先将训练图像和对应标签
        合在一起，增强后再将其分开
    '''

    def __init__(self, train_path, label_path, merge_path, 
        aug_merge_path, aug_train_path, aug_label_path, img_type='tif'):
        """
        train_path: 原始训练图像路径
        label_path: 原始训练图像对应的标签路径
        merge_path: 生成合并图像的目标路径
        aug_merge_path: 对合并图像增强后的目标输出路径
        aug_train_path: 从增强合并图像分离出的增强后的训练
        """
        self.train_imgs = glob.glob(train_path + '/*.' + img_type)  # glob.glob返回一个列表
        self.label_imgs = glob.glob(label_path + '/*.' + img_type)
        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aut_label_path = aug_label_path
        self.slices = len(self.train_imgs)
        self.datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def Augmentation(self):
        """
        此方法用于数据增强
        """
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        # 首先判断原图像与标签是否数量上对应
        if len(trains) != len(labels) or len(trains) == 0 or len(labels) == 0:
            print("Trains don't match labels")
            return 0
        # 依次处理每一张图像与对应标签，将其合并
        for i in range(len(trains)):
            img_t = load_img(path_train + "/" + str(i) + "." + imgtype)
            img_l = load_img(path_label + "/" + str(i) + "." + imgtype)
            # load_img方法返回的是一个PIL的Image对象，若要转换为矩阵需要调用img_to_array方法
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)
            # 将原图与对应标签合并
            x_t[:,:,2] = x_l[:,:,0]
            # 将合并后的图像保存
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge + "/" + str(i) + "." + imgtype)
            # 
            img = x_t
            img = img.reshape((1,) + img.shape)
            savedir = path_aug_merge + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            self.doAugmentate(img, savedir, str(i))

    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):
        """
        对一张图片进行增强，默认一张图片生成30张
        """
        
