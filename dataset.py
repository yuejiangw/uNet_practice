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
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)
        # 有关数据增强，请参考：https://keras.io/zh/preprocessing/image/
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
            # 每一张图像都有自己独立的一个文件夹存储增强后的图片
            savedir = path_aug_merge + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            self.doAugmentate(img, savedir, str(i))

    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):
        """
        对一张图片进行增强，默认一张图片生成30张
        """
        datagen = self.datagen
        i = 0
        # 参考：https://blog.csdn.net/mieleizhi0522/article/details/82191331
        for batch in datagen.flow(
                img,
                batch_size=batch_size,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format
            ):
            i += 1
            if i > imgnum:
                break
    
    def splitMerge(self):
        """
        将合在一起的图片分开
        """
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_label = self.aug_label_path

        for i in range(self.slices):
            path = path_merge + "/" + str(i)
            train_imgs = glob.glob(path + "/*." + self.img_type)
            savedir = path_train + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            savedir = path_label + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            
            for imgname in train_imgs:
                midname = imgname[imgname.rindex("/")+1: imgname.rindex("."+self.img_type)]
                img = cv2.imread(imgname)
                # cv2 读取图片是BGR格式，要转换为RGB
                # 参考：https://www.jianshu.com/p/0e462b4c7a93
                img_train = img[:,:,2]
                img_label = img[:,:,0]
                cv2.imwrite(path_train + "/" + str(i) + "/" + midname + "_train" + "." + self.img_type, img_train)
                cv2.imwrite(path_label + "/" + str(i) + "/" + midname + "_label" + "." + self.img_type, img_label)

    def splitTransform(self):
        """
        将透视变换后的图像拆分
        """
        path_merge = './images/deform/de_merge'
        path_train = './images/deform/de_train/'
        path_label = './images/deform/de_label/'

        train_imgs = glob.glob(path_merge + '/*.' + self.img_type)
        for imgname in train_imgs:
            midname = imgname[imgname.rindex("/") + 1 : imgname.rindex("." + self.img_type)]
            img = cv2.imread(imgname)
            img_train = img[:,:,2]
            img_label = img[:,:,0]
            cv2.write(path_train + midname + '.' + self.img_type, img_train)
            cv2.write(path_label + midname + '.' + self.img_type, img_label)

class dataProcess(object):
    '''
    此类用于数据处理
    '''
    def __init__(self, out_rows, out_cols, data_path='./images/deform/de_train',
        label_path='./images/deform/de_label', test_path='./images/test', npy_path='./npy_data', img_type='tif'):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.test_path = test_path
        self.npy_path = npy_path
        self.img_type = img_type
    
    # 创建训练数据
    def create_train_data(self):
        i = 0
        print('-' * 30)
        print("Creating train images...")
        print('-' * 30)
        imgs = glob.glob(self.data_path + '/.*' + self.img_type)
        print('The number of training images is: ' + len(imgs))

        img_data = np.array((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        img_label = np.array((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for img_name in imgs:
            midname = img_name[img_name.rindex("/") + 1:]
            # 读取图像和对应标签，grayscale=True代表以灰度图像形式读取
            # load_img返回一个Image对象，需要手动转换为矩阵
            img = load_img(self.data_path + "/" + midname, grayscale=True)      
            label = load_img(self.label_path + "/" + midname, grayscale=True)
            img = img_to_array(img)
            label = img_to_array(label)
            img_data[i] = img
            img_label[i] = label
            if i % 10 == 0:
                print("Done: {0}/{1} images".format(i, len(imgs)))
            i += 1
        print("Loading train images done, start saving...")
        np.save(self.npy_path + '/imgs_train.npy', img_data)
        print('Saving to imgs_train.npy files done')
        np.save(self.npy_path + './imgs_label.npy', img_label)
        print("Saving to imgs_label.npy files done")
    
    # 创建测试数据
    def create_test_data(self):
        i = 0
        print('-' * 30)
        print("Creating test images...")
        print('-' * 30)
        imgs = glob.glob(self.test_path + '/*.' + self.img_type)
        print('The number of test images is: ' + len(imgs))

        img_data = np.array((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for img_name in imgs:
            midname = img_name[img_name.rindex("/") + 1:]
            img = load_img(self.test_path + '/' + midname, grayscale=True)
            img = img_to_array(img)
            img_data[i] = img
            i += 1
        print('Loading test images done, start saving...')
        np.save(self.npy_path + '/imgs_test.npy', img_data)
        print('Saving to imgs_test.npy files done')
    
    # 加载训练图片与mask
    def load_train_data(self):
        print("-" * 30)
        print("Load train images and labels...")
        print("-" * 30)
        imgs_train = np.load(self.npy_path + '/imgs_train.npy')
        imgs_label = np.load(self.npy_path + '/imgs_label.npy')
        # 将像素值归一化处理，参考：https://blog.csdn.net/qq_38784454/article/details/80449445
        imgs_train = imgs_train.astype('float32')
        imgs_label = imgs_label.astype('float32')
        imgs_train /= 255
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        # 将mask进行阈值处理，认为输出概率大于0.5的为1，否则为0
        imgs_label /= 255
        imgs_label[imgs_label > 0.5] = 1
        imgs_label[imgs_label <= 0.5] = 0
        
        return imgs_train, imgs_label

    # 加载测试图片
    def load_test_data(self):
        print('-' * 30)
        print("Load test images...") 
        print('-' * 30)
        imgs_test = np.load(self.npy_path + '/imgs_test.npy')
        # 归一化处理
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean
        
        return imgs_test

if __name__ == "__main__":
    # 数据增强
    aug = dataAugmentation()
    aug.Augmentation()
    aug.splitMerge()
    aug.splitTransform()

    # 数据处理
    mydata = dataProcess(512, 512)
    mydata.create_train_data()
    mydata.create_test_data()

    imgs_train, imgs_label = mydata.load_train_data()
    print(imgs_train.shape, imgs_label.shape)
