from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
import cv2


class HelmetDataset(Dataset):
    # 加载数据集定义
    def load_dataset(self, dataset_dir, is_train=True):
        # 定义识别需要的两个类
        self.add_class("dataset", 1, "PersonWithHelmet")
        self.add_class("dataset", 2, "PersonWithoutHelmet")
        # 数据所在位置
        images_dir = dataset_dir + '/JPEGImages/' + '/1-1668/'
        annotations_dir = dataset_dir + '/Annotations/'
        # 定位到所有图像
        for filename in listdir(images_dir):
            image_id = filename[:-4]
            # 训练集略过 350 序号之后的所有图像
            if is_train and int(image_id) >= 800:
                continue
            # 测试集略过 350 序号之前的所有图像
            if not is_train and int(image_id) < 800:
                continue
            if int(image_id) >= 1000:
                continue
            if int(image_id) == 72:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # 添加到数据集
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # 从注解文件中提取边框值
    def extract_boxes(self, filename):
        # 加载并解析文件
        tree = ElementTree.parse(filename)
        # 获取文档根元素
        root = tree.getroot()
        # 提取出图像尺寸及每个 bounding box 元素
        boxes = list()
        for object in root.findall('.//object'):
            name = object.find('name').text
            xmin = int(object.find('.//bndbox/xmin').text)
            ymin = int(object.find('.//bndbox/ymin').text)
            xmax = int(object.find('.//bndbox/xmax').text)
            ymax = int(object.find('.//bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # 加载图像掩膜
    def load_mask(self, image_id):
        # 获取图像详细信息
        info = self.image_info[image_id]
        # 定义盒文件位置
        path = info['annotation']
        # 加载 XML
        boxes, w, h = self.extract_boxes(path)
        # 为所有掩膜创建一个数组，每个数组都位于不同的通道
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # 创建掩膜
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            if box[4] == 'hat':
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('PersonWithHelmet'))
            else:
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('PersonWithoutHelmet'))
        return masks, asarray(class_ids, dtype='int32')

    # 加载图像引用
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# 定义预测配置
class PredictionConfig(Config):
    # 定义配置名
    NAME = "Helmet_cfg"
    # 类的数量
    NUM_CLASSES = 1 + 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# 绘制多张带有真实和预测边框的图像
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    # 加载图像和掩膜
    for i in range(n_images):
        # 加载图像和掩膜
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # 转换像素值（例如居中）
        scaled_image = mold_image(image, cfg)
        # 将图像转换为样本
        sample = expand_dims(scaled_image, 0)
        # 作出预测
        yhat = model.detect(sample, verbose=0)[0]
        # 定义子图
        pyplot.subplot(n_images, 2, i * 2 + 1)
        # 绘制原始像素数据
        pyplot.imshow(image)
        pyplot.title('Actual')
        # 绘制掩膜
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # 获取绘图框的上下文
        pyplot.subplot(n_images, 2, i * 2 + 2)
        # 绘制原始像素数据
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        # 绘制每个绘图框
        for box in yhat['rois']:
            # 获取坐标
            y1, x1, y2, x2 = box
            # 计算绘图框的宽度和高度
            width, height = x2 - x1, y2 - y1
            # 创建形状对象

            rect = Rectangle((x1, y1), width, height, fill=False, color='green')

            # 绘制绘图框
            ax.add_patch(rect)
            # 显示绘制结果
        pyplot.show()
    # 显示绘制结果



train_set = HelmetDataset()
train_set.load_dataset(r'D:\PersonalDocuments\VOC2028', is_train=True)
train_set.prepare()
test_set = HelmetDataset()
test_set.load_dataset(r'D:\PersonalDocuments\VOC2028', is_train=False)
test_set.prepare()

cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model_path = 'mask_rcnn_helmet_cfg_0002.h5'
model.load_weights(model_path, by_name=True)

plot_actual_vs_predicted(test_set, model, cfg)
