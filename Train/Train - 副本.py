from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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


# 定义模型配置
class HelmetConfig(Config):
    # 定义配置名
    NAME = "Helmet_cfg"
    # 类的数量
    NUM_CLASSES = 1 + 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 200
    IMAGE_MAX_DIM = 256
    TRAIN_ROIS_PER_IMAGE = 50
    # 每轮训练的迭代数量
    STEPS_PER_EPOCH = 760
    VALIDATION_STEPS = 189


# 训练集
train_set = HelmetDataset()
train_set.load_dataset(r'D:\PersonalDocuments\VOC2028', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# 测试集
test_set = HelmetDataset()
test_set.load_dataset(r'D:\PersonalDocuments\VOC2028', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# 配置信息
config = HelmetConfig()
config.display()

# 定义模型
model = MaskRCNN(mode='training', model_dir='./', config=config)
model.keras_model.metrics_tensors = []
# 加载 mscoco 权重信息，排除输出层
model.load_weights('mask_rcnn_coco.h5', by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# 训练
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
