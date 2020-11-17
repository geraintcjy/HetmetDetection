from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image


class HelmetDataset(Dataset):
    # 加载数据集定义
    def load_dataset(self, dataset_dir, is_train=True):
        # 定义识别需要的两个类
        self.add_class("dataset", 1, "PersonWithHelmet")
        self.add_class("dataset", 2, "PersonWithoutHelmet")
        # 数据所在位置
        images_dir = dataset_dir + '/JPEGImages/' + '/Selected_Revised/'
        annotations_dir = dataset_dir + '/Annotations/'
        # 定位到所有图像
        for filename in listdir(images_dir):
            if filename[4:5] == '2' and filename[0:1] == 'p':
                image_id = str(int(filename[-9:-4]) + 10000)
            elif filename[4:5] == 'A':
                image_id = str(int(filename[-9:-4]) + 20000)
            elif filename[4:5] == 'B':
                image_id = str(int(filename[-9:-4]) + 30000)
            else:
                image_id = filename[-9:-4]

            if is_train and int(image_id) > 0:
                continue

            if not is_train and int(image_id) <= 120:
                continue
            if int(image_id) == 72:
                continue
            if int(image_id) > 150:
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + filename[:-4] + '.xml'
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
    NAME = "helmet_cfg"
    # 类的数量（背景中的 + 袋鼠）
    NUM_CLASSES = 1 + 2
    # 简化 GPU 配置
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# 计算给定数据集中模型的 mAP
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # 加载指定 image id 的图像、边框和掩膜
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # 转换像素值（例如居中）
        scaled_image = mold_image(image, cfg)
        # 将图像转换为样本
        sample = expand_dims(scaled_image, 0)
        # 作出预测
        yhat = model.detect(sample, verbose=0)
        # 为第一个样本提取结果
        r = yhat[0]
        # 统计计算，包括计算 AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # 保存
        APs.append(AP)
    # 计算所有图片的平均 AP
    mAP = mean(APs)
    return mAP


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

# 创建配置
cfg = PredictionConfig()
# 定义模型
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# 加载模型权重
model.load_weights('mask_rcnn_helmet_cfg_0004.h5', by_name=True)

# 评估测试集上的模型
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
