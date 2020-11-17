from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.model import MaskRCNN


# 定义并加载安全帽数据集的类
class HelmetDataset(Dataset):
    # 加载数据集定义
    def load_dataset(self, dataset_dir, is_train=True):
        # 定义一个类
        self.add_class("dataset", 1, "Helmet")
        # 定义数据所在位置
        images_dir = dataset_dir + '/JPEGImages/'
        annotations_dir = dataset_dir + '/Annotations/'
        # 定位到所有图像
        for filename in listdir(images_dir):
            # 提取图像 id
            image_id = filename[:-4]
            # 如果我们正在建立的是训练集，略过 350 序号之后的所有图像
            if is_train and int(image_id) >= 350:
                continue
            # 如果我们正在建立的是测试/验证集，略过 350 序号之前的所有图像
            if not is_train and int(image_id) < 350:
                continue
            if int(image_id) >= 500:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # 添加到数据集
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # 从注解文件中提取边框值
    def get_position(self, filename):
        # 加载并解析文件
        tree = ElementTree.parse(filename)
        # 获取文档根元素
        root = tree.getroot()
        # 提取出每个 bounding box 元素
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # 提取出图像尺寸
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
        boxes, w, h = self.get_position(path)
        # 为所有掩膜创建一个数组，每个数组都位于不同的通道
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # 创建掩膜
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('Helmet'))
        return masks, asarray(class_ids, dtype='int32')

    # 加载图像引用
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# 定义模型配置
class HelmetConfig(Config):
    # 给配置对象命名
    NAME = "helmet_cfg"
    # 类的数量（背景中的 + 袋鼠）
    NUM_CLASSES = 1 + 1
    # 每轮训练的迭代数量
    STEPS_PER_EPOCH = 300


# 训练集
train_set = HelmetDataset()
train_set.load_dataset(r'D:\PersonalDocuments\VOC2028', is_train=True)
train_set.prepare()
# 准备测试/验证集
test_set = HelmetDataset()
test_set.load_dataset(r'D:\PersonalDocuments\VOC2028', is_train=False)
test_set.prepare()
# 准备配置信息
config = HelmetConfig()
config.display()
# 定义模型
model = MaskRCNN(mode='training', model_dir='./', config=config)
# 加载 mscoco 权重信息，排除输出层
model.load_weights('mask_rcnn_coco.h5', by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# 训练权重（输出层，或者说‘头部’）
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

# 加载图像
image_id = 0
image = train_set.load_image(image_id)
print(image.shape)
# 加载图像掩膜
mask, class_ids = train_set.load_mask(image_id)
print(mask.shape)

"""
# 绘制最开始的几张图像
for i in range(9):
    # 定义子图
    pyplot.subplot(330 + 1 + i)
    # 绘制原始像素数据
    image = train_set.load_image(i)
    pyplot.imshow(image)
    # 绘制所有掩膜
    mask, _ = train_set.load_mask(i)
    for j in range(mask.shape[2]):
        pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
# 展示绘制结果
pyplot.show()
"""