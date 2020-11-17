import cv2
from numpy import expand_dims
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image



class PredictionConfig(Config):
    # 定义配置名
    NAME = "Helmet_cfg"
    # 类的数量
    NUM_CLASSES = 1 + 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


cfg = PredictionConfig()
cfg.display()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('mask_rcnn_helmet_cfg_good.h5', by_name=True)

image = []

for i in range(16):
    image.append(cv2.imread(r'D:\PersonalDocuments\VOC2028\Test\image\0000' + str(i + 1) + '.jpg'))
    scaled_image = mold_image(image[i], cfg)
    # 将图像转换为样本
    sample = expand_dims(scaled_image, 0)
    results = model.detect(sample, verbose=0)[0]
    j = 0
    for box in results['rois']:
        y1, x1, y2, x2 = box
        if results['class_ids'][j] == 1:
            cv2.rectangle(image[i], (x1, y1), (x2, y2), (22, 255, 95), 2)
        if results['class_ids'][j] == 2:
            cv2.rectangle(image[i], (x1, y1), (x2, y2), (26, 70, 249), 2)
        j = j + 1
    cv2.imshow('image', image[i])
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.imwrite(r'D:\PersonalDocuments\VOC2028\Test\detected\0000' + str(i + 1) + '.jpg', image[i])
