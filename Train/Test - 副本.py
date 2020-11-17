import cv2
from mrcnn.config import Config
from mrcnn.model import MaskRCNN


class PredictionConfig(Config):
    # 定义配置名
    NAME = "Helmet_cfg"
    # 类的数量
    NUM_CLASSES = 1 + 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


image = []
image.append(cv2.imread(r'D:\PersonalDocuments\VOC2028\Test\image\00001.jpg'))

cfg = PredictionConfig()
cfg.display()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('mask_rcnn_helmet_cfg_0004.h5', by_name=True)

results = model.detect(image, verbose=1)[0]

i = 0
for box in results['rois']:
    y1, x1, y2, x2 = box
    if results['class_ids'][i] == 1:
        cv2.rectangle(image[0], (x1, y1), (x2, y2), (22, 255, 95), 2)
    if results['class_ids'][i] == 2:
        cv2.rectangle(image[0], (x1, y1), (x2, y2), (26, 70, 249), 2)
    i = i + 1

cv2.imshow('image', image[0])
cv2.waitKey(5000)
cv2.destroyAllWindows()
