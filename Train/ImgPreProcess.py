import os
import cv2
from xml.dom.minidom import parse


def all_path(dirname):
    path = []

    for maindir, subdir, file_name_list in os.walk(dirname):

        print("目录:", maindir)
        print("次目录:", subdir)
        print("文件:", file_name_list)

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            path.append(apath)

    return path


def preprocess(k, init):
    img = cv2.imread(imgpath[k], 1)
    domTree = parse(xmlpath[k])
    rootNode = domTree.documentElement
    objects = rootNode.getElementsByTagName('object')
    for object in objects:
        name = object.getElementsByTagName('name')[0].firstChild.data
        bndbox = object.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0].firstChild.data
        xmax = bndbox.getElementsByTagName('xmax')[0].firstChild.data
        ymin = bndbox.getElementsByTagName('ymin')[0].firstChild.data
        ymax = bndbox.getElementsByTagName('ymax')[0].firstChild.data

        cut = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        if name == 'hat':
            cv2.imwrite('D:/PersonalDocuments/VOC2028/Train/Hat/' + str(init[0] + 1) + '.jpg', cut)
            init[0] = init[0] + 1
        else:
            cv2.imwrite('D:/PersonalDocuments/VOC2028/Train/Person/' + str(init[1] + 1) + '.jpg', cut)
            init[1] = init[1] + 1
    return init


initial = [0, 0]
imgpath = all_path(r'D:\PersonalDocuments\VOC2028\JPEGImages')
xmlpath = all_path(r'D:\PersonalDocuments\VOC2028\Annotations')
for num in range(0, 1000):
    initial = preprocess(num, initial)
