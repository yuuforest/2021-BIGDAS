''' 2021-1 Face Detection - Yurim & Jieun '''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import time
import numpy as np
import pickle
import cv2
import pandas as pd
import os
from keras.preprocessing import image as im

import keras.backend as K
from keras.layers import Input
from keras.models import Sequential, Model, load_model

from frcnn.RPN import rpn_layer
from frcnn.classifier import classifier_layer
from frcnn.loss import non_max_suppression_fast, apply_regr, rpn_to_roi
from frcnn.test import get_real_coordinates, format_img
from frcnn.vgg import nn_base

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from decode_np import Decode
import copy


# Face Dictionary
crowd = {}

image_num = 0
num_female = 0
num_male = 0

# Timer
st_total = time.time()
st = time.time()


################ YOLOv4 ################

def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

model_path = './yolo4_weight.h5'
anchors_path = 'model/yolov4_model_data/yolo4_anchors.txt'
classes_path = 'model/yolov4_model_data/coco_classes.txt'

class_names = get_class(classes_path)
anchors = get_anchors(anchors_path)

num_anchors = len(anchors)
num_classes = len(class_names)

model_image_size = (608, 608)

# 分数阈值和nms_iou阈值
conf_thresh = 0.3
nms_thresh = 0.45

yolo4_model = yolo4_body(Input(shape=model_image_size+(3,)), num_anchors//3, num_classes)

model_path = os.path.expanduser(model_path)
assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

print('Loading YOLOv4 weights from {} \nTesting...'.format(model_path))
yolo4_model.load_weights(model_path)



################ Faster-RCNN ################

base_path = '.'
config_output_filename = os.path.join(base_path, 'model/pickle/model_vgg_config_person4.pickle')

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

# Load the records
record_df = pd.read_csv(C.record_path)
r_epochs = len(record_df)
num_features = 512

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (VGG here, can be Resnet50, Inception, etc)
shared_layers = nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = rpn_layer(shared_layers, num_anchors)

classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading Faster-RCNN weights from {} \nTesting...'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

# Switch key value for class mapping
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}



###################### YOLOv4 Person Detection Model ######################

print('--------------------------------')
print('Waiting Person Detection...')

base_path = '.'
# Directory to save the test images
test_base_path = 'input'
img_path = test_base_path

st = time.time()

if __name__ == '__main__':
    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

    for idx, img_name in enumerate(os.listdir(img_path)):

        filepath = os.path.join(test_base_path, img_name)

        image = cv2.imread(filepath)

        image, boxes, scores, classes = _decode.detect_image(image, True, img_name)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # yolo4_model.close_session()
    # print('Elapsed time = {}'.format(time.time() - st))

print('Elapsed time = {}'.format(time.time() - st))
print('Person Detection Completed!')



###################### Faster-RCNN Face Detection Model ######################

print('--------------------------------')
print('Waiting Face Detection...')

base_path = '.'
test_base_path = 'data/person_yolo'

all_imgs = []

classes = {}

# If the box classification value is less than this, we ignore this box
bbox_threshold = 0.2

visualise = True

img_path = test_base_path

st = time.time()

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    filepath = os.path.join(test_base_path, img_name)
    file_image = im.load_img(filepath)

    img = cv2.imread(filepath)

    X, ratio = format_img(img, C)

    X = np.transpose(X, (0, 2, 3, 1))

    # get output layer Y1, Y2 from the RPN and the feature maps F
    # Y1: y_rpn_cls
    # Y2: y_rpn_regr
    [Y1, Y2, F] = model_rpn.predict(X)

    # Get bboxes by applying NMS
    # R.shape = (300, 4)
    R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):
            # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]

            # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            frame_num = img_name.split('_')[0]


            ################################## Save Cropped Image ##################################

            f = open("txt/face/" + frame_num + ".txt", "a")
            f.write('{}_{}'.format(img_name.split('.')[0], jk) + '.png' + ',' + str(real_x1) + ',' + str(real_y1) + ',' + str(real_x2) + ',' + str(real_y2) + '\n')

            croppedImage = file_image.crop((real_x1, real_y1, real_x2, real_y2))
            img_name2 = img_name.split('.')
            cropped_name = '{}_{}'.format(img_name2[0], jk)
            croppedImage.save('./data/face_frcnn/' + cropped_name + '.png')
            image_num += 1

            face_coordinate = (real_x1, real_y1, real_x2, real_y2)

            # Update Crowd Dictionary
            crowd['{}'.format(cropped_name)] = [face_coordinate, croppedImage]

            new_name = 'data/face_frcnn/' + cropped_name + '.png'

print('Elapsed time = {}'.format(time.time() - st))
print("{} Faces Detection Completed!".format(image_num))



###################### Drawing Box ######################

# Face Dictionary
bbox_dict = {}

while True:
    frame_num = input('Input image frame name (0 is break) :')
    if frame_num == '0':
        break
    frame_num = str(1)

# frame_num = '5'

f_person = open("txt/person/" + frame_num + ".txt", "r")
f_face = open("txt/face/" + frame_num + ".txt", "r")

while True:
    line_person = f_person.readline()
    if not line_person:
        break

    line_person = line_person.split('\n')[0]
    person_ID = line_person.split(',')[0]

    # Update Crowd Dictionary
    coord = int(line_person.split(',')[1]), int(line_person.split(',')[2]), int(line_person.split(',')[3]), int(line_person.split(',')[4])
    bbox_dict[person_ID] = [coord]

key_list2 = bbox_dict.keys()

while True:
    line_face = f_face.readline()
    if not line_face:
        break

    new_line_face = line_face.split(',')[0]
    new_line_face = new_line_face.split('_')[0] + '_' + new_line_face.split('_')[1] + '.png'

    for idx in key_list2:
        if new_line_face == idx:
            line_face = line_face.split('\n')[0]

            coord = int(line_face.split(',')[1]), int(line_face.split(',')[2]), int(line_face.split(',')[3]), int(line_face.split(',')[4])
            bbox_dict[idx].append(coord)


key_list2 = bbox_dict.keys()


########### Draw Bounding Box ###########

img_path = 'input'
path_dir = os.listdir(img_path)

f = open("txt/frame/" + frame_num + ".txt", "a")

for img_name in path_dir:
    image_path = os.path.join(img_path, img_name)
    img = cv2.imread(image_path)

    # if frame is right
    if img_name.split('.')[0] == frame_num:
        for idx in key_list2:
            for idx2 in range(1, len(bbox_dict[idx])):
                face_coord = (int(bbox_dict[idx][0][0] + bbox_dict[idx][idx2][0]),
                              int(bbox_dict[idx][0][1] + bbox_dict[idx][idx2][1]),
                              int(bbox_dict[idx][0][0] + bbox_dict[idx][idx2][2]),
                              int(bbox_dict[idx][0][1] + bbox_dict[idx][idx2][3]))
                f.write(str(face_coord[0]) + ',' + str(face_coord[1]) + ',' + str(face_coord[2]) + ',' + str(face_coord[3]) + '\n')
                # cv2.rectangle(img, (face_coord[0], face_coord[1]), (face_coord[2], face_coord[3]), (0, 255, 255), 1)
                # cv2.imwrite('data/output/result_' + img_name, img)

bbox_dict.clear()

f.close()
f_person.close()
f_face.close()


################################## Nearest Bounding Box ##################################

i = 0
info = {}
matching = {}
none = {}

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union

def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h

def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


file = open("txt/frame/" + frame_num + ".txt", "r")
file_new = open("txt/add_bbox/" + frame_num + ".txt", "w")
img = cv2.imread('input/' + frame_num + '.png')
string = file.readlines()

height, width, channel = img.shape

# info.clear()
# matching.clear()
# none.clear()

for idx, value in enumerate(string):
    # print(string[idx])
    box_x1 = int(string[idx].split(',')[0])
    box_y1 = int(string[idx].split(',')[1])
    box_x2 = int(string[idx].split(',')[2])
    box_y2 = int(string[idx].split(',')[3].replace('\n', ''))
    box = (box_x1, box_y1, box_x2, box_y2)

    info[idx] = box

while True:
    if not info:
        break

    matching.clear()
    none.clear()
    num = 0
    num_matching = 0
    num_none = 0

    matching[num] = info[0]
    num_matching += 1

    for num2 in range(1, len(info)):
        prob = iou(info[num], info[num2])
        if prob >= 0.3:
            matching[num_matching] = info[num2]
            num_matching += 1
        elif prob < 0.5:
            none[num_none] = info[num2]
            num_none += 1

    x1 = width
    y1 = height
    x2 = 0
    y2 = 0

    for num3 in range(0, len(matching)):
        if int(matching[num3][0]) < x1:
            x1 = int(matching[num3][0])
        if int(matching[num3][1]) < y1:
            y1 = int(matching[num3][1])
        if int(matching[num3][2]) > x2:
            x2 = int(matching[num3][2])
        if int(matching[num3][3]) > y2:
            y2 = int(matching[num3][3])

    width_bb = x2 - x1
    height_bb = y2 - y1
    #
    side = (width_bb * (30/100)) / 2
    up = height_bb * (30/100)

    new_x1 = int(x1 - side)
    if new_x1 < 0:
        new_x1 = 0
    new_y1 = int(y1 - up)
    if new_y1 < 0:
        new_y1 = 0
    new_x2 = int(x2 + side)
    if new_x2 > width:
        new_x2 = width
    new_y2 = int(y2)

    file_new.write(str(new_x1) + ',' + str(new_y1) + ',' + str(new_x2) + ',' + str(new_y2) + '\n')

    # new info
    info = copy.deepcopy(none)

info.clear()
matching.clear()
none.clear()

file.close()
file_new.close()


################################# Face Update #################################

test_base_path = 'input'

img_path = test_base_path
path_dir = os.listdir(img_path)

# New crowd dictionary
new_crowd = {}

imgs = []
key_imgs=[]

st = time.time()

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    print(img_name)
    filepath = os.path.join(test_base_path, img_name)
    file_image = im.load_img(filepath)

    img = cv2.imread(filepath)

    new_file = open("txt/add_bbox/" + img_name.split('.')[0] + ".txt", "r")

    i = 0

    while True:
        line_new = new_file.readline()
        if not line_new:
            break

        line_new = line_new.split('\n')[0]

        coordinate = (line_new.split(',')[0], line_new.split(',')[1], line_new.split(',')[2], line_new.split(',')[3])

        croppedImage = file_image.crop((int(coordinate[0]), int(coordinate[1]), int(coordinate[2]), int(coordinate[3])))
        img_name2 = img_name.split('.')[0]
        cropped_name = '{}_{}'.format(img_name2, i)
        croppedImage.save('./data/new_face/' + cropped_name + '.png')
        i += 1
        # print('new face name :', cropped_name)

        new_crowd[cropped_name] = [coordinate,croppedImage]


    new_file.close()

imgs = np.array(imgs)
key_imgs = np.array(key_imgs)

print('Elapsed time = {}'.format(time.time() - st))


################################## Drawing Gender Bounding Box ##################################

img_path = 'input'
test_base_path = 'input'

key_list = new_crowd.keys()

print('--------------------------------')
print('Waiting Drawing Box...')

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    # print('img_name :', img_name)
    filepath = os.path.join(test_base_path, img_name)
    # file_image = image.load_img(filepath)
    img = cv2.imread(filepath)

    for idx2 in key_list:
        # print('idx2 :', idx2)
        if img_name.split('.')[0] == idx2.split('_')[0]:

            x1 = int(new_crowd[idx2][0][0])
            y1 = int(new_crowd[idx2][0][1])
            x2 = int(new_crowd[idx2][0][2])
            y2 = int(new_crowd[idx2][0][3])

            color = (0, 255, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.imwrite('output/result_' + img_name, img)


print('All Completed!')
print('Total elapsed time = {}'.format(time.time() - st_total))

