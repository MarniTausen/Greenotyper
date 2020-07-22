#!/usr/bin/env python

import numpy as np
import sys
import os
import tensorflow as tf
#tf.get_logger().setLevel('INFO')
#from matplotlib import pyplot as plt
from skimage.color import hsv2rgb, rgb2hsv, lab2rgb, rgb2lab
from skimage.transform import resize
from skimage import io
from PIL import Image
from skimage.draw import line
from optparse import OptionParser
from datetime import datetime
from time import time, sleep
from multiprocessing import Pool
import numpy as np
import fcntl
import errno
from itertools import product
from skimage.transform import resize

## Unet imports
#from keras.models import load_model
#from keras import backend as K

Image.MAX_IMAGE_PIXELS = None

class ArgumentError(Exception):
    pass

class pipeline_settings:

    def __init__(self):
        self.initialize_mask_settings()
        self.initialize_placement_settings()
        self.initialize_identification_settings()

    def initialize_mask_settings(self):
        self.mask_settings = {}
        self.mask_settings['HSV'] = {}
        self.mask_settings['LAB'] = {}

        self.mask_settings['HSV']['enabled'] = True
        self.mask_settings['LAB']['enabled'] = True

        default_hsv = [('hue', (0.08, 0.40)),
                       ('sat', (0.2, 1)),
                       ('val', (0.2, 1))]
        default_lab = [('L', (10, 90)),
                       ('a', (-128, -4)),
                       ('b', (4, 128))]

        for key, value in default_hsv:
            self.mask_settings['HSV'][key] = value
        for key, value in default_lab:
            self.mask_settings['LAB'][key] = value
    def initialize_placement_settings(self):
        self.placement_settings = {}
        self.placement_settings['rows'] = 2
        self.placement_settings['columns'] = 5
        self.placement_settings['PlantLabel'] = 'POT'
        self.placement_settings['GroupIdentifier'] = 'QRCODE'
    def initialize_identification_settings(self):
        self.identification_settings = {}
        self.identification_settings['ColorReference'] = 'QRCODE'
        self.identification_settings['ColorCorrect'] = True
        self.identification_settings['ColorCorrectType'] = "maximum"
        self.identification_settings['dimension'] = 250
        self.identification_settings['Cameramap'] = None
        self.identification_settings['Namemap'] = None
        self.identification_settings['Network'] = None
        self.identification_settings['TimestampFormat'] = "MT%Y%m%d%H%M%S"
        self.identification_settings['TimestampOutput'] = "%Y/%m/%d - %H:%M"
        self.identification_settings['FilenamePattern'] = "{ID}_{Timestamp}_*.jpg"
    def read(self, filename):
        import ast
        input_file = open(filename)
        settings = {}
        for line in input_file:
            key, value = line.split(";")
            settings[key] = value
        if "mask_settings" in settings:
            self.mask_settings = ast.literal_eval(settings["mask_settings"])
        if "placement_settings" in settings:
            self.placement_settings = ast.literal_eval(settings["placement_settings"])
        if "identification_settings" in settings:
            self.identification_settings = ast.literal_eval(settings["identification_settings"])
        input_file.close()
    def write(self, filename):
        output_file = open(filename, "w")
        output_file.write("mask_settings;{}".format(self.mask_settings)+"\n")
        output_file.write("placement_settings;{}".format(self.placement_settings)+"\n")
        output_file.write("identification_settings;{}".format(self.identification_settings)+"\n")
        output_file.close()

class Pipeline:

    def __get_version__(self):
        self.__version__ = "0.7.0"
        return self.__version__

    ## Initialization codes and file reading
    def __init__(self, graph=None, label_file=None, pipeline=None):
        self.boxes = {}
        if not graph is None: self.load_graph(graph)
        if not label_file is None: self.read_pbtxt(label_file)
        if not pipeline is None:
            self.load_pipeline(pipeline)
        else:
            pipeline = pipeline_settings()
            self.load_pipeline(pipeline)
        self.init_output_settings()
        #self.DefaultMaskSettings()
        self.__get_version__()
    def init_output_settings(self):
        self.measure_size = (False, "")
        self.measure_greenness = (False, "")
        self.mask_output = (False, "")
        self.crop_output = (False, "")
        self.substructure = (False, "")
        self.group_identified = False
    def load_pipeline(self, pipeline):
        self.pipeline_settings = pipeline
        self.HSV = pipeline.mask_settings['HSV']
        self.LAB = pipeline.mask_settings['LAB']
        self.ncol = pipeline.placement_settings['columns']
        self.nrow = pipeline.placement_settings['rows']
        self.PlantLabel = pipeline.placement_settings['PlantLabel']
        self.GroupIdentifier = pipeline.placement_settings['GroupIdentifier']
        self.ColorReference = pipeline.identification_settings['ColorReference']
        self.ColorCorrect = pipeline.identification_settings['ColorCorrect']
        self.ColorCorrectType = pipeline.identification_settings['ColorCorrectType']
        self.dim = pipeline.identification_settings['dimension']
        self.timestamp_format = pipeline.identification_settings['TimestampFormat']
        self.timestamp_output = pipeline.identification_settings['TimestampOutput']
        if not pipeline.identification_settings['Cameramap'] is None:
            self.read_camera_map(pipeline.identification_settings['Cameramap'])
        if not pipeline.identification_settings['Namemap'] is None:
            self.read_name_map(pipeline.identification_settings['Namemap'])
        if not pipeline.identification_settings['Network'] is None:
            network_dir = pipeline.identification_settings['Network']
            if os.path.isdir(network_dir):
                files = os.listdir(network_dir)
                graph = os.path.join(network_dir,next(filter(lambda x: ".pb" in x and "txt" not in x, files)))
                label = os.path.join(network_dir,next(filter(lambda x: ".pbtxt" in x, files)))
                if hasattr(self, "loaded_graph"):
                    if graph!=self.loaded_graph:
                        self.load_graph(graph)
                else:
                    self.load_graph(graph)
                if hasattr(self, "loaded_label"):
                    if label!=self.loaded_label:
                        self.read_pbtxt(label)
                else:
                    self.read_pbtxt(label)
                self.loaded_graph = graph
                self.loaded_label = label
    def load_graph(self, graph_file):
        '''
        Read graph from <graph_file>
        Saves graph as self.detection_graph.
        '''
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            gd = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(graph_file, 'rb') as fg:
                sg = fg.read()
                gd.ParseFromString(sg)
                tf.import_graph_def(gd, name='')
        self.loaded_graph = graph_file
    def read_pbtxt(self, label_file):
        '''
        Read pbtxt file from <label_file>
        Saves pbtxt file as self.label_map
        '''
        pbtxt_file = open(label_file)
        items = pbtxt_file.read().split("item")
        pbtxt_file.close()
        self.label_map = {}
        for item in items:
            elements = item.split("\n")
            cid = None
            for element in elements:
                if "name" in element:
                    name = element.split('"')[1]
                    self.boxes[name] = list()
                if "id" in element:
                    cid = element.split(':')[-1]
                    cid = cid.replace(" ", "")
                    break
            if cid == None: continue
            self.label_map[int(cid)] = name
        self.loaded_label = label_file
    def open_image(self, image_filename):
        '''
        Open an image using PILLOW
        Image available as self.image
        Supports: jpg and png
        If png is loaded the fourth channel for alpha values is dropped.
        '''
        self._image_filename = image_filename

        self.image = np.array(Image.open(image_filename))
        if 4==self.image.shape[2]:
            self.image = np.array(self.image[:, :, :3])

        self._reset_inferred_boxes()

        self.height, self.width, _ = self.image.shape
    def _reset_inferred_boxes(self):
        '''
        Clears any inferred boxes
        '''
        keys = self.boxes.keys()
        self.boxes = {}
        for key in keys:
            self.boxes[key] = list()
    def save_image(self, image_filename):
        '''
        Save self.image as <image_filename>.jpeg
        '''
        Image.fromarray(self.image).save(image_filename, "JPEG")
    def infer_network_on_image(self):
        '''
        Applies the loaded graph on loaded image.
        The inferred bounding boxes in self.boxes
        '''
        # config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}) ## Disable gpu if running on a gpu
        with self.detection_graph.as_default():
            with tf.compat.v1.Session(graph=self.detection_graph) as sess:
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                image_np_expanded = np.expand_dims(self.image, axis=0)

                boxes, classes = sess.run(
                    [detection_boxes, detection_classes],
                    feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes) ## Location of the pots
                classes = np.squeeze(classes).astype(np.int32)

                for i, box in enumerate(boxes):
                    if sum(box)==0: break
                    top, left, bottom, right = self._transform_values(box)
                    self.boxes[self.label_map[classes[i]]].append([left, right, top, bottom])

    ## Pot identification and filteration functions
    def identify_group(self, row_threshold=300):
        if self.PlantLabel not in self.boxes:
            raise Exception("No plants to filter (No detected class of value {})".format(self.PlantLabel))
        if self.GroupIdentifier not in self.boxes:
            raise Exception("Missing group identifier (No detected class of {})".format(self.GroupIdentifier))

        Pots = self.boxes[self.PlantLabel]
        rows = self._divide_into_rows(Pots, row_threshold)
        row_groups = self._permutate_rows(rows)
        group = self._most_likely_group(row_groups)

        self.boxes[self.PlantLabel] = group

        self.group_identified = True
    def _divide_into_rows(self, Pots, row_threshold):
        rows = []
        while len(Pots)>0:
            top_left = self._center(sorted(Pots, key=lambda x: x[0]+x[2])[0])
            row, rest = list(), list()
            for pot in Pots:
                if abs(self._center(pot)[1]-top_left[1])<=row_threshold: row.append(pot)
                else: rest.append(pot)
            Pots = rest
            rows.append(row)
        return rows
    def _permutate_rows(self, rows):
        rows = [sorted(row, key=lambda x: x[0]) for row in rows]
        row_groups = list()
        for row in rows:
            i, groups = 1, [row[0:self.ncol]]
            while i+self.ncol<=len(row): groups.append(row[i:(i+self.ncol)]); i += 1
            row_groups.append(groups)
        return row_groups
    def _most_likely_group(self, row_groups):
        ## Getting all combinations
        # Dividing into groups of rows
        i, nrow_row_groups = 1, [row_groups[0:self.nrow]]
        while i+self.nrow<=len(row_groups):
            nrow_row_groups.append(row_groups[i:(i+self.nrow)])
            i += 1
        # Generating each combination of groups
        combinations = []
        group_combination = []
        for i, nrg in enumerate(nrow_row_groups):
            group_combination.append([])
            for j, rg in enumerate(nrg):
                group_combination[i].append([])
                for g in rg:
                    if (j-1)>=0:
                        for val in group_combination[i][j-1]:
                            group_combination[i][j].append(val+[g])
                    else:
                        group_combination[i][j].append([g])
        # Retreiving the combinations
        for row in group_combination:
            for combos in row[-1]:
                combinations.append(combos)

        ## Compute the most likely group
        group = []
        min_dist = float("inf")
        #print(combinations)
        identify_codes = self.boxes.get(self.GroupIdentifier, [])
        #print(identify_codes)
        for combo in combinations:
            full_region = self._combined_region(combo)
            n = sum([self._center(code) in full_region for code in identify_codes])
            if n>0:
                # Calculate distance!
                dist = 0
                for i in range(self.nrow-1):
                    dist += self._between_group_distance(combo[i], combo[i+1])
                if dist<min_dist:
                    group = []
                    for row in combo:
                        group += row
        return group
    def _combined_region(self, combination):
        minleft, maxright, mintop, maxbottom = float("inf"), float("-inf"), float("inf"), float("-inf")
        for row in combination:
            for left, right, top, bottom in row:
                if left<minleft: minleft = left
                if right>maxright: maxright = right
                if top<mintop: mintop = top
                if bottom>maxbottom: maxbottom = bottom
        return self._region(minleft, maxright, mintop, maxbottom)
    def _distance(self, p1, p2):
        return ((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5
    def _between_group_distance(self, group1, group2):
        return sum([self._distance(self._center(box1), self._center(box2))
                    for box1, box2 in zip(group1, group2)])
    def _transform_values(self, values):
        return [int(values[0]*self.height), int(values[1]*self.width),
                int(values[2]*self.height), int(values[3]*self.width)]
    def _center(self, x):
        return [(x[0]+x[1])//2, (x[2]+x[3])//2]
    class _region:
        def __init__(self, left, right, top, bottom):
            self.left = left
            self.right = right
            self.top = top
            self.bottom = bottom
        def __contains__(self, item):
            return self.left<=item[0] and self.right>=item[0] and self.top<=item[1] and self.bottom>=item[1]
        def __str__(self):
            return "REGION CLASS - left:{}, right:{}, top:{}, bottom:{}".format(self.left, self.right, self.top, self.bottom)

    ## Draw and mask options
    def draw_line(self, points, width = 1, color=(255, 0, 0)):
        last_point = points[0]
        for point in points[1:]:
            for t in range(0-width, 0+width):
                rr, cc = line(int(last_point[1]+t), int(last_point[0]+t), int(point[1]+t), int(point[0]+t))
                try: self.image[rr, cc] = color
                except: pass
            last_point = point
    def draw_bounding_boxes(self):
        for box in self.boxes["QRCODE"]:
            left, right, top, bottom = box
            self.draw_line([(left, top), (left, bottom), (right, bottom), (right, top),
                       (left, top)], width = 3, color=(255, 0, 0))
        for box in self.boxes["QRCOLOR"]:
            left, right, top, bottom = box
            self.draw_line([(left, top), (left, bottom), (right, bottom), (right, top),
                       (left, top)], width = 3, color=(0, 0, 255))
        for box in self.boxes["POT"]:
            c = self._center(box)
            dim = self.dim
            left, right, top, bottom = c[0]-dim, c[0]+dim, c[1]-dim, c[1]+dim
            self.draw_line([(left, top), (left, bottom), (right, bottom), (right, top),
                       (left, top)], width = 3, color=(0, 255, 0))
    def setHSVmasksettings(self, hue, sat, val):
        self.HSV['hue'][0], self.HSV['hue'][1] = hue[0], hue[1]
        self.HSV['sat'][0], self.HSV['sat'][1] = sat[0], sat[1]
        self.HSV['val'][0], self.HSV['val'][1] = val[0], val[1]
    def setLABmasksettings(self, L, a, b):
        self.LAB['L'][0], self.LAB['L'][1] = L[0], L[1]
        self.LAB['a'][0], self.LAB['a'][1] = a[0], a[1]
        self.LAB['b'][0], self.LAB['b'][1] = b[0], b[1]
    def _base_mask(self, converter, c1, c2, c3, img=None):
        if img is None: img = self.image
        Nimg = converter(img)
        r = np.zeros(Nimg.shape[:2], dtype=int)
        minv, maxv = [c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]]
        tmask = (Nimg>=minv)==(Nimg<=maxv)
        tmask = tmask.sum(2)==3
        r[tmask] = 1
        return r
    def HSVmask(self, img=None, hue=None, sat=None, val=None):
        if hue is None: hue = self.HSV['hue']
        if sat is None: sat = self.HSV['sat']
        if val is None: val = self.HSV['val']
        return self._base_mask(rgb2hsv, hue, sat, val, img)
    def LABmask(self, img=None, L=None, a=None, b=None):
        if L is None: L = self.LAB['L']
        if a is None: a = self.LAB['a']
        if b is None: b = self.LAB['b']
        return self._base_mask(rgb2lab, L, a, b, img)
    def combinemasks(self, mask1, mask2):
        combined = mask1+mask2
        combined[combined<2] = 0
        combined[combined==2] = 1
        return combined
    def mask_image(self, img=None, outputmask=False):
        HSV, LAB = None, None
        if self.HSV['enabled']: HSV = self.HSVmask(img)
        if self.LAB['enabled']: LAB = self.LABmask(img)
        if HSV is None:
            if outputmask:
                return LAB
            else:
                self._mask = LAB
        elif LAB is None:
            if outputmask:
                return HSV
            else:
                self._mask = HSV
        else:
            if outputmask:
                return self.combinemasks(HSV, LAB)
            else:
                self._mask = self.combinemasks(HSV, LAB)
    def fancy_overlay(self, changes=(55, -25, -25)):
        joined = self.image + changes
        self.image[self._mask==1] = joined[self._mask==1]
    def inverse_mask(self):
        self.image[self._mask==0] = (0, 0, 0)
    def basic_mask(self, color=(0,0,0)):
        self.image[self._mask==1] = color
    def HEXtoRGB(self, hexstring):
        hexstring = hexstring.split("#")[-1]
        return (int(hexstring[0:2], 16), int(hexstring[2:4], 16), int(hexstring[4:6], 16))
    def apply_masks_to_crops(self, crops):
        cshape = crops.shape
        n = cshape[0]
        predicted_masks = np.ndarray(cshape[:3])
        for i in range(n):
            mini_img = np.copy(crops[i])
            mini_hsv, mini_lab = None, None
            if self.HSV['enabled']: mini_hsv = self.HSVmask(img = mini_img)
            if self.LAB['enabled']: mini_lab = self.LABmask(img = mini_img)
            if mini_hsv is None:
                mini_mask = mini_lab
            elif mini_lab is None:
                mini_mask = mini_hsv
            else:
                mini_mask = self.combinemasks(mini_hsv, mini_lab)

            predicted_masks[i] = mini_mask

        return predicted_masks

    class Unet:
        def __init__(self, traindata, labeldata, dim=512):
            self.traindata = traindata
            self.labeldata = labeldata
            if self.validate_factor_of_2(dim):
                self.img_rows = dim
                self.img_cols = dim
            else:
                raise("Dimension provided is not a factor of 2!")
        def validate_factor_of_2(self, dim):
            temp_value = dim
            while temp_value>4:
                if temp_value / 2 % 2 != 0: return False
                temp_value = temp_value / 2
            return True
        def create_unet(self):
            inputs = tf.keras.layers.Input((self.img_rows, self.img_cols, 3))

            conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)

            conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)

            conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv3)

            conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = tf.keras.layers.Dropout(0.5)(conv4)
            pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(drop4)

            conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = tf.keras.layers.Dropout(0.5)(conv5)

            up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2,2))(drop5))
            merge6 = tf.keras.layers.concatenate([drop4, up6])
            conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2,2))(conv6))
            merge7 = tf.keras.layers.concatenate([conv3, up7])
            conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2,2))(conv7))
            merge8 = tf.keras.layers.concatenate([conv2, up8])
            conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                tf.keras.layers.UpSampling2D(size=(2,2))(conv8))
            merge9 = tf.keras.layers.concatenate([conv1, up9])
            conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

            conv10 = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

            model = tf.keras.Model(inputs=inputs, outputs=conv10)

            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy', self.recall_m, self.precision_m])

            return model
        def train(self, unet_file, validation_img=None, validation_labels=None, validation_split=0.2, epochs=20):
            self.model = self.create_unet()
            self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(unet_file, monitor='loss',
                                                                       verbose=1, save_best_only=True)
            logdir = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            if validation_img is None:
                self.model.fit(self.traindata, self.labeldata, batch_size=2, epochs=epochs,
                               validation_split=validation_split,
                               validation_freq=1, shuffle=True,
                               callbacks=[self.model_checkpoint, self.tensorboard_callback])
            else:
                self.model.fit(self.traindata, self.labeldata, batch_size=2, epochs=epochs,
                               validation_data=(validation_img, validation_labels),
                               validation_freq=1, shuffle=True,
                               callbacks=[self.model_checkpoint, self.tensorboard_callback])
        def save_model(self, filename):
            self.model.save(filename+".h5")
        def recall_m(self, y_true, y_pred):
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
            return recall
        def precision_m(self, y_true, y_pred):
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
            return precision

    def _flatten_label_image(self, labelimg):
        return labelimg.sum(2)//3
    def _normalise_label_image(self, labelimg):
        labelimg[labelimg<255] = 1
        labelimg[labelimg==255] = 0
        return labelimg
    def load_train_data(self, traindir, width=512, height=512):
        train_imgs = sorted(filter(lambda x: ".jpg" in x, os.listdir(os.path.join(traindir, "image"))))
        label_imgs = sorted(filter(lambda x: ".jpg" in x, os.listdir(os.path.join(traindir, "label"))))

        n_samples = len(train_imgs)

        traindata = np.ndarray((n_samples, width, height, 3), dtype=np.float32)
        labeldata = np.ndarray((n_samples, width, height, 1), dtype=np.float32)

        i = 0

        for trainname, labelname in zip(train_imgs, label_imgs):
            if trainname!=labelname:
                raise Exception("Train and label names do not match! Make sure all train and label names match exactly, and that there is an equal amount of images in the image and label directories.")
            trainimg = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(os.path.join(traindir, "image", trainname), grayscale=False, target_size=[512, 512]))
            labelimg = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(os.path.join(traindir, "label", labelname), grayscale=False, target_size=[512, 512]))

            labelimg = self._flatten_label_image(labelimg)
            labelimg = self._normalise_label_image(labelimg)
            labelimg = labelimg.reshape((512, 512, 1))

            traindata[i] = trainimg/255
            labeldata[i] = labelimg

            i += 1

        return traindata, labeldata, train_imgs

    def _generate_corner_crops(self,img_dim, cropsize):
        hoff, woff = img_dim[0]-cropsize[0], img_dim[1]-cropsize[1]
        h_crops = [(0,cropsize[0]),(hoff,img_dim[0])]
        w_crops = [(0,cropsize[1]),(woff,img_dim[1])]
        return list(product(*[h_crops, w_crops]))
    def augment_data(self, traindata, labeldata, Flips=True, Rotations=True, Crops=True, Cropdimension=(460,460)):
        n_samples, width, height = traindata.shape[:3]

        if Flips:
            Flips = [False, True]
        else:
            Flips = [False]

        if Rotations:
            Rotations = [0,1,2,3]
        else:
            Rotations = [0]

        if Crops:
            Crops = [((0,height),(0,width))]+self._generate_corner_crops((height,width), Cropdimension)
        else:
            Crops = [((0,height),(0,width))]

        augmentations = list(product(*[Flips, Rotations, Crops]))

        n_augments = len(augmentations)

        n_total = n_samples*n_augments

        augmented_traindata = np.ndarray((n_total, height, width, 3), dtype=np.float32)
        augmented_labeldata = np.ndarray((n_total, height, width, 1), dtype=np.float32)

        i = 0

        for j in range(n_samples):
            current_train = traindata[j]
            current_label = labeldata[j]
            for flip, rot, crop in augmentations:
                curtrain_img = current_train[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
                curtrain_img = resize(curtrain_img, (height, width, 1))

                curlabel_img = current_label[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
                curlabel_img = resize(curlabel_img, (height, width, 1))

                if flip:
                    curtrain_img = np.flip(curtrain_img, axis=(0,1))
                    curtrain_img = np.rot90(curtrain_img, rot)

                    curlabel_img = np.flip(curlabel_img, axis=(0,1))
                    curlabel_img = np.rot90(curlabel_img, rot)
                else:
                    curtrain_img = np.rot90(curtrain_img, rot)
                    curlabel_img = np.rot90(curlabel_img, rot)

                augmented_traindata[i] = curtrain_img/255
                augmented_labeldata[i] = curlabel_img

                i += 1

        return augmented_traindata, augmented_labeldata

    def _iou_score(self, target, prediction):
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        return np.sum(intersection)/np.sum(union)
    def _pixel_accuracy(self, target, prediction):
        n = target.shape[0]*target.shape[1]
        TP = np.sum(np.logical_and(target, prediction))
        union = np.sum(np.logical_or(target, prediction))
        FPFN = union - TP
        TN = n - TP - FPFN
        return (TP+TN)/n
    def _precision(self, target, prediction):
        n = target.shape[0]*target.shape[1]
        FULL_TRUTH = target.sum()
        TP = np.sum(np.logical_and(target, prediction))
        union = np.sum(np.logical_or(target, prediction))
        FP = union - FULL_TRUTH
        FN = union - TP - FP
        TN = n - TP - FP - FN
        return TP/(TP+FP)
    def _recall(self, target, prediction):
        n = target.shape[0]*target.shape[1]
        FULL_TRUTH = target.sum()
        TP = np.sum(np.logical_and(target, prediction))
        union = np.sum(np.logical_or(target, prediction))
        FP = union - FULL_TRUTH
        FN = union - TP - FP
        TN = n - TP - FP - FN
        return TP/(TP+FN)
    def _dice_coefficient(self, target, prediction):
        intersection = np.sum(np.logical_and(target, prediction))
        return 2*intersection/(target.sum()+prediction.sum())
    def _f1_score(self, target, prediction):
        precision = self._precision(target, prediction)
        recall = self._recall(target, prediction)
        return 2*(precision*recall)/(precision+recall)
    def _PASCAL_VOC_AP(self, precisions, recalls, ious, IoU=0.5):
        ## AP (Area under curve AUC)
        pascal_info = list()
        for iou, pre, rec in zip(ious, precisions, recalls):
            pascal_info.append((iou>=IoU, pre, rec))

        ## Remove False-positives based on the IoU score:
        pascal_info = filter(lambda x: x[0]==True, pascal_info)

        ## Rank values by recall
        pascal_info = sorted(pascal_info, key=lambda x: x[2])

        i = 0
        r1 = 0
        r2 = 0
        sum_list = list()
        while i<len(pascal_info):
            i += max(range(len(pascal_info[i:])), key=lambda j: pascal_info[i:][j][1])
            r2 = pascal_info[i][2]
            p = pascal_info[i][1]
            sum_list.append((r2-r1)*p)
            r1 = r2
            i += 1
        sum_list.append((1-r2)*p)

        return sum(sum_list)

    ## Unet segmentation functions
    def load_unet(self, filename):
        self.unet_model = tf.keras.models.load_model(filename, custom_objects={'recall_m': self.recall_m,
                                                                        'precision_m': self.precision_m})
    def unet_predict(self, image_data):
        if not hasattr(self, "unet_model"):
            raise Exception("No Unet has been loaded")
        predicted_masks = self.unet_model.predict(image_data, batch_size=1, verbose=1, workers=1)
        predicted_masks[predicted_masks>=0.5] = 1
        predicted_masks[predicted_masks<0.5] = 0
        return predicted_masks.astype(np.uint8)
    def unet_prepare_images(self, images):
        n = len(images)
        imagedata = np.ndarray((n, 512, 512, 3), dtype=np.float32)
        for i in range(n):
            imagedata[i] = images[i]/255
        return imagedata
    def unet_apply_on_images(self, images):
        #images, filenames = self.unet_join_image_data(image_data)

        print("Applying U-net to: {} images".format(images.shape[0]))
        #print(filenames)

        predicted_masks = self.unet_predict(images)

        print("Done Applying U-net to: {} images".format(predicted_masks.shape[0]))

        return predicted_masks
    def unet_output_data(self, images, predicted_masks, filenames):
        if not self.group_identified:
            raise Exception("Group not identfied: identify_group() has not been run beforehand! Please run identify_group() before.")

        if self.measure_size[0]:
            growth_file = self.filelocking_csv_writer(os.path.join(self.measure_size[1],"database.size.csv"))
            #growth_file.init_header(["Name", "Time", "Size"])
            growth_rows = []
        if self.measure_greenness[0]:
            greenness_file = self.filelocking_csv_writer(os.path.join(self.measure_greenness[1],"database.greenness.csv"))
            #greenness_file.init_header(["Name", "Time", "MeanHue", "VarinaceHue", "Size"])
            greenness_rows = []

        for i, output_name in enumerate(filenames):
            name = output_name.split("/")[-1].split("_")[0]
            timestamp = output_name.split("_")[-1]
            time_dir = self._format_time(timestamp, "%YY%mM%dD")
            mini_img = (images[i]*255).astype(np.uint8)
            mini_mask = predicted_masks[i].reshape((512,512))

            if self.substructure[0]:
                if self.substructure[1]=="Sample":
                    for active_dir in self._get_active_dirs():
                        new_dir = os.path.join(active_dir, name)
                        if not os.path.isdir(new_dir):
                            try:
                                os.mkdir(new_dir)
                            except:
                                pass
                    output_name = os.path.join(name,output_name)
                if self.substructure[1]=="Time":
                    for active_dir in self._get_active_dirs():
                        new_dir = os.path.join(active_dir, time_dir)
                        #print(new_dir)
                        if not os.path.isdir(new_dir):
                            try:
                                os.mkdir(new_dir)
                            except:
                                pass
                    output_name = os.path.join(time_dir,output_name)

            if self.measure_size[0]:
                size = str(int(mini_mask.sum()))
                growth_rows.append([name, timestamp, size])

            if self.crop_output[0]:
                Image.fromarray(mini_img).save(os.path.join(self.crop_output[1],output_name+".jpg"), "JPEG")

            if self.measure_greenness[0]:
                if mini_mask.sum()>(512*512)*0.02 :
                    mean_degree, var_degree, n, plot_image = self.__circular_hsv(mini_img, mini_mask, plot=False)
                    greenness_rows.append([name, timestamp, str(mean_degree), str(var_degree), str(n)])
                else:
                    greenness_rows.append([name, timestamp, "NaN"])

            if self.mask_output[0]:
                crop_img = np.copy(mini_img)
                crop_img[mini_mask==0] = (0,0,0)
                Image.fromarray(crop_img).save(os.path.join(self.mask_output[1],output_name+"mask.jpg"), "JPEG")

        if self.measure_size[0]:
            growth_file.write_rows(growth_rows)
            growth_file.close()
        if self.measure_greenness[0]:
            greenness_file.write_rows(greenness_rows)
            greenness_file.close()

    def unet_join_image_data(self, image_data):
        images_list = []
        filenames_list = []
        if image_data is None: return None, None
        for images, filenames in image_data:
            if images is None: continue
            images_list.append(images)
            filenames_list += filenames
        if len(images_list)==0: return None, None
        return np.concatenate(images_list), filenames_list
    def collect_crop_data(self, dim):
        pots = self.boxes[self.PlantLabel]
        if len(pots)<(self.nrow*self.ncol):
            return None
        images = np.ndarray((len(pots), dim, dim, 3), dtype=np.uint8)
        for i, pot in enumerate(pots):
            c = self._center(pot)
            #left, right, top, bottom = self.__get_region_of_center(c, int(dim/2))
            left, right, top, bottom = self.__get_border_aware_region(c, int(dim/2))
            images[i] = np.copy(self.image[top:bottom, left:right])
            #io.imsave("test_outputs/crops/{}.jpg".format(i), images[i])
        return images

    def recall_m(self, y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall
    def precision_m(self, y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    def __circular_hsv(self, mini_img, mini_mask, plot=True):
        mini_img[mini_mask==0] = (0,0,0)
        mini_hsv, mini_lab = None, None
        if self.HSV['enabled']: mini_hsv = self.HSVmask(img = mini_img)
        if self.LAB['enabled']: mini_lab = self.LABmask(img = mini_img)
        if mini_hsv is None:
            mini_mask = mini_lab
        elif mini_lab is None:
            mini_mask = mini_hsv
        else:
            mini_mask = self.combinemasks(mini_hsv, mini_lab)

        mini_hsv = rgb2hsv(mini_img)
        mini_h = mini_hsv[mini_mask==1, 0]*360
        mean_degree = mini_h.mean()
        var_degree = mini_h.var()
        n = len(mini_h)

        plot_image = None

        if plot:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt

            counts, bins = np.histogram(mini_h, bins=180, range=(0,360))
            frequency = (counts/float(counts.sum()))*100
            bins = np.linspace(0.0, 2 * np.pi, len(bins), endpoint=False)
            full_bin = np.copy(bins)
            full_bin = np.append(full_bin, 6.28318531)
            width = (2 * np.pi)/len(bins)
            colors = plt.cm.hsv(bins/2/np.pi)

            my_dpi = 100
            fig = plt.figure(figsize=(500/my_dpi, 500/my_dpi), dpi=my_dpi)

            ax = fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
            ax.set_ylim(0,12)
            ax.set_yticks(np.arange(0, 12,2))
            bars = ax.bar(bins[:-1], frequency, width=width, color=colors)
            ax.fill_between(full_bin, 0, 100, where=2.61799388<full_bin, facecolor="grey")
            ax.fill_between(full_bin, 0, 100, where=0.523598776>full_bin, facecolor="grey")
            #print(label)

            fig.canvas.draw()

            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            plot_image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close(fig)


        return mean_degree, var_degree, n, plot_image
    def _get_region_of_center(self, center, dim):
        return center[0]-dim, center[0]+dim, center[1]-dim, center[1]+dim
    def __get_border_aware_region(self, center, dim):
        left, right = center[0]-dim, center[0]+dim
        top, bottom = center[1]-dim, center[1]+dim
        if left<0:
            right += abs(0-left)
            left = 0
        if right>self.width:
            left -= abs(self.width-right)
            right = self.width
        if top<0:
            bottom += abs(0-top)
            top = 0
        if bottom>self.height:
            top -= abs(self.height-bottom)
            bottom = self.height
        return left, right, top, bottom

    ## COLOR CORRECTION
    def color_correction(self):
        if self.ColorCorrect:
            if self.ColorReference not in self.boxes:
                raise Exception("Missing color reference class (No detected object of class {})".format(self.ColorReference))
            identify_codes = self.boxes.get(self.ColorReference, [])

            if len(identify_codes)==0: return None

            left, right, top, bottom = identify_codes[0]

            color_correct_base = self.image[top:bottom,left:right]

            if self.ColorCorrectType=="maximum":
                self.image[..., 0] = self._imadjust(self.image[..., 0],
                                                    self._get_mean_max(color_correct_base[..., 0]))
                self.image[..., 1] = self._imadjust(self.image[..., 1],
                                                    self._get_mean_max(color_correct_base[..., 1]))
                self.image[..., 2] = self._imadjust(self.image[..., 2],
                                                    self._get_mean_max(color_correct_base[..., 2]))
            if self.ColorCorrectType=="minimum":
                self.image[..., 0] = self._imadjust(self.image[..., 0],
                                                    self._get_mean_min(color_correct_base[..., 0]))
                self.image[..., 1] = self._imadjust(self.image[..., 1],
                                                    self._get_mean_min(color_correct_base[..., 1]))
                self.image[..., 2] = self._imadjust(self.image[..., 2],
                                                    self._get_mean_min(color_correct_base[..., 2]))
            if self.ColorCorrectType=="both":
                self.image[..., 0] = self._imadjust(self.image[..., 0],
                                                    self._get_mean_min_max(color_correct_base[..., 0]))
                self.image[..., 1] = self._imadjust(self.image[..., 1],
                                                    self._get_mean_min_max(color_correct_base[..., 1]))
                self.image[..., 2] = self._imadjust(self.image[..., 2],
                                                    self._get_mean_min_max(color_correct_base[..., 2]))
    def _imadjust(self, channel, vin, vout=(0,255)):
        #assert len(channel) == 2, 'Channel must be 2-dimenisonal'

        scale = (vout[1] - vout[0]) / float(vin[1] - vin[0])
        vs = channel-vin[0]
        vs[vs<vin[0]] = 0
        vd = vs*scale+vout[0]
        vd[vd>vout[1]] = vout[1]

        return vd.astype(int)
    def _get_mean_min_max(self, channel, t=15):
        minv = np.median(channel[channel<=(channel.min()+t)])
        maxv = np.median(channel[channel>=(channel.max()-t)])
        return minv, maxv
    def _get_mean_min(self, channel, t=15):
        return np.median(channel[channel<=(channel.min()+t)]), 0
    def _get_mean_max(self, channel, t=15):
        return 0, np.median(channel[channel>=(channel.max()-t)])

    ## Pot labelling and identification
    def _read_csv(self, filename, header=True, sep=","):
        _file = open(filename)
        table = {}
        bad_characters = ["\n", " ", "\r"]
        if header:
            _header = _file.readline()
            for bad_char in bad_characters:
                _header = _header.replace(bad_char, "")
            _header = _header.split(sep)
        else:
            firstline = _file.readline().split(",")
            _header = list(range(len(firstline)))
            _file.seek(0)
        for name in _header:
            table[name] = []
        for line in _file:
            for bad_char in bad_characters:
                line = line.replace(bad_char, "")
            line = line.split(sep)
            for column, value in zip(_header, line):
                table[column].append(value)
        _file.close()
        return table
    def read_name_map(self, filename):
        table = self._read_csv(filename)
        n = len(table[list(table.keys())[0]])

        Columns = ["Name", "NS", "EW"]
        for colname in Columns:
            if colname not in table:
                errmsg = "Column '{}' missing in name map file '{}'".format(colname, filename)
                raise Exception(errmsg)

        self.name_map = {}
        for i in range(n):
            ns = int(table["NS"][i])
            ew = int(table["EW"][i])
            name = table["Name"][i]
            if ns not in self.name_map:
                self.name_map[ns] = {}
            self.name_map[ns][ew] = name
    def read_camera_map(self, filename):
        table = self._read_csv(filename)
        n = len(table[list(table.keys())[0]])

        Columns = ["Camera", "NS", "EW", "Orientation"]
        for colname in Columns:
            if colname not in table:
                errmsg = "Column '{}' missing in camera map file '{}'".format(colname, filename)
                raise Exception(errmsg)

        self.camera_map = {}
        for i in range(n):
            camera = table["Camera"][i]
            self.camera_map[camera] = {}

            self.camera_map[camera]["NS"] = [int(ns) for ns in table["NS"][i].split("-")]
            self.camera_map[camera]["EW"] = [int(ew) for ew in table["EW"][i].split("-")]

            self.camera_map[camera]["orient"] = table["Orientation"][i]
    def prettifyTime(self, time_stamp):
        string = time_stamp.split(".")[0]
        return datetime.strptime(string, self.timestamp_format).strftime(self.timestamp_output)
    def _format_time(self, string, outformat):
        return datetime.strptime(string, self.timestamp_format).strftime(outformat)
    def _get_pot_labels(self, NS, EW, orientatation):

        labels = []

        if orientatation=="E":
            for ew in range(EW[0], EW[1]+1):
                for ns in range(NS[0], NS[1]+1):
                    labels.append([ns, ew])
        if orientatation=="W":
            for ew in range(EW[1], EW[0]-1, -1):
                for ns in range(NS[1], NS[0]-1, -1):
                    labels.append([ns, ew])
        if orientatation=="N":
            for ns in range(NS[1], NS[0]-1, -1):
                for ew in range(EW[0], EW[1]+1):
                    labels.append([ns, ew])
        if orientatation=="S":
            for ns in range(NS[0], NS[1]+1):
                for ew in range(EW[1], EW[0]-1, -1):
                    labels.append([ns, ew])

        return(labels)
    class _interval:
        def __init__(self, start, stop):
            self.start = start
            self.stop = stop
        def __contains__(self, item):
            return self.start<=item and item<=self.stop
        def __getitem__(self, item):
            if item==0: return self.start
            if item==1: return self.stop
            return None
    def _get_camera_id_and_time_stamp(self):
        fn_pattern = self.pipeline_settings.identification_settings['FilenamePattern']
        ID_search = "{ID}"
        Timestamp_search = "{Timestamp}"
        ID_index_start = fn_pattern.index(ID_search)
        Timestamp_index_start = fn_pattern.index(Timestamp_search)
        ID_region = self._interval(ID_index_start,
                                   ID_index_start+len(ID_search)-1)
        Timestamp_region = self._interval(Timestamp_index_start,
                            Timestamp_index_start+len(Timestamp_search)-1)
        if ID_region[0]==0:
            if fn_pattern[ID_region[1]+1]=="{" and ID_region[1]+1 in Timestamp_region:
                print("REPORT ERROR! No distingushing feature at end")
            ID_start_end_char = ("", fn_pattern[ID_region[1]+1])
        elif ID_region[1]==len(fn_pattern):
            if fn_pattern[ID_region[0]-1]=="}" and ID_region[0]-1 in Timestamp_region:
                print("REPORT ERROR! No distingushing feature at start")
            ID_start_end_char = (fn_pattern[ID_region[0]-1], "")
        else:
            ID_start_end_char = (fn_pattern[ID_region[0]-1], fn_pattern[ID_region[1]+1])

        if Timestamp_region[0]==0:
            if fn_pattern[Timestamp_region[1]+1]=="{" and Timestamp_region[1]+1 in ID_region:
                print("REPORT ERROR! No distingushing feature at end")
            Timestamp_start_end_char = ("", fn_pattern[Timestamp_region[1]+1])
        elif Timestamp_region[1]==len(fn_pattern):
            if fn_pattern[Timestamp_region[0]-1]=="}" and Timestamp_region[0]-1 in ID_region:
                print("REPORT ERROR! No distingushing feature at start")
            Timestamp_start_end_char = (fn_pattern[Timestamp_region[0]-1], "")
        else:
            Timestamp_start_end_char = (fn_pattern[Timestamp_region[0]-1], fn_pattern[Timestamp_region[1]+1])

        if ID_index_start<Timestamp_index_start: id_first = True
        else: id_first = False

        directory = "/".join(self._image_filename.split("/")[:-1])
        basename = self._image_filename.split("/")[-1]

        if id_first:
            id_base, basename = basename.split(ID_start_end_char[1], 1)
            if ID_start_end_char[0]!="":
                id_base = id_base.split(ID_start_end_char[0])[-1]

            time_base, basename = basename.split(Timestamp_start_end_char[1], 1)
            if Timestamp_start_end_char[0]!="":
                time_base = time_base.split(Timestamp_start_end_char[0])[-1]

        else:
            time_base, basename = basename.split(Timestamp_start_end_char[1], 1)
            if Timestamp_start_end_char[0]!="":
                time_base = time_base.split(Timestamp_start_end_char[0])[-1]

            id_base, basename = basename.split(ID_start_end_char[1], 1)
            if ID_start_end_char[0]!="":
                id_base = id_base.split(ID_start_end_char[0])[-1]

        return id_base, time_base, self.prettifyTime(time_base)
    def _get_active_dirs(self):
        active_dirs = []
        if self.mask_output[0]: active_dirs.append(self.mask_output[1])
        #if self.measure_size[0]: active_dirs.append(self.measure_size[1])
        if self.crop_output[0]: active_dirs.append(self.crop_output[1])
        #if self.measure_greenness[0]: active_dirs.append(self.measure_greenness[1])
        return active_dirs
    def get_filename_labels(self):
        camera, time_base, timestamp = self._get_camera_id_and_time_stamp()

        time_dir = self._format_time(time_base, "%YY%mM%dD")

        NS = self.camera_map[camera]['NS']
        EW = self.camera_map[camera]['EW']
        orient = self.camera_map[camera]['orient']

        labels = self._get_pot_labels(NS, EW, orient)

        filename_labels = []
        for label in labels:
            name = self.name_map[label[0]][label[1]]

            filename_labels.append(name+"_"+time_base)

        return filename_labels
    def crop_and_label_pots(self, return_crop_list=False, return_greenness_figures=False):
        if not hasattr(self, "camera_map"):
            raise Exception("No camera map has been loaded! Unable to label objects")
        if not hasattr(self, "name_map"):
            raise Exception("No name map has been loaded! Unable to label objects")
        if self.PlantLabel not in self.boxes:
            raise Exception("No plants available to label")
        if len(self.boxes[self.PlantLabel])!=(self.nrow*self.ncol):
            return("WARNING: Missing plants! Cannot output for image: {}".format(self._image_filename))
        if not self.group_identified:
            raise Exception("Group not identfied: identify_group() has not been run beforehand! Please run identify_group() before.")

        camera, time_base, timestamp = self._get_camera_id_and_time_stamp()

        time_dir = self._format_time(time_base, "%YY%mM%dD")

        NS = self.camera_map[camera]['NS']
        EW = self.camera_map[camera]['EW']
        orient = self.camera_map[camera]['orient']

        labels = self._get_pot_labels(NS, EW, orient)

        pots = self.boxes[self.PlantLabel]

        crops = self.collect_crop_data(self.dim*2)
        predicted_masks = self.apply_masks_to_crops(crops)

        if return_crop_list:
            crop_list = []
            sample_list = []
            if return_greenness_figures:
                circular_hist_list = []
        else:
            if self.measure_size[0]:
                growth_file = self.filelocking_csv_writer(os.path.join(self.measure_size[1],"database.size.csv"))
            if self.measure_greenness[0]:
                greenness_file = self.filelocking_csv_writer(os.path.join(self.measure_greenness[1],"database.greenness.csv"))

            growth_rows = []
            greenness_rows = []

        for i, label in enumerate(labels):
            name = self.name_map[label[0]][label[1]]

            if self.substructure[0]:
                if self.substructure[1]=="Sample":
                    for active_dir in self._get_active_dirs():
                        new_dir = os.path.join(active_dir, name)
                        if not os.path.isdir(new_dir):
                            try:
                                os.mkdir(new_dir)
                            except:
                                pass
                    base_dir = name
                if self.substructure[1]=="Time":
                    for active_dir in self._get_active_dirs():
                        new_dir = os.path.join(active_dir, time_dir)
                        #print(new_dir)
                        if not os.path.isdir(new_dir):
                            try:
                                os.mkdir(new_dir)
                            except:
                                pass
                    base_dir = time_dir
            else:
                base_dir = ""

            output_name = os.path.join(base_dir, name+"_"+time_base)

            if return_crop_list:
                crop_list.append(np.copy(crops[i]))
                sample_list.append([camera, timestamp, name])
                if return_greenness_figures:
                    mini_img = crops[i]
                    mini_mask = predicted_masks[i].reshape((self.dim*2, self.dim*2))

                    mean_degree, var_degree, n, plot_image = self.__circular_hsv(mini_img, mini_mask, plot=True)
                    circular_hist_list.append(plot_image)
            else:
                mini_img = np.copy(crops[i])
                mini_mask = predicted_masks[i].reshape((self.dim*2, self.dim*2))

                if self.measure_size[0]:
                    size = str(int(mini_mask.sum()))
                    growth_rows.append([name, timestamp, size])

                if self.measure_greenness[0]:
                    if mini_mask.sum()>(self.dim*self.dim)*0.05:
                        mean_degree, var_degree, n, plot_image = self.__circular_hsv(mini_img, mini_mask, plot=False)

                        greenness_rows.append([name, timestamp, str(mean_degree), str(var_degree), str(n)])
                    else:
                        greenness_rows.append([name, timestamp, "NaN"])

                if self.crop_output[0]:
                    Image.fromarray(crops[i]).save(os.path.join(self.crop_output[1],output_name+".jpg"), "JPEG")

                if self.mask_output[0]:
                    crop_img = np.copy(crops[i])
                    crop_img[mini_mask==0] = (0,0,0)
                    Image.fromarray(crop_img).save(os.path.join(self.mask_output[1],output_name+"mask.jpg"), "JPEG")
                    #io.imsave(os.path.join(self.mask_output[1],output_name+"mask.jpg"),
                    #          mini_img)




        if return_crop_list:
            if return_greenness_figures:
                return crop_list, sample_list, circular_hist_list
            return crop_list, sample_list
        else:
            if self.measure_size[0]:
                growth_file.write_rows(growth_rows)
                growth_file.close()
            if self.measure_greenness[0]:
                greenness_file.write_rows(greenness_rows)
                greenness_file.close()
    def scan_directory(self, directory):
        items = os.listdir(directory)
        files = []
        for item in items:
            if '.'==item[0]: continue
            item_fp = os.path.join(directory, item)
            if os.path.isfile(item_fp):
                files.append(item_fp)
            else:
                internal_items = self.scan_directory(item_fp)
                for in_item in internal_items:
                    files.append(in_item)
        return files

    class filelocking_csv_writer:
        def __init__(self, filename, sep=","):
            self._file = open(filename, "a", buffering=1)
            self.sep = sep
        def write_rows(self, rows):
            while True:
                try:
                    fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError as e:
                    # raise on unrelated IOErrors
                    if e.errno != errno.EAGAIN:
                        raise
                    else:
                        sleep(0.01)
            #fcntl.flock(self._file, fcntl.LOCK_EX)
            for row in rows:
                self._file.write(self.sep.join(row))
                self._file.write("\n")
            self._file.flush()
            os.fsync(self._file.fileno())
            fcntl.flock(self._file, fcntl.LOCK_UN)
        def close(self):
            self._file.close()
    class csv_writer:
        def __init__(self, filename, sep=",", mode="a"):
            self._file = open(filename, mode, buffering=1)
            self.rows = []
            self.sep = sep
        def init_header(self, header):
            self.header = [str(col) for col in header]
            self.ncol = len(self.header)
        def write_header(self):
            self._file.write(self.sep.join(self.header))
            self._file.write("\n")
        def add_row(self,row):
            if len(row)!=self.ncol:
                raise "Wrong dimension of row. Row has {} columns, while csv table has {} columns".format(len(row),
                                                                                                          self.ncol)
            self.rows.append([str(col) for col in row])
        def write_row(self, row):
            self._file.write(self.sep.join(row))
            self._file.write("\n")
        def close(self):
            self._file.close()

    class simple_csv_reader:
        def __init__(self, filename, sep=","):
            self._file = open(filename)
            self.sep = sep
            self.pos = self._file.tell()
        def readline(self):
            return self._file.readline()[:-1].split(self.sep)
        def __iter__(self):
            return self
        def __next__(self):
            data = self.readline()
            if self.pos==self._file.tell():
                raise StopIteration
            self.pos = self._file.tell()
            return data
        def close(self):
            self._file.close()
    def organize_output(self, filename, output_file):
        datafile = self.simple_csv_reader(filename)
        unsorted_output = {}
        samples = set()
        for line in datafile:
            if len(line)!=3: continue
            name, time, measure = line
            try:
                time = self.prettifyTime(time)
            except:
                pass
            if time not in unsorted_output:
                unsorted_output[time] = {}
            unsorted_output[time][name] = measure
            samples.add(name)
        datafile.close()
        outfile = self.csv_writer(output_file, mode="w")
        time_series = sorted(unsorted_output.keys())
        samples = sorted(list(samples))
        outfile.init_header(["Time"]+samples)
        outfile.write_header()

        for time_point in time_series:
            row = [time_point]
            for sample in samples:
                if sample in unsorted_output[time_point]:
                    row.append(unsorted_output[time_point][sample])
                else:
                    row.append("NA")
            outfile.write_row(row)
            del unsorted_output[time_point]

    def greenness_output(self, filename, output_file):
        datafile = self.simple_csv_reader(filename)
        base_stats = {}
        times = set()
        for line in datafile:
            if len(line)!=5: continue
            name, time, mean, var, n = line
            try:
                time = self.prettifyTime(time)
            except:
                pass
            if name not in base_stats:
                base_stats[name] = {}
            base_stats[name][time] = {'mean': float(mean),
                                      'var': float(var),
                                      'n': int(n)}
            times.add(time)
        times = sorted(list(times))
        samples = list(base_stats.keys())
        #print(times)
        #print(samples)

        outfile_files = {'mean': self.csv_writer(output_file+".mean.csv", mode="w"),
                         'var': self.csv_writer(output_file+".var.csv", mode="w"),
                         'n': self.csv_writer(output_file+".n.csv", mode="w")}
        for name in outfile_files:
             outfile_files[name].init_header(["Time"]+samples)
             outfile_files[name].write_header()

        out_names = ['mean', 'var', 'n']

        for time in times:
            row = {'mean': [time],
                    'var': [time],
                    'n': [time]}
            for sample in samples:
                if time in base_stats[sample]:
                    for name in out_names:
                        row[name].append(str(base_stats[sample][time][name]))
                else:
                    for name in out_names:
                        row[name].append('NA')
            for name in out_names:
                outfile_files[name].write_row(row[name])

        for name in out_names:
            outfile_files[name].close()
