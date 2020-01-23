#!/usr/bin/env python

import numpy as np
import sys
import os
import tensorflow as tf
#from matplotlib import pyplot as plt
from skimage.color import hsv2rgb, rgb2hsv, lab2rgb, rgb2lab
#from skimage import io
from PIL import Image
from skimage.draw import line
from optparse import OptionParser
from datetime import datetime
from time import time, sleep
from multiprocessing import Pool
import numpy as np
import fcntl
import errno

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
        self.__version__ = "0.6.0.rc3"
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

        self.boxes["POT"] = group
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
        r = np.full(Nimg.shape[:2], 1, dtype=int)
        minv, maxv = [c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]]
        tmask = (Nimg>=minv)==(Nimg<=maxv)
        tmask = tmask.sum(2)==3
        r[tmask] = 0
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
        combined[combined>255] = 255
        return combined
    def mask_image(self):
        HSV, LAB = None, None
        if self.HSV['enabled']: HSV = self.HSVmask()
        if self.LAB['enabled']: LAB = self.LABmask()
        if HSV is None:
            self._mask = LAB
        elif LAB is None:
            self._mask = HSV
        else:
            self._mask = self.combinemasks(HSV, LAB)
    def fancy_overlay(self, changes=(55, -25, -25)):
        joined = self.image + changes
        self.image[self._mask==0] = joined[self._mask==0]
    def inverse_mask(self):
        self.image[self._mask!=0] = (0, 0, 0)
    def basic_mask(self, color=(0,0,0)):
        self.image[self._mask==0] = color
    def HEXtoRGB(self, hexstring):
        hexstring = hexstring.split("#")[-1]
        return (int(hexstring[0:2], 16), int(hexstring[2:4], 16), int(hexstring[4:6], 16))

    def __circular_hsv(self, mini_img, mini_mask_0, plot=True):
        mini_hsv = rgb2hsv(mini_img)
        mini_h = mini_hsv[mini_mask_0, 0]*360
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

    ## COLOR CORRECTION
    def color_correction(self):
        if self.ColorCorrect:
            if self.ColorReference not in self.boxes:
                raise Exception("Missing color reference class (No detected object of class {})".format(self.ColorReference))
            identify_codes = self.boxes.get(self.ColorReference, [])

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
    def crop_and_label_pots(self, return_crop_list=False):
        if not hasattr(self, "camera_map"):
            raise Exception("No camera map has been loaded! Unable to label objects")
        if not hasattr(self, "name_map"):
            raise Exception("No name map has been loaded! Unable to label objects")
        if self.PlantLabel not in self.boxes:
            raise Exception("No plants available to label")
        if len(self.boxes[self.PlantLabel])!=(self.nrow*self.ncol):
            print("WARNING: Missing plants! Skipping this step.")
            return None

        camera, time_base, timestamp = self._get_camera_id_and_time_stamp()

        time_dir = self._format_time(time_base, "%YY%mM%dD")

        NS = self.camera_map[camera]['NS']
        EW = self.camera_map[camera]['EW']
        orient = self.camera_map[camera]['orient']

        labels = self._get_pot_labels(NS, EW, orient)

        pots = self.boxes[self.PlantLabel]

        if return_crop_list:
            crop_list = []
            sample_list = []
        else:
            if self.measure_size[0]:
                growth_file = self.filelocking_csv_writer(os.path.join(self.measure_size[1],"database.size.csv"))
            if self.measure_greenness[0]:
                greenness_file = self.filelocking_csv_writer(os.path.join(self.measure_greenness[1],"database.greenness.csv"))

            growth_rows = []
            greenness_rows = []

        for pot, label in zip(pots, labels):
            name = self.name_map[label[0]][label[1]]
            c = self._center(pot)
            dim = self.dim
            left, right, top, bottom = self._get_region_of_center(c, dim)

            if self.substructure[0]:
                if self.substructure[1]=="Sample":
                    for active_dir in self._get_active_dirs():
                        new_dir = os.path.join(active_dir, name)
                        if not os.path.isdir(new_dir):
                            os.mkdir(new_dir)
                    base_dir = name
                if self.substructure[1]=="Time":
                    for active_dir in self._get_active_dirs():
                        new_dir = os.path.join(active_dir, time_dir)
                        #print(new_dir)
                        if not os.path.isdir(new_dir):
                            os.mkdir(new_dir)
                    base_dir = time_dir
            else:
                base_dir = ""

            output_name = os.path.join(base_dir, name+"_"+time_base)

            if return_crop_list:
                crop_list.append(np.copy(self.image[top:bottom, left:right]))
                sample_list.append([camera, timestamp, name])
            else:
                mini_img = np.copy(self.image[top:bottom, left:right])
                mini_hsv, mini_lab = None, None
                if self.HSV['enabled']: mini_hsv = self.HSVmask(img = mini_img)
                if self.LAB['enabled']: mini_lab = self.LABmask(img = mini_img)
                if mini_hsv is None:
                    mini_mask = mini_lab
                elif mini_lab is None:
                    mini_mask = mini_hsv
                else:
                    mini_mask = self.combinemasks(mini_hsv, mini_lab)
                mini_mask_0 = mini_mask==0

                if self.measure_size[0]:
                    blackpixels = np.where(mini_mask_0)
                    blackpixels = np.array(blackpixels)
                    size = str(blackpixels.shape[1])

                    growth_rows.append([name, timestamp, size])

                if self.measure_greenness[0]:
                    if mini_mask_0.sum()>(self.dim*self.dim)*0.05:
                        mean_degree, var_degree, n, plot_image = self.__circular_hsv(mini_img, mini_mask_0, plot=False)

                        greenness_rows.append([name, timestamp, str(mean_degree), str(var_degree), str(n)])

                    else:
                        greenness_rows.append([name, timestamp, "NaN"])

                if self.mask_output[0]:
                    mini_img[~mini_mask_0] = (0,0,0)
                    Image.fromarray(mini_img).save(os.path.join(self.mask_output[1],output_name+"mask.jpg"), "JPEG")
                    #io.imsave(os.path.join(self.mask_output[1],output_name+"mask.jpg"),
                    #          mini_img)

                if self.crop_output[0]:
                    Image.fromarray(self.image[top:bottom, left:right]).save(os.path.join(self.crop_output[1],output_name+".jpg"), "JPEG")
                    #io.imsave(os.path.join(self.crop_output[1],output_name+".jpg"),
                    #          self.image[top:bottom, left:right])


        if return_crop_list:
            return crop_list, sample_list
        else:
            if self.measure_size[0]:
                growth_file.write_rows(growth_rows)
                growth_file.close()
            if self.measure_greenness[0]:
                greenness_file.write_rows(greenness_rows)
                greenness_file.close()

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
        def __init__(self, filename, sep=","):
            self._file = open(filename, "w")
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
            if len(row)!=self.ncol:
                raise "Wrong dimension of row. Row has {} columns, while csv table has {} columns".format(len(row),
                                                                                                          self.ncol)
            self._file.write(self.sep.join(row))
            self._file.write("\n")
        def write(self, sep=","):
            self._file.write(sep.join(self.header))
            for row in self.rows:
                self._file.write("\n")
                self._file.write(sep.join(row))
            self._file.close()
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
        def next(self):
            data = self.readline()
            if self.pos==self._file.tell():
                raise StopIteration
            self.pos = self._file.tell()
            return data
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
            if time not in unsorted_output:
                unsorted_output[time] = {}
            unsorted_output[time][name] = measure
            samples.add(name)
        datafile.close()
        outfile = self.csv_writer(output_file)
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

        outfile_files = {'mean': self.csv_writer(output_file+".mean.csv"),
                         'var': self.csv_writer(output_file+".var.csv"),
                         'n': self.csv_writer(output_file+".n.csv")}
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
