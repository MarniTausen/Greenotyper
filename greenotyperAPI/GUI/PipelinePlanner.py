#!/usr/bin/env python

from sys import argv, exit
from PyQt5.QtWidgets import (QApplication, QLabel, QDialog, QMainWindow,
                             QGridLayout, QWidget, QHBoxLayout, QPushButton,
                             QFileDialog, QSlider, QMessageBox, QVBoxLayout,
                             QGroupBox, QCheckBox, QSpinBox, QRadioButton,
                             QLineEdit, QSizePolicy, QAction, QStatusBar,
                             QMenuBar, QTabWidget)
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import (pyqtSlot, Qt, QTimer, QObject, QRunnable,
                          QThreadPool, pyqtSignal)
from skimage import io
from greenotyperAPI import GREENOTYPER
import greenotyperAPI
from os import listdir, path, getcwd
from greenotyperAPI.GUI.qrangeslider import QRangeSlider
import traceback
#from multiprocessing import SimpleQueue

#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#import matplotlib.pyplot as plt

from time import time

class ImageLabel(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.p = None
        #self.setMinimumSize(300,200)
    def setPixmap(self, p):
        self.p = p
        #self.aspect = self.p.width()/self.p.height()
        #self.setMinimumSize(200,200/self.aspect)
        self.update()
    def paintEvent(self, event):
        if self.p:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            crect = self.rect()
            rendering = self.p.scaled(crect.size(),
                                      Qt.KeepAspectRatio, Qt.FastTransformation)
            painter.drawPixmap(rendering.rect(), self.p)

class PipelinePlanner(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        self.GUI = GUI(self)
        self.setCentralWidget(self.GUI)
        self.init_menu_bar()

    def init_menu_bar(self):

        #self.status = self.statusBar()
        #self.status.setWindowTitle("GREENOTYPER")
        #self.parent.setStatusBar(self.status)

        QuitAction = QAction("Exit", self)
        QuitAction.triggered.connect(exit)

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(QuitAction)

class GUI(QWidget):

    def __init__(self, parent):
        QWidget.__init__(self)
        self.parent = parent
        self.imglabel = ImageLabel(self)
        self.imglabel.setAutoFillBackground(True)
        #self.imglabel.setScaledContents(1)
        self.process_text = QLabel()
        #self.process_text.setMinimumHeight(25)
        self.process_text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.imglayout = QVBoxLayout()
        self.imglayout.addWidget(self.imglabel, 1)
        self.imglayout.addWidget(self.process_text, 0)

        self.PL = GREENOTYPER.Pipeline()
        self.PS = GREENOTYPER.pipeline_settings()

        self.threadpool = QThreadPool()
        self.masking_is_running = False
        self.network_is_running = False
        self.running_process_waring = {"mask": "Apply mask process already running!",
                                       "network": "Find Plants is already running!"}

        self.detected = False

        self.init_top_button_bar()
        self.init_sliders()
        self.init_mask_control_panel()
        self.init_placement_control_panel()

        self.make_right_bar()

        self.init_placement_control_panel()
        self.init_network_control_panel()
        self.init_identification_panel()

        self.make_left_bar()

        mainLayout = QGridLayout()
        mainLayout.addLayout(self.topbar, 0, 1)
        mainLayout.addLayout(self.imglayout, 1, 1, 2, 1)
        mainLayout.addLayout(self.rightbar, 0, 2, 2, 2)
        mainLayout.addLayout(self.leftbar, 0, 0, 2, 1)

        self.setLayout(mainLayout)

        self.parent.setWindowTitle("Greenotyper Pipeline Planner (v{})".format(self.PL.__version__))

    def init_top_button_bar(self):
        self.openfile = QPushButton("Open Image")
        self.openfile.clicked.connect(self.openImage)
        self.opennetwork = QPushButton("Open Network")
        self.opennetwork.clicked.connect(self.openNetwork)
        self.apply_mask = QPushButton("Apply Mask")
        self.apply_mask.setDisabled(True)
        self.apply_mask.clicked.connect(self.ImageMask)
        self.find_plants = QPushButton("Find Plants")
        self.find_plants.setDisabled(True)
        self.find_plants.clicked.connect(self.FindPlants)
        self.openpipeline = QPushButton("Open Pipeline")
        self.openpipeline.clicked.connect(self.openPipeline)
        self.exportpipeline = QPushButton("Export Pipeline")
        self.exportpipeline.clicked.connect(self.exportPipeline)
        #apply_mask.clicked.connect(self.ImageMask)

        self.topbar = QHBoxLayout()
        self.topbar.addWidget(self.openfile)
        self.topbar.addWidget(self.apply_mask)
        self.topbar.addWidget(self.opennetwork)
        self.topbar.addWidget(self.find_plants)
        self.topbar.addWidget(self.openpipeline)
        self.topbar.addWidget(self.exportpipeline)
        self.topbar.addStretch(1)
    def init_sliders(self):
        self.hsv_label = QCheckBox("HSV mask")
        self.hsv_label.setMinimumSize(230,0)
        self.hsv_label.setChecked(True)

        self.h_label = QLabel()
        self.h_label.setText("Hue setting")
        self.h_slider = QRangeSlider()
        self.h_slider.setMin(0)
        self.h_slider.setMax(360)
        self.h_slider.setRange(0, 360) ## Automatically update

        self.s_label = QLabel()
        self.s_label.setText("Saturation setting")
        self.s_slider = QRangeSlider()
        self.s_slider.setMin(0)
        self.s_slider.setMax(100)
        self.s_slider.setRange(0, 100)

        self.v_label = QLabel()
        self.v_label.setText("Value setting")
        self.v_slider = QRangeSlider()
        self.v_slider.setMin(0)
        self.v_slider.setMax(100)
        self.v_slider.setRange(0, 100)

        self.lab_label = QCheckBox("LAB mask")
        #self.lab_label.setText("LAB mask")
        self.lab_label.setChecked(True)

        self.l_label = QLabel()
        self.l_label.setText("Luminous setting")
        self.l_slider = QRangeSlider()
        self.l_slider.setMin(0)
        self.l_slider.setMax(100)
        self.l_slider.setRange(0, 100)

        self.a_label = QLabel()
        self.a_label.setText("a (green-red) color")
        self.a_slider = QRangeSlider()
        self.a_slider.setMin(-128)
        self.a_slider.setMax(128)
        self.a_slider.setRange(-128, 128)

        self.b_label = QLabel()
        self.b_label.setText("b (blue-yellow) color")
        self.b_slider = QRangeSlider()
        self.b_slider.setMin(-128)
        self.b_slider.setMax(128)
        self.b_slider.setRange(-128, 128)
    def init_mask_control_panel(self):
        self.reset_settings = QPushButton("Reset settings")
        self.reset_settings.clicked.connect(self.setDefaultValues)

        self.reset_image = QPushButton("Reset image")
        self.reset_image.clicked.connect(self.reload_image)

        self.reset_buttons = QHBoxLayout()
        self.reset_buttons.addWidget(self.reset_settings)
        self.reset_buttons.addWidget(self.reset_image)

        self.fancy_overlay = QRadioButton("Overlay mask")
        self.crop_mask = QRadioButton("Inverse mask")
        self.negative_mask = QRadioButton("Basic mask")
        self.fancy_overlay.setChecked(True)

        self.overlay_input = QLineEdit()
        self.overlay_input.setText("70,-30,90")
        self.overlay_input.setMaxLength(14)

        self.overlay_mask_group = QHBoxLayout()
        self.overlay_mask_group.addWidget(self.fancy_overlay)
        self.overlay_mask_group.addWidget(self.overlay_input)

        self.color_input = QLineEdit()
        self.color_input.setText("#22CC22")
        self.color_input.setMaxLength(7)

        self.basic_mask_group = QHBoxLayout()
        self.basic_mask_group.addWidget(self.negative_mask)
        self.basic_mask_group.addWidget(self.color_input)

        self.pot_filteration = QCheckBox("Pot filteration")
        self.pot_filteration.setChecked(True)
        self.pot_filteration.setSizePolicy(QSizePolicy.Fixed,
                                           QSizePolicy.Fixed)

        self.draw_bounding_boxes = QCheckBox("Draw bounding boxes")
        self.draw_bounding_boxes.setChecked(True)
        self.draw_bounding_boxes.setSizePolicy(QSizePolicy.Fixed,
                                               QSizePolicy.Fixed)

        self.color_correct_check = QCheckBox("")
        self.color_correct_check.setChecked(True)
        self.color_correct_check.setSizePolicy(QSizePolicy.Fixed,
                                               QSizePolicy.Fixed)

        self.color_correction = QPushButton("Run color correction")
        self.color_correction.clicked.connect(self.ColorCorrect)

        self.color_correct = QHBoxLayout()
        self.color_correct.addWidget(self.color_correct_check)
        self.color_correct.addWidget(self.color_correction)

        self.save_image = QPushButton("Save image")
        self.save_image.clicked.connect(self.saveImage)
    def make_right_bar(self):
        self.rightbar = QVBoxLayout()
        self.rightbar.addWidget(self.hsv_label)
        self.rightbar.addWidget(self.h_label)
        self.rightbar.addWidget(self.h_slider)
        self.rightbar.addWidget(self.s_label)
        self.rightbar.addWidget(self.s_slider)
        self.rightbar.addWidget(self.v_label)
        self.rightbar.addWidget(self.v_slider)

        self.rightbar.addWidget(self.lab_label)
        self.rightbar.addWidget(self.l_label)
        self.rightbar.addWidget(self.l_slider)
        self.rightbar.addWidget(self.a_label)
        self.rightbar.addWidget(self.a_slider)
        self.rightbar.addWidget(self.b_label)
        self.rightbar.addWidget(self.b_slider)

        #self.rightbar.addWidget(self.reset_settings)
        #self.rightbar.addWidget(self.reset_image)
        self.rightbar.addLayout(self.reset_buttons)

        self.rightbar.addLayout(self.overlay_mask_group)
        self.rightbar.addWidget(self.crop_mask)
        self.rightbar.addLayout(self.basic_mask_group)
        self.rightbar.addWidget(self.pot_filteration)
        self.rightbar.addWidget(self.draw_bounding_boxes)
        self.rightbar.addLayout(self.color_correct)
        self.rightbar.addWidget(self.save_image)

        #self.rightbar.addWidget(self.placement_label)
        #self.rightbar.addLayout(self.labels)
        #self.rightbar.addLayout(self.spinboxes)
        #self.rightbar.addWidget(self.figcanvas)

        self.rightbar.addStretch(1)

    def init_placement_control_panel(self):
        #self.placement_label = QLabel()
        #self.placement_label.setText("Placement setup")
        self.placement_group = QWidget()


        self.column_label = QLabel()
        self.column_label.setText("Number of Columns")
        #self.column_label.setSizePolicy(QSizePolicy.Fixed,
        #                                QSizePolicy.Fixed)

        self.row_label = QLabel()
        self.row_label.setText("Number of Rows")
        #self.row_label.setSizePolicy(QSizePolicy.Fixed,
        #                             QSizePolicy.Fixed)

        self.labels = QHBoxLayout()
        self.labels.addWidget(self.column_label)
        self.labels.addWidget(self.row_label)

        self.columns = QSpinBox()
        self.columns.setValue(5)
        self.columns.setRange(1, 40)

        self.rows = QSpinBox()
        self.rows.setValue(2)
        self.rows.setRange(1, 40)

        self.spinboxes = QHBoxLayout()
        self.spinboxes.addWidget(self.columns)
        self.spinboxes.addWidget(self.rows)

        self.filename_pattern_text = QLabel()
        self.filename_pattern_text.setText("Filename pattern")
        self.filename_pattern_text.setSizePolicy(QSizePolicy.Fixed,
                                                 QSizePolicy.Fixed)


        self.filename_pattern = QLineEdit()
        #self.filename_pattern.setText("{ID}_{Timestamp}_*")
        self.filename_pattern.setSizePolicy(QSizePolicy.Fixed,
                                           QSizePolicy.Fixed)
        self.filename_pattern.setMaximumSize(260,30)

        self.filename_pattern_group = QHBoxLayout()
        self.filename_pattern_group.addWidget(self.filename_pattern_text)
        self.filename_pattern_group.addWidget(self.filename_pattern)

        self.time_stamp_text = QLabel()
        self.time_stamp_text.setText("Timestamp format")

        self.time_stamp_input = QLineEdit()
        #self.time_stamp_input.setText("MT%Y%m%d%H%M%S")
        self.time_stamp_input.setSizePolicy(QSizePolicy.Fixed,
                                            QSizePolicy.Fixed)


        self.time_out_text = QLabel()
        self.time_out_text.setText("Time output")

        self.time_stamp_output = QLineEdit()
        #self.time_stamp_output.setText("%Y/%m/%d - %H:%M")
        self.time_stamp_output.setSizePolicy(QSizePolicy.Fixed,
                                             QSizePolicy.Fixed)

        self.time_labels = QHBoxLayout()
        self.time_labels.addWidget(self.time_stamp_text)
        self.time_labels.addWidget(self.time_out_text)

        self.time_inputs = QHBoxLayout()
        self.time_inputs.addWidget(self.time_stamp_input)
        self.time_inputs.addWidget(self.time_stamp_output)

        bg_color = "#ffffff"
        margin = "2.5px"
        br_color = "#000000"
        br_width = "1px"
        css_style = "padding:{};".format(margin)
        css_style += "background-color:{};".format(bg_color)
        css_style += "border: {} solid {};".format(br_width, br_color)

        self.cammap_button = QPushButton("Camera map")
        self.cammap_button.setSizePolicy(QSizePolicy.Fixed,
                                         QSizePolicy.Fixed)
        self.cammap_button.clicked.connect(self.OpenCameraMap)
        self.cammap_button.setMinimumWidth(120)
        self.cammap_label = QLabel()
        self.cammap_label.setText("No file chosen")
        self.cammap_label.setStyleSheet(css_style)
        self.cammap_label.setMaximumWidth(140)
        self.cammap_group = QHBoxLayout()
        self.cammap_group.addWidget(self.cammap_button)
        self.cammap_group.addWidget(self.cammap_label)

        self.idmap_button = QPushButton("ID map")
        self.idmap_button.setSizePolicy(QSizePolicy.Fixed,
                                        QSizePolicy.Fixed)
        self.idmap_button.clicked.connect(self.OpenIDMap)
        self.idmap_button.setMinimumWidth(120)
        self.idmap_label = QLabel()
        self.idmap_label.setText("No file chosen")
        self.idmap_label.setStyleSheet(css_style)
        self.idmap_label.setMaximumWidth(140)
        self.idmap_group = QHBoxLayout()
        self.idmap_group.addWidget(self.idmap_button)
        self.idmap_group.addWidget(self.idmap_label)


        self.placement_layout = QVBoxLayout()
        #self.placement_layout.addWidget(self.placement_label)
        self.placement_layout.addLayout(self.labels)
        self.placement_layout.addLayout(self.spinboxes)

        self.placement_layout.addWidget(self.filename_pattern_text)
        self.placement_layout.addWidget(self.filename_pattern)
        self.placement_layout.addLayout(self.time_labels)
        self.placement_layout.addLayout(self.time_inputs)
        self.placement_layout.addLayout(self.cammap_group)
        self.placement_layout.addLayout(self.idmap_group)

        self.placement_group.setLayout(self.placement_layout)
    def init_network_control_panel(self):
        self.network_control_panel = QWidget()

        self.plant_class_label = QLabel("Plant Class")
        self.plant_class_text = QLineEdit()
        self.plant_class_text.setSizePolicy(QSizePolicy.Fixed,
                                            QSizePolicy.Fixed)
        self.plant_class_text.setMaximumSize(120,30)

        self.group_identifier = QLabel("Group identifier")
        self.group_identifier_text = QLineEdit()
        self.group_identifier_text.setSizePolicy(QSizePolicy.Fixed,
                                                 QSizePolicy.Fixed)
        self.group_identifier_text.setMaximumSize(120,20)

        self.color_reference = QLabel("Color reference")
        self.color_reference_text = QLineEdit()
        self.color_reference_text.setSizePolicy(QSizePolicy.Fixed,
                                                QSizePolicy.Fixed)
        self.color_reference_text.setMaximumSize(160,20)

        self.class_labels = QHBoxLayout()
        self.class_labels.addWidget(self.plant_class_label)
        self.class_labels.addWidget(self.group_identifier)

        self.class_text_boxes = QHBoxLayout()
        self.class_text_boxes.addWidget(self.plant_class_text)
        self.class_text_boxes.addWidget(self.group_identifier_text)

        self.color_reference_layout = QHBoxLayout()
        self.color_reference_layout.addWidget(self.color_reference)
        self.color_reference_layout.addWidget(self.color_reference_text)

        self.white_balancing = QRadioButton("White balancing")
        self.black_balancing = QRadioButton("Black balancing")
        self.full_colorcorrect = QRadioButton("Full color correction")
        self.full_colorcorrect.setChecked(True)

        self.network_control_layout = QVBoxLayout()
        self.network_control_layout.addLayout(self.class_labels)
        self.network_control_layout.addLayout(self.class_text_boxes)
        self.network_control_layout.addLayout(self.color_reference_layout)
        self.network_control_layout.addWidget(self.white_balancing)
        self.network_control_layout.addWidget(self.black_balancing)
        self.network_control_layout.addWidget(self.full_colorcorrect)

        self.network_control_layout.addStretch(1)

        self.network_control_panel.setLayout(self.network_control_layout)

    def init_identification_panel(self):
        self.identification_label = QLabel()
        self.identification_label.setText("Identification setup")

        bg_color = "#ffffff"
        margin = "2.5px"
        br_color = "#000000"
        br_width = "1px"
        css_style = "padding:{};".format(margin)
        css_style += "background-color:{};".format(bg_color)
        css_style += "border: {} solid {};".format(br_width, br_color)

        self.crop_and_label_pots = QPushButton("Test crop and label pots")
        self.crop_and_label_pots.clicked.connect(self.TestCrop)


        self.dimension_text = QLabel()
        self.dimension_text.setText("Dimension size x2")

        self.dimension_size = QLineEdit()
        #elf.dimension_size.setText("200")
        self.onlyInt = QtGui.QIntValidator()
        self.dimension_size.setValidator(self.onlyInt)
        self.dimension_size.setSizePolicy(QSizePolicy.Fixed,
                                          QSizePolicy.Fixed)

        self.dimensions = QHBoxLayout()
        self.dimensions.addWidget(self.dimension_text)
        self.dimensions.addWidget(self.dimension_size)

        self.croplabel = ImageLabel(self)
        self.croplabel.setAutoFillBackground(True)
        self.croplabel.setMaximumSize(180,180)
        self.croplabel.setMinimumSize(180,180)
        self.croplabel.setStyleSheet("background-color:#000000;")

        self.previous = QPushButton("Previous")
        self.previous.setDisabled(True)
        self.previous.clicked.connect(self.PreviousCrop)
        #self.previous.setSizePolicy(QSizePolicy.Fixed,
        #                            QSizePolicy.Fixed)
        self.next = QPushButton("Next")
        self.next.setDisabled(True)
        self.next.clicked.connect(self.NextCrop)
        #self.next.setSizePolicy(QSizePolicy.Fixed,
        #                        QSizePolicy.Fixed)

        self.prevnext = QHBoxLayout()
        self.prevnext.addWidget(self.previous)
        self.prevnext.addWidget(self.next)

        self.crop_info = QLabel()
        self.crop_info.setStyleSheet(css_style)
    def make_left_bar(self):
        self.leftbar = QVBoxLayout()

        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(self.placement_group, "Placement setup")
        self.left_tabs.addTab(self.network_control_panel, "Network settings")
        self.left_tabs.setSizePolicy(QSizePolicy.Fixed,
                                     QSizePolicy.Fixed)

        self.leftbar.addWidget(self.left_tabs)
        self.leftbar.addWidget(self.crop_and_label_pots)
        self.leftbar.addLayout(self.dimensions)
        self.leftbar.addWidget(self.croplabel)
        self.leftbar.addLayout(self.prevnext)
        self.leftbar.addWidget(self.crop_info)

        self.leftbar.addStretch(1)

    def setDefaultValues(self):
        HSV = self.PS.mask_settings['HSV']
        LAB = self.PS.mask_settings['LAB']
        self.h_slider.setRange(int(HSV['hue'][0]*360), int(HSV['hue'][1]*360))
        self.s_slider.setRange(int(HSV['sat'][0]*100), int(HSV['sat'][1]*100))
        self.v_slider.setRange(int(HSV['val'][0]*100), int(HSV['val'][1]*100))
        self.l_slider.setRange(LAB['L'][0], LAB['L'][1])
        self.a_slider.setRange(LAB['a'][0], LAB['a'][1])
        self.b_slider.setRange(LAB['b'][0], LAB['b'][1])

        self.h_slider.update()
        self.s_slider.update()
        self.v_slider.update()
        self.l_slider.update()
        self.a_slider.update()
        self.b_slider.update()

        self.hsv_label.setChecked(HSV['enabled'])
        self.lab_label.setChecked(LAB['enabled'])

        self.color_correct_check.setChecked(self.PS.identification_settings['ColorCorrect'])

        self.rows.setValue(self.PS.placement_settings['rows'])
        self.columns.setValue(self.PS.placement_settings['columns'])
        self.dimension_size.setText(str(self.PS.identification_settings['dimension']))

        self.filename_pattern.setText(self.PS.identification_settings['FilenamePattern'])

        self.time_stamp_input.setText(self.PS.identification_settings['TimestampFormat'])
        self.time_stamp_output.setText(self.PS.identification_settings['TimestampOutput'])

        if not self.PS.identification_settings['Cameramap'] is None:
            fileName = self.PS.identification_settings['Cameramap']
            wd = getcwd()
            self.cammap_label.setText("./"+path.relpath(fileName, wd))
            #self.PL.read_camera_map(fileName)
        if not self.PS.identification_settings['Namemap'] is None:
            fileName = self.PS.identification_settings['Namemap']
            wd = getcwd()
            self.idmap_label.setText("./"+path.relpath(fileName, wd))
            #self.PL.read_name_map(fileName)
        if not self.PS.identification_settings['Network'] is None:
            self.find_plants.setDisabled(False)

        self.color_correct_check.setChecked(self.PS.identification_settings['ColorCorrect'])

        cctype = self.PS.identification_settings['ColorCorrectType']
        if cctype=="maximum": self.white_balancing.setChecked(True)
        if cctype=="minimum": self.black_balancing.setChecked(True)
        if cctype=="both": self.full_colorcorrect.setChecked(True)

        self.plant_class_text.setText(self.PS.placement_settings['PlantLabel'])
        self.group_identifier_text.setText(self.PS.placement_settings['GroupIdentifier'])
        self.color_reference_text.setText(self.PS.identification_settings['ColorReference'])

    def reload_image(self):
        if self.network_is_running or self.masking_is_running:
            if self.masking_is_running: warningmsg = self.running_process_waring["mask"]
            if self.network_is_running: warningmsg = self.running_process_waring["network"]
            QMessageBox.question(self, '', warningmsg,
                                 QMessageBox.Ok, QMessageBox.Ok)
            return None
        if hasattr(self.PL, "image"):
            self.read_image(self._prev_filename)
            self.__onload_image(self.PL.image)
        else:
            QMessageBox.question(self, '', "No image is loaded",
                                 QMessageBox.Ok, QMessageBox.Ok)
    def read_image(self, filename):
        self._prev_filename = filename
        self.PL.open_image(filename)
        self.masked = False
        self.detected = False
    def __onload_image(self, IMG):
        height, width, channel = IMG.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(IMG.data, width, height, bytesPerLine,
                            QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.imglabel.setPixmap(pixmap)
        self.imglabel.repaint()
        self.imglabel.update()

    @pyqtSlot()
    def openImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "",
                                                  "Imagefiles (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if fileName:
            self.read_image(fileName)
            self.__onload_image(self.PL.image)
            self.apply_mask.setDisabled(False)
    @pyqtSlot()
    def openNetwork(self):
        #options = QFileDialog.Options()
        fileName = str(QFileDialog.getExistingDirectory(self, "Select Network Directory", "Network Directory"))
        if fileName:
            path = fileName+"/"
            files = listdir(fileName)
            pb = filter(lambda x: ".pb" in x and "txt" not in x, files)
            pbtxt = filter(lambda x: ".pbtxt" in x, files)
            self.PL.load_graph(path+next(pb))
            self.PL.read_pbtxt(path+next(pbtxt))
            self.find_plants.setDisabled(False)
            self.PS.identification_settings['Network'] = path
    @pyqtSlot()
    def saveImage(self):
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getOpenFileName()", "",
                                                  "Imagefiles (*.png *.jpg *.jpeg);;All Files (*)")
        if fileName:
            self.PL.save_image(fileName)

    @pyqtSlot()
    def openPipeline(self):
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "",
                                                  "Pipeline file (*.pipeline);;All Files (*)")
        if fileName:
            try:
                self.PS.read(fileName)
                #self.PL.load_pipeline(self.PS)
                self.setDefaultValues()
            except:
                QMessageBox.about(self, "openPipeline", "Error in the format of {}".format(fileName))
    @pyqtSlot()
    def exportPipeline(self):
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getOpenFileName()", "",
                                                  "Pipeline file (*.pipeline);;All Files (*)")
        if fileName:
            self.UpdatePipelineSettings()
            self.PS.write(fileName)
    def UpdatePipelineSettings(self):
        self.PS.mask_settings['HSV']['hue'] = list(self.normalize_range(self.h_slider.getRange(), 360))
        self.PS.mask_settings['HSV']['sat'] = list(self.normalize_range(self.s_slider.getRange(), 100))
        self.PS.mask_settings['HSV']['val'] = list(self.normalize_range(self.v_slider.getRange(), 100))

        self.PS.mask_settings['LAB']['L'] = list(self.l_slider.getRange())
        self.PS.mask_settings['LAB']['a'] = list(self.a_slider.getRange())
        self.PS.mask_settings['LAB']['b'] = list(self.b_slider.getRange())

        self.PS.mask_settings['HSV']['enabled'] = self.hsv_label.isChecked()
        self.PS.mask_settings['LAB']['enabled'] = self.lab_label.isChecked()

        self.PS.identification_settings['ColorCorrect'] = self.color_correct_check.isChecked()
        self.PS.identification_settings['ColorReference'] = self.color_reference_text.text()
        if self.white_balancing.isChecked():
            self.PS.identification_settings['ColorCorrectType'] = "maximum"
        if self.black_balancing.isChecked():
            self.PS.identification_settings['ColorCorrectType'] = "minimum"
        if self.full_colorcorrect.isChecked():
            self.PS.identification_settings['ColorCorrectType'] = "both"


        self.PS.placement_settings['rows'] = self.rows.value()
        self.PS.placement_settings['columns'] = self.columns.value()
        self.PS.identification_settings['dimension'] = int(self.dimension_size.text())

        self.PS.identification_settings['TimestampFormat'] = self.time_stamp_input.text()
        self.PS.identification_settings['TimestampOutput'] = self.time_stamp_output.text()

        self.PS.identification_settings['FilenamePattern'] = self.filename_pattern.text()

        self.PS.placement_settings['PlantLabel'] = self.plant_class_text.text()
        self.PS.placement_settings['GroupIdentifier'] = self.group_identifier_text.text()

    def _mask_process(self, progress_callback):
        progress_callback.emit("producing mask (Apply Mask)")
        self.UpdateMaskSettings()

        self.PL.mask_image()

        progress_callback.emit("mask done, applying overlay (Apply Mask)")

        if self.fancy_overlay.isChecked():
            overlay_settings = self.overlay_input.text()
            failed = False
            try:
                overlay_settings = [int(i) for i in overlay_settings.split(",")]
            except:
                QMessageBox.about(self, "Overlay mask fail",
                                  "Overlay input is formatted incorrect!\nMust be format: (-)R,(-)G,(-)B with optional negative markings.")
                failed = True
            if not failed:
                if len(overlay_settings)!=3:
                    QMessageBox.about(self, "Overlay mask fail",
                                      "Overlay input is formatted incorrect!\nMust be format: (-)R,(-)G,(-)B with optional negative markings.")
                else:
                    self.PL.fancy_overlay(overlay_settings)
        if self.crop_mask.isChecked():
            self.PL.inverse_mask()
        if self.negative_mask.isChecked():
            self.PL.basic_mask(self.PL.HEXtoRGB(self.color_input.text()))
        progress_callback.emit("process complete! (Apply Mask)")
    def _mask_update(self):
        self.__onload_image(self.PL.image)
        self.masked = True
        self.masking_is_running = False
    def _mask_error(self, error):
        exctype, value, errormsg = error
        print(exctype)
        print(value)
        print(errormsg)
        QMessageBox.about(self, "Error in Masking", str(value))
        self.network_is_running = False
        self.process_text.setText("Process failed!(Apply Mask)")
    @pyqtSlot()
    def ImageMask(self, multithread=True):
        if hasattr(self.PL, "image"):
            if self.masking_is_running or self.network_is_running:
                if self.masking_is_running: warningmsg = self.running_process_waring["mask"]
                if self.network_is_running: warningmsg = self.running_process_waring["network"]
                QMessageBox.question(self, '', warningmsg,
                                QMessageBox.Ok, QMessageBox.Ok)
                return None
            if self.masked:
                self.read_image(self._prev_filename)
                self.__onload_image(self.PL.image)
                self.detected = False
                self.masked = False
            if self.hsv_label.isChecked()==False and self.lab_label.isChecked()==False:
                 QMessageBox.question(self, '', "No mask type is checked!\nUnable to produce mask",
                                 QMessageBox.Ok, QMessageBox.Ok)
            else:
                self.masking_is_running = True
                if multithread:
                    worker = greenotyperAPI.GUI.PipelineRunner.Worker(self._mask_process)
                    self.threadpool.start(worker)
                    worker.signals.progress.connect(self._write_progress)
                    worker.signals.error.connect(self._mask_error)
                    worker.signals.finished.connect(self._mask_update)
                else:
                    empty_signal = greenotyperAPI.GUI.PipelineRunner.WorkerSignals()
                    self._mask_process(empty_signal.progress)
                    self._mask_update()
        else:
            QMessageBox.question(self, '', "No image is loaded",
                                 QMessageBox.Ok, QMessageBox.Ok)
    def normalize_range(self, in_range, max_val):
        return in_range[0]/float(max_val), in_range[1]/float(max_val)
    def UpdateMaskSettings(self):
        self.UpdatePipelineSettings()
        self.PL.load_pipeline(self.PS)
    def _network_process(self, progress_callback):
        progress_callback.emit("finding plants (Find Plants)")
        self.UpdateMaskSettings()
        self.PL.infer_network_on_image()
        if self.pot_filteration.isChecked():
            progress_callback.emit("running pot filteration (Find Plants)")
            self.PL.identify_group()
        if self.color_correct_check.isChecked():
            progress_callback.emit("applying color correction (Find Plants)")
            self.PL.color_correction()
        if self.draw_bounding_boxes.isChecked():
            progress_callback.emit("drawing bounding boxes (Find Plants)")
            self.PL.draw_bounding_boxes()
            self.__onload_image(self.PL.image)
        progress_callback.emit("process complete! (Find Plants)")
        self.detected = True
        self.network_is_running = False
    def _network_error(self, error):
        exctype, value, errormsg = error
        print(exctype)
        print(value)
        print(errormsg)
        QMessageBox.about(self, "Error in Filteration", str(value))
        self.network_is_running = False
        self.process_text.setText("Process failed!(Find Plants)")
    def _write_progress(self, value):
        self.process_text.setText(value)
    @pyqtSlot()
    def FindPlants(self, multithread=True):
        if hasattr(self.PL, "image"):
            if self.network_is_running or self.masking_is_running:
                if self.masking_is_running: warningmsg = self.running_process_waring["mask"]
                if self.network_is_running: warningmsg = self.running_process_waring["network"]
                QMessageBox.question(self, '', warningmsg,
                                     QMessageBox.Ok, QMessageBox.Ok)
                return None
            self.network_is_running = True
            if multithread:
                worker = greenotyperAPI.GUI.PipelineRunner.Worker(self._network_process)
                worker.signals.error.connect(self._network_error)
                worker.signals.progress.connect(self._write_progress)
                self.threadpool.start(worker)
            else:
                empty_signal = greenotyperAPI.GUI.PipelineRunner.WorkerSignals()
                self._network_process(empty_signal.progress)
        else:
            QMessageBox.question(self, '', "No image is loaded",
                                 QMessageBox.Ok, QMessageBox.Ok)
    @pyqtSlot()
    def ColorCorrect(self):
        if hasattr(self.PL, "image"):
            if self.detected:
                self.UpdatePipelineSettings()
                self.PL.load_pipeline(self.PS)
                self.PL.color_correction()
                self.__onload_image(self.PL.image)
            else:
                QMessageBox.question(self, '', "No detection has been run or identifying class not found!",
                                     QMessageBox.Ok, QMessageBox.Ok)
            pass
        else:
            QMessageBox.question(self, '', "No image is loaded",
                                 QMessageBox.Ok, QMessageBox.Ok)

    @pyqtSlot()
    def OpenCameraMap(self):
        ## Test if
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "",
                                                  "CSV (*.csv);;All Files (*)")
        if fileName:
            wd = getcwd()
            self.cammap_label.setText("./"+path.relpath(fileName, wd))
            self.PL.read_camera_map(fileName)
            self.PS.identification_settings["Cameramap"] = fileName
    @pyqtSlot()
    def OpenIDMap(self):
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "",
                                                  "CSV (*.csv);;All Files (*)")
        if fileName:
            wd = getcwd()
            self.idmap_label.setText("./"+path.relpath(fileName, wd))
            self.PL.read_name_map(fileName)
            self.PS.identification_settings["Namemap"] = fileName
    @pyqtSlot()
    def TestCrop(self):
        failed = False
        if not hasattr(self.PL, "detection_graph"):
            failed = True
            QMessageBox.question(self, '', "No network loaded!",
                                 QMessageBox.Ok, QMessageBox.Ok)
        if not self.detected:
            failed = True
            QMessageBox.question(self, '', "No plants have been detected! Try running Find Plants.",
                                 QMessageBox.Ok, QMessageBox.Ok)
        if not hasattr(self.PL, "camera_map"):
            failed = True
            QMessageBox.question(self, '', "No camera map loaded!",
                                 QMessageBox.Ok, QMessageBox.Ok)
        if not hasattr(self.PL, "name_map"):
            failed = True
            QMessageBox.question(self, '', "No id map loaded!",
                                 QMessageBox.Ok, QMessageBox.Ok)
        if not len(self.PL.boxes[self.PS.placement_settings['PlantLabel']])>0:
            failed = True
            QMessageBox.question(self, '', "No plants have been detected! Try running Find Plants.",
                                 QMessageBox.Ok, QMessageBox.Ok)
        ## TEST IF FILENAME PATTERN COUPLING IS MATCHING

        if not failed:
            self.crop_list, self.sample_list = self.PL.crop_and_label_pots(return_crop_list=True)
            self.current_crop = 0
            self.max_crop = len(self.crop_list)-1
            self.next.setDisabled(False)
            self.previous.setDisabled(True)
            self.__set_crop_image(self.crop_list[self.current_crop])
            print(self.sample_list[self.current_crop])
            self.crop_info.setText(" ".join(self.sample_list[self.current_crop]))
    def __set_crop_image(self, IMG):
        height, width, channel = IMG.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(IMG.data, width, height, bytesPerLine,
                            QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.croplabel.setPixmap(pixmap)
    def NextCrop(self):
        self.current_crop += 1
        if self.current_crop==self.max_crop:
            self.next.setDisabled(True)
        if self.current_crop>0:
            self.previous.setDisabled(False)
        self.__set_crop_image(self.crop_list[self.current_crop])
        self.crop_info.setText(" ".join(self.sample_list[self.current_crop]))
    def PreviousCrop(self):
        self.current_crop -= 1
        if self.current_crop==0:
            self.previous.setDisabled(True)
        if self.current_crop<self.max_crop:
            self.next.setDisabled(False)
        self.__set_crop_image(self.crop_list[self.current_crop])
        self.crop_info.setText(" ".join(self.sample_list[self.current_crop]))

# if __name__=="__main__":
#
#     PF = GREENOTYPER.Pipeline()
#
#     app = QApplication([])
#     app.setApplicationName("GREENOTYPER (v{})".format(PF.__version__))
#     scriptDir = path.dirname(path.realpath(__file__))
#     app.setWindowIcon(QtGui.QIcon(scriptDir + path.sep + 'icon/icon.png'))
#     gui = GUI()
#     gui.show()
#
#     exit(app.exec_())
