from PyQt5.QtWidgets import (QApplication, QLabel, QDialog, QMainWindow,
                             QGridLayout, QWidget, QHBoxLayout, QPushButton,
                             QFileDialog, QSlider, QMessageBox, QVBoxLayout,
                             QGroupBox, QCheckBox, QSpinBox, QRadioButton,
                             QLineEdit, QSizePolicy, QProgressBar, QTextEdit)
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import (pyqtSlot, Qt, QTimer, QObject, QRunnable,
                          QThreadPool, pyqtSignal)
from greenotyperAPI import *
from greenotyperAPI.GUI import PipelinePlanner
import os
import multiprocessing as mp
import traceback, sys
import time
import datetime

def _single_process(progress_callback, image, PS, Interface):
    print("Making process")

    start = time.time()

    PL = GRAPE.Pipeline()
    PL.load_pipeline(PS)

    Interface.setOutputSettings(PL)

    print("Reading image!")
    PL.open_image(image)
    print("Searching for plants")
    PL.infer_network_on_image()
    print("Applying color correction")
    PL.color_correction()
    print("Identifying group")
    PL.identify_group()
    print("Cropping and labelling pots")
    PL.crop_and_label_pots()

    return time.time() - start

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(str)
class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):

        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()

class PipelineRunner(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.PL = GREENOTYPER.Pipeline()
        self.PS = GREENOTYPER.pipeline_settings()

        self.PipelinePlannerApp = PipelinePlanner.PipelinePlanner()
        #self.PipelinePlannerApp = PipelinePlanner.GUI()
        #self.LabelImgApp = MainWindow()

        self.threadpool = QThreadPool()

        self.init_dirs()
        self.wd = os.getcwd()

        self.init_pipeline_buttons()
        self.init_other_apps()
        self.init_main_body()
        self.init_settings_bar()
        self.init_commandline_bar()

        mainLayout = QGridLayout()
        mainLayout.addLayout(self.pipeline_buttons, 0, 0)
        mainLayout.addLayout(self.other_apps, 0, 1)
        mainLayout.addLayout(self.main_body, 1, 0, 2, 2)
        mainLayout.addLayout(self.settings_bar, 0, 2, 3, 3)
        mainLayout.addLayout(self.commandline_group, 3, 0, 3, 0)

        self.setLayout(mainLayout)

        self.setWindowTitle("Greenotyper (v{})".format(self.PL.__version__))

    def init_dirs(self):
        self.pipeline_file = None
        self.inputdir = None
        self.maskdir = None
        self.cropdir = None
        self.sizedir = None
        self.greennessdir = None

    def init_pipeline_buttons(self):
        self.openpipeline = QPushButton("Open Pipeline")
        self.openpipeline.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.openpipeline.clicked.connect(self.openPipeline)
        self.openpipeline.setMinimumWidth(150)

        self.testpipeline = QPushButton("Test Pipeline")
        self.testpipeline.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.testpipeline.setMinimumWidth(150)
        self.testpipeline.clicked.connect(self.testPipeline)

        self.runpipeline = QPushButton("Run pipeline")
        self.runpipeline.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.runpipeline.setMinimumWidth(150)
        self.runpipeline.clicked.connect(self.runPipeline)

        self.pipeline_buttons = QVBoxLayout()
        self.pipeline_buttons.addWidget(self.openpipeline)
        self.pipeline_buttons.addWidget(self.testpipeline)
        self.pipeline_buttons.addWidget(self.runpipeline)
        self.pipeline_buttons.addStretch(1)
    def init_other_apps(self):
        self.other_apps_label = QLabel()
        self.other_apps_label.setText("Additonal Applications")
        self.PipelinePlannerButton = QPushButton("Pipeline Planner")
        self.PipelinePlannerButton.clicked.connect(self.OpenPipelinePlanner)
        #self.PipelinePlannerButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.PipelinePlannerButton.setMinimumWidth(170)

        self.LabelImgButton = QPushButton("LabelImg")
        #self.LabelImgButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.LabelImgButton.setMinimumWidth(170)
        self.LabelImgButton.setDisabled(True)
        #self.LabelImgButton.clicked.connect(self.OpenLabelImg)

        self.other_apps = QVBoxLayout()
        self.other_apps.addWidget(self.other_apps_label)
        self.other_apps.addWidget(self.PipelinePlannerButton)
        self.other_apps.addWidget(self.LabelImgButton)
        self.other_apps.addStretch(1)
    def init_main_body(self):
        bg_color = "#ffffff"
        margin = "2.5px"
        br_color = "#000000"
        br_width = "1px"
        css_style = "padding:{};".format(margin)
        css_style += "background-color:{};".format(bg_color)
        css_style += "border: {} solid {};".format(br_width, br_color)

        button_size = 160

        self.pipeline_label = QLabel()
        self.pipeline_label.setText("Pipeline:")
        self.pipeline_label.setSizePolicy(QSizePolicy.Fixed,
                                         QSizePolicy.Fixed)
        self.pipeline_file_label = QLabel()
        self.pipeline_file_label.setText("Default Pipeline")
        self.pipeline_file_label.setStyleSheet(css_style)
        self.pipeline_labels = QHBoxLayout()
        self.pipeline_labels.addWidget(self.pipeline_label)
        self.pipeline_labels.addWidget(self.pipeline_file_label)

        self.inputdirectory_button = QPushButton("Input file directory")
        self.inputdirectory_button.setSizePolicy(QSizePolicy.Fixed,
                                                 QSizePolicy.Fixed)
        self.inputdirectory_button.clicked.connect(self.openInputDirectory)
        self.inputdirectory_button.setMinimumWidth(button_size+15)
        self.inputdir_label = QLabel()
        self.inputdir_label.setText("No directory chosen")
        self.inputdir_label.setStyleSheet(css_style)
        self.input_group = QHBoxLayout()
        self.input_group.addWidget(self.inputdirectory_button)
        self.input_group.addWidget(self.inputdir_label)
        #self.input_group.addStretch(1)

        self.mask_check = QCheckBox("")
        self.mask_check.setChecked(False)
        self.mask_check.setDisabled(True)
        self.mask_check.setSizePolicy(QSizePolicy.Fixed,
                                      QSizePolicy.Fixed)
        self.maskoutput_button = QPushButton("Mask output")
        self.maskoutput_button.setSizePolicy(QSizePolicy.Fixed,
                                             QSizePolicy.Fixed)
        self.maskoutput_button.setMinimumWidth(button_size)
        self.maskoutput_button.clicked.connect(self.openMaskDirectory)
        self.maskout_label = QLabel()
        self.maskout_label.setText("No directory chosen")
        self.maskout_label.setStyleSheet(css_style)
        self.mask_group = QHBoxLayout()
        self.mask_group.addWidget(self.mask_check)
        self.mask_group.addWidget(self.maskoutput_button)
        self.mask_group.addWidget(self.maskout_label)

        self.crop_check = QCheckBox("")
        self.crop_check.setChecked(False)
        self.crop_check.setDisabled(True)
        self.crop_check.setSizePolicy(QSizePolicy.Fixed,
                                      QSizePolicy.Fixed)
        self.cropoutput_button = QPushButton("Crop output")
        #self.cropoutput_button.setWidth()
        self.cropoutput_button.setSizePolicy(QSizePolicy.Fixed,
                                             QSizePolicy.Fixed)
        self.cropoutput_button.clicked.connect(self.openCropDirectory)
        self.cropoutput_button.setMinimumWidth(button_size)
        self.cropout_label = QLabel()
        self.cropout_label.setText("No directory chosen")
        self.cropout_label.setStyleSheet(css_style)
        self.crop_group = QHBoxLayout()
        self.crop_group.addWidget(self.crop_check)
        self.crop_group.addWidget(self.cropoutput_button)
        self.crop_group.addWidget(self.cropout_label)

        self.size_check = QCheckBox("")
        self.size_check.setChecked(False)
        self.size_check.setDisabled(True)
        self.size_check.setSizePolicy(QSizePolicy.Fixed,
                                        QSizePolicy.Fixed)
        self.sizeoutput_button = QPushButton("Size output")
        self.sizeoutput_button.setSizePolicy(QSizePolicy.Fixed,
                                               QSizePolicy.Fixed)
        self.sizeoutput_button.clicked.connect(self.openSizeDirectory)
        self.sizeoutput_button.setMinimumWidth(button_size)
        self.sizeout_label = QLabel()
        self.sizeout_label.setText("No directory chosen")
        self.sizeout_label.setStyleSheet(css_style)
        self.size_group = QHBoxLayout()
        self.size_group.addWidget(self.size_check)
        self.size_group.addWidget(self.sizeoutput_button)
        self.size_group.addWidget(self.sizeout_label)


        self.greenness_check = QCheckBox("")
        self.greenness_check.setChecked(False)
        self.greenness_check.setDisabled(True)
        self.greenness_check.setSizePolicy(QSizePolicy.Fixed,
                                        QSizePolicy.Fixed)
        self.greennessoutput_button = QPushButton("Greenness output")
        self.greennessoutput_button.setSizePolicy(QSizePolicy.Fixed,
                                               QSizePolicy.Fixed)
        self.greennessoutput_button.clicked.connect(self.openGreennessDirectory

                                                    )
        self.greennessoutput_button.setMinimumWidth(button_size)
        self.greennessout_label = QLabel()
        self.greennessout_label.setText("No directory chosen")
        self.greennessout_label.setStyleSheet(css_style)
        self.greenness_group = QHBoxLayout()
        self.greenness_group.addWidget(self.greenness_check)
        self.greenness_group.addWidget(self.greennessoutput_button)
        self.greenness_group.addWidget(self.greennessout_label)


        self.progress = QProgressBar()
        self.progress.setGeometry(0,0,150,40)
        self.progress.setMaximum(1)
        self.progress.setValue(1)

        self.time_left = QLabel()
        self.time_left.setText("--h--m--s")

        self.progress_estimator = QHBoxLayout()
        self.progress_estimator.addWidget(self.progress)
        self.progress_estimator.addWidget(self.time_left)

        self.main_body = QVBoxLayout()
        self.main_body.addLayout(self.pipeline_labels)
        self.main_body.addLayout(self.input_group)
        self.main_body.addLayout(self.mask_group)
        self.main_body.addLayout(self.crop_group)
        self.main_body.addLayout(self.size_group)
        self.main_body.addLayout(self.greenness_group)
        self.main_body.addLayout(self.progress_estimator)
        self.main_body.addStretch(1)
    def init_settings_bar(self):
        self.multicore_label = QLabel()
        self.multicore_label.setText("Number of CPU")

        self.multicore = QLineEdit()
        self.multicore.setText(str(mp.cpu_count()))
        self.onlyInt = QtGui.QIntValidator()
        self.multicore.setValidator(self.onlyInt)
        self.multicore.setSizePolicy(QSizePolicy.Fixed,
                                     QSizePolicy.Fixed)
        self.multicore.setMaximumWidth(40)
        self.multicore.setMaxLength(4)

        self.multicore_group = QHBoxLayout()
        self.multicore_group.addWidget(self.multicore_label)
        self.multicore_group.addWidget(self.multicore)
        self.multicore_group.addStretch(1)

        self.output_organisation = QGroupBox("Output organistation")

        self.no_subfolders = QRadioButton("No subfolders")
        self.divide_by_day = QRadioButton("Divide by day")
        self.divide_by_individual = QRadioButton("Divide by individual")

        self.no_subfolders.setChecked(True)

        self.division_options = QVBoxLayout()
        self.division_options.addWidget(self.no_subfolders)
        self.division_options.addWidget(self.divide_by_day)
        self.division_options.addWidget(self.divide_by_individual)

        self.output_organisation.setLayout(self.division_options)

        self.test_options = QGroupBox("Testing options")

        self.test_images_label = QLabel()
        self.test_images_label.setText("Test images")

        self.test_images = QLineEdit()
        self.test_images.setText(str(10))
        self.test_images.setValidator(self.onlyInt)
        self.test_images.setSizePolicy(QSizePolicy.Fixed,
                                       QSizePolicy.Fixed)
        self.test_images.setMaximumWidth(60)
        self.test_images.setMaxLength(6)

        self.test_images_group = QHBoxLayout()
        self.test_images_group.addWidget(self.test_images_label)
        self.test_images_group.addWidget(self.test_images)
        self.test_images_group.addStretch(1)

        self.test_id_label = QLabel()
        self.test_id_label.setText("Test from")
        self.test_id = QLineEdit()
        self.test_id.setMaximumWidth(75)

        self.test_id_layout = QHBoxLayout()
        self.test_id_layout.addWidget(self.test_id_label)
        self.test_id_layout.addWidget(self.test_id)
        self.test_id_layout.addStretch(1)

        self.test_options_layout = QVBoxLayout()
        self.test_options_layout.addLayout(self.test_images_group)
        self.test_options_layout.addLayout(self.test_id_layout)

        self.test_options.setLayout(self.test_options_layout)

        self.runtime_output = QGroupBox("Commandline runtime options")

        self.relative_paths = QCheckBox("Use relative paths")
        self.relative_paths.setChecked(False)

        self.use_bash = QRadioButton("Use bash")
        self.use_gwf = QRadioButton("Use gwf")

        self.use_bash.setChecked(True)

        self.use_cases = QHBoxLayout()
        self.use_cases.addWidget(self.use_bash)
        self.use_cases.addWidget(self.use_gwf)

        #self.environment_label = QLabel()
        #self.environment_label.setText("Environment")

        textbox_height = 25
        textbox_width = 170

        self.environment_textbox = QTextEdit()
        self.environment_default = "Environment loading - Example using conda: conda activate greenotyperenv"
        self.environment_textbox.setText(self.environment_default)
        self.environment_textbox.setMaximumHeight(textbox_height)
        self.environment_textbox.setMaximumWidth(textbox_width)

        self.cluster_submit = QTextEdit()
        self.cluster_default = 'Cluster submit command, replaces {command} with the greenotyper command, example srun "{command}"'
        self.cluster_submit.setText(self.cluster_default)
        self.cluster_submit.setMaximumHeight(textbox_height)
        self.cluster_submit.setMaximumWidth(textbox_width)

        self.cluster_button = QPushButton("output workflow script")
        self.cluster_button.clicked.connect(self.saveWorkflow)

        self.runtime_output_layout = QVBoxLayout()
        self.runtime_output_layout.addWidget(self.relative_paths)
        self.runtime_output_layout.addLayout(self.use_cases)
        self.runtime_output_layout.addWidget(self.environment_textbox)
        self.runtime_output_layout.addWidget(self.cluster_submit)
        self.runtime_output_layout.addWidget(self.cluster_button)

        self.runtime_output.setLayout(self.runtime_output_layout)


        self.settings_bar = QVBoxLayout()
        self.settings_bar.addLayout(self.multicore_group)
        self.settings_bar.addWidget(self.output_organisation)
        self.settings_bar.addWidget(self.test_options)
        self.settings_bar.addWidget(self.runtime_output)

        self.settings_bar.addStretch(1)
    def init_commandline_bar(self):
        bg_color = "#222222"
        margin = "2.5px"
        color = "#ffffff"
        css_style = "padding:{};".format(margin)
        css_style += "background-color:{};".format(bg_color)
        css_style += "color:{};".format(color)

        self.commandbox = QTextEdit()
        self.commandbox.setMaximumHeight(50)
        self.commandbox.setText("No inputs defined")
        self.commandbox.setStyleSheet(css_style)

        self.system_clipboard = QApplication.clipboard()

        self.update_command_line = QPushButton("Update Commandline")
        self.update_command_line.clicked.connect(self.updateCommandline)

        self.copyToClipboard = QPushButton("Copy to Clipboard")
        self.copyToClipboard.clicked.connect(self.copy_to_clipboard)

        self.commandline_buttons = QVBoxLayout()
        self.commandline_buttons.addWidget(self.update_command_line)
        self.commandline_buttons.addWidget(self.copyToClipboard)

        self.commandline_group = QHBoxLayout()
        self.commandline_group.addWidget(self.commandbox)
        self.commandline_group.addLayout(self.commandline_buttons)

    @pyqtSlot()
    def openPipeline(self, fileName=None):
        if fileName is None:
            fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "",
                                                      "Pipeline file (*.pipeline);;All Files (*)")
        if fileName:
            self.PS.read(fileName)
            wd = os.getcwd()
            self.pipeline_file = fileName
            self.pipeline_file_label.setText("./"+os.path.relpath(fileName, wd))
            self.PL.load_pipeline(self.PS)
    @pyqtSlot()
    def testPipeline(self, multiprocess=True):
        if self.inputdir is None:
            QMessageBox.about(self, "Input directory missing",
                              "No input directory selected!")
        else:
            #self.setOutputSettings(self.PL)
            images = self.PL.scan_directory(self.inputdir)
            number_test_images = int(self.test_images.text())
            if number_test_images<1:
                QMessageBox.about(self, "Test file number failure",
                                  "Number of test images must be positive")
            else:
                self.total_input = number_test_images
                self.time_mean = 30
                self.progress.setMaximum(self.total_input)
                self.progress.setValue(1)
                self._update_time(30)
                self.progress.setValue(0)
                self.time_mean = 0
                #self.time_left.setText("--h--m--s")
                #self.image_counter = 0

                if multiprocess:
                    self.multi_process_images(images[:number_test_images])
                else:
                    self.process_images(images[:number_test_images])
    @pyqtSlot()
    def runPipeline(self):
        if self.inputdir is None:
            QMessageBox.about(self, "Input directory missing",
                              "No input directory selected!")
        else:
            images = self.PL.scan_directory(self.inputdir)
            self.total_input = len(images)
            self.time_mean = 30
            self.progress.setMaximum(self.total_input)
            self.progress.setValue(1)
            self._update_time(30)
            self.progress.setValue(0)
            self.time_mean = 0
            #self.time_left.setText("--h--m--s")
            self.multi_process_images(images)

    def setOutputSettings(self, PL):
        if self.mask_check.isChecked():
            PL.mask_output = (True, self.maskdir)
        if self.size_check.isChecked():
            PL.measure_size = (True, self.sizedir)
        if self.crop_check.isChecked():
            PL.crop_output = (True, self.cropdir)
        if self.greenness_check.isChecked():
            PL.measure_greenness = (True, self.greennessdir)
        if self.no_subfolders.isChecked():
            PL.substructure = (False, "")
        if self.divide_by_day.isChecked():
            PL.substructure = (True, "Time")
        if self.divide_by_individual.isChecked():
            PL.substructure = (True, "Sample")
    def process_images(self, images):
        for image in images:
            start = time.time()
            try:
                print("Reading image!")
                self.PL.open_image(image)
                print("Searching for plants")
                self.PL.infer_network_on_image()
                print("Applying color correction")
                self.PL.color_correction()
                print("Identifying group")
                self.PL.identify_group()
                print("Cropping and labelling pots")
                self.PL.crop_and_label_pots()
            except:
                traceback.print_exc()
                exctype, value = sys.exc_info()[:2]
                self._report_error((exctype, value, traceback.format_exc()))
            self._process_complete()
            self._update_time(time.time()-start)
    def _process_complete(self):
        self.progress.setValue(self.progress.value()+1)
        if self.progress.value()==self.total_input:
            self.time_left.setText("Complete")
    def _prettify_time(self, seconds):
        base_time = str(datetime.timedelta(seconds=round(seconds)))
        return datetime.datetime.strptime(base_time, "%H:%M:%S").strftime("%Hh%Mm%Ss")
    def update_mean(self, value):
        if self.time_mean==0:
            self.time_mean = value
        else:
            self.time_mean = (self.time_mean * (self.progress.value()-1) + value) / (self.progress.value())
    def _update_time(self, rtime):
        left = self.total_input-self.progress.value()
        if left == 0:
            self.time_left.setText("Completed")
        else:
            self.update_mean(rtime)
            time_left = (self.time_mean*left)/int(self.multicore.text())
            self.time_left.setText(self._prettify_time(time_left))
    def _report_error(self, error):
        exctype, value, errormsg = error
        print(errormsg)
    def multi_process_images(self, images):
        n_process = int(self.multicore.text())
        self.threadpool.setMaxThreadCount(n_process)
        self.time_mean = 0
        for image in images:
            worker = Worker(_single_process, image=image, PS=self.PS, Interface=self)
            worker.signals.finished.connect(self._process_complete)
            worker.signals.error.connect(self._report_error)
            worker.signals.result.connect(self._update_time)
            self.threadpool.start(worker)

    @pyqtSlot()
    def OpenPipelinePlanner(self):
        self.PipelinePlannerApp.show()
        self.PipelinePlannerApp.GUI.setDefaultValues()
        self.PipelinePlannerApp.setWindowIcon(self.windowicon)

    @pyqtSlot()
    def openInputDirectory(self, dir_name=None):
        if dir_name is None:
            dir_name = str(QFileDialog.getExistingDirectory(self, "Select Network Directory", "Network Directory"))
        if dir_name:
            self.inputdir_label.setText("./"+os.path.relpath(dir_name, self.wd))
            self.inputdir = dir_name
    @pyqtSlot()
    def openMaskDirectory(self, dir_name=None):
        if dir_name is None:
            dir_name = str(QFileDialog.getExistingDirectory(self, "Select Mask output directory", "Mask Directory"))
        if dir_name:
            self.maskout_label.setText("./"+os.path.relpath(dir_name, self.wd))
            self.maskdir = dir_name
            self.mask_check.setChecked(True)
            self.mask_check.setDisabled(False)
    @pyqtSlot()
    def openCropDirectory(self, dir_name=None):
        if dir_name is None:
            dir_name = str(QFileDialog.getExistingDirectory(self, "Select Crop output Directory", "Crop Directory"))
        if dir_name:
            self.cropout_label.setText("./"+os.path.relpath(dir_name, self.wd))
            self.cropdir = dir_name
            self.crop_check.setChecked(True)
            self.crop_check.setDisabled(False)
    @pyqtSlot()
    def openSizeDirectory(self, dir_name=None):
        if dir_name is None:
            dir_name = str(QFileDialog.getExistingDirectory(self, "Select Size output Directory", "Size Directory"))
        if dir_name:
            self.sizeout_label.setText("./"+os.path.relpath(dir_name, self.wd))
            self.sizedir = dir_name
            self.size_check.setChecked(True)
            self.size_check.setDisabled(False)
    @pyqtSlot()
    def openGreennessDirectory(self, dir_name=None):
        if dir_name is None:
            dir_name = str(QFileDialog.getExistingDirectory(self, "Select Greenness output Directory", "Greenness Directory"))
        if dir_name:
            self.greennessout_label.setText("./"+os.path.relpath(dir_name, self.wd))
            self.greennessdir = dir_name
            self.greenness_check.setChecked(True)
            self.greenness_check.setDisabled(False)

    @pyqtSlot()
    def copy_to_clipboard(self):
        self.system_clipboard.setText(self.commandbox.toPlainText())

    def _fix_spaces(self, string):
        return string.replace(" ", "\\ ")
    def _set_relative_path(self, string):
        if self.relative_paths.isChecked():
            return "./"+os.path.relpath(string, self.wd)
        else:
            return string
    def generateCommandline(self, multi_image=True):
        s = "greenotyper"
        if multi_image:
            s += " -t {}".format(int(self.multicore.text()))
        if not self.pipeline_file is None:
            s += " -p {}".format(self._set_relative_path(self._fix_spaces(self.pipeline_file)))
        if not self.inputdir is None:
            s += " -i {}".format(self._set_relative_path(self._fix_spaces(self.inputdir)))
        if self.size_check.isChecked():
            s += " -s {}".format(self._set_relative_path(self._fix_spaces(self.sizedir)))
        if self.greenness_check.isChecked():
            s += " -g {}".format(self._set_relative_path(self._fix_spaces(self.greennessdir)))
        if self.crop_check.isChecked():
            s += " -c {}".format(self._set_relative_path(self._fix_spaces(self.cropdir)))
        if self.mask_check.isChecked():
            s += " -m {}".format(self._set_relative_path(self._fix_spaces(self.maskdir)))
        if self.divide_by_day.isChecked():
            s += " --by_day"
        if self.divide_by_individual.isChecked():
            s += " --by_individual"

        return s
    @pyqtSlot()
    def updateCommandline(self):
        s = self.generateCommandline()
        self.commandbox.setText(s)

    def _make_gwf_template(self, commandline, format):
        tab = "    "
        gwf_template = "def process_image(image, outputs):\n"
        gwf_template += "{tab}inputs = [image]\n{tab}outputs = outputs\n".format(tab=tab)
        gwf_template += "{tab}options = {{\n{tab}{tab}'memory': '2g',\n{tab}{tab}'walltime': '00:10:00'\n{tab}}}\n\n".format(tab=tab)
        gwf_template += "{tab}spec = '''\n".format(tab=tab)
        gwf_template += tab+commandline+"\n"
        gwf_template += "{tab}'''.format({format})\n\n".format(tab=tab, format=format)
        gwf_template += "{tab}return inputs, outputs, options, spec\n\n".format(tab=tab)
        return gwf_template
    def _bash_commandline(self, commandline):
        before, after = commandline.split("-i", 1)
        remaning_options = after.split("-", 1)
        if len(remaning_options)==1:
            after = ""
            generalized_commandline = before+"-i"+" $IMAGE "
        else:
            after = remaning_options[-1]
            generalized_commandline = before+"-i"+" $IMAGE "+"-"+after

        return generalized_commandline
    def _generalize_commandline(self, commandline):
        environment = self.environment_textbox.toPlainText()
        before, after = commandline.split("-i", 1)
        remaning_options = after.split("-", 1)
        if len(remaning_options)==1:
            after = ""
            generalized_commandline = before+"-i"+" {image} "
        else:
            after = remaning_options[-1]
            generalized_commandline = before+"-i"+" {image} "+"-"+after

        if environment!=self.environment_default:
            tab = "    "
            generalized_commandline = environment+"\n\n"+tab+generalized_commandline

        return generalized_commandline, "image=image"
    def generateWorkflowFile(self):
        if self.inputdir is None:
            QMessageBox.about(self, "Input directory missing",
                              "No input directory selected!")
            return None
        if self.use_bash.isChecked():
            commandline = self.generateCommandline(False)
            bash_workflow = "#!/bin/bash\n\n"
            tab = "    "
            cluster_submit_command = self.cluster_submit.toPlainText()

            environment = self.environment_textbox.toPlainText()
            if environment==self.environment_default:
                environment = ""
                #bash_workflow += environment
                #bash_workflow += "\n\n"

            bash_workflow += 'IMAGES=$(find {dir} -type f \\( -name "*.jpg" -or -name "*" \\))\n\n'.format(dir = self._set_relative_path(self.inputdir))

            bash_workflow += "for IMAGE in $IMAGES;\ndo\n"

            bash_command = self._bash_commandline(commandline)
            #bash_workflow += tab+"echo $IMAGE\n"
            if cluster_submit_command==self.cluster_default:
                bash_workflow += tab+bash_command+"\n"
            else:
                bash_workflow += tab+cluster_submit_command.format(command=bash_command,environment=environment)+"\n"
            bash_workflow += "done\n"

            return bash_workflow
        if self.use_gwf.isChecked():
            commandline = self.generateCommandline(False)
            tab = "    "

            gwf_workflow = "from gwf import Workflow\nimport os\n"
            gwf_workflow += "gwf = Workflow()\n\n"

            generalized_commandline, format = self._generalize_commandline(commandline)

            gwf_workflow += self._make_gwf_template(generalized_commandline, format)

            gwf_workflow += "images = os.listdir({})\n".format(self._set_relative_path(self.inputdir))

            outputs = []
            gwf_workflow += "for image in images:\n"
            gwf_workflow += "{tab}gwf.target_from_template('{{}}_process'.format(image),\n{tab}{tab}{tab}{tab}process_image(image, outputs={outputs}))".format(tab=tab, outputs=outputs)

            return gwf_workflow
    def saveWorkflow(self):
        workflow = self.generateWorkflowFile()
        if self.use_bash.isChecked():
            fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getOpenFileName()", "",
                                                      "bash script (*.sh);;All Files (*)")
        if self.use_gwf.isChecked():
            fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getOpenFileName()", "",
                                                      "python script (*.py);;All Files (*)")
        if fileName:
            _file = open(fileName, "w")
            _file.write(workflow)
            _file.close()

    # @pyqtSlot()
    # def OpenLabelImg(self):
    #     #from PlantFinderAPI.GUI import labelImg
    #     self.LabelImgApp = labelImg.MainWindow()
    #     self.LabelImgApp.show()

if __name__=="__main__":
    pass
