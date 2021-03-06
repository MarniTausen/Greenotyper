from greenotyperAPI import *
import greenotyperAPI
import os
import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
import sys

app = QApplication(sys.argv)

class TestFileReading(unittest.TestCase):

    def setUp(self):
        self.PL = GREENOTYPER.Pipeline()
        self.network_dir = "sample_data/network"
        files = os.listdir(self.network_dir)
        self.graph = os.path.join(self.network_dir,next(filter(lambda x: ".pb" in x and "txt" not in x, files)))
        self.label = os.path.join(self.network_dir,next(filter(lambda x: ".pbtxt" in x, files)))
    def tearDown(self):
        del self.PL
        del self.network_dir
        del self.graph
        del self.label

    def test_version(self):
        self.assertEqual(self.PL.__get_version__(), self.PL.__version__)
    def test_graph_load(self):
        self.PL.load_graph(self.graph)
        self.assertTrue(hasattr(self.PL, "detection_graph"))
    def test_pbtxt_load(self):
        self.PL.read_pbtxt(self.label)
        self.assertTrue(hasattr(self.PL, "label_map"))
    def test_load_image(self):
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
        self.assertTrue(hasattr(self.PL, "image"))
        self.assertTrue(hasattr(self.PL, "_image_filename"))
        self.assertTrue(self.PL._image_filename=="sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
    def test_load_from_file_pipeline(self):
        PipelineSettings = GREENOTYPER.pipeline_settings()
        PipelineSettings.read("sample_data/sample.pipeline")
        self.PL.load_pipeline(PipelineSettings)
        self.assertTrue(hasattr(self.PL, "pipeline_settings"))
        self.assertTrue(hasattr(self.PL, "HSV"))
        self.assertTrue(hasattr(self.PL, "LAB"))
        self.assertTrue(hasattr(self.PL, "ncol"))
        self.assertTrue(hasattr(self.PL, "nrow"))
    def test_load_default_pipeline(self):
        PipelineSettings = GREENOTYPER.pipeline_settings()
        self.PL.load_pipeline(PipelineSettings)
        self.assertTrue(hasattr(self.PL, "pipeline_settings"))
        self.assertTrue(hasattr(self.PL, "HSV"))
        self.assertTrue(hasattr(self.PL, "LAB"))
        self.assertTrue(hasattr(self.PL, "ncol"))
        self.assertTrue(hasattr(self.PL, "nrow"))
    def test_png_file_reading(self):
        png_file = "test_data/png_test.png"
        self.PL.open_image(png_file)
        self.assertTrue(hasattr(self.PL, "image"))
        self.assertTrue(hasattr(self.PL, "_image_filename"))
        self.assertEqual(self.PL._image_filename,png_file)
        self.assertEqual(self.PL.image.shape[2],3)
    def test_write_pipeline(self):
        PipelineSettings = GREENOTYPER.pipeline_settings()
        PipelineSettings.read("sample_data/sample.pipeline")
        PipelineSettings.write("test_data/test.pipeline")

        sample_file = open("sample_data/sample.pipeline")
        sample_contents = sample_file.read()
        sample_file.close()
        write_file = open("test_data/test.pipeline")
        write_contents = write_file.read()
        write_file.close()

        self.assertTrue(sample_contents==write_contents)

class TestDetection(unittest.TestCase):

    def setUp(self):
        network_dir = "sample_data/network"
        files = os.listdir(network_dir)
        graph = os.path.join(network_dir,next(filter(lambda x: ".pb" in x and "txt" not in x, files)))
        label = os.path.join(network_dir,next(filter(lambda x: ".pbtxt" in x, files)))

        self.PL = GREENOTYPER.Pipeline(graph, label)
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
    def tearDown(self):
        del self.PL

    def test_infer_network(self):
        self.PL.infer_network_on_image()
        self.assertTrue(hasattr(self.PL, "boxes"))
        self.assertEqual(len(self.PL.boxes), len(self.PL.label_map))
        self.assertEqual(len(self.PL.boxes["POT"]), 10)
        expected_bounding_boxes = [[567, 1056, 1332, 1897],
                                   [2203, 2660, 1344, 1944],
                                   [1126, 1597, 248, 843],
                                   [606, 1082, 264, 846],
                                   [1625, 2082, 275, 882],
                                   [1120, 1583, 1345, 1933],
                                   [1659, 2145, 1365, 1951],
                                   [2634, 3103, 268, 865],
                                   [2769, 3210, 1371, 1956],
                                   [2167, 2608, 286, 873]]
        for box in self.PL.boxes["POT"]:
            self.assertTrue(box in expected_bounding_boxes)
        self.assertEqual(len(self.PL.boxes['QRCODE']), 1)
        self.assertEqual(len(self.PL.boxes['QRCOLOR']), 2)

class TestFilteration(unittest.TestCase):

    def setUp(self):
        network_dir = "sample_data/network"
        files = os.listdir(network_dir)
        graph = os.path.join(network_dir,next(filter(lambda x: ".pb" in x and "txt" not in x, files)))
        label = os.path.join(network_dir,next(filter(lambda x: ".pbtxt" in x, files)))

        self.PL = GREENOTYPER.Pipeline(graph, label)
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
        self.PL.infer_network_on_image()
    def tearDown(self):
        del self.PL

    def test_basic_filteration(self):
        self.PL.identify_group()
        self.assertTrue(hasattr(self.PL, "boxes"))
        self.assertEqual(len(self.PL.boxes), len(self.PL.label_map))
        self.assertEqual(len(self.PL.boxes["POT"]), 10)
        expected_bounding_boxes = [[567, 1056, 1332, 1897],
                                   [2203, 2660, 1344, 1944],
                                   [1126, 1597, 248, 843],
                                   [606, 1082, 264, 846],
                                   [1625, 2082, 275, 882],
                                   [1120, 1583, 1345, 1933],
                                   [1659, 2145, 1365, 1951],
                                   [2634, 3103, 268, 865],
                                   [2769, 3210, 1371, 1956],
                                   [2167, 2608, 286, 873]]
        for box in self.PL.boxes["POT"]:
            self.assertTrue(box in expected_bounding_boxes)

class TestLabelling(unittest.TestCase):

    def setUp(self):
        self.PL = GREENOTYPER.Pipeline()
    def tearDown(self):
        del self.PL

    def test_name_map_fail(self):
        filename = "sample_data/camera_map.csv"
        try:
            self.PL.read_name_map(filename)
        except Exception as inst:
            self.assertEqual(str(inst),
                             "Column 'Name' missing in name map file '{}'".format(filename))
    def test_camera_map_fail(self):
        filename = "sample_data/id_map.csv"
        try:
            self.PL.read_camera_map(filename)
        except Exception as inst:
            self.assertEqual(str(inst),
                             "Column 'Camera' missing in camera map file '{}'".format(filename))
    def test_name_map(self):
        filename = "sample_data/id_map.csv"
        self.PL.read_name_map(filename)
        self.assertTrue(hasattr(self.PL, "name_map"))
        self.assertEqual(self.PL.name_map[20][48], "3600")
        self.assertEqual(self.PL.name_map[40][1], "1801")
        self.assertEqual(self.PL.name_map[13][35], "3333")
        self.assertEqual(self.PL.name_map[1][40], "3421")

        self.assertEqual(len(self.PL.name_map), 40)
        self.assertEqual(len(self.PL.name_map[1]),48)
        self.assertEqual(len(self.PL.name_map[40]),42)
    def test_camera_map(self):
        filename = "sample_data/camera_map.csv"
        self.PL.read_camera_map(filename)
        self.assertTrue(hasattr(self.PL, "camera_map"))
        for i in range(1,180):
            cam_name = "Cam"+str(i)
            self.assertTrue(cam_name in self.PL.camera_map)
            self.assertTrue("NS" in self.PL.camera_map[cam_name])
            self.assertTrue("EW" in self.PL.camera_map[cam_name])
            self.assertTrue("orient" in self.PL.camera_map[cam_name])
    def test_get_pot_labels(self):
        NS = [6,10]
        EW = [3,4]

        E_result = [[6, 3], [7, 3], [8, 3], [9, 3], [10, 3],
                    [6, 4], [7, 4], [8, 4], [9, 4], [10, 4]]
        W_result = [[10, 4], [9, 4], [8, 4], [7, 4], [6, 4],
                    [10, 3], [9, 3], [8, 3], [7, 3], [6, 3]]
        N_result = [[10, 3], [10, 4], [9, 3], [9, 4], [8, 3],
                    [8, 4], [7, 3], [7, 4], [6, 3], [6, 4]]
        S_result = [[6, 4], [6, 3], [7, 4], [7, 3], [8, 4],
                    [8, 3], [9, 4], [9, 3], [10, 4], [10, 3]]

        self.assertEqual(self.PL._get_pot_labels(NS, EW, "E"), E_result)
        self.assertEqual(self.PL._get_pot_labels(NS, EW, "W"), W_result)
        self.assertEqual(self.PL._get_pot_labels(NS, EW, "N"), N_result)
        self.assertEqual(self.PL._get_pot_labels(NS, EW, "S"), S_result)

class TestThresholding(unittest.TestCase):

    def setUp(self):
        network_dir = "sample_data/network"
        files = os.listdir(network_dir)
        graph = os.path.join(network_dir,next(filter(lambda x: ".pb" in x and "txt" not in x, files)))
        label = os.path.join(network_dir,next(filter(lambda x: ".pbtxt" in x, files)))
        PS = GREENOTYPER.pipeline_settings()
        PS.read("sample_data/sample.pipeline")

        self.PL = GREENOTYPER.Pipeline(graph, label, PS)
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
    def tearDown(self):
        del self.PL

    def test_change_hsv_settings(self):
        ohue1, ohue2 = self.PL.HSV['hue']
        osat1, osat2 = self.PL.HSV['sat']
        oval1, oval2 = self.PL.HSV['val']
        nhue1, nhue2 = 0, 1
        nsat1, nsat2 = 0, 0.8
        nval1, nval2 = 0, 0.6
        self.assertNotEqual(nhue1, ohue1)
        self.assertNotEqual(nhue2, ohue2)
        self.assertNotEqual(nsat1, osat1)
        self.assertNotEqual(nsat2, osat2)
        self.assertNotEqual(nval1, oval1)
        self.assertNotEqual(nval2, oval2)
        self.PL.setHSVmasksettings((nhue1, nhue2),
                                   (nsat1, nsat2),
                                   (nval1, nval2))
        hue1, hue2 = self.PL.HSV['hue']
        sat1, sat2 = self.PL.HSV['sat']
        val1, val2 = self.PL.HSV['val']
        self.assertEqual(nhue1, hue1)
        self.assertEqual(nhue2, hue2)
        self.assertEqual(nsat1, sat1)
        self.assertEqual(nsat2, sat2)
        self.assertEqual(nval1, val1)
        self.assertEqual(nval2, val2)
    def test_change_lab_settings(self):
        oL1, oL2 = self.PL.LAB['L']
        oa1, oa2 = self.PL.LAB['a']
        ob1, ob2 = self.PL.LAB['b']
        nL1, nL2 = 0, 100
        na1, na2 = 0, 128
        nb1, nb2 = -128, 0
        self.assertNotEqual(nL1, oL1)
        self.assertNotEqual(nL2, oL2)
        self.assertNotEqual(na1, oa1)
        self.assertNotEqual(na2, oa2)
        self.assertNotEqual(nb1, ob1)
        self.assertNotEqual(nb2, ob2)
        self.PL.setLABmasksettings((nL1, nL2),
                                   (na1, na2),
                                   (nb1, nb2))
        L1, L2 = self.PL.LAB['L']
        a1, a2 = self.PL.LAB['a']
        b1, b2 = self.PL.LAB['b']
        self.assertEqual(nL1, L1)
        self.assertEqual(nL2, L2)
        self.assertEqual(na1, a1)
        self.assertEqual(na2, a2)
        self.assertEqual(nb1, b1)
        self.assertEqual(nb2, b2)

class TestColorCorrection(unittest.TestCase):

    def setUp(self):
        network_dir = "sample_data/network"
        files = os.listdir(network_dir)
        graph = os.path.join(network_dir,next(filter(lambda x: ".pb" in x and "txt" not in x, files)))
        label = os.path.join(network_dir,next(filter(lambda x: ".pbtxt" in x, files)))

        self.PL = GREENOTYPER.Pipeline(graph, label)
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
        self.PL.infer_network_on_image()
    def tearDown(self):
        del self.PL

    def test_imadjust(self):

        t = np.asarray([5, 245])
        corrected = self.PL._imadjust(t, (5, 245))

        self.assertEqual(corrected.min(), 0)
        self.assertEqual(corrected.max(), 255)

        t = np.asarray([7, 50, 234])
        corrected = self.PL._imadjust(t, (7, 234))

        self.assertEqual(corrected.min(), 0)
        self.assertEqual(corrected.max(), 255)
    def test_min_color_correct(self):
        self.PL.ColorCorrectType = "minimum"
        lowest = max(self.PL.image[..., 0].min(), self.PL.image[..., 1].min(),
                     self.PL.image[..., 2].min())
        self.assertTrue(lowest>0)
        self.PL.color_correction()
        lowest = max(self.PL.image[..., 0].min(), self.PL.image[..., 1].min(),
                     self.PL.image[..., 2].min())
        self.assertTrue(lowest==0)
    def test_max_color_correct(self):
        self.PL.ColorCorrectType = "maximum"
        highest = min(self.PL.image[..., 0].max(), self.PL.image[..., 1].max(),
                     self.PL.image[..., 2].max())
        self.assertTrue(highest<255)
        self.PL.color_correction()
        highest = min(self.PL.image[..., 0].max(), self.PL.image[..., 1].max(),
                     self.PL.image[..., 2].max())
        self.assertTrue(highest==255)
    def test_min_max_color_correct(self):
        self.PL.ColorCorrectType = "both"
        highest = min(self.PL.image[..., 0].max(), self.PL.image[..., 1].max(),
                     self.PL.image[..., 2].max())
        lowest = max(self.PL.image[..., 0].min(), self.PL.image[..., 1].min(),
                     self.PL.image[..., 2].min())
        self.assertTrue(highest<255)
        self.assertTrue(lowest>0)
        self.PL.color_correction()
        highest = min(self.PL.image[..., 0].max(), self.PL.image[..., 1].max(),
                     self.PL.image[..., 2].max())
        lowest = max(self.PL.image[..., 0].min(), self.PL.image[..., 1].min(),
                     self.PL.image[..., 2].min())
        self.assertTrue(highest==255)
        self.assertTrue(lowest==0)

class TestMasking(unittest.TestCase):

    def setUp(self):
        PS = GREENOTYPER.pipeline_settings()
        PS.read("sample_data/sample.pipeline")

        self.PL = GREENOTYPER.Pipeline(pipeline=PS)
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
        self.PL.infer_network_on_image()
    def tearDown(self):
        del self.PL

    def test_mask(self):
        self.PL.mask_image()
        self.assertTrue(hasattr(self.PL, "_mask"))
    # def test_fancy_overlay(self):
    #     self.PL.mask_image()
    #     self.PL.fancy_overlay()
    #
    #     #self.PL.save_image("test_data/fancy_overlay_image.jpeg")
    #
    #     PS = GREENOTYPER.pipeline_settings()
    #     PS.read("sample_data/sample.pipeline")
    #     TestImage = GREENOTYPER.Pipeline(pipeline=PS)
    #     TestImage.open_image("test_data/fancy_overlay_image.jpeg")
    #
    #     #print(self.PL.image==TestImage.image)
    #     #print(self.PL.image.shape)
    #     #print(TestImage.image.shape)
    #
    #     self.assertTrue((self.PL.image==TestImage.image).all())
    # def test_inverse_mask(self):
    #     self.PL.mask_image()
    #     self.PL.inverse_mask()
    #
    #     #self.PL.save_image("test_data/inverse_mask_image.jpeg")
    #
    #     PS = GREENOTYPER.pipeline_settings()
    #     PS.read("sample_data/sample.pipeline")
    #     TestImage = GREENOTYPER.Pipeline(pipeline=PS)
    #     TestImage.open_image("test_data/inverse_mask_image.jpeg")
    #
    #     print(self.PL.image==TestImage.image)
    #     print(self.PL.image.shape)
    #     print(TestImage.image.shape)
    #
    #     self.assertTrue((self.PL.image==TestImage.image).all())
    # def test_basic_mask(self):
    #     self.PL.mask_image()
    #     self.PL.basic_mask(self.PL.HEXtoRGB("#AA2222"))
    #
    #     #self.PL.save_image("test_data/basic_mask_image.jpeg")
    #
    #     PS = GREENOTYPER.pipeline_settings()
    #     PS.read("sample_data/sample.pipeline")
    #     TestImage = GREENOTYPER.Pipeline(pipeline=PS)
    #     TestImage.open_image("test_data/basic_mask_image.jpeg")
    #
    #     print(self.PL.image==TestImage.image)
    #     print(self.PL.image.shape)
    #     print(TestImage.image.shape)
    #
    #     self.assertTrue((self.PL.image==TestImage.image).all())

class TestBoundingBoxDrawing(unittest.TestCase):

    def setUp(self):
        PS = GREENOTYPER.pipeline_settings()
        PS.read("sample_data/sample.pipeline")

        self.PL = GREENOTYPER.Pipeline(pipeline=PS)
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
        self.PL.infer_network_on_image()
    def tearDown(self):
        del self.PL

    def test_draw_bounding_box(self):
        self.PL.draw_bounding_boxes()

        self.assertTrue(True)

class TestOutputs(unittest.TestCase):

    def setUp(self):
        PS = GREENOTYPER.pipeline_settings()
        PS.read("sample_data/sample.pipeline")

        self.PL = GREENOTYPER.Pipeline(pipeline=PS)
    def tearDown(self):
        del self.PL

    def test_crop_and_label_pots_1_image_size_thresholding(self):
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
        self.PL.infer_network_on_image()
        self.PL.identify_group()
        self.PL.color_correction()
        self.PL.measure_size = (True, "test_outputs")
        self.PL.mask_output = (True, "test_outputs")
        self.PL.crop_output = (True, "test_outputs")

        # reset file
        _file = open("test_outputs/database.size.csv", "w")
        _file.write("")
        _file.close()

        self.PL.crop_and_label_pots()

        self.PL.organize_output("test_outputs/database.size.csv", "test_outputs/sizes.csv")

        True_results = {"Time": "2018/06/16 - 11:34",
                        "2171": "142874", "2172": "11876",
                        "2173": "146116", "2174": "80369",
                        "2175": "91804", "2191": "16865",
                        "2192": "128383", "2193": "9830",
                        "2194": "109220", "2195": "87013"}

        _file = open("test_outputs/sizes.csv")
        header = _file.readline().replace("\n","").split(",")
        dataline = _file.readline().replace("\n","").split(",")
        results = {}
        #print(header)
        #print(dataline)
        for name, data in zip(header, dataline):
            results[name] = data

        for name in header:
            self.assertEqual(results[name],True_results[name])

    def test_crop_and_label_pots_1_image_greenness(self):
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
        self.PL.infer_network_on_image()
        self.PL.identify_group()
        self.PL.color_correction()
        self.PL.measure_greenness = (True, "test_outputs")

        # reset file
        _file = open("test_outputs/database.greenness.csv", "w")
        _file.write("")
        _file.close()

        self.PL.crop_and_label_pots()

        self.PL.greenness_output("test_outputs/database.greenness.csv", "test_outputs/greenness")

        self.assertTrue(True)

    #def test_crop_outputs(self): -> substructure sample
    #def test_mask_outputs(self): -> substructure time
    def test_circular_hsv_figure(self):
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
        self.PL.infer_network_on_image()
        self.PL.identify_group()
        self.PL.color_correction()
        self.PL.measure_greenness = (True, "test_outputs")

        # reset file
        _file = open("test_outputs/database.greenness.csv", "w")
        _file.write("")
        _file.close()

        self.PL.crop_and_label_pots(return_crop_list=True, return_greenness_figures=True)

class TestsPipelineRunner(unittest.TestCase):

    def setUp(self):
        from greenotyperAPI.GUI import PipelineRunner
        self.mainwindow = PipelineRunner.PipelineRunner()
        self.mainwindow.windowicon = QtGui.QIcon(os.path.join(sys.prefix,'icon.png'))
    def tearDown(self):
        del self.mainwindow

    def test_defaults(self):
        self.mainwindow.setOutputSettings(self.mainwindow.PL)
        self.mainwindow.updateCommandline()

        self.assertTrue(True)

    def test_open_pipeline(self):
        self.mainwindow.openPipeline("sample_data/sample.pipeline")

        self.assertEqual(self.mainwindow.pipeline_file, "sample_data/sample.pipeline")
        self.assertEqual(self.mainwindow.pipeline_file_label.text(), "./sample_data/sample.pipeline")

    def test_pipeline_preparation(self):
        self.mainwindow.openPipeline("sample_data/sample.pipeline")

        self.mainwindow.openInputDirectory("sample_data/Cam41")

        self.assertEqual(self.mainwindow.inputdir, "sample_data/Cam41")
        self.assertEqual(self.mainwindow.inputdir_label.text(), "./sample_data/Cam41")

        self.mainwindow.openMaskDirectory("test_outputs")
        self.assertEqual(self.mainwindow.maskdir, "test_outputs")
        self.assertEqual(self.mainwindow.maskout_label.text(), "./test_outputs")
        self.assertTrue(self.mainwindow.mask_check.isChecked())

        self.mainwindow.openCropDirectory("test_outputs")
        self.assertEqual(self.mainwindow.cropdir, "test_outputs")
        self.assertEqual(self.mainwindow.cropout_label.text(), "./test_outputs")
        self.assertTrue(self.mainwindow.crop_check.isChecked())

        self.mainwindow.openSizeDirectory("test_outputs")
        self.assertEqual(self.mainwindow.sizedir, "test_outputs")
        self.assertEqual(self.mainwindow.sizeout_label.text(), "./test_outputs")
        self.assertTrue(self.mainwindow.size_check.isChecked())

        self.mainwindow.openGreennessDirectory("test_outputs")
        self.assertEqual(self.mainwindow.greennessdir, "test_outputs")
        self.assertEqual(self.mainwindow.greennessout_label.text(), "./test_outputs")
        self.assertTrue(self.mainwindow.greenness_check.isChecked())

        self.mainwindow.divide_by_day.setChecked(True)
        self.mainwindow.setOutputSettings(self.mainwindow.PL)
        self.mainwindow.updateCommandline()

        self.mainwindow.divide_by_individual.setChecked(True)
        self.mainwindow.relative_paths.setChecked(True)
        self.mainwindow.setOutputSettings(self.mainwindow.PL)
        self.mainwindow.updateCommandline()

        self.mainwindow.divide_by_day.setChecked(False)
        self.mainwindow.divide_by_individual.setChecked(False)
        self.mainwindow.no_subfolders.setChecked(True)

        self.mainwindow.generateWorkflowFile()

        self.mainwindow.use_bash.setChecked(False)
        self.mainwindow.use_gwf.setChecked(True)

        self.mainwindow.generateWorkflowFile()

    def test_pipeline_test(self):
        self.mainwindow.openPipeline("sample_data/sample.pipeline")

        self.mainwindow.openInputDirectory("test_data/test_images")
        self.mainwindow.openMaskDirectory("test_outputs")
        self.mainwindow.openCropDirectory("test_outputs")
        self.mainwindow.openSizeDirectory("test_outputs")
        self.mainwindow.openGreennessDirectory("test_outputs")

        self.mainwindow.mask_check.setChecked(False)
        self.mainwindow.crop_check.setChecked(False)
        self.mainwindow.size_check.setChecked(False)
        self.mainwindow.greenness_check.setChecked(False)

        self.mainwindow.setOutputSettings(self.mainwindow.PL)

        self.mainwindow.test_images.setText("3")

        self.mainwindow.testPipeline(False)

    def test_open_pipeline_planner(self):
        self.mainwindow.OpenPipelinePlanner()


class TestsPipelinePlanner(unittest.TestCase):
    def setUp(self):
        from greenotyperAPI.GUI import PipelinePlanner
        self.mainwindow = PipelinePlanner.PipelinePlanner()
    def tearDown(self):
        del self.mainwindow

    def test_defaults(self):

        self.mainwindow.GUI.setDefaultValues()
        self.mainwindow.GUI.read_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
        self.mainwindow.GUI.reload_image()

        self.mainwindow.GUI.UpdatePipelineSettings()
        self.mainwindow.GUI.UpdateMaskSettings()

        self.mainwindow.GUI.PS.read("sample_data/sample.pipeline")
        self.mainwindow.GUI.setDefaultValues()

        self.mainwindow.GUI.FindPlants(multithread=False)
        self.mainwindow.GUI.ImageMask(multithread=False)

        self.mainwindow.GUI.TestCrop()

        self.assertTrue(True)

class TestsUnet(unittest.TestCase):
    def setUp(self):
        PS = GREENOTYPER.pipeline_settings()
        PS.read("sample_data/sample.pipeline")
        self.PL = GREENOTYPER.Pipeline(pipeline=PS)
        self.PL.open_image("sample_data/Cam41/Cam41_MT20180616113420_C39_69.jpg")
        self.PL.infer_network_on_image()
        self.PL.identify_group()
        self.PL.color_correction()

    def tearDown(self):
        del self.PL

    def test_load_unet(self):
        self.assertFalse(hasattr(self.PL, "unet_model"))

        self.PL.load_unet("training_data/u-net/network/currentbest.unet.hdf5")

        self.assertTrue(hasattr(self.PL, "unet_model"))

    def test_unet_prepare_data(self):
        crops = self.PL.collect_crop_data(512)
        self.assertEqual(crops.shape[0], 10)
        self.assertEqual(crops.shape[1], 512)
        self.assertEqual(crops.shape[2], 512)
        self.assertEqual(crops.shape[3], 3)

        imagedata = self.PL.unet_prepare_images(crops)
        self.assertEqual(imagedata.shape[0], 10)
        self.assertEqual(imagedata.shape[1], 512)
        self.assertEqual(imagedata.shape[2], 512)
        self.assertEqual(imagedata.shape[3], 3)

        for i in range(imagedata.shape[0]):
            self.assertLessEqual(imagedata[i].max(), 1)

        True_labels = ['2175_MT20180616113420', '2174_MT20180616113420',
                       '2173_MT20180616113420', '2172_MT20180616113420',
                       '2171_MT20180616113420', '2195_MT20180616113420',
                       '2194_MT20180616113420', '2193_MT20180616113420',
                       '2192_MT20180616113420', '2191_MT20180616113420']

        filename_labels = self.PL.get_filename_labels()

        for tlabel, clabel in zip(True_labels, filename_labels):
            self.assertEqual(tlabel, clabel)

    def test_unet_predict_and_output(self):
        self.PL.measure_size = (True, "test_outputs/u-net")
        self.PL.mask_output = (True, "test_outputs/u-net")
        self.PL.crop_output = (True, "test_outputs/u-net")
        self.PL.measure_greenness = (True, "test_outputs/u-net")

        # reset files
        _file = open("test_outputs/u-net/database.size.csv", "w")
        _file.write("")
        _file.close()
        _file = open("test_outputs/u-net/database.greenness.csv", "w")
        _file.write("")
        _file.close()

        self.PL.load_unet("training_data/u-net/network/currentbest.unet.hdf5")

        crops = self.PL.collect_crop_data(512)
        imagedata = self.PL.unet_prepare_images(crops)
        filename_labels = self.PL.get_filename_labels()

        predicted_masks = self.PL.unet_apply_on_images(imagedata)

        self.PL.unet_output_data(imagedata, predicted_masks, filename_labels)

        self.PL.organize_output("test_outputs/u-net/database.size.csv", "test_outputs/u-net/unet.sizes.csv")

        True_results = {"Time": "2018/06/16 - 11:34",
                        "2171": "151528", "2172": "13637",
                        "2173": "151743", "2174": "86893",
                        "2175": "100536", "2191": "18501",
                        "2192": "138264", "2193": "10389",
                        "2194": "113808", "2195": "89083"}

        _file = open("test_outputs/u-net/unet.sizes.csv")
        header = _file.readline().replace("\n","").split(",")
        dataline = _file.readline().replace("\n","").split(",")
        results = {}
        #print(header)
        #print(dataline)
        for name, data in zip(header, dataline):
            results[name] = data

        for name in header:
            self.assertEqual(results[name],True_results[name])


        self.PL.greenness_output("test_outputs/u-net/database.greenness.csv", "test_outputs/u-net/unet.greenness")

    def test_unet_creation_parameters(self):
        Unet_model = self.PL.Unet([],[])

        model = Unet_model.create_unet()

        number_of_params = model.count_params()

        expected_number_of_params = 31032837

        self.assertEqual(number_of_params, expected_number_of_params)

    def test_dim_sizes(self):
        Unet_model = self.PL.Unet([],[])

        good_resolutions = [32, 64, 128, 256, 512, 1024,
                            2048, 4096, 8192, 16384, 32768,
                            65536, 131072, 262144]

        for i in good_resolutions:
            self.assertTrue(Unet_model.validate_factor_of_2(i))

    def test_load_train_data_and_augmentations(self):
        traindata, labeldata, filenames = self.PL.load_train_data("training_data/u-net/validation")

        self.assertEqual(8, traindata.shape[0])

        augment_train, augment_label = self.PL.augment_data(traindata, labeldata)

        self.assertEqual(8*40, augment_train.shape[0])

    def tests_no_augmentations(self):
        traindata, labeldata, filenames = self.PL.load_train_data("training_data/u-net/validation")

        self.assertEqual(8, traindata.shape[0])

        augment_train, augment_label = self.PL.augment_data(traindata, labeldata, Flips=False,
                                                            Rotations=False, Crops=False)

        self.assertEqual(8, augment_train.shape[0])

    def test_segmentation_measures(self):
        truth_array = [[0,0,1,0,0],
                       [0,1,1,1,0],
                       [1,1,1,1,1],
                       [0,1,1,1,0],
                       [0,0,1,0,0]]
        truth_array = np.array(truth_array)
        prediction_array = [[0,0,0,0,0],
                            [0,1,1,1,0],
                            [0,1,1,1,0],
                            [0,1,1,1,0],
                            [0,1,1,1,0]]
        prediction_array = np.array(prediction_array)

        TN, TP = 10, 10
        FN, FP = 3, 2

        true_precision = TP/(TP+FP)
        true_recall = TP/(TP+FN)
        true_pixel_accuracy = (TP+TN)/(TP+TN+FN+FP)
        true_iou = TP/(TP+FN+FP)
        true_dice_coefficient = (2*TP)/((TP+FN)+(TP+FP))

        self.assertEqual(true_precision, self.PL._precision(truth_array, prediction_array))
        self.assertEqual(true_recall, self.PL._recall(truth_array, prediction_array))
        self.assertEqual(true_pixel_accuracy, self.PL._pixel_accuracy(truth_array, prediction_array))
        self.assertEqual(true_iou, self.PL._iou_score(truth_array, prediction_array))
        self.assertEqual(true_dice_coefficient, self.PL._dice_coefficient(truth_array, prediction_array))

    def test_pascal_voc_ap(self):
        recalls = [0.4, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1]
        precisions = [1, 1, 0.95, 0.85, 0.90, 0.80, 0.75, 0.85, 0.70, 0.75, 0.70, 0.65]
        ious = [0.3, 0.5, 0.6, 0.55, 0.60, 0.65, 0.50, 0.70, 0.4, 0.60, 0.55, 0.60]

        true_all_AP = ((0.5-0)*1)+((0.55-0.5)*0.95)+((0.7-0.55)*0.90)+((0.85-0.7)*0.85)+((0.95-0.85)*0.75)+((0.99-0.95)*0.70)+((1-0.99)*0.65)
        true_all_50IOU = ((0.5-0)*1)+((0.55-0.5)*0.95)+((0.7-0.55)*0.90)+((0.85-0.7)*0.85)+((0.95-0.85)*0.75)+((0.99-0.95)*0.70)+((1-0.99)*0.65)
        true_all_60IOU = ((0.55-0)*0.95)+((0.7-0.55)*0.90)+((0.85-0.7)*0.85)+((0.95-0.85)*0.75)+((1-0.95)*0.65)

        self.assertEqual(true_all_AP, self.PL._PASCAL_VOC_AP(precisions, recalls, ious, IoU=0))

        self.assertEqual(true_all_50IOU, self.PL._PASCAL_VOC_AP(precisions, recalls, ious, IoU=0.5))

        self.assertEqual(true_all_60IOU, self.PL._PASCAL_VOC_AP(precisions, recalls, ious, IoU=0.6))

    def test_join_unet_data(self):
        traindata, _, filenames = self.PL.load_train_data("training_data/u-net/validation")
        testdata, _, testfilenames = self.PL.load_train_data("training_data/u-net/test")

        self.PL.unet_join_image_data([(traindata, filenames), (testdata, testfilenames)])
