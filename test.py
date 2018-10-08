import unittest

from PIL import Image

from vehicle_detection import load_data
import matplotlib.pyplot as plt

class TestLoadingData(unittest.TestCase):

    def test_Loading(self):
        test_list = load_data('../cityscapes_samples','../cityscapes_samples_labels')
        print(test_list[0])
        print(test_list[0]['file_path'])
        print(test_list[0]['label'][0]['class'])
        print(test_list[0]['label'][0]['position'][0])
        print(test_list[0]['label'][0]['position'][1])
        print(test_list[0]['label'][1]['class'])
        print(test_list[0]['label'][1]['position'][0])
        print(test_list[0]['label'][1]['position'][1])
        img = Image.open(test_list[0]['file_path'])
        plt.imshow(img)
        plt.show()
        self.assertEqual('foo'.upper(), 'FOO')
