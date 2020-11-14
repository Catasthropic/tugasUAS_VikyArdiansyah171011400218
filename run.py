"""
Tugas UAS Viky Ardiansyah
171011400218
07TPLE004

Image Processing menggunakan OpenCV untuk mendeteksi objek sederhana
IDE: Pycharm

"""

import os
from ShapeClassifier import ShapeClassifier
import argparse


# handle Command Line Arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--images_path', default='./images', type=str, help='path to the directory containing images')
parser.add_argument('--verbose', type=int, default=1, help='verbosity to how visualize results')
args = parser.parse_args()

# raise an error if wrong arguments are passed
assert args.verbose in [0, 1], "verbose be 0 or 1"
assert os.path.exists(args.images_path), "path doesn't exists"


if __name__ == "__main__":
    classifier = ShapeClassifier(verbose=args.verbose)
    # run the classifier on all images present in the directory
    for _path, _dirs, _images in os.walk(args.images_path):
        for img in _images:
            classifier.predict_shape(os.path.join(_path, img))


"""
Some comments !!!

This Detector has been tested on 35 images containing cropped road signs with 7 different shapes:
1. Square
2. Horizontal Rectangle
3. Vertical Rectangle
4. Diamond
5. Octagon
6. Pentagon
7. Circle

The detector classifier all 35 images with 100% accuracy. With OpenCV functions e.g. cv2.inRange(), cv2.contours(), 
cv2.erode() etc. are pretty fast. Instead of training a deep learning model, simple image processing does a great 
job with high speed.  
"""