from tkinter import Button, Tk, HORIZONTAL
from tkinter import filedialog
from tkinter.ttk import Progressbar
import time
import threading
import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from PIL import Image
import os 


class MonApp(Tk):
    def __init__(self):
        super().__init__()


        self.btn = Button(self, text='Source', command=self.open_src)
        self.btn2 = Button(self, text='Destination', command=self.open_dst)
        self.btn3 = Button(self, text='Start', command=self.start)
        self.btn.grid(row=0,column=0)
        self.btn2.grid(row=1,column=0)
        self.btn3.grid(row=2,column=0)
        self.progress = Progressbar(self, orient=HORIZONTAL,length=100,  mode='determinate')
        self.progress['value'] = 0
        self.progress.grid(row=3,column=0)

    # Open the source directory
    def open_src(self):
        self.filename = filedialog.askdirectory(title="Select a source directory")

    # Open the deestination directory
    def open_dst(self):
        self.filename2 = filedialog.askdirectory(title="Select a destination directory")

    # method for beginning photo processing 
    def start(self):
        self.btn['state']='disabled'
        self.btn2['state']='disabled'
        self.btn3['state']='disabled'
        self.progress['value'] = 0
        if os.path.isdir(self.filename) and os.path.isdir(self.filename2):
            self.exceute(self.filename, self.filename2)
            self.btn['state']='normal'
            self.btn2['state']='normal'
            self.btn3['state']='normal'

    
    def gamma_correct_lab(self, img, gamma):
        out = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        out[:, :, 0] = np.power((out[:, :, 0])/255, (1/gamma)) * 255
        out = cv.cvtColor(out, cv.COLOR_LAB2BGR)
        return out

    # Gamma correct image (to deepen shadows/blacks)
    def gamma_correct(self, img, gamma):
        out = img.copy()
        out[:, :, 0] = np.power((out[:, :, 0])/255, (1/gamma)) * 255
        out[:, :, 1] = np.power((out[:, :, 1])/255, (1/gamma)) * 255
        out[:, :, 2] = np.power((out[:, :, 2])/255, (1/gamma)) * 255
        return out

    # Auto white balance based on grayworld assumption
    def white_balance(self, img):
        result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
        return result

    # Adjust saturation given a factor
    def saturation_adjustment(self, img, factor):
        result = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        result[:, :, 1] = result[:, :, 1] * factor
        result = cv.cvtColor(result, cv.COLOR_HSV2BGR)
        return result

    # Decrease the green of an image (by 5%)
    def reduce_green(self, im):
        out = im.copy()
        out[:, :, 1] = out[:, :, 1] * .95
        return out

    # draw an image with detected objects
    def face_boxes(self, filename, result_list):
        data = pyplot.imread(filename)
        w, h = Image.open(filename).size
        a = w * h
        ax = pyplot.gca()
        
        # create each box
        area = 0
        for result in result_list:
            # get box coordinates
            x, y, width, height = result['box']

            # Calculate the area taken up by the face box
            area = area + ((height * width) / a)

        return area
    
    # Classify if the given image is a portrait
    def classify_portrait(self, filename):
        pixels = pyplot.imread(filename)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        faces = detector.detect_faces(pixels)
        x = self.face_boxes(filename, faces)
        
        # If the face boxes are at least .8% of the image, it should be classiied as a portrait
        if x >= .008:
            return True
        else:
            return False

    # Main logic loop of the program
    def exceute(self, src, dst):
        os.chdir(src)
        # Get all the files in the source directory that are jpegs
        pics = glob.glob("./*.jpg")

        num_pics = len(list(pics))
        i = 0
        # Loop through pics 
        for pic in pics:
            img = cv.imread(pic)
            img2 = img.copy()
            lst = pic.split(".jpg")
            # Check if the image is a portrait
            if self.classify_portrait(pic):
                os.chdir(dst)
                # If it is a portrait blur it more (to even out skin tones) and label it a portrait in the destination directory
                cv.imwrite((lst[0] + "-EditedPortrait.jpg"), cv.GaussianBlur(self.gamma_correct(self.saturation_adjustment(self.white_balance(self.reduce_green(img2)), 1), .9), (7, 7), 0))
            else:
                os.chdir(dst)
                # If it isn't a portrait edit the image accordingly
                cv.imwrite((lst[0] + "-Edited.jpg"), cv.GaussianBlur(self.gamma_correct(self.saturation_adjustment(self.white_balance(self.reduce_green(img2)), 1), .9), (3, 3), 0))
            os.chdir(src)
            i = i + 1
            # Update the progressbar 
            self.progress['value'] = round((i / num_pics) * 100) 
            self.progress.update()
        return
if __name__ == '__main__':

    app = MonApp()
    app.mainloop()