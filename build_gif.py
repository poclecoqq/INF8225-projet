import glob

from PIL import Image
import cv2
import numpy as np

import os

def get_image_num(name):
    print(name)
    return int(name.split('.')[0])

def make_gif(frame_folder):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('video.mp4', fourcc, 100, (640, 480))

    for image in sorted(os.listdir(frame_folder), key=get_image_num):
        
        img = cv2.imread(str(os.path.join(frame_folder, image)))
       #print(image)
        video.write(img)


    cv2.destroyAllWindows()
    video.release()
    

if __name__ == "__main__":
    make_gif(".\\generated_images\\images\\")