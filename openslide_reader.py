import openslide
import numpy as np
import cv2

class slide_reader():
    def __init__(self, slide_path):
        self.slide = openslide.OpenSlide(slide_path)
        self.width = self.slide.dimensions[0]
        self.height = self.slide.dimensions[1]
        self.row = self.height//256
        self.col = self.width//256
    def get_region(self, row, col, stride, need_array=True):
        low = -(stride // 2)
        row = max(row, low)
        col = max(col, low)
        level = self.slide.get_best_level_for_downsample(stride)
        level = max(level - 1,0)
        rate = self.slide.level_downsamples[level]
        length = min(int(256//rate)*stride,self.slide.level_dimensions[level][0],self.slide.level_dimensions[level][1])
        image = self.slide.read_region(location=(int((col+low)*256), int((row+low)*256)), level=level, size=(length, length))
        if need_array:
            res = np.array(image)
            res = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
        else:
            res = image
            res = res.convert('RGB')
        return res