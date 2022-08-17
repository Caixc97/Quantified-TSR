import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import QtGui,QtCore
from PyQt5.QtGui import QPixmap,QImage
import cv2
from time import sleep
from ConvertImg import GetCoordinate, GetName
from openslide_reader import slide_reader
import numpy as np

MAX_STRIDE = 25
NUM_CATEGORY = 3
scale = 1.6
#1600,1200
DEBUG = False


def gen_grid(img,stride):
    interval = int(img.shape[0]/stride)
    width = int(img.shape[0]/224)
    for i in range(stride-1):
        img[:, (i + 1) * interval:(i + 1) * interval+width, :] = 0
        img[(i + 1) * interval:(i + 1) * interval+width, :, :] = 0


RGB_dict = {-1:[0,1,2],-2:[0,1,2],0:[1],1:[0,1],2:[0],3:[2],4:[1,2],5:[0,2],6:[1],7:[0,1],8:[0,2],9:[2],10:[1,2]}

def gen_labeled_img(img,stride,label,target_label,RGB=-1):
    interval = int(img.shape[0] / stride)
    flag = False
    RGB = RGB_dict[RGB]
    if flag is False:
        for i in range(min(stride,label.shape[0])):
            for j in range(min(stride,label.shape[1])):
                if label[i,j,target_label] == 1:
                    img[interval*i:interval*(i+1),interval*j:interval*(j+1),RGB] = img[interval*i:interval*(i+1),interval*j:interval*(j+1),RGB] / 1.5
    else:
        for i in range(stride):
            for j in range(stride):
                if label[i,j,target_label] == 1:
                    img[interval*i:interval*(i+1),interval*j:interval*(j+1),RGB] = (255 - img[interval*i:interval*(i+1),interval*j:interval*(j+1),RGB]) * 0.8 + img[interval*i:interval*(i+1),interval*j:interval*(j+1),RGB]



class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.parent = parent
        self.state = None
    def mousePressEvent(self, event):
        if event.buttons () == QtCore.Qt.LeftButton:
            width = (1000*scale) / self.parent.stride
            self.ini_row = int(event.y()/width)
            self.ini_col = int(event.x()/width)
            self.state = 'left'
            #self.parent.change_label(row-center,col-center, False)
        elif event.buttons () == QtCore.Qt.RightButton:
            width = (1000*scale) / self.parent.stride
            self.ini_row = int(event.y() / width)
            self.ini_col = int(event.x() / width)
            self.state = 'right'
            #self.parent.change_label(row - center, col - center, True)
    def mouseReleaseEvent(self, event):
        if self.state == 'left':
            width = (1000*scale) / self.parent.stride
            self.end_row = int(event.y()/width)
            self.end_col = int(event.x()/width)
            center = self.parent.stride // 2
            start_row = min(self.ini_row,self.end_row)
            end_row = max(self.ini_row,self.end_row)
            start_col = min(self.ini_col,self.end_col)
            end_col = max(self.ini_col,self.end_col)
            for row in range(start_row,end_row+1):
                for col in range(start_col,end_col+1):

                    self.parent.change_label(row - center,col-center, False)
            self.parent.update_image()
            self.state = None
        elif self.state == 'right':
            width = (1000*scale) // self.parent.stride
            self.end_row = int(event.y() / width)
            self.end_col = int(event.x() / width)
            center = self.parent.stride // 2
            start_row = min(self.ini_row,self.end_row)
            end_row = max(self.ini_row,self.end_row)
            start_col = min(self.ini_col,self.end_col)
            end_col = max(self.ini_col,self.end_col)
            for row in range(start_row,end_row+1):
                for col in range(start_col,end_col+1):
                    self.parent.change_label(row - center,col-center, True)
            self.parent.update_image()
            self.state = None
        else:
            return
    def wheelEvent(self, event):
        if self.parent.dir_path is None:
            return
        angle = event.angleDelta().y()//120
        width = 1000 / self.parent.stride
        row = int(event.y() / width)
        col = int(event.x() / width)
        step = angle * max(int(self.parent.stride/10+0.5), 1)
        ori_stride = self.parent.stride
        ori_center = ori_stride//2
        row = int(event.y() / width)
        col = int(event.x() / width)
        res_row = row - ori_center
        res_col = col - ori_center
        ratio_row = (res_row + np.sign(res_row))/ori_stride
        ratio_col = (res_col + np.sign(res_col))/ori_stride
        self.parent.stride += step
        if self.parent.stride < 1:
            self.parent.stride -= step
            return
        elif self.parent.stride > self.parent.max_stride:
            self.parent.stride -= step
            return
        new_res_row = ratio_row * self.parent.stride - np.sign(res_row)
        new_res_col = ratio_col * self.parent.stride - np.sign(res_col)
        self.parent.row += int(res_row - new_res_row)
        self.parent.col += int(res_col - new_res_col)
        self.parent.row = max(self.parent.row, self.parent.stride//2)
        self.parent.col = max(self.parent.col, self.parent.stride//2)
        self.parent.row = min(self.parent.row, self.parent.max_row - self.parent.stride // 2)
        self.parent.col = min(self.parent.col, self.parent.max_col - self.parent.stride//2)
        if self.parent.mode == 0:
            self.parent.image = np.ones((256 * self.parent.stride, 256 * self.parent.stride, 3), dtype='uint8') * 255
        if self.parent.stride > 50:
            self.parent.flag_showgrid = False
        self.parent.update_image()





class MainForm(QWidget):
    def __init__(self, name='MainForm'):
        super(MainForm, self).__init__()
        self.mode = -1
        self.setWindowTitle(name)
        self.cwd = os.getcwd()
        self.resize(1600*scale, 1200*scale)
        self.dir_path = None
        self.img_list = None
        self.flag_showall = False
        self.stride = 10
        self.flag_showgrid = True
        self.flag_showdone = False
        self.flag_annotated = False
        self.flag_recentlabel = -1
        self.max_stride = MAX_STRIDE
        self.image = np.ones((256*self.stride, 256*self.stride, 3), dtype='uint8') * 255

        # btn chooseFile
        self.btn_chooseFold = QPushButton(self)
        self.btn_chooseFold.setObjectName("btn_chooseFile")
        self.btn_chooseFold.setText("choose fold")
        self.btn_chooseFold.setGeometry(100*scale, 15*scale, 200*scale, 50*scale)

        # btn chooseTif
        self.btn_chooseTif = QPushButton(self)
        self.btn_chooseTif.setObjectName("btn_chooseTif")
        self.btn_chooseTif.setText("choose tif/mrxs")
        self.btn_chooseTif.setGeometry(350*scale, 15*scale, 200*scale, 50*scale)

        # btn generateLabel
        self.btn_generateLabel = QPushButton(self)
        self.btn_generateLabel.setObjectName("btn_generateLabel")
        self.btn_generateLabel.setText("generate label")
        self.btn_generateLabel.setGeometry(600*scale, 15*scale, 200*scale, 50*scale)

        # btn up
        self.btn_up = QPushButton(self)
        self.btn_up.setObjectName("btn_up")
        self.btn_up.setText("↑")
        self.btn_up.setGeometry(1200*scale, 50*scale, 100*scale, 100*scale)

        # btn down
        self.btn_down = QPushButton(self)
        self.btn_down.setObjectName("btn_down")
        self.btn_down.setText("↓")
        self.btn_down.setGeometry(1200*scale, 250*scale, 100*scale, 100*scale)

        # btn left
        self.btn_left = QPushButton(self)
        self.btn_left.setObjectName("btn_left")
        self.btn_left.setText("←")
        self.btn_left.setGeometry(1100*scale, 150*scale, 100*scale, 100*scale)

        # btn right
        self.btn_right = QPushButton(self)
        self.btn_right.setObjectName("btn_right")
        self.btn_right.setText("→")
        self.btn_right.setGeometry(1300*scale, 150*scale, 100*scale, 100*scale)

        # btn zoomin
        self.btn_zoomin = QPushButton(self)
        self.btn_zoomin.setObjectName("btn_zoomin")
        self.btn_zoomin.setText("zoom in")
        self.btn_zoomin.setGeometry(1450*scale, 60*scale, 100*scale, 80*scale)

        # btn zoomout
        self.btn_zoomout = QPushButton(self)
        self.btn_zoomout.setObjectName("btn_zoomout")
        self.btn_zoomout.setText("zoom out")
        self.btn_zoomout.setGeometry(1450*scale, 260*scale, 100*scale, 80*scale)

        # btn hidegride
        self.btn_hidegrid = QPushButton(self)
        self.btn_hidegrid.setObjectName("btn_hidegrid")
        self.btn_hidegrid.setText("hide/show grid")
        self.btn_hidegrid.setGeometry(1450*scale, 160*scale, 100*scale, 80*scale)

        self.exclusive_btn_list = []
        # btn tumor
        self.btn_tumor = QPushButton(self)
        self.btn_tumor.setObjectName("btn_tumor")
        self.btn_tumor.setCheckable(True)
        self.btn_tumor.setText("肿瘤")
        self.btn_tumor.setGeometry(1100*scale, 400*scale, 200*scale, 80*scale)
        self.exclusive_btn_list.append(self.btn_tumor)

        # btn neg
        self.btn_neg = QPushButton(self)
        self.btn_neg.setObjectName("btn_neg")
        self.btn_neg.setCheckable(True)
        self.btn_neg.setText("非肿瘤")
        self.btn_neg.setGeometry(1350*scale, 400*scale, 200*scale, 80*scale)
        self.exclusive_btn_list.append(self.btn_neg)

        # btn negse
        self.btn_negse = QPushButton(self)
        self.btn_negse.setObjectName("btn_negse")
        self.btn_negse.setCheckable(True)
        self.btn_negse.setText("非肿瘤（深染）")
        self.btn_negse.setGeometry(1100*scale, 500*scale, 200*scale, 80*scale)
        self.exclusive_btn_list.append(self.btn_negse)

        # btn done
        self.btn_done = QPushButton(self)
        self.btn_done.setObjectName("btn_done")
        self.btn_done.setCheckable(True)
        self.btn_done.setText("done")
        self.btn_done.setGeometry(1225*scale, 1100*scale, 200*scale, 80*scale)

        # btn annotated region
        self.btn_annotated = QPushButton(self)
        self.btn_annotated.setObjectName("btn_annotated")
        self.btn_annotated.setCheckable(True)
        self.btn_annotated.setText("choose annotated region")
        self.btn_annotated.setGeometry(1225*scale, 900*scale, 200*scale, 80*scale)

        # btn show all
        self.btn_showall = QPushButton(self)
        self.btn_showall.setObjectName("btn_showall")
        self.btn_showall.setCheckable(True)
        self.btn_showall.setText("show all annotation")
        self.btn_showall.setGeometry(1225*scale, 1000*scale, 200*scale, 80*scale)

        # image
        self.image_viewer = ImageViewer(self)
        self.image_viewer.setText("未选择样本")
        self.image_viewer.setFixedSize(1000*scale, 1000*scale)
        self.image_viewer.move(50*scale, 100*scale)

        #coordinate
        self.label_coor = QLabel(self)
        self.label_coor.resize(300*scale, 100*scale)
        self.label_coor.setText("")
        self.label_coor.move(50*scale, 1100*scale)

        # slot
        self.btn_chooseFold.clicked.connect(self.slot_btn_chooseFold)
        self.btn_chooseTif.clicked.connect(self.slot_btn_chooseTif)
        self.btn_generateLabel.clicked.connect(self.slot_generatelabel)
        self.btn_up.clicked.connect(self.slot_btn_up)
        self.btn_down.clicked.connect(self.slot_btn_down)
        self.btn_left.clicked.connect(self.slot_btn_left)
        self.btn_right.clicked.connect(self.slot_btn_right)
        self.btn_zoomin.clicked.connect(self.slot_zoomin)
        self.btn_zoomout.clicked.connect(self.slot_zoomout)
        self.btn_hidegrid.clicked.connect(self.slot_hidegrid)
        self.btn_tumor.clicked.connect(self.slot_tumor)
        self.btn_neg.clicked.connect(self.slot_neg)
        self.btn_negse.clicked.connect(self.slot_negse)
        self.btn_done.clicked.connect(self.slot_done)
        self.btn_annotated.clicked.connect(self.slot_annotated)
        self.btn_showall.clicked.connect(self.slot_showall)

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(850*scale, 30*scale, 200*scale, 30*scale)


    def slot_btn_chooseFold(self):
        dir_path = QFileDialog.getExistingDirectory(None, "选取文件夹", ".")
        if len(self.dir_path) == 0:
            return
        self.dir_path = dir_path
        self.reset()
        self.img_list = os.listdir(self.dir_path)
        self.mode = 0
        for file in self.img_list:
            if file.endswith('jpeg'):
                self.row, self.col = GetCoordinate(file)
                while self.row < self.stride//2:
                    self.row += 1
                while self.col < self.stride//2:
                    self.col += 1
                self.name = file[:file.rfind('(')]
                self.update_image()
                self.max_row = 0
                self.max_col = 0
                for file in self.img_list:
                    if file.endswith('jpeg'):
                        row, col = GetCoordinate(file)
                        if row > self.max_row:
                            self.max_row = row
                        if col > self.max_col:
                            self.max_col = col
                self.image = np.ones((256 * self.stride, 256 * self.stride, 3), dtype='uint8') * 255
                self.label = np.ones(shape=(self.max_row+self.max_stride, self.max_col+self.max_stride, NUM_CATEGORY), dtype='uint8') * 255
                self.done = np.zeros(shape=(self.max_row+self.max_stride, self.max_col+self.max_stride, 1), dtype='uint8')
                self.annotated = np.zeros(shape=(self.max_row + self.max_stride, self.max_col + self.max_stride, 1),
                                          dtype='uint8')
                if self.mode == 0:
                    save_path = os.path.join(self.dir_path, 'LabeledImage')
                elif self.dir_path.endswith('.tif'):
                    save_path = os.path.join(self.dir_path[:self.dir_path.rfind('.tif')] + ' annotations-SE',
                                             'LabeledImage')
                elif self.dir_path.endswith('.mrxs'):
                    save_path = os.path.join(self.dir_path[:self.dir_path.rfind('.mrxs')] + ' annotations-SE',
                                             'LabeledImage')
                try:
                    pre_label = np.load(os.path.join(save_path, 'all_label.npy'))
                    assert (pre_label.shape == self.label.shape)
                    self.label = pre_label
                    self.done[self.label[:, :, 0] != 255] = 1
                except:
                    pass
                self.max_stride = MAX_STRIDE
                return
        self.image_viewer.setText(self.dir_path+"没有有效文件")

    def slot_btn_chooseTif(self):
        dir_path, filetype = QFileDialog.getOpenFileName(None, "选取tif/mrxs文件",'.','*.tif;*.mrxs;*.ndpi')
        if len(dir_path) == 0:
            return
        self.reset()
        self.dir_path = dir_path
        self.mode = 1
        if self.dir_path.endswith('.tif'):
            self.name = os.path.basename(self.dir_path[:self.dir_path.rfind('.tif')])
        elif self.dir_path.endswith('.mrxs'):
            self.name = os.path.basename(self.dir_path[:self.dir_path.rfind('.mrxs')])
        elif self.dir_path.endswith('.ndpi'):
            self.name = os.path.basename(self.dir_path[:self.dir_path.rfind('.ndpi')])
        self.reader = slide_reader(self.dir_path)
        if self.reader.slide.level_count != 1:
            self.max_stride = min(self.reader.row, self.reader.col)
        else:
            self.max_stride = MAX_STRIDE
        self.max_row = self.reader.row
        self.max_col = self.reader.col
        self.row = self.stride//2
        self.col = self.stride//2
        self.label = np.ones(shape=(self.max_row, self.max_col, NUM_CATEGORY),
                             dtype='uint8') * 255
        self.annotated = np.zeros(shape=(self.max_row, self.max_col, 1),
                                  dtype='uint8')
        self.done = np.zeros(shape=(self.max_row, self.max_col, 1), dtype='uint8')
        if self.mode == 0:
            save_path = os.path.join(self.dir_path, 'LabeledImage')
        elif self.dir_path.endswith('.tif'):
            save_path = os.path.join(self.dir_path[:self.dir_path.rfind('.tif')] + ' annotations-SE',
                                     'LabeledImage')
        elif self.dir_path.endswith('.mrxs'):
            save_path = os.path.join(self.dir_path[:self.dir_path.rfind('.mrxs')] + ' annotations-SE',
                                     'LabeledImage')
        try:
            pre_label = np.load(os.path.join(save_path, 'all_label.npy'))
            self.label = pre_label
            self.done = np.zeros(shape=(self.label.shape[0], self.label.shape[1], 1), dtype='uint8')
            self.done[self.label[:, :, 0] != 255] = 1
        except:
            pass
        self.update_image()

    def update_image(self):
        if self.dir_path is None:
            return
        if self.stride % 2 == 1:
            low = -(self.stride // 2)
            high = (self.stride // 2) + 1
        else:
            low = -(self.stride // 2)
            high = self.stride // 2
        del self.image
        self.image = self.reader.get_region(self.row,self.col,self.stride)
        if self.flag_showgrid:
            gen_grid(self.image, self.stride)
        if self.flag_showdone:
            gen_labeled_img(self.image, self.stride,
                            self.done[
                            self.row + low:self.row + high,
                            self.col + low:self.col + high
                            ], target_label=0, RGB=-1)
        if self.flag_showall is True:
            for i in range(NUM_CATEGORY):
                gen_labeled_img(self.image, self.stride,
                                self.label[
                                self.row + low:self.row + high,
                                self.col + low:self.col + high
                                ], target_label=i, RGB = i)
        elif self.flag_recentlabel != -1:
            gen_labeled_img(self.image, self.stride,
                            self.label[
                            self.row + low:self.row + high,
                            self.col + low:self.col + high
                            ], target_label=self.flag_recentlabel, RGB = self.flag_recentlabel)
        if self.flag_annotated is True:
            gen_labeled_img(self.image, self.stride,
                            self.annotated[
                            self.row + low:self.row + high,
                            self.col + low:self.col + high
                            ], target_label=0, RGB = -2)
        self.label_coor.setText('row:'+str(self.row)+', '+'col:'+str(self.col)+', width:'+str(self.stride))
        #qimage = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3, QtGui.QImage.Format_BGR888)
        if self.mode == 0:
            img2 = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            img2 = cv2.cvtColor(self.image, cv2.COLOR_RGBA2RGB)
        qimage = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg = QPixmap.fromImage(qimage)
        jpg = jpg.scaled(self.image_viewer.width(),self.image_viewer.height())
        self.image_viewer.setPixmap(jpg)


    def slot_btn_up(self):
        if self.dir_path is None:
            return
        if self.row < self.stride:
            self.row = self.stride//2
            self.update_image()
            return
        self.row -= max(self.stride//2,1)
        self.update_image()

    def slot_btn_down(self):
        if self.dir_path is None:
            return
        if self.row >= self.max_row - 1.5 * self.stride:
            self.row = self.max_row - self.stride//2
            self.update_image()
            return
        self.row += max(self.stride // 2,1)
        self.update_image()

    def slot_btn_left(self):
        if self.dir_path is None:
            return
        if self.col < self.stride:
            self.col = self.stride//2
            self.update_image()
            return
        self.col -= max(self.stride//2,1)
        self.update_image()

    def slot_btn_right(self):
        if self.dir_path is None:
            return
        if self.col >= self.max_col - 1.5 * self.stride:
            self.col = self.max_col - self.stride//2
            self.update_image()
            return
        self.col += max(self.stride//2,1)
        self.update_image()

    def slot_zoomin(self):
        if self.dir_path is None:
            return
        if self.stride == 1:
            return
        self.stride -= 1
        if self.mode == 0:
            del self.image
            self.image = np.ones((256 * self.stride, 256 * self.stride, 3), dtype='uint8') * 255
        self.update_image()

    def slot_zoomout(self):
        if self.dir_path is None:
            return
        if self.stride == self.max_stride:
            return
        self.stride += 1
        if self.mode == 0:
            del self.image
            self.image = np.ones((256 * self.stride, 256 * self.stride, 3), dtype='uint8') * 255
        while self.row < self.stride // 2:
            self.row += 1
        while self.col < self.stride // 2:
            self.col += 1
        if self.stride > 50:
            self.flag_showgrid = False
        self.update_image()

    def slot_hidegrid(self):
        if self.dir_path is None:
            return
        self.flag_showgrid = not self.flag_showgrid
        self.update_image()

    def slot_tumor(self):
        label = 0
        if self.flag_recentlabel == label:
            self.flag_recentlabel = -1
        else:
            self.flag_recentlabel = label
            for i in range(NUM_CATEGORY):
                if i == label:
                    continue
                self.exclusive_btn_list[i].setChecked(False)
        self.update_image()

    def slot_neg(self):
        label = 1
        if self.flag_recentlabel == label:
            self.flag_recentlabel = -1
        else:
            self.flag_recentlabel = label
            for i in range(NUM_CATEGORY):
                if i == label:
                    continue
                self.exclusive_btn_list[i].setChecked(False)
        self.update_image()

    def slot_negse(self):
        label = 2
        if self.flag_recentlabel == label:
            self.flag_recentlabel = -1
        else:
            self.flag_recentlabel = label
            for i in range(NUM_CATEGORY):
                if i == label:
                    continue
                self.exclusive_btn_list[i].setChecked(False)
        self.update_image()

    def slot_done(self):
        self.flag_showdone = not self.flag_showdone
        self.update_image()

    def slot_annotated(self):
        self.flag_annotated = not self.flag_annotated
        self.update_image()

    def slot_showall(self):
        self.flag_showall = not self.flag_showall
        self.update_image()

    def slot_generatelabel(self):
        # import line_profiler
        # profile = line_profiler.LineProfiler(self.temp)
        # profile.enable()
        self.temp()
        # profile.disable()  # 停止分析
        # profile.print_stats(sys.stdout)

    def temp(self):
        if self.dir_path is None:
            return
        if self.mode == 0:
            save_path = os.path.join(self.dir_path, 'LabeledImage')
        elif self.dir_path.endswith('.tif'):
            save_path = os.path.join(self.dir_path[:self.dir_path.rfind('.tif')]+' annotations-SE', 'LabeledImage')
            if not os.path.exists(self.dir_path[:self.dir_path.rfind('.tif')]+' annotations-SE'):
                os.makedirs(self.dir_path[:self.dir_path.rfind('.tif')]+' annotations-SE')
        elif self.dir_path.endswith('.mrxs'):
            save_path = os.path.join(self.dir_path[:self.dir_path.rfind('.mrxs')]+' annotations-SE', 'LabeledImage')
            if not os.path.exists(self.dir_path[:self.dir_path.rfind('.mrxs')]+' annotations-SE'):
                os.makedirs(self.dir_path[:self.dir_path.rfind('.mrxs')]+' annotations-SE')
        elif self.dir_path.endswith('.ndpi'):
            save_path = os.path.join(self.dir_path[:self.dir_path.rfind('.ndpi')]+' annotations-SE', 'LabeledImage')
            if not os.path.exists(self.dir_path[:self.dir_path.rfind('.ndpi')]+' annotations-SE'):
                os.makedirs(self.dir_path[:self.dir_path.rfind('.ndpi')]+' annotations-SE')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_list = os.listdir(save_path)
        self.pbar.setValue(0)
        total = np.sum(self.annotated[:,:,0])
        count = 0
        for row in range(1, self.max_row-1):
            for col in range(1, self.max_col-1):
                if self.annotated[row, col, 0] != 1:
                    continue
                if self.done[row, col] == 1:
                    for file in file_list:
                        if self.name + '(' + str(row) + ',' + str(col) + ')' in file:
                            os.remove(os.path.join(save_path,file))
                temp = self.reader.get_region(row, col, 1, False)
                count += 1
                self.pbar.setValue(int(count / total * 100))
                label = self.label[row, col, :]
                label[label == 255] = 0
                if label[0] == 1:
                    label_name = '1'
                elif label[1] == 1:
                    label_name = '0'
                elif label[2] == 1:
                    label_name = '2'
                else:
                    continue
                file_name = self.name + '(' + str(row) + ',' + str(col) + ')' + label_name + '.jpeg'
                if self.mode == 0:
                    cv2.imwrite(os.path.join(save_path, file_name), temp)
                else:
                    with open(os.path.join(save_path, file_name), 'w') as file_output:
                        temp.save(file_output, format='jpeg')
                self.done[row, col, :] = 1
        self.pbar.setValue(100)
        np.save(os.path.join(save_path, 'all_label.npy'), self.label)


    def change_label(self, relative_row, relative_col, is_cancel):
        if self.dir_path is None:
            return
        if self.flag_recentlabel == -1 and self.flag_annotated is False:
            return
        if is_cancel is False:
            target = 1
        else:
            target = 0
        if self.flag_annotated is True:
            if self.annotated[self.row+relative_row, self.col+relative_col, 0] == target:
                return
            self.annotated[self.row+relative_row, self.col+relative_col, 0] = target
        else:
            if self.label[self.row+relative_row, self.col+relative_col, self.flag_recentlabel] == target:
                return
            self.label[self.row+relative_row, self.col+relative_col, self.flag_recentlabel] = target

    def reset(self):
        for i in range(NUM_CATEGORY):
            self.exclusive_btn_list[i].setChecked(False)
        self.img_list = None
        self.stride = 10
        self.flag_showall = False
        self.flag_showgrid = True
        self.flag_showdone = False
        self.flag_recentlabel = -1
        self.flag_annotated = False
        self.max_stride = MAX_STRIDE
        self.image = np.ones((256*self.stride, 256*self.stride, 3), dtype='uint8') * 255






if __name__=="__main__":
    app = QApplication(sys.argv)
    mainForm = MainForm('测试QFileDialog')
    mainForm.show()
    sys.exit(app.exec_())