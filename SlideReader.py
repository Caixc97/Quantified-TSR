import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import QtGui,QtCore
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
import cv2
import pickle
from openslide_reader import slide_reader
import numpy as np
import time
from tqdm import tqdm


threshold = 2
RGB_dict = {-1:[1,1,1], #white
            -2:[255,255,255],#black
            0:[139,26,26], #firebrick
            1:[255,160,122], #lightsalmon
            2:[255,231,186], #wheat
            3:[1,255,255], #cyan
            4:[1,1,255], #blue
            5:[34,139,34], #ForestGreen
            6:[255,1,1], #hotpink
            7:[160,32,240], #purple
            8:[255,246,143], #khaiki
            9:[255,215,1] #gold
            }

MAX_STRIDE = 25
NUM_CATEGORY = 3
scale = 0.8
#1600,1200

def get_tsr_map(tumor_label, tumor_region, radius=40, stride=1):
    roi_minrow = np.min(tumor_region.nonzero()[0])
    roi_maxrow = np.max(tumor_region.nonzero()[0])
    roi_mincol = np.min(tumor_region.nonzero()[1])
    roi_maxcol = np.max(tumor_region.nonzero()[1])
    tsr_map = np.zeros((tumor_label.shape[0]//stride, tumor_label.shape[1]//stride))
    for row in range(max(roi_minrow - radius, 0), roi_maxrow + radius):
        for col in range(max(roi_mincol - radius, 0), roi_maxcol + radius):
            window_label = tumor_label[max(row - radius, 0):row + radius, max(col - radius, 0):col + radius]
            tsr_map[row//stride][col//stride] = np.sum(window_label)/(window_label.shape[0] * window_label.shape[1])
    return tsr_map

def find_min_region(tumor_label, tumor_region, winLength=264, winWidth=154):
        roi = tumor_region.nonzero()
        min_coor = (-1,-1)
        min_tsr = 1
        winAera = winWidth * winLength
        for row in np.unique(roi[0]):
            for col in roi[1][roi[0]==row]:
                window_label = tumor_label[max(row - winWidth//2, 0):row + winWidth//2, max(col - winLength//2, 0):col + winLength//2]
                window_region = tumor_region[max(row - winWidth//2, 0):row + winWidth//2, max(col - winLength//2, 0):col + winLength//2]
                if np.sum(window_region)/winAera < 0.8:
                    continue
                if np.sum(window_label[:winWidth // 2, :]) * np.sum(window_label[winWidth // 2:, :]) == 0:
                    continue
                if np.sum(window_label[:, :winLength // 2]) * np.sum(window_label[:, winLength // 2:]) == 0:
                    continue
                tsr = np.sum(window_label[window_region])/np.sum(window_region)
                if tsr < min_tsr:
                    min_tsr = tsr
                    min_coor = (row,col)
        return min_coor, min_tsr


class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.parent = parent
        self.state = None
    def wheelEvent(self, event):
        if self.parent.dir_path == None:
            return
        angle = event.angleDelta().y()//120
        width = 1000 / self.parent.stride
        step = angle * max(int(self.parent.stride/10+0.5), 1)
        ori_stride = self.parent.stride
        ori_center = ori_stride/2
        row = event.y() / width
        col = event.x() / width
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
        self.parent.col = min(self.parent.col, self.parent.max_col - self.parent.stride // 2)
        self.parent.update_image()
    def mousePressEvent(self, event):
        if self.parent.flag_isedit is False and self.parent.flag_select is False:
            return
        elif self.parent.cache_path is None:
            return
        width = (1000 * scale) / (self.parent.stride * 8)
        if event.buttons () == QtCore.Qt.LeftButton:
            self.ini_row = int(event.y()//width)
            self.ini_col = int(event.x()//width)
            self.state = 'left'
            #self.parent.change_label(row-center,col-center, False)
        elif event.buttons () == QtCore.Qt.RightButton:
            self.ini_row = int(event.y() // width)
            self.ini_col = int(event.x() // width)
            self.state = 'right'
            #self.parent.change_label(row - center, col - center, True)
    def mouseReleaseEvent(self, event):
        width = (1000 * scale) / (self.parent.stride * 8)
        if self.state == 'left':
            self.end_row = int(event.y()//width)
            self.end_col = int(event.x()//width)
            start_row = min(self.ini_row,self.end_row)
            end_row = max(self.ini_row,self.end_row)
            start_col = min(self.ini_col,self.end_col)
            end_col = max(self.ini_col,self.end_col)
            if self.parent.flag_isedit == True:
                self.parent.change_label(start_row, end_row+1, start_col, end_col+1, False)
                self.parent.update_image()
            elif self.parent.flag_select == True:
                self.parent.select_window(self.end_row, self.end_col)
                self.parent.update_image()
            self.state = None
        elif self.state == 'right':
            self.end_row = int(event.y() // width)
            self.end_col = int(event.x() // width)
            start_row = min(self.ini_row,self.end_row)
            end_row = max(self.ini_row,self.end_row)
            start_col = min(self.ini_col,self.end_col)
            end_col = max(self.ini_col,self.end_col)
            if self.parent.flag_isedit == True:
                self.parent.change_label(start_row, end_row+1, start_col, end_col+1, True)
                self.parent.update_image()
            self.state = None
        else:
            return

class MainForm(QWidget):
    def __init__(self, name='MainForm'):
        super(MainForm, self).__init__()
        self.threshold = 2
        self.setWindowTitle(name)
        self.cwd = os.getcwd()
        self.resize(int(1800*scale), int(1200*scale))
        self.exclusive_btn_list = []

        # info panel
        self.info_panel = ImageViewer(self)
        self.info = "No slide"
        self.info_panel.setText(self.info)
        self.info_panel.setFixedSize(int(300*scale), int(400*scale))
        self.info_panel.move(int(1600*scale), int(50*scale))

        # btn chooseAnno
        self.btn_chooseAnno = QPushButton(self)
        self.btn_chooseAnno.setObjectName("btn_chooseFile")
        self.btn_chooseAnno.setText("load annotations")
        self.btn_chooseAnno.setGeometry(int(100*scale), int(15*scale), int(200*scale), int(50*scale))

        # btn chooseTif
        self.btn_chooseTif = QPushButton(self)
        self.btn_chooseTif.setObjectName("btn_chooseTif")
        self.btn_chooseTif.setText("load tif/mrxs/ndpi")
        self.btn_chooseTif.setGeometry(int(350*scale), int(15*scale), int(200*scale), int(50*scale))

        # btn up
        self.btn_up = QPushButton(self)
        self.btn_up.setObjectName("btn_up")
        self.btn_up.setText("↑")
        self.btn_up.setGeometry(int(1200*scale), int(50*scale), int(100*scale), int(100*scale))

        # btn down
        self.btn_down = QPushButton(self)
        self.btn_down.setObjectName("btn_down")
        self.btn_down.setText("↓")
        self.btn_down.setGeometry(int(1200*scale), int(250*scale), int(100*scale), int(100*scale))

        # btn left
        self.btn_left = QPushButton(self)
        self.btn_left.setObjectName("btn_left")
        self.btn_left.setText("←")
        self.btn_left.setGeometry(int(1100*scale), int(150*scale), int(100*scale), int(100*scale))

        # btn right
        self.btn_right = QPushButton(self)
        self.btn_right.setObjectName("btn_right")
        self.btn_right.setText("→")
        self.btn_right.setGeometry(int(1300*scale), int(150*scale), int(100*scale), int(100*scale))

        # btn zoomin
        self.btn_zoomin = QPushButton(self)
        self.btn_zoomin.setObjectName("btn_zoomin")
        self.btn_zoomin.setText("zoom in")
        self.btn_zoomin.setGeometry(int(1450*scale), int(60*scale), int(100*scale), int(80*scale))

        # btn zoomout
        self.btn_zoomout = QPushButton(self)
        self.btn_zoomout.setObjectName("btn_zoomout")
        self.btn_zoomout.setText("zoom out")
        self.btn_zoomout.setGeometry(int(1450*scale), int(260*scale), int(100*scale), int(80*scale))

        # btn tumor
        self.btn_tumor = QPushButton(self)
        self.btn_tumor.setObjectName("btn_tumor")
        self.btn_tumor.setCheckable(True)
        self.btn_tumor.setText("Tumor")
        self.btn_tumor.setGeometry(int(1100*scale), int(400*scale), int(200*scale), int(80*scale))
        self.exclusive_btn_list.append(self.btn_tumor)

        # btn show tumor region
        self.btn_show_tumor_region = QPushButton(self)
        self.btn_show_tumor_region.setObjectName("btn_show_tumor_region")
        self.btn_show_tumor_region.setCheckable(True)
        self.btn_show_tumor_region.setText("Tumor region")
        self.btn_show_tumor_region.setGeometry(int(1100*scale), int(500*scale), int(200*scale), int(80*scale))
        self.exclusive_btn_list.append(self.btn_show_tumor_region)

        # btn show nest region
        self.btn_show_nest_region = QPushButton(self)
        self.btn_show_nest_region.setObjectName("btn_show_nest_region")
        self.btn_show_nest_region.setCheckable(True)
        self.btn_show_nest_region.setText("Nest")
        self.btn_show_nest_region.setGeometry(int(1100*scale), int(600*scale), int(200*scale), int(80*scale))
        self.exclusive_btn_list.append(self.btn_show_nest_region)

        # btn show margin region
        self.btn_show_margin_region = QPushButton(self)
        self.btn_show_margin_region.setObjectName("btn_show_margin_region")
        self.btn_show_margin_region.setCheckable(True)
        self.btn_show_margin_region.setText("ITF")
        self.btn_show_margin_region.setGeometry(int(1100*scale), int(700*scale), int(200*scale), int(80*scale))
        self.exclusive_btn_list.append(self.btn_show_margin_region)

        # btn show tsr map
        self.btn_show_tsr_map = QPushButton(self)
        self.btn_show_tsr_map.setObjectName("btn_show_tsr_map")
        self.btn_show_tsr_map.setCheckable(True)
        self.btn_show_tsr_map.setText("TSR heatmap")
        self.btn_show_tsr_map.setGeometry(int(1100*scale), int(800*scale), int(200*scale), int(80*scale))
        self.exclusive_btn_list.append(self.btn_show_tsr_map)

        # btn choose window
        self.btn_choose_window = QPushButton(self)
        self.btn_choose_window.setObjectName("btn_choose_window")
        self.btn_choose_window.setCheckable(True)
        self.btn_choose_window.setText("Window")
        self.btn_choose_window.setGeometry(int(1100*scale), int(1000*scale), int(100*scale), int(40*scale))

        # btn make image
        self.btn_mkimage = QPushButton(self)
        self.btn_mkimage.setObjectName("btn_mkimage")
        self.btn_mkimage.setText("Make img")
        self.btn_mkimage.setGeometry(int(1100 * scale), int(1050 * scale), int(100 * scale), int(40 * scale))
        # self.btn_mkimage.clicked.connect(self.slot_mkimage)
        self.btn_mkimage.clicked.connect(self.save_fig)

        # btn former
        self.btn_former = QPushButton(self)
        self.btn_former.setObjectName("btn_former")
        self.btn_former.setText("Last slide")
        self.btn_former.setGeometry(int(1350*scale), int(400*scale), int(200*scale), int(80*scale))

        # btn next
        self.btn_next = QPushButton(self)
        self.btn_next.setObjectName("btn_next")
        self.btn_next.setText("Next slide")
        self.btn_next.setGeometry(int(1350*scale), int(500*scale), int(200*scale), int(80*scale))

        # btn edit
        self.btn_edit = QPushButton(self)
        self.btn_edit.setObjectName("btn_edit")
        self.btn_edit.setCheckable(True)
        self.btn_edit.setText("Modification mode")
        self.btn_edit.setGeometry(int(1350*scale), int(600*scale), int(200*scale), int(80*scale))

        # btn recalculate
        self.btn_recalculate = QPushButton(self)
        self.btn_recalculate.setObjectName("btn_recalculate")
        self.btn_recalculate.setText("Recalculate")
        self.btn_recalculate.setGeometry(int(1350 * scale), int(700 * scale), int(200 * scale), int(80 * scale))

        # btn save label
        self.btn_savelabel = QPushButton(self)
        self.btn_savelabel.setObjectName("btn_savelabel")
        self.btn_savelabel.setText("Save modification")
        self.btn_savelabel.setGeometry(int(1350 * scale), int(800 * scale), int(200 * scale), int(80 * scale))

        # btn save label
        self.btn_switchmargin = QPushButton(self)
        self.btn_switchmargin.setObjectName("btn_switchmargin")
        self.btn_switchmargin.setText("switch margin")
        self.btn_switchmargin.setGeometry(int(1570 * scale), int(400 * scale), int(200 * scale), int(80 * scale))
        self.btn_switchmargin.clicked.connect(self.switch_margin)

        # alpha slider
        self.slider_alpha = QSlider(self)
        self.slider_alpha.setGeometry(int(1450*scale), int(900*scale), int(30*scale), int(200*scale))
        self.slider_alpha.setMaximum(100)
        self.slider_alpha.setMinimum(0)
        self.slider_alpha.setSingleStep(5)
        self.slider_alpha.setValue(50)

        # image
        self.image_viewer = ImageViewer(self)
        self.image_viewer.setText("No slide")
        self.image_viewer.setFixedSize(int(1000*scale), int(1000*scale))
        self.image_viewer.move(int(50*scale), int(100*scale))

        # slot
        self.btn_chooseAnno.clicked.connect(self.slot_btn_chooseAnno)
        self.btn_chooseTif.clicked.connect(self.slot_btn_chooseTif)
        self.btn_up.clicked.connect(self.slot_btn_up)
        self.btn_down.clicked.connect(self.slot_btn_down)
        self.btn_left.clicked.connect(self.slot_btn_left)
        self.btn_right.clicked.connect(self.slot_btn_right)
        self.btn_zoomin.clicked.connect(self.slot_zoomin)
        self.btn_zoomout.clicked.connect(self.slot_zoomout)
        self.btn_tumor.clicked.connect(self.slot_tumor)
        self.btn_show_nest_region.clicked.connect(self.slot_show_nest_region)
        self.btn_show_margin_region.clicked.connect(self.slot_show_margin_region)
        self.btn_show_tumor_region.clicked.connect(self.slot_show_tumor_region)
        self.btn_show_tsr_map.clicked.connect(self.slot_show_tsr_map)
        self.btn_choose_window.clicked.connect(self.slot_choose_window)
        self.slider_alpha.valueChanged.connect(self.change_alpha)
        self.btn_edit.clicked.connect(self.slot_edit)
        self.btn_recalculate.clicked.connect(self.slot_recalculate)
        self.btn_savelabel.clicked.connect(self.slot_savelabel)
        self.btn_next.clicked.connect(self.slot_next)
        self.btn_former.clicked.connect(self.slot_former)


        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(int(850*scale), int(30*scale), int(200*scale), int(30*scale))
        self.reset()

        # label
        # coordinate
        self.label_coor = QLabel(self)
        self.label_coor.resize(int(500*scale), int(100*scale))
        self.label_coor.setText("")
        self.label_coor.move(int(50*scale), int(1100*scale))
        # win length
        self.label_winLength = QLabel(self)
        self.label_winLength.resize(int(250*scale), int(30*scale))
        self.label_winLength.setText("window length:{0:.2f}mm".format(self.winLength*8.75/1000))
        self.label_winLength.move(int(1100 * scale), int(900 * scale))
        self.btn_editLength = QPushButton(self)
        self.btn_editLength.setText("edit")
        self.btn_editLength.setGeometry(int(1330*scale), int(905*scale), int(80*scale), int(30*scale))
        self.btn_editLength.clicked.connect(self.editLength)
        # win width
        self.label_winWidth = QLabel(self)
        self.label_winWidth.resize(int(250*scale), int(30*scale))
        self.label_winWidth.setText("window width:{0:.2f}mm".format(self.winWidth*8.75/1000))
        self.label_winWidth.move(int(1100 * scale), int(930 * scale))
        self.btn_editWidth = QPushButton(self)
        self.btn_editWidth.setText("edit")
        self.btn_editWidth.setGeometry(int(1330*scale), int(935*scale), int(80*scale), int(30*scale))
        self.btn_editWidth.clicked.connect(self.editWidth)
        # win size
        self.label_winSize = QLabel(self)
        self.label_winSize.resize(int(250*scale), int(30*scale))
        self.label_winSize.setText("window aera:{0:.2f}mm²".format(self.winWidth*8.75*self.winLength*8.75/1000000))
        self.label_winSize.move(int(1100 * scale), int(960 * scale))

    def gen_labeled_img(self, img, label_img):
        label_img = cv2.resize(label_img, (img.shape[1], img.shape[0]))
        slice = label_img[:,:,0] != 0
        img[slice] = (1 - self.alpha_ratio) * img[slice] + self.alpha_ratio * label_img[slice]
        img[img < 0] = 0

    def gen_window(self, img, label_img):
        label_img = cv2.resize(label_img, (img.shape[0], img.shape[1]))
        slice = label_img[:, :, 0] != 0
        img[slice] = 0

    def change_alpha(self):
        # self.threshold = round(self.slider_alpha.value()/20, 1)
        # print(self.threshold)
        # self.update_image()
        last_alpha = self.alpha_ratio
        self.alpha_ratio = round(self.slider_alpha.value()/100, 1)
        if last_alpha != self.alpha_ratio:
            self.update_image()

    def slot_btn_chooseAnno(self):
        dir_path = QFileDialog.getExistingDirectory(None, "选取文件夹", ".")
        if len(dir_path) == 0:
            return
        self.cache_path = dir_path
        self.load_cache()
        # self.label = np.load(self.dir_path)

    def load_cache(self):
        try:
            self.tissue_label = np.load(os.path.join(os.path.dirname(self.cache_path),'tissue_label.npy'))
            with open(os.path.join(self.cache_path, 'label'), 'rb') as file_obj:
                self.label = pickle.load(file_obj)
            with open(os.path.join(self.cache_path, 'label_images'), 'rb') as file_obj:
                self.label_images = pickle.load(file_obj)
            self.label_images.append(np.zeros((self.label_images[0].shape[0],self.label_images[0].shape[1],3)))
            with open(os.path.join(self.cache_path, 'tumor_region'), 'rb') as file_obj:
                self.tumor_region = pickle.load(file_obj)
            with open(os.path.join(self.cache_path, 'nest_region'), 'rb') as file_obj:
                self.nest_region = pickle.load(file_obj)
            with open(os.path.join(self.cache_path, 'margin_region'), 'rb') as file_obj:
                self.margin_region = pickle.load(file_obj)
            with open(os.path.join(self.cache_path, 'tsr_map'), 'rb') as file_obj:
                self.tsr_map = pickle.load(file_obj)
            with open(os.path.join(self.cache_path, 'min_coor'), 'rb') as file_obj:
                min_coor = pickle.load(file_obj)
                if type(min_coor[0]) == tuple:
                    self.selected_window[0] = min_coor[0][0]
                    self.selected_window[1] = min_coor[0][1]
                    self.winLength = min_coor[1][0]
                    self.winWidth = min_coor[1][1]
                else:
                    self.selected_window[0] = min_coor[0]
                    self.selected_window[1] = min_coor[1]
            self.label_images[3][:] = 0
            self.label_images[3][self.margin_region] = RGB_dict[4]
            self.margin_flag = 'new'
            self.reset_panel()
            self.gen_window_line()
            self.label_winLength.setText(
                "window Length:{0:.2f}mm".format(self.winLength * 8.75 / 1000))
            self.label_winWidth.setText(
                "window Width:{0:.2f}mm".format(self.winWidth * 8.75 / 1000))
            self.label_winSize.setText(
                "window aera:{0:.2f}mm²".format(self.winWidth * 8.75 * self.winLength * 8.75 / 1000000))
        except:
            print('Anno load fail')

    def slot_btn_chooseTif(self):
        dir_path, filetype = QFileDialog.getOpenFileName(None, "选取tif/mrxs/sys/ndpi文件",'.','*.tif;*.mrxs;*.svs;*.ndpi')
        if len(dir_path) == 0:
            return
        self.reset()
        self.dir_path = dir_path
        self.load_data()

    def load_data(self):
        if self.dir_path.endswith('.tif'):
            self.name = os.path.basename(self.dir_path[:self.dir_path.rfind('.tif')])
        else:
            self.name = os.path.basename(self.dir_path[:self.dir_path.rfind('.mrxs')])
        self.reader = slide_reader(self.dir_path)
        if self.reader.slide.level_count != 1:
            self.max_stride = min(self.reader.row, self.reader.col)
        else:
            self.max_stride = MAX_STRIDE
        self.max_row = self.reader.row
        self.max_col = self.reader.col
        self.row = int(self.max_row / 2)
        self.col = int(self.max_col / 2)
        self.stride = 100
        self.reset_panel()
        self.update_image()

    def update_image(self):
        if self.dir_path is None:
            return
        if self.stride % 2 == 1:
            low = -(self.stride // 2)
            high = (self.stride // 2) + 1
        else:
            low = -self.stride // 2
            high = self.stride // 2

        del self.image
        self.image = self.reader.get_region(self.row,self.col,self.stride)
        if type(self.label) != type(None):
            for i, flag in enumerate(self.flag_showlabel):
                if flag == True:
                    # this_label = self.tissue_label[self.row*8 + low*8:self.row*8 + high*8, self.col*8 + low*8:self.col*8 + high*8]
                    # this_label = this_label > self.threshold
                    # label_img = np.zeros((this_label.shape[0], this_label.shape[1], 3), dtype='uint8')
                    # label_img[this_label] = RGB_dict[1]
                    # self.gen_labeled_img(self.image, label_img)
                    self.gen_labeled_img(self.image, self.label_images[i][
                                    self.row*8 + low*8:self.row*8 + high*8,
                                    self.col*8 + low*8:self.col*8 + high*8
                                    ])
            if self.flag_select == True:
                self.gen_window(self.image, self.label_images[5][
                                                 self.row * 8 + low * 8:self.row * 8 + high * 8,
                                                 self.col * 8 + low * 8:self.col * 8 + high * 8
                                                 ])
        self.label_coor.setText(str(self.name)+': row:'+str(self.row)+', '+'col:'+str(self.col)+', width:'+str(self.stride))
        #qimage = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3, QtGui.QImage.Format_BGR888)
        img2 = cv2.cvtColor(self.image, cv2.COLOR_RGBA2RGB)
        qimage = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg = QPixmap.fromImage(qimage)
        jpg = jpg.scaled(self.image_viewer.width(),self.image_viewer.height())
        self.image_viewer.setPixmap(jpg)

    def save_fig(self):
        self.image = self.reader.get_region(self.row, self.col, self.stride)
        img = cv2.cvtColor(self.image, cv2.COLOR_RGBA2BGR)
        img_name = '{0}_{1}_{2}_ori.png'.format(self.row,self.col,self.stride)
        cv2.imwrite(os.path.join(os.path.dirname(self.cache_path), img_name), img)
        if self.stride % 2 == 1:
            low = -(self.stride // 2)
            high = (self.stride // 2) + 1
        else:
            low = -self.stride // 2
            high = self.stride // 2
        img = self.tissue_label[
              self.row * 8 + low * 8:self.row * 8 + high * 8,
              self.col * 8 + low * 8:self.col * 8 + high * 8]
        def sigmoid(x):
            return 1/(1+(np.exp((-x))))
        img = sigmoid(img)
        img[img==0.5] = 0
        plt.imshow(img,cmap='jet')
        plt.axis('off')
        plt.savefig(os.path.join(os.path.dirname(self.cache_path), img_name.replace('ori','label')),
                    bbox_inches='tight',pad_inches = 0, dpi=300)


    def editLength(self):
        value, ok = QInputDialog.getDouble(self, 'edit', '修改选框长度', value=self.winLength*8.75/1000, min=0, decimals=2,flags=QtCore.Qt.WindowCloseButtonHint)
        if ok:
            self.winLength = int(value*1000/8.75)
            self.label_winLength.setText(
                "window length:{0:.2f}mm".format(self.winLength * 8.75 / 1000))
            self.label_winSize.setText(
                "window aera:{0:.2f}mm²".format(self.winWidth * 8.75 * self.winLength * 8.75 / 1000000))

    def editWidth(self):
        value, ok = QInputDialog.getDouble(self, 'edit', '修改选框宽度', value=self.winWidth * 8.75 / 1000, min=0,
                                           decimals=2, flags=QtCore.Qt.WindowCloseButtonHint)
        if ok:
            self.winWidth = int(value * 1000 / 8.75)
            self.label_winWidth.setText(
                "window Width:{0:.2f}mm".format(self.winWidth * 8.75 / 1000))
            self.label_winSize.setText(
                "window aera:{0:.2f}mm²".format(self.winWidth * 8.75 * self.winLength * 8.75 / 1000000))

    def slot_mkimage(self):
        if self.selected_window[0] == -1:
            return
        elif self.dir_path == None:
            return
        loc_x = (self.selected_window[1]-self.winLength//2) * 32
        loc_y = (self.selected_window[0]-self.winWidth//2) * 32
        stride = np.sqrt(self.selected_window[1]*self.selected_window[0])
        level = self.reader.slide.get_best_level_for_downsample(stride)
        level = max(level - 8, 0)
        rate = self.reader.slide.level_downsamples[level]
        image = self.reader.slide.read_region(
            location=(loc_x, loc_y), level=level, size=(int(self.winLength*32//rate), int(self.winWidth*32//rate)))
        image = cv2.cvtColor(np.array(image),cv2.COLOR_RGBA2BGR)
        if type(self.label) != type(None):
            for i, flag in enumerate(self.flag_showlabel):
                if flag == True:
                    self.gen_labeled_img(image, cv2.cvtColor(self.label_images[i][
                                    self.selected_window[0] - self.winWidth//2:self.selected_window[0] + self.winWidth//2,
                                    self.selected_window[1] - self.winLength//2:self.selected_window[1] + self.winLength//2
                                    ],cv2.COLOR_RGB2BGR))
        selected_aera = \
            np.sum(
            self.label[self.selected_window[0] - self.winWidth // 2:self.selected_window[0] + self.winWidth // 2,
            self.selected_window[1] - self.winLength // 2:self.selected_window[1] + self.winLength // 2]) \
                        / (self.winLength * self.winWidth) * 100
        img_name = time.strftime('Img%H%M', time.localtime()) + '(TSR{0:.2f}).png'.format(selected_aera)
        cv2.imwrite(os.path.join(os.path.dirname(self.cache_path),img_name), image)

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
        self.update_image()

    def slot_zoomout(self):
        if self.dir_path is None:
            return
        if self.stride == self.max_stride:
            return
        self.stride += 1
        while self.row < self.stride // 2:
            self.row += 1
        while self.col < self.stride // 2:
            self.col += 1
        self.update_image()

    def slot_tumor(self):
        label = 0
        self.flag_showlabel[label] = not self.flag_showlabel[label]
        if self.flag_isedit is True:
            for i, btn in enumerate(self.exclusive_btn_list):
                if i == label:
                    continue
                btn.setChecked(False)
                self.flag_showlabel[i] = False
        self.update_image()

    def slot_show_tumor_region(self):
        label = 1
        self.flag_showlabel[label] = not self.flag_showlabel[label]
        if self.flag_isedit is True:
            for i, btn in enumerate(self.exclusive_btn_list):
                if i == label:
                    continue
                btn.setChecked(False)
                self.flag_showlabel[i] = False
        self.update_image()

    def slot_show_nest_region(self):
        label = 2
        self.flag_showlabel[label] = not self.flag_showlabel[label]
        if self.flag_isedit is True:
            for i, btn in enumerate(self.exclusive_btn_list):
                if i == label:
                    continue
                btn.setChecked(False)
                self.flag_showlabel[i] = False
        self.update_image()

    def slot_show_margin_region(self):
        label = 3
        self.flag_showlabel[label] = not self.flag_showlabel[label]
        if self.flag_isedit is True:
            for i, btn in enumerate(self.exclusive_btn_list):
                if i == label:
                    continue
                btn.setChecked(False)
                self.flag_showlabel[i] = False
        self.update_image()

    def slot_show_tsr_map(self):
        label = 4
        self.flag_showlabel[label] = not self.flag_showlabel[label]
        if self.flag_isedit is True:
            for i, btn in enumerate(self.exclusive_btn_list):
                if i == label:
                    continue
                btn.setChecked(False)
                self.flag_showlabel[i] = False
        self.update_image()

    def slot_choose_window(self):
        if self.flag_isedit == True:
            self.flag_isedit = False
            self.btn_edit.setChecked(False)
        self.flag_select = not self.flag_select
        self.update_image()

    def slot_edit(self):
        if self.flag_select == True:
            self.flag_select = False
            self.btn_choose_window.setChecked(False)
        if self.flag_isedit == True:
            self.flag_isedit = False
        else:
            self.flag_isedit = True
            if np.sum(self.flag_showlabel) > 1:
                for i, btn in enumerate(self.exclusive_btn_list):
                    btn.setChecked(False)
                    self.flag_showlabel[i] = False
        self.update_image()

    def slot_recalculate(self):
        if self.cache_path is None:
            return
        tissue_label = self.label
        output_path = os.path.dirname(self.cache_path)
        slide = self.reader.slide
        # 计算肿瘤区域、tsr热图
        tumor_region, margin_region, nest_region = self.cal_tumor_region(tissue_label)
        tsr_map = self.get_tsr_map(tissue_label, tumor_region, radius=60)
        # 寻找最低点
        min_coor, tsr_min = self.find_min_region(tissue_label, tumor_region, winLength=self.winLength, winWidth=self.winWidth)
        # 生成标记图像
        cache_dir = os.path.join(output_path, 'reader_cache')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        label_images = [np.zeros(shape=(tissue_label.shape[0], tissue_label.shape[1], 3), dtype='uint8')
                        for _ in range(5)]
        label_images[0][tissue_label] = RGB_dict[1]
        label_images[1][tumor_region] = RGB_dict[5]
        label_images[2][nest_region] = RGB_dict[6]
        label_images[3][margin_region] = RGB_dict[4]
        res = np.vectorize(viridis_dict.get)((tsr_map * 256).astype('int').tolist())
        label_images[4] = (np.transpose(np.array(res), (1, 2, 0)) * 256).astype('uint8')
        roi_minrow = tumor_region.nonzero()[0][0]
        roi_maxrow = tumor_region.nonzero()[0][-1]
        roi_mincol = np.min(tumor_region.nonzero()[1])
        roi_maxcol = np.max(tumor_region.nonzero()[1])
        for row in range(roi_minrow - 1, roi_maxrow + 2):
            for col in range(roi_mincol - 1, roi_maxcol + 2):
                if self._is_border(row, col, tumor_region, tumor_side=False):
                    label_images[4][row, col] = [255, 255, 255]
        # 保存结果
        with open(os.path.join(cache_dir, 'label'), 'wb') as file_obj:
            pickle.dump(tissue_label, file_obj)
        with open(os.path.join(cache_dir, 'label_images'), 'wb') as file_obj:
            pickle.dump(label_images, file_obj)
        with open(os.path.join(cache_dir, 'tumor_region'), 'wb') as file_obj:
            pickle.dump(tumor_region, file_obj)
        with open(os.path.join(cache_dir, 'margin_region'), 'wb') as file_obj:
            pickle.dump(margin_region, file_obj)
        with open(os.path.join(cache_dir, 'nest_region'), 'wb') as file_obj:
            pickle.dump(nest_region, file_obj)
        with open(os.path.join(cache_dir, 'tsr_map'), 'wb') as file_obj:
            pickle.dump(tsr_map, file_obj)
        with open(os.path.join(cache_dir, 'min_coor'), 'wb') as file_obj:
            pickle.dump((min_coor,(self.winLength,self.winWidth)), file_obj)
        self.label = tissue_label
        self.label_images = label_images
        self.label_images.append(np.zeros((self.label_images[0].shape[0], self.label_images[0].shape[1], 3)))
        self.tumor_region = tumor_region
        self.nest_region = nest_region
        self.margin_region = margin_region
        self.tsr_map = tsr_map
        self.selected_window[0] = min_coor[0]
        self.selected_window[1] = min_coor[1]
        self.gen_window_line()
        # 生成预览图
        preview_dir = os.path.join(output_path, 'preview image')
        if not os.path.exists(preview_dir):
            os.mkdir(preview_dir)
        # tsr map
        cv2.imwrite(os.path.join(preview_dir, 'tsr_map.png'),
                cv2.cvtColor(label_images[4][roi_minrow:roi_maxrow, roi_mincol:roi_maxcol], cv2.COLOR_RGB2BGR))
        w = (roi_maxrow - roi_minrow) * 32
        h = (roi_maxcol - roi_mincol) * 32
        level = max(slide.level_count - 6, 0)
        ratio = int(slide.level_downsamples[level])
        slide_image = slide.read_region((roi_mincol * 32, roi_minrow * 32), level, (h // ratio, w // ratio))
        slide_image = cv2.cvtColor(np.array(slide_image), cv2.COLOR_RGBA2RGB)
        # original
        cv2.imwrite(os.path.join(preview_dir, 'original.png'), cv2.cvtColor(slide_image, cv2.COLOR_RGB2BGR))
        # tumor
        label_image = cv2.resize(label_images[0][roi_minrow:roi_maxrow, roi_mincol:roi_maxcol],
                             (h // ratio, w // ratio))
        slice = label_image[:, :, 0] != 0
        image = slide_image.copy()
        image[slice] = 0.5 * image[slice] + 0.5 * label_image[slice]
        image = image.astype('float32')
        cv2.imwrite(os.path.join(preview_dir, 'tumor.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # overlook
        label_image = 0.3 * label_images[0] + 0.3 * label_images[1] + 0.3 * label_images[2]
        label_image = cv2.resize(label_image[roi_minrow:roi_maxrow, roi_mincol:roi_maxcol],
                             (h // ratio, w // ratio))
        slice = label_image[:, :, 0] != 0
        image = slide_image.copy()
        image[slice] = 0.5 * image[slice] + 0.5 * label_image[slice]
        image = image.astype('float32')
        cv2.imwrite(os.path.join(preview_dir, 'overlook.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # min window
        loc_x = (min_coor[1] - 264 // 2) * 32
        loc_y = (min_coor[0] - 154 // 2) * 32
        stride = np.sqrt(min_coor[1] * min_coor[0])
        level = slide.get_best_level_for_downsample(stride)
        level = max(level - 6, 0)
        rate = slide.level_downsamples[level]
        image = slide.read_region(
            location=(loc_x, loc_y), level=level,
            size=(int(264 * 32 // rate), int(154 * 32 // rate)))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(preview_dir, 'min_region.png'), image)
        self.update_image()
        self.reset_panel()

    def slot_next(self):
        if self.dir_path is None:
            return
        dir_path = self.dir_path
        cache_path = self.cache_path
        self.reset()
        current_data = os.path.basename(dir_path)
        data_dir = os.path.dirname(dir_path)
        data_list = []
        for file in os.listdir(data_dir):
            if file.endswith('mrxs') or file.endswith('ndpi'):
                data_list.append(file)
        index = data_list.index(current_data)
        index = (index + 1) % len(data_list)
        next_data = data_list[index]
        self.dir_path = dir_path.replace(current_data, next_data)
        self.load_data()
        if cache_path is not None:
            self.cache_path = cache_path.replace(current_data[:current_data.rfind('.')],
                                                      next_data[:next_data.rfind('.')])
            self.load_cache()

    def slot_former(self):
        if self.dir_path is None:
            return
        dir_path = self.dir_path
        cache_path = self.cache_path
        self.reset()
        current_data = os.path.basename(dir_path)
        data_dir = os.path.dirname(dir_path)
        data_list = []
        for file in os.listdir(data_dir):
            if file.endswith('mrxs') or file.endswith('ndpi'):
                data_list.append(file)
        index = data_list.index(current_data)
        index = (index - 1) % len(data_list)
        next_data = data_list[index]
        self.dir_path = dir_path.replace(current_data, next_data)
        self.load_data()
        if cache_path is not None:
            self.cache_path = cache_path.replace(current_data[:current_data.rfind('.')],
                                                      next_data[:next_data.rfind('.')])
            self.load_cache()

    def slot_savelabel(self):
        if self.cache_path is None:
            return
        with open(os.path.join(self.cache_path, 'label'), 'wb') as file_obj:
            pickle.dump(self.label, file_obj)
        with open(os.path.join(self.cache_path, 'label_images'), 'wb') as file_obj:
            pickle.dump(self.label_images[:5], file_obj)
        with open(os.path.join(self.cache_path, 'tumor_region'), 'wb') as file_obj:
            pickle.dump(self.tumor_region, file_obj)
        with open(os.path.join(self.cache_path, 'nest_region'), 'wb') as file_obj:
            pickle.dump(self.nest_region, file_obj)
        if self.margin_flag == 'new':
            with open(os.path.join(self.cache_path, 'margin_region'), 'wb') as file_obj:
                pickle.dump(self.margin_region, file_obj)
        else:
            with open(os.path.join(self.cache_path, 'old_margin_region'), 'wb') as file_obj:
                pickle.dump(self.margin_region, file_obj)
        with open(os.path.join(self.cache_path, 'tsr_map'), 'wb') as file_obj:
            pickle.dump(self.tsr_map, file_obj)

        # roi_minrow = np.min(self.tumor_region.nonzero()[0])
        # roi_maxrow = np.max(self.tumor_region.nonzero()[0])
        # roi_mincol = np.min(self.tumor_region.nonzero()[1])
        # roi_maxcol = np.max(self.tumor_region.nonzero()[1])
        # h = (roi_maxcol - roi_mincol) * 32
        # w = (roi_maxrow - roi_minrow) * 32
        # level = max(self.reader.slide.level_count - 6, 0)
        # ratio = int(self.reader.slide.level_downsamples[level])
        # slide_image = self.reader.slide.read_region((roi_mincol * 32, roi_minrow * 32), level, (h // ratio, w // ratio))
        # slide_image = cv2.cvtColor(np.array(slide_image), cv2.COLOR_RGBA2RGB)
        # label_image1 = cv2.resize(self.label_images[0][roi_minrow:roi_maxrow, roi_mincol:roi_maxcol],
        #                       (h // ratio, w // ratio))
        # label_image2 = cv2.resize(self.label_images[2][roi_minrow:roi_maxrow, roi_mincol:roi_maxcol],
        #                       (h // ratio, w // ratio))
        # label_image3 = cv2.resize(self.label_images[3][roi_minrow:roi_maxrow, roi_mincol:roi_maxcol],
        #                       (h // ratio, w // ratio))
        # image = 0.5 * label_image1 + 0.5 * label_image2 + 0.5 * label_image3
        # image = (0.5 * slide_image + 0.5 * image).astype('float32')
        # tsr_total = np.sum(self.label[self.tumor_region]) / np.sum(self.tumor_region)
        # output_path = os.path.dirname(self.cache_path)
        # cv2.imwrite(os.path.join(output_path, 'TSR' + str(round(tsr_total * 100, 2)) + '.png'),
        #         cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def reset(self):
        self.img_list = None
        self.stride = 10
        self.flag_showlabel = [False for _ in range(5)]
        self.flag_isedit = False
        self.flag_select = False
        self.max_stride = MAX_STRIDE
        self.label = None
        self.image = np.ones((256*self.stride, 256*self.stride, 3), dtype='uint8') * 255
        self.dir_path = None
        self.cache_path = None
        self.alpha_ratio = 0.5
        self.threshold = 2
        for btn in self.exclusive_btn_list:
            btn.setChecked(False)
        self.btn_edit.setChecked(False)
        self.btn_choose_window.setChecked(False)
        self.selected_window = [-1, -1]
        self.winLength = 66
        self.winWidth = 39
        self.reset_panel()

    def reset_panel(self):
        if self.dir_path is None:
            self.info = '未选择切片数据'
            self.info_panel.setText(self.info)
        elif self.cache_path is not None:
            if self.selected_window[0] != -1:
                selected_aera = np.sum(self.label[self.selected_window[0] - self.winWidth//2:self.selected_window[0] + self.winWidth//2,
                        self.selected_window[1] - self.winLength//2:self.selected_window[1] + self.winLength//2])/ (self.winLength*self.winWidth) * 100
            else:
                selected_aera = 0
            self.info = '{name}\n' \
                        '------------------\n' \
                        'Info\n' \
                        'TSR total：{r1:.2f}\n' \
                        'TSR nest: {r2:.2f}\n' \
                        'TSR ITF: {r3:.2f}\n' \
                        'margin ratio: {r5:.2f}\n' \
                        'TSR selected: {r4:.2f}' \
                .format(name=os.path.basename(self.dir_path),
                        r1=np.sum(self.label[self.tumor_region]) / np.sum(self.tumor_region) * 100,
                        r2=np.sum(self.label[self.nest_region]) / np.sum(self.nest_region) * 100,
                        r3=np.sum(self.label[self.margin_region]) / np.sum(self.margin_region) * 100,
                        r4=selected_aera,
                        r5=np.sum(self.margin_region) / np.sum(self.tumor_region))
            self.info_panel.setText(self.info)
        else:
            self.info = '{name}\n' \
                        '------------------\n' \
                        '未选择标记数据' \
                .format(name=os.path.basename(self.dir_path))
            self.info_panel.setText(self.info)

    def change_label(self, relative_start_row,relative_end_row, relative_start_col, relative_end_col, is_cancel):
        if self.dir_path is None or self.cache_path is None:
            return
        if np.sum(self.flag_showlabel) == 0 or self.flag_isedit is False:
            return
        if is_cancel is False:
            target = True
        else:
            target = False
        low = -(self.stride // 2)
        start_row = self.row*8 + low*8
        start_col = self.col*8 + low*8
        if self.flag_showlabel[0] is True:
            self.label[
            start_row+relative_start_row:start_row+relative_end_row,
            start_col+relative_start_col:start_col+relative_end_col] = target
            self.label_images[0][
            start_row+relative_start_row:start_row+relative_end_row,
            start_col+relative_start_col:start_col+relative_end_col] = RGB_dict[1] if target else [0, 0, 0]
            if target == False:
                self.tissue_label[
                start_row+relative_start_row:start_row+relative_end_row,
                start_col+relative_start_col:start_col+relative_end_col] = 0
            else:
                self.tissue_label[
                start_row+relative_start_row:start_row+relative_end_row,
                start_col+relative_start_col:start_col+relative_end_col] = 1
        elif self.flag_showlabel[1] is True:
            self.tumor_region[
            start_row + relative_start_row:start_row + relative_end_row,
            start_col + relative_start_col:start_col + relative_end_col] = target
            self.label_images[1][
            start_row + relative_start_row:start_row + relative_end_row,
            start_col + relative_start_col:start_col + relative_end_col] = RGB_dict[5] if target else [0, 0, 0]
        elif self.flag_showlabel[2] is True:
            self.nest_region[
            start_row + relative_start_row:start_row + relative_end_row,
            start_col + relative_start_col:start_col + relative_end_col] = target
            self.label_images[2][
            start_row + relative_start_row:start_row + relative_end_row,
            start_col + relative_start_col:start_col + relative_end_col] = RGB_dict[6] if target else [0, 0, 0]
        elif self.flag_showlabel[3] is True:
            self.margin_region[
            start_row + relative_start_row:start_row + relative_end_row,
            start_col + relative_start_col:start_col + relative_end_col] = target
            self.label_images[3][
            start_row + relative_start_row:start_row + relative_end_row,
            start_col + relative_start_col:start_col + relative_end_col] = RGB_dict[4] if target else [0, 0, 0]
        self.reset_panel()

    def select_window(self, retive_row, retive_col):
        self.selected_window[0] = self.row * 8 - (self.stride // 2)*8 + retive_row
        self.selected_window[1] = self.col * 8 - (self.stride // 2)*8 + retive_col
        self.gen_window_line()
        self.update_image()
        self.reset_panel()

    def gen_window_line(self):
        self.label_images[5][:] = 0
        self.label_images[5][
        max(self.selected_window[0] - self.winWidth//2,0),
        max(self.selected_window[1] - self.winLength//2,0):
        min(self.selected_window[1] + self.winLength//2,self.label_images[5].shape[1]-1), :] = 1
        self.label_images[5][
        min(self.selected_window[0] + self.winWidth//2,self.label_images[5].shape[0]-1),
        max(self.selected_window[1] - self.winLength//2, 0):
        min(self.selected_window[1] + self.winLength//2, self.label_images[5].shape[1]-1), :] = 1
        self.label_images[5][
        max(self.selected_window[0] - self.winWidth//2,0):
        min(self.selected_window[0] + self.winWidth//2,self.label_images[5].shape[0]-1),
        max(self.selected_window[1] - self.winLength//2,0),:] = 1
        self.label_images[5][
        max(self.selected_window[0] - self.winWidth//2,0):
        min(self.selected_window[0] + self.winWidth//2,self.label_images[5].shape[0]-1),
        min(self.selected_window[1] + self.winLength//2,self.label_images[5].shape[1]-1), :] = 1

    def cal_tumor_region(self, label, expand_radius=57, margin_dis=114):
        roi_minrow = np.min(label.nonzero()[0])
        roi_maxrow = np.max(label.nonzero()[0])
        roi_mincol = np.min(label.nonzero()[1])
        roi_maxcol = np.max(label.nonzero()[1])
        tumor_region = label.copy()
        # check border
        border_list = []
        for row in range(roi_minrow-1, roi_maxrow+2):
            for col in range(roi_mincol-1, roi_maxcol+2):
                if self._is_border(row, col, tumor_region, tumor_side=True):
                    border_list.append((row, col))
        # expand
        for i, pixel in enumerate(border_list):
            row, col = pixel
            tumor_region[max(row - expand_radius, 0):row + expand_radius,
            max(col - expand_radius, 0):col + expand_radius] = True
        # check border
        border_list = []
        for row in range(roi_minrow-expand_radius-1, roi_maxrow+expand_radius+2):
            for col in range(roi_mincol-expand_radius-1, roi_maxcol+expand_radius+2):
                if self._is_border(row, col, tumor_region, tumor_side=False):
                    border_list.append((row,col))
        # contrast
        expand_radius -= 5
        for pixel in border_list:
            row, col = pixel
            tumor_region[max(row - expand_radius, 0):row + expand_radius,
            max(col - expand_radius, 0):col + expand_radius] = False
        # calculate margin
        nest_region = tumor_region.copy()
        # check border
        border_list = []
        for row in range(roi_minrow-expand_radius-1, roi_maxrow+expand_radius+2):
            for col in range(roi_mincol-expand_radius-1, roi_maxcol+expand_radius+2):
                if self._is_border(row, col, tumor_region, tumor_side=False):
                    border_list.append((row, col))
        # contrast
        for pixel in border_list:
            row, col = pixel
            nest_region[max(row - margin_dis, 0):row + margin_dis,
            max(col - margin_dis, 0):col + margin_dis] = False
        margin_region = (nest_region == False) * tumor_region
        return tumor_region, margin_region, nest_region

    def _is_border(self, row, col, tumor_region, tumor_side):
        """
        返回给定坐标是否是肿瘤边界
        tumor_side指示要检查的是边界的肿瘤侧还是正常组织侧
        注意传入的row,col不能在图像边界
        """
        if tumor_region[row, col] == True and tumor_side == True:
            target = False
        elif tumor_region[row, col] == False and tumor_side == False:
            target = True
        else:
            return False
        if tumor_region[row-1, col] == target:
            return True
        elif tumor_region[row, col-1] == target:
            return True
        elif tumor_region[row, col+1] == target:
            return True
        elif tumor_region[row+1, col] == target:
            return True
        return False

    def get_tsr_map(self, tumor_label, tumor_region, radius=77, stride=1):
        roi = tumor_region.nonzero()
        tsr_map = np.zeros((tumor_label.shape[0]//stride, tumor_label.shape[1]//stride))
        for row in tqdm(np.unique(roi[0])):
            for col in roi[1][roi[0]==row]:
                window_label = tumor_label[max(row - radius, 0):row + radius, max(col - radius, 0):col + radius]
                window_region = tumor_region[max(row - radius, 0):row + radius, max(col - radius, 0):col + radius]
                tsr_map[row//stride][col//stride] = np.sum(window_label[window_region])/np.sum(window_region)
        return tsr_map

    def find_min_region(self, tumor_label, tumor_region, winLength=264, winWidth=154):
        roi = tumor_region.nonzero()
        min_coor = (0, 0)
        min_tsr = 1
        winAera = winWidth * winLength
        for row in tqdm(np.unique(roi[0])):
            for col in roi[1][roi[0]==row]:
                window_label = tumor_label[max(row - winWidth//2, 0):row + winWidth//2, max(col - winLength//2, 0):col + winLength//2]
                window_region = tumor_region[max(row - winWidth//2, 0):row + winWidth//2, max(col - winLength//2, 0):col + winLength//2]
                if np.sum(window_region)/winAera < 0.8:
                    continue
                tsr = np.sum(window_label[window_region])/np.sum(window_region)
                if tsr < min_tsr:
                    min_tsr = tsr
                    min_coor = (row,col)
        return min_coor, min_tsr

    def switch_margin(self):
        if self.margin_flag == 'new':
            self.margin_flag = 'old'
            with open(os.path.join(self.cache_path, 'old_margin_region'), 'rb') as file_obj:
                self.margin_region = pickle.load(file_obj)
        else:
            self.margin_flag = 'new'
            with open(os.path.join(self.cache_path, 'margin_region'), 'rb') as file_obj:
                self.margin_region = pickle.load(file_obj)
        self.label_images[3][:] = 0
        self.label_images[3][self.margin_region] = RGB_dict[4]
        self.update_image()


viridis_data = [[ 0.26700401,  0.00487433,  0.32941519],
  [ 0.26851048,  0.00960483,  0.33542652],
  [ 0.26994384,  0.01462494,  0.34137895],
  [ 0.27130489,  0.01994186,  0.34726862],
  [ 0.27259384,  0.02556309,  0.35309303],
  [ 0.27380934,  0.03149748,  0.35885256],
  [ 0.27495242,  0.03775181,  0.36454323],
  [ 0.27602238,  0.04416723,  0.37016418],
  [ 0.2770184 ,  0.05034437,  0.37571452],
  [ 0.27794143,  0.05632444,  0.38119074],
  [ 0.27879067,  0.06214536,  0.38659204],
  [ 0.2795655 ,  0.06783587,  0.39191723],
  [ 0.28026658,  0.07341724,  0.39716349],
  [ 0.28089358,  0.07890703,  0.40232944],
  [ 0.28144581,  0.0843197 ,  0.40741404],
  [ 0.28192358,  0.08966622,  0.41241521],
  [ 0.28232739,  0.09495545,  0.41733086],
  [ 0.28265633,  0.10019576,  0.42216032],
  [ 0.28291049,  0.10539345,  0.42690202],
  [ 0.28309095,  0.11055307,  0.43155375],
  [ 0.28319704,  0.11567966,  0.43611482],
  [ 0.28322882,  0.12077701,  0.44058404],
  [ 0.28318684,  0.12584799,  0.44496   ],
  [ 0.283072  ,  0.13089477,  0.44924127],
  [ 0.28288389,  0.13592005,  0.45342734],
  [ 0.28262297,  0.14092556,  0.45751726],
  [ 0.28229037,  0.14591233,  0.46150995],
  [ 0.28188676,  0.15088147,  0.46540474],
  [ 0.28141228,  0.15583425,  0.46920128],
  [ 0.28086773,  0.16077132,  0.47289909],
  [ 0.28025468,  0.16569272,  0.47649762],
  [ 0.27957399,  0.17059884,  0.47999675],
  [ 0.27882618,  0.1754902 ,  0.48339654],
  [ 0.27801236,  0.18036684,  0.48669702],
  [ 0.27713437,  0.18522836,  0.48989831],
  [ 0.27619376,  0.19007447,  0.49300074],
  [ 0.27519116,  0.1949054 ,  0.49600488],
  [ 0.27412802,  0.19972086,  0.49891131],
  [ 0.27300596,  0.20452049,  0.50172076],
  [ 0.27182812,  0.20930306,  0.50443413],
  [ 0.27059473,  0.21406899,  0.50705243],
  [ 0.26930756,  0.21881782,  0.50957678],
  [ 0.26796846,  0.22354911,  0.5120084 ],
  [ 0.26657984,  0.2282621 ,  0.5143487 ],
  [ 0.2651445 ,  0.23295593,  0.5165993 ],
  [ 0.2636632 ,  0.23763078,  0.51876163],
  [ 0.26213801,  0.24228619,  0.52083736],
  [ 0.26057103,  0.2469217 ,  0.52282822],
  [ 0.25896451,  0.25153685,  0.52473609],
  [ 0.25732244,  0.2561304 ,  0.52656332],
  [ 0.25564519,  0.26070284,  0.52831152],
  [ 0.25393498,  0.26525384,  0.52998273],
  [ 0.25219404,  0.26978306,  0.53157905],
  [ 0.25042462,  0.27429024,  0.53310261],
  [ 0.24862899,  0.27877509,  0.53455561],
  [ 0.2468114 ,  0.28323662,  0.53594093],
  [ 0.24497208,  0.28767547,  0.53726018],
  [ 0.24311324,  0.29209154,  0.53851561],
  [ 0.24123708,  0.29648471,  0.53970946],
  [ 0.23934575,  0.30085494,  0.54084398],
  [ 0.23744138,  0.30520222,  0.5419214 ],
  [ 0.23552606,  0.30952657,  0.54294396],
  [ 0.23360277,  0.31382773,  0.54391424],
  [ 0.2316735 ,  0.3181058 ,  0.54483444],
  [ 0.22973926,  0.32236127,  0.54570633],
  [ 0.22780192,  0.32659432,  0.546532  ],
  [ 0.2258633 ,  0.33080515,  0.54731353],
  [ 0.22392515,  0.334994  ,  0.54805291],
  [ 0.22198915,  0.33916114,  0.54875211],
  [ 0.22005691,  0.34330688,  0.54941304],
  [ 0.21812995,  0.34743154,  0.55003755],
  [ 0.21620971,  0.35153548,  0.55062743],
  [ 0.21429757,  0.35561907,  0.5511844 ],
  [ 0.21239477,  0.35968273,  0.55171011],
  [ 0.2105031 ,  0.36372671,  0.55220646],
  [ 0.20862342,  0.36775151,  0.55267486],
  [ 0.20675628,  0.37175775,  0.55311653],
  [ 0.20490257,  0.37574589,  0.55353282],
  [ 0.20306309,  0.37971644,  0.55392505],
  [ 0.20123854,  0.38366989,  0.55429441],
  [ 0.1994295 ,  0.38760678,  0.55464205],
  [ 0.1976365 ,  0.39152762,  0.55496905],
  [ 0.19585993,  0.39543297,  0.55527637],
  [ 0.19410009,  0.39932336,  0.55556494],
  [ 0.19235719,  0.40319934,  0.55583559],
  [ 0.19063135,  0.40706148,  0.55608907],
  [ 0.18892259,  0.41091033,  0.55632606],
  [ 0.18723083,  0.41474645,  0.55654717],
  [ 0.18555593,  0.4185704 ,  0.55675292],
  [ 0.18389763,  0.42238275,  0.55694377],
  [ 0.18225561,  0.42618405,  0.5571201 ],
  [ 0.18062949,  0.42997486,  0.55728221],
  [ 0.17901879,  0.43375572,  0.55743035],
  [ 0.17742298,  0.4375272 ,  0.55756466],
  [ 0.17584148,  0.44128981,  0.55768526],
  [ 0.17427363,  0.4450441 ,  0.55779216],
  [ 0.17271876,  0.4487906 ,  0.55788532],
  [ 0.17117615,  0.4525298 ,  0.55796464],
  [ 0.16964573,  0.45626209,  0.55803034],
  [ 0.16812641,  0.45998802,  0.55808199],
  [ 0.1666171 ,  0.46370813,  0.55811913],
  [ 0.16511703,  0.4674229 ,  0.55814141],
  [ 0.16362543,  0.47113278,  0.55814842],
  [ 0.16214155,  0.47483821,  0.55813967],
  [ 0.16066467,  0.47853961,  0.55811466],
  [ 0.15919413,  0.4822374 ,  0.5580728 ],
  [ 0.15772933,  0.48593197,  0.55801347],
  [ 0.15626973,  0.4896237 ,  0.557936  ],
  [ 0.15481488,  0.49331293,  0.55783967],
  [ 0.15336445,  0.49700003,  0.55772371],
  [ 0.1519182 ,  0.50068529,  0.55758733],
  [ 0.15047605,  0.50436904,  0.55742968],
  [ 0.14903918,  0.50805136,  0.5572505 ],
  [ 0.14760731,  0.51173263,  0.55704861],
  [ 0.14618026,  0.51541316,  0.55682271],
  [ 0.14475863,  0.51909319,  0.55657181],
  [ 0.14334327,  0.52277292,  0.55629491],
  [ 0.14193527,  0.52645254,  0.55599097],
  [ 0.14053599,  0.53013219,  0.55565893],
  [ 0.13914708,  0.53381201,  0.55529773],
  [ 0.13777048,  0.53749213,  0.55490625],
  [ 0.1364085 ,  0.54117264,  0.55448339],
  [ 0.13506561,  0.54485335,  0.55402906],
  [ 0.13374299,  0.54853458,  0.55354108],
  [ 0.13244401,  0.55221637,  0.55301828],
  [ 0.13117249,  0.55589872,  0.55245948],
  [ 0.1299327 ,  0.55958162,  0.55186354],
  [ 0.12872938,  0.56326503,  0.55122927],
  [ 0.12756771,  0.56694891,  0.55055551],
  [ 0.12645338,  0.57063316,  0.5498411 ],
  [ 0.12539383,  0.57431754,  0.54908564],
  [ 0.12439474,  0.57800205,  0.5482874 ],
  [ 0.12346281,  0.58168661,  0.54744498],
  [ 0.12260562,  0.58537105,  0.54655722],
  [ 0.12183122,  0.58905521,  0.54562298],
  [ 0.12114807,  0.59273889,  0.54464114],
  [ 0.12056501,  0.59642187,  0.54361058],
  [ 0.12009154,  0.60010387,  0.54253043],
  [ 0.11973756,  0.60378459,  0.54139999],
  [ 0.11951163,  0.60746388,  0.54021751],
  [ 0.11942341,  0.61114146,  0.53898192],
  [ 0.11948255,  0.61481702,  0.53769219],
  [ 0.11969858,  0.61849025,  0.53634733],
  [ 0.12008079,  0.62216081,  0.53494633],
  [ 0.12063824,  0.62582833,  0.53348834],
  [ 0.12137972,  0.62949242,  0.53197275],
  [ 0.12231244,  0.63315277,  0.53039808],
  [ 0.12344358,  0.63680899,  0.52876343],
  [ 0.12477953,  0.64046069,  0.52706792],
  [ 0.12632581,  0.64410744,  0.52531069],
  [ 0.12808703,  0.64774881,  0.52349092],
  [ 0.13006688,  0.65138436,  0.52160791],
  [ 0.13226797,  0.65501363,  0.51966086],
  [ 0.13469183,  0.65863619,  0.5176488 ],
  [ 0.13733921,  0.66225157,  0.51557101],
  [ 0.14020991,  0.66585927,  0.5134268 ],
  [ 0.14330291,  0.66945881,  0.51121549],
  [ 0.1466164 ,  0.67304968,  0.50893644],
  [ 0.15014782,  0.67663139,  0.5065889 ],
  [ 0.15389405,  0.68020343,  0.50417217],
  [ 0.15785146,  0.68376525,  0.50168574],
  [ 0.16201598,  0.68731632,  0.49912906],
  [ 0.1663832 ,  0.69085611,  0.49650163],
  [ 0.1709484 ,  0.69438405,  0.49380294],
  [ 0.17570671,  0.6978996 ,  0.49103252],
  [ 0.18065314,  0.70140222,  0.48818938],
  [ 0.18578266,  0.70489133,  0.48527326],
  [ 0.19109018,  0.70836635,  0.48228395],
  [ 0.19657063,  0.71182668,  0.47922108],
  [ 0.20221902,  0.71527175,  0.47608431],
  [ 0.20803045,  0.71870095,  0.4728733 ],
  [ 0.21400015,  0.72211371,  0.46958774],
  [ 0.22012381,  0.72550945,  0.46622638],
  [ 0.2263969 ,  0.72888753,  0.46278934],
  [ 0.23281498,  0.73224735,  0.45927675],
  [ 0.2393739 ,  0.73558828,  0.45568838],
  [ 0.24606968,  0.73890972,  0.45202405],
  [ 0.25289851,  0.74221104,  0.44828355],
  [ 0.25985676,  0.74549162,  0.44446673],
  [ 0.26694127,  0.74875084,  0.44057284],
  [ 0.27414922,  0.75198807,  0.4366009 ],
  [ 0.28147681,  0.75520266,  0.43255207],
  [ 0.28892102,  0.75839399,  0.42842626],
  [ 0.29647899,  0.76156142,  0.42422341],
  [ 0.30414796,  0.76470433,  0.41994346],
  [ 0.31192534,  0.76782207,  0.41558638],
  [ 0.3198086 ,  0.77091403,  0.41115215],
  [ 0.3277958 ,  0.77397953,  0.40664011],
  [ 0.33588539,  0.7770179 ,  0.40204917],
  [ 0.34407411,  0.78002855,  0.39738103],
  [ 0.35235985,  0.78301086,  0.39263579],
  [ 0.36074053,  0.78596419,  0.38781353],
  [ 0.3692142 ,  0.78888793,  0.38291438],
  [ 0.37777892,  0.79178146,  0.3779385 ],
  [ 0.38643282,  0.79464415,  0.37288606],
  [ 0.39517408,  0.79747541,  0.36775726],
  [ 0.40400101,  0.80027461,  0.36255223],
  [ 0.4129135 ,  0.80304099,  0.35726893],
  [ 0.42190813,  0.80577412,  0.35191009],
  [ 0.43098317,  0.80847343,  0.34647607],
  [ 0.44013691,  0.81113836,  0.3409673 ],
  [ 0.44936763,  0.81376835,  0.33538426],
  [ 0.45867362,  0.81636288,  0.32972749],
  [ 0.46805314,  0.81892143,  0.32399761],
  [ 0.47750446,  0.82144351,  0.31819529],
  [ 0.4870258 ,  0.82392862,  0.31232133],
  [ 0.49661536,  0.82637633,  0.30637661],
  [ 0.5062713 ,  0.82878621,  0.30036211],
  [ 0.51599182,  0.83115784,  0.29427888],
  [ 0.52577622,  0.83349064,  0.2881265 ],
  [ 0.5356211 ,  0.83578452,  0.28190832],
  [ 0.5455244 ,  0.83803918,  0.27562602],
  [ 0.55548397,  0.84025437,  0.26928147],
  [ 0.5654976 ,  0.8424299 ,  0.26287683],
  [ 0.57556297,  0.84456561,  0.25641457],
  [ 0.58567772,  0.84666139,  0.24989748],
  [ 0.59583934,  0.84871722,  0.24332878],
  [ 0.60604528,  0.8507331 ,  0.23671214],
  [ 0.61629283,  0.85270912,  0.23005179],
  [ 0.62657923,  0.85464543,  0.22335258],
  [ 0.63690157,  0.85654226,  0.21662012],
  [ 0.64725685,  0.85839991,  0.20986086],
  [ 0.65764197,  0.86021878,  0.20308229],
  [ 0.66805369,  0.86199932,  0.19629307],
  [ 0.67848868,  0.86374211,  0.18950326],
  [ 0.68894351,  0.86544779,  0.18272455],
  [ 0.69941463,  0.86711711,  0.17597055],
  [ 0.70989842,  0.86875092,  0.16925712],
  [ 0.72039115,  0.87035015,  0.16260273],
  [ 0.73088902,  0.87191584,  0.15602894],
  [ 0.74138803,  0.87344918,  0.14956101],
  [ 0.75188414,  0.87495143,  0.14322828],
  [ 0.76237342,  0.87642392,  0.13706449],
  [ 0.77285183,  0.87786808,  0.13110864],
  [ 0.78331535,  0.87928545,  0.12540538],
  [ 0.79375994,  0.88067763,  0.12000532],
  [ 0.80418159,  0.88204632,  0.11496505],
  [ 0.81457634,  0.88339329,  0.11034678],
  [ 0.82494028,  0.88472036,  0.10621724],
  [ 0.83526959,  0.88602943,  0.1026459 ],
  [ 0.84556056,  0.88732243,  0.09970219],
  [ 0.8558096 ,  0.88860134,  0.09745186],
  [ 0.86601325,  0.88986815,  0.09595277],
  [ 0.87616824,  0.89112487,  0.09525046],
  [ 0.88627146,  0.89237353,  0.09537439],
  [ 0.89632002,  0.89361614,  0.09633538],
  [ 0.90631121,  0.89485467,  0.09812496],
  [ 0.91624212,  0.89609127,  0.1007168 ],
  [ 0.92610579,  0.89732977,  0.10407067],
  [ 0.93590444,  0.8985704 ,  0.10813094],
  [ 0.94563626,  0.899815  ,  0.11283773],
  [ 0.95529972,  0.90106534,  0.11812832],
  [ 0.96489353,  0.90232311,  0.12394051],
  [ 0.97441665,  0.90358991,  0.13021494],
  [ 0.98386829,  0.90486726,  0.13689671],
  [ 0.99324789,  0.90615657,  0.1439362 ]]
viridis_dict = {i:tuple(viridis_data[i]) for i in range(256)}

if __name__=="__main__":
    app = QApplication(sys.argv)
    mainForm = MainForm('SilideReader')
    mainForm.show()
    sys.exit(app.exec_())

