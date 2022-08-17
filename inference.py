from model import getModel
from DataLoader import ImageDataLoader
import shutil
import numpy as np
from torch import nn
from torch import cuda, load, matmul, no_grad
import os
import openslide
import pickle
import multiprocessing
import time
import traceback
from SlideSeg import seg_slide
from tqdm import tqdm
from cv2 import resize, imwrite, cvtColor, COLOR_RGB2BGR, COLOR_RGBA2RGB, COLOR_RGBA2BGR
import sys
import yaml
with open('config.yaml', 'r',encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

RGB_dict = config['RGB_dict']
threshold = config['threshold']

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = logfile
    def write(self, message):
        self.terminal.write(message+'\n')
        with open(self.log, 'a+') as f:
            f.write(message+'\n')
    def flush(self):
        pass

viridis_dict = {i:tuple(config['viridis_data'][i]) for i in range(256)}


class tif_classifier():
    def __init__(self, logger_path=None):
        if logger_path == None:
            logger_path = time.strftime('Log%H%M.txt', time.localtime())
        self.logger = Logger(logger_path)
        self.flag_cuda = cuda.is_available()
        self.model = getModel(backbone='resnet34', num_class=2, dropout=False)
        self.model.load_state_dict(load('best_state2'))
        self.model.eval()
        self.logger.write('model loaded successfully')
        if self.flag_cuda:
            self.logger.write('gpu mode: gpu')
            self.model = self.model.cuda()
        else:
            self.logger.write('gpu mode: cpu')
        self.weights = self.model.classifier.state_dict()['0.weight'][1,:].detach().T
        self.bias = self.model.classifier.state_dict()['0.bias'][1].detach()

    def classify_patches(self, patches, return_segmentation=False):
        if self.flag_cuda:
            patches = patches.cuda()
        l, f = self.model.predict(patches)
        patch_pred = nn.functional.softmax(l, dim=1).detach().cpu().numpy()
        if return_segmentation:
            res = matmul(f, self.weights) + self.bias/64
            for j in range(res.size(0)):
                if patch_pred[j, 1] < 0.5:
                    res[j, :] = 0
            label = res.reshape(res.size(0), 8, 8)
            return patch_pred, label.detach().cpu().numpy()
        else:
            return patch_pred

    def classify_WSI(self, slide_path, output_path, num_workers=2, batch_size=5):
        try:
            basename = os.path.basename(slide_path)
            name = basename[:basename.rfind('.')]
            output_path = os.path.join(output_path, name)
            seg_path = os.path.join(output_path, 'seg')
            label_path = os.path.join(output_path, 'label')
            tissue_label_path = os.path.join(label_path, 'tissue')
            slide = openslide.OpenSlide(slide_path)
            # self.logger.write('---------------------------------')
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: ', time.localtime()) + name + ' classification:')
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            # seg
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'seg and filter...')
            if not seg_slide(slide_path, seg_path, num_workers=num_workers):
                self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'seg fail!')
                return 0
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'seg and filter done!')
            # inference
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'infer...')
            dataloader = ImageDataLoader(mode='inference', shuffle=False, batch_size=batch_size, num_workers=num_workers, img_path=seg_path)
            if not os.path.exists(label_path):
                os.mkdir(label_path)
            if not os.path.exists(tissue_label_path):
                os.mkdir(tissue_label_path)
            with no_grad():
                for input, path in tqdm(dataloader):
                    l, f = self.classify_patches(input, return_segmentation=True)
                    for i in range(input.size(0)):
                        np.save(os.path.join(tissue_label_path, path[i]), f[i])
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'inference done')
            # aggregate label
            # aggregate label
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'aggregate result...')
            max_row = int(slide.dimensions[1] // 256)
            max_col = int(slide.dimensions[0] // 256)
            tissue_label = np.zeros((max_row * 8, max_col * 8), dtype='float32')
            count = np.zeros((max_row * 8, max_col * 8), dtype='uint8')
            for row in tqdm(range(max_row * 2)):
                row = row / 2
                for col in range(max_col * 2):
                    col = col / 2
                    label_name = name + '(' + str(row) + ',' + str(col) + ')' + '.npy'
                    if os.path.exists(os.path.join(tissue_label_path, label_name)):
                        tissue_label[int(row * 8):int(row * 8) + 8, int(col * 8):int(col * 8) + 8] += np.load(
                            os.path.join(tissue_label_path, label_name))
                        count[int(row * 8):int(row * 8) + 8, int(col * 8):int(col * 8) + 8] += 1
            if np.max(tissue_label) == 0:
                return
            count[count == 0] = 1
            tissue_label = tissue_label / count
            np.save(os.path.join(output_path, 'tissue_label.npy'), tissue_label)
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'aggregation done')
            tissue_label = np.load(os.path.join(output_path, 'tissue_label.npy'))
            tissue_label = (tissue_label > threshold)
            # 计算肿瘤区域、tsr热图
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'calculate region...')
            label_smoothed = self.smooth_boundary(tissue_label)
            tumor_region, margin_region, nest_region = self.cal_tumor_region(label_smoothed)
            tsr_map = self.get_tsr_map(tissue_label, tumor_region, radius=60)
            # 寻找最低点
            min_coor, tsr_min = self.find_min_region(tissue_label, tumor_region)
            min_coor = (min_coor, (int(66.), int(39.)))
            # 生成标记图像
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'generate image...')
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
                pickle.dump(min_coor, file_obj)
            # 生成预览图
            preview_dir = os.path.join(output_path, 'preview image')
            if not os.path.exists(preview_dir):
                os.mkdir(preview_dir)
            # tsr map
            imwrite(os.path.join(preview_dir, 'tsr_map.png'),
                    cvtColor(label_images[4][roi_minrow:roi_maxrow, roi_mincol:roi_maxcol], COLOR_RGB2BGR))
            w = (roi_maxrow - roi_minrow) * 32
            h = (roi_maxcol - roi_mincol) * 32
            level = max(slide.level_count - 6, 0)
            ratio = int(slide.level_downsamples[level])
            slide_image = slide.read_region((roi_mincol * 32, roi_minrow * 32), level, (h // ratio, w // ratio))
            slide_image = cvtColor(np.array(slide_image), COLOR_RGBA2RGB)
            # original
            imwrite(os.path.join(preview_dir, 'original.png'), cvtColor(slide_image, COLOR_RGB2BGR))
            # tumor
            label_image = resize(label_images[0][roi_minrow:roi_maxrow, roi_mincol:roi_maxcol],
                                 (h // ratio, w // ratio))
            slice = label_image[:, :, 0] != 0
            image = slide_image.copy()
            image[slice] = 0.5 * image[slice] + 0.5 * label_image[slice]
            image = image.astype('float32')
            imwrite(os.path.join(preview_dir, 'tumor.png'), cvtColor(image, COLOR_RGB2BGR))
            # overlook
            label_image = 0.3 * label_images[0] + 0.3 * label_images[1] + 0.3 * label_images[2]
            label_image = resize(label_image[roi_minrow:roi_maxrow, roi_mincol:roi_maxcol],
                                 (h // ratio, w // ratio))
            slice = label_image[:, :, 0] != 0
            image = slide_image.copy()
            image[slice] = 0.5 * image[slice] + 0.5 * label_image[slice]
            image = image.astype('float32')
            imwrite(os.path.join(preview_dir, 'overlook.png'), cvtColor(image, COLOR_RGB2BGR))
            # min window
            loc_x = (min_coor[0][1] - 66 // 2) * 32
            loc_y = (min_coor[0][0] - 39 // 2) * 32
            stride = np.sqrt(min_coor[0][1] * min_coor[0][0])
            level = slide.get_best_level_for_downsample(stride)
            level = max(level - 6, 0)
            rate = slide.level_downsamples[level]
            image = slide.read_region(
                location=(loc_x, loc_y), level=level,
                size=(int(264 * 32 // rate), int(154 * 32 // rate)))
            image = cvtColor(np.array(image), COLOR_RGBA2BGR)
            imwrite(os.path.join(preview_dir, 'min_region.png'), image)
            slide.close()
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'all done')
            # 清理临时文件
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'clean...')
            if os.path.exists(label_path):
                shutil.rmtree(label_path)
            if os.path.exists(seg_path):
                shutil.rmtree(seg_path)
            self.logger.write(time.strftime('%Y-%m-%d %H:%M:%S: \t', time.localtime()) + 'clean done')
            # 返回结果
            tsr_total = np.sum(tissue_label[tumor_region]) / np.sum(tumor_region)
            tsr_nest = np.sum(tissue_label[nest_region]) / np.sum(nest_region)
            tsr_margin = np.sum(tissue_label[margin_region]) / np.sum(margin_region)
            return round(tsr_total * 100, 2), round(tsr_min * 100, 2), round(tsr_nest * 100, 2), round(tsr_margin * 100,
                                                                                                       2)
        except:
            self.logger.write('**************Error**************')
            self.logger.write(traceback.format_exc())
            self.logger.write('*********************************')
            return 0

    def smooth_boundary(self, label, expand_radius=4):
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
        for pixel in border_list:
            row, col = pixel
            tumor_region[max(row - expand_radius, 0):row + expand_radius,
            max(col - expand_radius, 0):col + expand_radius] = False
        # check border
        border_list = []
        for row in range(roi_minrow-expand_radius-1, roi_maxrow+expand_radius+2):
            for col in range(roi_mincol-expand_radius-1, roi_maxcol+expand_radius+2):
                if self._is_border(row, col, tumor_region, tumor_side=False):
                    border_list.append((row,col))
        # contrast
        for pixel in border_list:
            row, col = pixel
            tumor_region[max(row - expand_radius, 0):row + expand_radius,
            max(col - expand_radius, 0):col + expand_radius] = False
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
        return tumor_region


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
        if row < 0 or col < 0 or row >= tumor_region.shape[0] or col >= tumor_region.shape[1]:
            return False
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
        for row in np.unique(roi[0]):
            for col in roi[1][roi[0]==row]:
                window_label = tumor_label[max(row - radius, 0):row + radius, max(col - radius, 0):col + radius]
                window_region = tumor_region[max(row - radius, 0):row + radius, max(col - radius, 0):col + radius]
                tsr_map[row//stride][col//stride] = np.sum(window_label[window_region])/np.sum(window_region)
        return tsr_map

    def find_min_region(self, tumor_label, tumor_region, winLength=66, winWidth=39):
        roi = tumor_region.nonzero()
        min_coor = (0,0)
        min_tsr = 1
        winAera = winWidth * winLength
        for row in np.unique(roi[0]):
            for col in roi[1][roi[0]==row]:
                window_label = tumor_label[max(row - winWidth//2, 0):row + winWidth//2, max(col - winLength//2, 0):col + winLength//2]
                window_region = tumor_region[max(row - winWidth//2, 0):row + winWidth//2, max(col - winLength//2, 0):col + winLength//2]
                if np.sum(window_region)/winAera < 0.8:
                    continue
                f = 5
                if np.sum(window_label[:winWidth // f, winLength // f:-winLength // f]) * \
                        np.sum(window_label[-winWidth // f:, winLength // f:-winLength // f]) == 0:
                    continue
                if np.sum(window_label[winWidth // f:-winWidth // f, :winLength // f]) * \
                        np.sum(window_label[winWidth // f:-winWidth // f, -winLength // f:]) == 0:
                    continue
                tsr = np.sum(window_label[window_region])/np.sum(window_region)
                if tsr < min_tsr:
                    min_tsr = tsr
                    min_coor = (row,col)
        return min_coor, min_tsr


if __name__ == '__main__':
    classifier = tif_classifier()
    classifier.classify_WSI(os.path.join('data/original_slide/2019-3905.mrxs'), 'predict_result', 20, 1024)
    # multiprocessing.freeze_support()
    # print("请根据电脑内存及CPU核数输入并行数（推荐：2-5）：")
    # try:
    #     num_workers = int(input())
    # except:
    #     num_workers = 2
    # try:
    #     assert(num_workers < multiprocessing.cpu_count())
    # except:
    #     num_workers = multiprocessing.cpu_count() - 1
    #
    # print("请根据电脑显卡显存输入batch size（如不支持显卡，则根据内存决定）（推荐：5-10）：")
    # try:
    #     batch_size = int(input())
    # except:
    #     batch_size = 5
    # print("输入要计算的文件(mrxs, tif, svs, ndpi)所在目录：\n"
    #       "(例：E:\\CK\\data\\CK-Original-section-211119)")
    # input_dir = input()
    # if len(input_dir) == 0:
    #     input_dir = "E:\\CK\\data\\CK-Original-section-211119"
    # while not os.path.exists(input_dir):
    #     print('目录不存在！请重新输入：')
    #     input_dir = input()
    # print("输入输出目录：\n"
    #       "(例：E:\\CK\\predict_result)")
    # output_dir = input()
    # if len(output_dir) == 0:
    #     output_dir = "E:\\CK\\predict_result"
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # logger_path = os.path.join('Log', time.strftime('Log%H%M.txt', time.localtime()))
    # logger = Logger(logger_path)
    # logger.write("num_workers: %d" % num_workers)
    # logger.write("batch size: %d" % batch_size)
    # logger.write("input dir: %s" % input_dir)
    # logger.write("output dir: %s" % output_dir)
    # classifier = tif_classifier(logger_path)
    # done_dict = {}
    # if os.path.exists('../done_list.csv'):
    #     with open('../done_list.csv', 'r') as file_donelist:
    #         content = file_donelist.readlines()
    #     title = content[0]
    #     content = content[1:]
    #     for line in content:
    #         ori_name = line.strip().split(',')[0]
    #         done_dict[ori_name] = line.strip().split(',')
    # else:
    #     title = 'original slide name,state,TSR,TSR(min),TSR(nest),TSR(margin),input dir,output dir,start time,time spent\n'
    # for file in os.listdir(input_dir):
    #     if file.endswith('.mrxs') or file.endswith('.ndpi') or file.endswith('.svs') or file.endswith('.tif'):
    #         slide_name = file[:file.rfind('.mrxs')]
    #         if slide_name in done_dict.keys():
    #             if done_dict[slide_name][1] == 'success':
    #                 continue
    #         start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    #         start_time = time.time()
    #         tsr1, tsr2, tsr3, tsr4 = classifier.classify_WSI(os.path.join(input_dir, file), output_dir,
    #                                       num_workers=num_workers, batch_size=batch_size)
    #         end_time = time.time()
    #         time_spent = round(end_time - start_time)
    #         if tsr1 == 0:
    #             state = 'fail'
    #         else:
    #             state = 'success'
    #         done_dict[slide_name] = [slide_name, state,
    #                                  str(round(tsr1, 3)), str(round(tsr2, 3)), str(round(tsr3, 3)), str(round(tsr4,3)),
    #                                  os.path.join(input_dir, file), os.path.join(output_dir, slide_name),
    #                                  start_time_str,
    #                                  str(time_spent//3600)+':'+str((time_spent % 3600)//60)+':'+str(time_spent % 60)]
    #         with open('../done_list.csv', 'w') as file_donelist:
    #             file_donelist.write(title)
    #             for value in done_dict.values():
    #                 file_donelist.write(','.join(value) + '\n')

