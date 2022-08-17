# for 3-29 demand
import numpy as np
import openslide
import os
import traceback
import multiprocessing
from tqdm import tqdm
import sys
import pickle
import cv2

WinSize = [(249, 187), (264, 153), (131, 78), (63, 41)]
WinName = ['field size','10x','20x','40x']
num_workers = 3

def relocate(tumor_label, tumor_region, boundary_region, winLength, winWidth):
    winLength = winLength
    winWidth = winWidth
    roi = tumor_region.nonzero()
    min_coor = (0, 0)
    min_tsr = 1
    winAera = winWidth * winLength
    flag_row = 0
    flag_col = 0
    for row in np.unique(roi[0]):
        flag_row += 1
        if flag_row < 4:
            continue
        else:
            flag_row = 0
        for col in roi[1][roi[0] == row]:
            flag_col += 1
            if flag_col < 4:
                continue
            else:
                flag_col = 0
            window_label = tumor_label[max(row - winWidth // 2, 0):row + winWidth // 2,
                           max(col - winLength // 2, 0):col + winLength // 2]
            window_region = tumor_region[max(row - winWidth // 2, 0):row + winWidth // 2,
                            max(col - winLength // 2, 0):col + winLength // 2]
            boundary = boundary_region[max(row - winWidth // 2, 0):row + winWidth // 2,
                            max(col - winLength // 2, 0):col + winLength // 2]
            if np.sum(window_region) / winAera < 0.9:
                continue
            if np.sum(boundary) / winAera < 0.8:
                continue
            f1 = 3
            f2 = 20
            # 四角
            if np.sum(window_label[:winWidth // f2, winLength // f1:-winLength // f1]) * \
                    np.sum(window_label[-winWidth // f2:, winLength // f1:-winLength // f1]) == 0:
                continue
            # 中三分之一
            if np.sum(window_label[winWidth // f1:-winWidth // f1, :winLength // f2]) * \
                    np.sum(window_label[winWidth // f1:-winWidth // f1, -winLength // f2:]) == 0:
                continue
            tsr = np.sum(window_label[window_region]) / np.sum(window_region)
            if tsr < min_tsr:
                min_tsr = tsr
                min_coor = (row, col)
    return min_tsr, min_coor

def split_region(tsr_map):
    tsr_BiRegion = np.zeros((tsr_map.shape[0],tsr_map.shape[1]))
    tsr_BiRegion[tsr_map > 0.5] = 1
    tsr_BiRegion[tsr_map == 0] = -1
    return tsr_BiRegion

def gen_window_line_image(slide, tumor_region, list_coor):
    roi_minrow = tumor_region.nonzero()[0][0]
    roi_maxrow = tumor_region.nonzero()[0][-1]
    roi_mincol = np.min(tumor_region.nonzero()[1])
    roi_maxcol = np.max(tumor_region.nonzero()[1])
    w = (roi_maxrow - roi_minrow) * 32
    h = (roi_maxcol - roi_mincol) * 32
    level = max(slide.level_count - 5, 0)
    ratio = int(slide.level_downsamples[level])
    slide_image = slide.read_region((roi_mincol * 32, roi_minrow * 32), level, (h // ratio, w // ratio))
    slide_image = slide_image.convert('RGB')
    slide_image = np.array(slide_image)
    window_img = np.zeros((tumor_region.shape[0],tumor_region.shape[1], 3), dtype='uint8')
    # window_img[tsr_map == 1] = 2
    # window_img[tsr_map == -1] = 3
    for i in range(4):
        window_img[
        max(list_coor[i][0] - WinSize[i][1] // 2, 0),
        max(list_coor[i][1] - WinSize[i][0] // 2, 0):
        min(list_coor[i][1] + WinSize[i][0] // 2, window_img.shape[1]-1), :] = 1
        window_img[
        max(list_coor[i][0] - WinSize[i][1]//2, 0),
        max(list_coor[i][1] - WinSize[i][0]//2, 0):
        min(list_coor[i][1] + WinSize[i][0]//2, window_img.shape[1]-1), :] = 1
        window_img[
        min(list_coor[i][0] + WinSize[i][1] // 2, window_img.shape[0] - 1),
        max(list_coor[i][1] - WinSize[i][0] // 2, 0):
        min(list_coor[i][1] + WinSize[i][0] // 2, window_img.shape[1] - 1), :] = 1
        window_img[
        max(list_coor[i][0] - WinSize[i][1] // 2, 0):
        min(list_coor[i][0] + WinSize[i][1] // 2, window_img.shape[0] - 1),
        max(list_coor[i][1] - WinSize[i][0] // 2, 0), :] = 1
        window_img[
        max(list_coor[i][0] - WinSize[i][1] // 2, 0):
        min(list_coor[i][0] + WinSize[i][1] // 2, window_img.shape[0] - 1),
        min(list_coor[i][1] + WinSize[i][0] // 2, window_img.shape[1] - 1), :] = 1
    window_img = window_img[roi_minrow:roi_maxrow, roi_mincol:roi_maxcol, :]
    window_img = cv2.resize(window_img,(slide_image.shape[1],slide_image.shape[0]))
    slice = window_img == 1
    slide_image[slice] = 0
    # slice = window_img == 2
    # slide_image[slice] = slide_image[slice]/3 + 255/3*2
    # slice = window_img == 3
    # slide_image[slice] = slide_image[slice]/2
    return slide_image

def get_window_images(slide, list_images, center, winsize):
    res = []
    # CK
    loc_x = (center[1] - winsize[0] // 2) * 32
    loc_y = (center[0] - winsize[1] // 2) * 32
    stride = np.sqrt(center[1] * center[0])
    level = slide.get_best_level_for_downsample(stride)
    level = max(level - 6, 0)
    rate = slide.level_downsamples[level]
    image = slide.read_region(
        location=(loc_x, loc_y), level=level,
        size=(int(winsize[0] * 32 // rate), int(winsize[1] * 32 // rate)))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
    res.append(image)
    # tsr map
    image = list_images[4][
            max(center[0] - winsize[1] // 2, 0):
            min(center[0] + winsize[1] // 2, list_images[4].shape[0]),
            max(center[1] - winsize[0] // 2, 0):
            min(center[1] + winsize[0] // 2, list_images[4].shape[1]),:
            ]
    image = cv2.resize(image,(res[0].shape[1],res[0].shape[0]))
    res.append(image)
    # mask map
    image = list_images[0][
            max(center[0] - winsize[1] // 2, 0):
            min(center[0] + winsize[1] // 2, list_images[4].shape[0]),
            max(center[1] - winsize[0] // 2, 0):
            min(center[1] + winsize[0] // 2, list_images[4].shape[1]),:
            ]
    image[image==0] = 255
    image = cv2.resize(image, (res[0].shape[1], res[0].shape[0]))
    res.append(image)
    return res


def ProcessRegion(slide_path, cache_dir, output_path, count):
    slide = openslide.OpenSlide(slide_path)
    slide_name = os.path.basename(slide_path)[:os.path.basename(slide_path).rfind('.mrxs')]
    if not os.path.exists(os.path.join(output_path, 'window_image')):
        os.mkdir(os.path.join(output_path, 'window_image'))
    image_path = os.path.join(output_path, 'window_image')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    with open(os.path.join(cache_dir, 'label'), 'rb') as file_obj:
        label = pickle.load(file_obj)
    with open(os.path.join(cache_dir, 'tumor_region'), 'rb') as file_obj:
        tumor_region = pickle.load(file_obj)
    try:
        with open(os.path.join(cache_dir, 'old_margin_region'), 'rb') as file_obj:
            boundary_region = pickle.load(file_obj)
    except:
        with open(os.path.join(cache_dir, 'margin_region'), 'rb') as file_obj:
            boundary_region = pickle.load(file_obj)
    with open(os.path.join(cache_dir, 'label_images'), 'rb') as file_obj:
        label_images = pickle.load(file_obj)
    with open(os.path.join(cache_dir, 'tsr_map'), 'rb') as file_obj:
        tsr_map = pickle.load(file_obj)
    tsr_map = split_region(tsr_map)
    centers = []
    res = [slide_name]
    for i in range(4):
        tsr, center = relocate(label, tumor_region, boundary_region, WinSize[i][0], WinSize[i][1])
        centers.append(center)
        res.append(str(round(tsr,3)))
        # cv2.imwrite(os.path.join(image_path, slide_name + '_CK '+WinName[i]+'.jpeg'), cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
        # cv2.imwrite(os.path.join(image_path, slide_name + '_TSR ' + WinName[i] + '.jpeg'), cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
        # cv2.imwrite(os.path.join(image_path, slide_name + '_MASK ' + WinName[i] + '.jpeg'), cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB))
    image = gen_window_line_image(slide, tumor_region, centers)
    cv2.imwrite(os.path.join(image_path, slide_name + '.png'), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with count.get_lock():
        with open(os.path.join(output_path,'result.csv'),'a+') as file_csv:
            file_csv.write(','.join(res))
            file_csv.write('\n')
        count.value += 1
        str_print = "{0}".format(count.value)
        sys.stdout.write('\r%s' % str_print)



def worker(slide_dir, predict_dir, output_path, index, num_workers, count):
    for i, predict_path in enumerate(os.listdir(predict_dir)):
        if i % num_workers != index:
            continue
        slide_name = predict_path
        slide_path = os.path.join(slide_dir, slide_name + '.mrxs')
        if not os.path.exists(slide_path):
            print(slide_path + ' not found!')
            continue
        else:
            ProcessRegion(slide_path, os.path.join(predict_dir, predict_path, 'reader_cache'), output_path, count)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('Input slide(mrxs) dir:')
    slide_dir = input()
    print('Input predict result dir')
    predict_dir = input()
    print('output path:')
    output_path = input()
    try:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with open(os.path.join(output_path,'result.csv'),'w') as file_csv:
            file_csv.write(','.join(['slide name','field size tsr','10x tsr','20x tsr','40x tsr']))
            file_csv.write('\n')
        count = multiprocessing.Value("i", 0)
        process_list = []
        for i in range(num_workers):
            p = multiprocessing.Process(target=worker,
                                        args=(slide_dir, predict_dir, output_path, i, num_workers, count))
            process_list.append(p)
            p.start()
        for p in process_list:
            p.join()
        print('done')
        input()
    except:
        print(traceback.format_exc())
        input()


