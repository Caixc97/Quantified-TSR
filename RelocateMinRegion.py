import numpy as np
from tqdm import tqdm
import openslide
import os
import pickle
from cv2 import resize, imwrite, cvtColor, COLOR_RGB2BGR, COLOR_RGBA2RGB, COLOR_RGBA2BGR

def relocate(slide_path, output_path, tumor_label, tumor_region, winLength=66, winWidth=39):
    slide = openslide.OpenSlide(slide_path)
    preview_dir = os.path.join(output_path, 'preview image')
    roi = tumor_region.nonzero()
    min_coor = (0, 0)
    min_tsr = 1
    winAera = winWidth * winLength
    flag_row = False
    flag_col = False
    for row in np.unique(roi[0]):
        flag_row = not flag_row
        if not flag_row:
            continue
        for col in roi[1][roi[0] == row]:
            flag_col = not flag_col
            if not flag_col:
                continue
            window_label = tumor_label[max(row - winWidth // 2, 0):row + winWidth // 2,
                           max(col - winLength // 2, 0):col + winLength // 2]
            window_region = tumor_region[max(row - winWidth // 2, 0):row + winWidth // 2,
                            max(col - winLength // 2, 0):col + winLength // 2]
            if np.sum(window_region) / winAera < 0.9:
                continue
            f = 10
            if np.sum(window_label[:winWidth // f, winLength // f:-winLength // f]) * \
                    np.sum(window_label[-winWidth // f:, winLength // f:-winLength // f]) == 0:
                continue
            if np.sum(window_label[winWidth // f:-winWidth // f, :winLength // f]) * \
                    np.sum(window_label[winWidth // f:-winWidth // f, -winLength // f:]) == 0:
                continue
            tsr = np.sum(window_label[window_region]) / np.sum(window_region)
            if tsr < min_tsr:
                min_tsr = tsr
                min_coor = (row, col)
    loc_x = (min_coor[1] - winLength // 2) * 32
    loc_y = (min_coor[0] - winWidth // 2) * 32
    stride = np.sqrt(min_coor[1] * min_coor[0])
    level = slide.get_best_level_for_downsample(stride)
    level = max(level - 6, 0)
    rate = slide.level_downsamples[level]
    image = slide.read_region(
        location=(loc_x, loc_y), level=level,
        size=(int(winLength * 32 // rate), int(winWidth * 32 // rate)))
    image = cvtColor(np.array(image), COLOR_RGBA2BGR)
    imwrite(os.path.join(preview_dir, 'min_region.png'), image)
    slide.close()
    return min_coor


if __name__ == '__main__':
    print("window length(mm):")
    win_length = float(input())
    print("window width(mm):")
    win_width = float(input())
    print("输入要计算的文件(mrxs, tif, svs, ndpi)所在目录：\n"
          "(例：E:\\CK\\data\\CK-Original-section-211119)")
    input_dir = input()
    if len(input_dir) == 0:
        input_dir = "E:\\CK\\data\\CK-Original-section-211119"
    while not os.path.exists(input_dir):
        print('目录不存在！请重新输入：')
        input_dir = input()
    print("输入输出目录：\n"
          "(例：E:\\CK\\predict_result)")
    output_dir = input()
    if len(output_dir) == 0:
        output_dir = "E:\\CK\\predict_result"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file in tqdm(os.listdir(input_dir)):
        if file.endswith('.mrxs') or file.endswith('.ndpi') or file.endswith('.svs') or file.endswith('.tif'):
            slide_path = os.path.join(input_dir, file)
            basename = os.path.basename(slide_path)
            name = basename[:basename.rfind('.')]
            output_path = os.path.join(output_dir, name)
            if not os.path.exists(output_path):
                continue
            cache_path = os.path.join(output_path, 'reader_cache')
            with open(os.path.join(cache_path, 'label'), 'rb') as file_obj:
                label = pickle.load(file_obj)
            with open(os.path.join(cache_path, 'tumor_region'), 'rb') as file_obj:
                tumor_region = pickle.load(file_obj)
            min_coor = relocate(slide_path, output_path, label, tumor_region, winLength=int(win_length*1000/8.75), winWidth=int(win_width*1000/8.75))
            with open(os.path.join(cache_path, 'min_coor'), 'wb') as file_obj:
                pickle.dump((min_coor, (int(win_length*1000/8.75), int(win_width*1000/8.75))), file_obj)
            print(file, ' done')