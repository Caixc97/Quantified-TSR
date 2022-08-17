import numpy as np
import openslide
import os
import traceback
import shutil
import multiprocessing
from tqdm import tqdm
import sys
import pickle
import cv2
from ImageAlgorithms import open_operation, close_operation, expand

num_workers = 3

def redefine_boundary(label, margin_region, tsr_map):
    ori_margin_tsr = np.sum(label[margin_region])/np.sum(margin_region)
    new_margin_region = np.zeros(shape=margin_region.shape, dtype=np.bool8)
    new_margin_region[tsr_map > ori_margin_tsr] = False
    new_margin_region[tsr_map <= ori_margin_tsr] = True
    new_margin_region[tsr_map == 0] = False
    return new_margin_region



def gen_preview_image(slide, tumor_region, region1, region2, output_path, slide_name):
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
    region1 = region1[roi_minrow:roi_maxrow, roi_mincol:roi_maxcol].astype(np.uint8)
    region1 = cv2.resize(region1, (slide_image.shape[1],slide_image.shape[0]))
    region2 = region2[roi_minrow:roi_maxrow, roi_mincol:roi_maxcol].astype(np.uint8)
    region2 = cv2.resize(region2, (slide_image.shape[1],slide_image.shape[0]))
    slice = region1 == 1
    slide_image[slice, 2] = slide_image[slice, 2]/2 + 255/2
    cv2.imwrite(os.path.join(output_path, slide_name + '-1.png'), cv2.cvtColor(slide_image, cv2.COLOR_BGR2RGB))
    slide_image[slice, 2] = slide_image[slice, 2] * 2 - 255
    slice = region2 == 1
    slide_image[slice, 1] = slide_image[slice, 1]/2 + 255/2
    cv2.imwrite(os.path.join(output_path, slide_name + '-2.png'), cv2.cvtColor(slide_image, cv2.COLOR_BGR2RGB))
    return



def ProcessRegion(slide_path, cache_dir, output_path, count):
    slide = openslide.OpenSlide(slide_path)
    slide_name = os.path.basename(slide_path)[:os.path.basename(slide_path).rfind('.mrxs')]
    with open(os.path.join(cache_dir, 'label'), 'rb') as file_obj:
        label = pickle.load(file_obj)
    with open(os.path.join(cache_dir, 'tumor_region'), 'rb') as file_obj:
        tumor_region = pickle.load(file_obj)
    with open(os.path.join(cache_dir, 'margin_region'), 'rb') as file_obj:
        margin_region = pickle.load(file_obj)
    with open(os.path.join(cache_dir, 'tsr_map'), 'rb') as file_obj:
        tsr_map = pickle.load(file_obj)
    new_boundary = redefine_boundary(label, margin_region, tsr_map)
    new_boundary = close_operation(new_boundary, radius=57, situ=True)
    new_boundary1 = open_operation(new_boundary, radius=17, situ=False)
    if np.sum(new_boundary1) != 0:
        new_boundary = new_boundary1
    new_boundary[~tumor_region] = False
    os.rename(os.path.join(cache_dir, 'margin_region'), os.path.join(cache_dir, 'old_margin_region'))
    with open(os.path.join(cache_dir, 'margin_region'), 'wb') as file_obj:
        pickle.dump(new_boundary, file_obj)
    bound_tsr = str(np.sum(label[new_boundary]) / np.sum(new_boundary))
    ori_tsr = str(np.sum(label[margin_region])/np.sum(margin_region))
    aera = np.sum(tumor_region)
    ori_ratio = str(np.sum(margin_region)/aera)
    ratio = str(np.sum(new_boundary)/aera)
    n_o_ratio = str(np.sum(new_boundary)/np.sum(margin_region))
    res = [slide_name, ori_tsr, bound_tsr,  ori_ratio, ratio, n_o_ratio]
    gen_preview_image(slide, tumor_region, margin_region, new_boundary, output_path, slide_name)

    with count.get_lock():
        with open(os.path.join(output_path,'result.csv'),'a+') as file_csv:
            file_csv.write(','.join(res))
            file_csv.write('\n')
        count.value += 1
        str_print = "{0}:{1}".format(count.value, slide_name)
        sys.stdout.write('\r%s' % str_print)



def worker(slide_dir, predict_dir, output_path, index, num_workers, count):
    for i, predict_path in enumerate(os.listdir(predict_dir)):
        if i % num_workers != index:
            continue
        slide_name = predict_path
        slide_path = os.path.join(slide_dir, slide_name + '.mrxs')
        if not os.path.exists(slide_path):
            print('\n' + slide_path + ' not found!')
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
            file_csv.write(','.join(['slide name',
                                     'ori boundary tsr','new boundary tsr',
                                     'ori boundary ratio', 'new boundary ratio', 'new boundary/old boundary']))
            file_csv.write('\n')
        count = multiprocessing.Value("i", 0)
        if num_workers == 1:
            worker(slide_dir, predict_dir, output_path, 0, num_workers, count)
        else:
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


