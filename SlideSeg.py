import openslide
import numpy as np
import sys
import multiprocessing
import os

def worker(num_workers, i, slide_path, save_path, row_count, flag_fail):
    try:
        basename = os.path.basename(slide_path)
        name = basename[:basename.rfind('.')]
        slide = openslide.OpenSlide(slide_path)
        max_row = slide.dimensions[1] // 256
        max_col = slide.dimensions[0] // 256
        for row in range((max_row - 1)*2):
            if row % num_workers != i:
                continue
            row = row/2
            for col in range((max_col - 1)*2):
                col = col/2
                patch_name = name + '(' + str(row) + ',' + str(col) + ')' + '.jpeg'
                if os.path.exists(os.path.join(save_path, patch_name)):
                    continue
                patch = slide.read_region(location=(int(col * 256), int(row * 256)), level=0, size=(256, 256))
                patch_hsv = patch.convert('HSV')
                h, s, v = patch_hsv.split()
                if np.average(s) < 15:
                    continue
                patch = patch.convert('RGB')
                patch_name = name + '(' + str(row) + ',' + str(col) + ')' + '.jpeg'
                patch.save(os.path.join(save_path, patch_name))
            with row_count.get_lock():
                row_count.value += 1
            str_print = "{0}/{1}".format(row_count.value,max_row*2)
            sys.stdout.write('\r%s' % str_print)
            if flag_fail.value != 0:
                break
        slide.close()
    except:
        flag_fail.value += 1


# def seg_slide(slide_path, save_path, num_workers=5):
#     slide = openslide.OpenSlide(slide_path)
#     max_row = slide.dimensions[1]//256
#     max_col = slide.dimensions[0]//256
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     pool = multiprocessing.Pool(processes=num_workers)
#     row_count = multiprocessing.Value("d", 0)
#     slice = max_row//(num_workers-1)
#     for i in range(num_workers):
#         pool.apply_async(worker, (slice*i, min(slice*(i+1), max_row - 1),slide_path, save_path, row_count))
#     pool.close()
#     pool.join()

def seg_slide(slide_path, save_path, num_workers=10):
    slide = openslide.OpenSlide(slide_path)
    max_row = slide.dimensions[1]//256
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    row_count = multiprocessing.Value("i", 0)
    flag_fail = multiprocessing.Value("i", 0)
    process_list = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(num_workers, i, slide_path, save_path, row_count, flag_fail))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()
    if flag_fail.value == 0:
        return True
    else:
        return False


if __name__ == '__main__':
    dir_path = 'data/unlabeled/raw/2014-HE-first-part'
    dst = dir_path.replace('raw', 'seg')
    if not os.path.exists(dst):
        os.mkdir(dst)
    total = 0
    for file in os.listdir(dir_path):
        if file.endswith('mrxs') or file.endswith('svs'):
            total += 1
    count = 0
    for file in os.listdir(dir_path):
        if file.endswith('mrxs') or file.endswith('svs'):
            save_dir = os.path.join(dst,file[:file.rfind('.')])
            if not os.path.join(save_dir):
                os.mkdir(save_dir)
            print(file+'({0}/{1})'.format(count,total))
            count += 1
            if not seg_slide(os.path.join(dir_path,file),dst,num_workers=10):
                print('fail!')

    dir_path = 'data/unlabeled/raw/TCGA'
    dst = dir_path.replace('raw', 'seg')
    if not os.path.exists(dst):
        os.mkdir(dst)
    total = 0
    for dir in os.listdir(dir_path):
        for file in os.listdir(os.path.join(dir_path,dir)):
            if file.endswith('mrxs') or file.endswith('svs'):
                total += 1
    count = 0
    for dir in os.listdir(dir_path):
        for file in os.listdir(os.path.join(dir_path,dir)):
            if file.endswith('mrxs') or file.endswith('svs'):
                save_dir = os.path.join(dst, file[:file.rfind('.')])
                if not os.path.join(save_dir):
                    os.mkdir(save_dir)
                print('\n'+file+'({0}/{1})'.format(count, total))
                count += 1
                if not seg_slide(os.path.join(dir_path, dir, file), dst, num_workers=20):
                    print('fail!')


# import os
# for dir in os.listdir('.'):
#     if not dir.startswith('2-5'):
#         continue
#     print(dir,len(os.listdir(dir)))