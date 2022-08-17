import os
import numpy as np
import pickle
import traceback

def getTSR(cache_path):
    with open(os.path.join(cache_path, 'label'), 'rb') as file_obj:
        label = pickle.load(file_obj)
    with open(os.path.join(cache_path, 'tumor_region'), 'rb') as file_obj:
        tumor_region = pickle.load(file_obj)
    with open(os.path.join(cache_path, 'nest_region'), 'rb') as file_obj:
        nest_region = pickle.load(file_obj)
    with open(os.path.join(cache_path, 'margin_region'), 'rb') as file_obj:
        margin_region = pickle.load(file_obj)
    with open(os.path.join(cache_path, 'min_coor'), 'rb') as file_obj:
        min_coor = pickle.load(file_obj)
    if type(min_coor[0]) == tuple:
        winWidth = min_coor[1][1]
        winLength = min_coor[1][0]
        min_coor = min_coor[0]
    else:
        winWidth = 39
        winLength = 66
    r1 = np.sum(label[tumor_region]) / np.sum(tumor_region) * 100
    r2 = np.sum(label[nest_region]) / np.sum(nest_region) * 100
    r3 = np.sum(label[margin_region]) / np.sum(margin_region) * 100
    r4 = np.sum(label[min_coor[0] - winWidth//2:min_coor[0] + winWidth//2,
        min_coor[1] - winLength//2:min_coor[1] + winLength//2])/ (winLength*winWidth) * 100
    return r1,r2,r3,r4


def main():
    try:
        print('Input predict result path:')
        process_dir = input()
        with open('info.csv','w') as file_csv:
            file_csv.write('name,tsr_total,tsr_nest,tsr_margin,tsr_window\n')
            for dir in os.listdir(process_dir):
                if not os.path.exists(os.path.join(process_dir,dir,'reader_cache')):
                    print(dir,' fail')
                    continue
                r1, r2, r3, r4 = getTSR(os.path.join(process_dir,dir,'reader_cache'))
                print(dir, ' succeed')
                file_csv.write(dir + ',')
                file_csv.write(str(r1) + ',')
                file_csv.write(str(r2) + ',')
                file_csv.write(str(r3) + ',')
                file_csv.write(str(r4) + '\n')
        input()
    except:
        with open('error_log.txt','w') as file_error:
            file_error.write(traceback.format_exc())
        print(traceback.format_exc())
        input()

if __name__ == '__main__':
    main()
