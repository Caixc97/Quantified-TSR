import numpy as np

def _is_border(row, col, tumor_region, tumor_side=True):
    """
    返回给定坐标是否是肿瘤边界
    tumor_side指示要检查的是边界的肿瘤侧还是正常组织侧
    注意传入的row,col不能在图像边界
    """
    try:
        if tumor_region[row, col] == True and tumor_side == True:
            target = False
        elif tumor_region[row, col] == False and tumor_side == False:
            target = True
        else:
            return False
        if row == 0 or col == 0 or row == tumor_region.shape[0] - 1 or col == tumor_region.shape[1] - 1:
            return True
        if tumor_region[row - 1, col] == target:
            return True
        elif tumor_region[row, col - 1] == target:
            return True
        elif tumor_region[row, col + 1] == target:
            return True
        elif tumor_region[row + 1, col] == target:
            return True
        return False
    except:
        return False

def check_border(arr, tumor_side=True):
    if np.sum(arr) == 0:
        return []
    roi_minrow = np.min(arr.nonzero()[0])
    roi_maxrow = np.max(arr.nonzero()[0])
    roi_mincol = np.min(arr.nonzero()[1])
    roi_maxcol = np.max(arr.nonzero()[1])
    border_list = []
    for row in range(max(roi_minrow - 1, 0), min(roi_maxrow + 2, arr.shape[0])):
        for col in range(max(roi_mincol - 1, 0), min(roi_maxcol + 2, arr.shape[1])):
            if _is_border(row, col, arr, tumor_side):
                border_list.append((row, col))
    return border_list

def expand(arr, radius, border_list):
    for i, pixel in enumerate(border_list):
        row, col = pixel
        arr[max(row - radius, 0):row + radius,
        max(col - radius, 0):col + radius] = True
    return arr

def contrast(arr, radius, border_list):
    for pixel in border_list:
        row, col = pixel
        arr[max(row - radius, 0):row + radius,
        max(col - radius, 0):col + radius] = False
    return arr


def close_operation(src_arr, radius, situ=False):
    if situ:
        target_arr = src_arr
    else:
        target_arr = src_arr.copy()
    border_list = check_border(target_arr,True)
    expand(target_arr, radius, border_list)
    border_list = check_border(target_arr,True)
    contrast(target_arr, radius, border_list)
    return target_arr

def open_operation(src_arr, radius, situ=False):
    if situ:
        target_arr = src_arr
    else:
        target_arr = src_arr.copy()
    border_list = check_border(target_arr,True)
    contrast(target_arr, radius, border_list)
    border_list = check_border(target_arr,True)
    expand(target_arr, radius, border_list)
    return target_arr
