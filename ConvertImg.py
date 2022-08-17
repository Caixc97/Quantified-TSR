import os

def GetName(dir_path):
    basename = os.path.basename(dir_path)
    return basename[:basename.rfind('(')]

def GetCoordinate(file):
    coordinate = file[file.rfind('(') + 1:file.rfind(')')].split(',')
    if '.' in coordinate[0]:
        row = float(coordinate[0])
        col = float(coordinate[1])
    else:
        row = int(coordinate[0])
        col = int(coordinate[1])
    return row, col





