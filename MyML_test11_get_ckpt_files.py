# encoding=utf8
import os
import re

def get_ckpt_file_seir():
    regtx = r'bird-classifier\.tfl\.ckpt-(.*?).meta'
    result = []
    for s in os.listdir(".\\"):
        seir = re.findall(regtx, s)
        if len(seir) == 1:
            result.append(seir[0])
    return result

def get_birds_files():
    result = []
    for s in os.listdir(".\\image\\birds\\"):
        file_name = "image\\birds\\" + s
        result.append(file_name)
    return result