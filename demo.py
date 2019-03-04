from network.rtpose_vgg import get_model, use_vgg
import cv2
import os
import torch
from utils import *
from config import Config
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
import json


model = get_model(trunk='vgg19')
model = model.cuda()
use_vgg(model, './model', 'vgg19')
model.load_state_dict(torch.load('./logs/logs-dr-pr-jose-1/checkpoint-last.pth'))
model.eval()
path_to_data = '/media/ssd_external/DR-V4'
# path_to_data = './data/JockDM'
# path_to_data = './data/real'
path_to_data = './data/JPEGImages'
# f = open(os.path.join(path_to_data, 'test.txt'))
path_to_sequences = [os.path.join(path_to_data, 'Real_{}.jpg' .format(i)) for i in range(len(os.listdir(path_to_data)))]

# path_to_sequences = [os.path.join(path_to_data, '%d.jpg' %(int(line.split('\n')[0]))) for line in f.readlines()]
# path_to_sequences = [os.path.join(path_to_data, '%.6d.png' %(int(line.split('\n')[0]))) for line in f.readlines()]
id = 0
for path in path_to_sequences:
    original_img = cv2.imread(path)
    original_img = crop(original_img)
    ratio = max(original_img.shape[:2]) / Config.crop_size
    img = cv2.resize(original_img,(Config.crop_size, Config.crop_size))
    img = preprocess(img).float()
    img = torch.unsqueeze(img, 0)
    out, _ = model(img.cuda())
    line = out[0]
    vertex = out[1]
    vertex = vertex.squeeze()
    # if id > 130:
    #     cv2.imshow('img', original_img)
    #     v = (vertex.sum(dim=0).detach().cpu().numpy()*255.0).astype(np.uint8)
    #     v = cv2.resize(v, (int(32*Config.stride*ratio), int(32*Config.stride*ratio)))
    #     v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    #     v = cv2.applyColorMap(v, cv2.COLORMAP_JET)
    #     v = cv2.addWeighted(v, 0.5, original_img, 0.5, 0)
    #     cv2.imshow('v', v)
    #     cv2.waitKey(0)
    #     exit()
    line = line.squeeze()
    objects, peaks = find_objects(vertex, line)
    if len(objects) > 0:
        for object in objects:
            centroid = tuple([int(point*Config.stride*ratio) for point in object[0]])
            vertexes = object[1]
            original_img = cv2.circle(original_img, centroid, 5, (0, 0, 255), -1)
            for point in vertexes:
                if point:
                    point = tuple([int(p*Config.stride*ratio) for p in point])
                    cv2.circle(original_img, point, 5, (0, 0, 255), -1)
            line_1 = [0, 1, 5, 4, 4, 0, 1, 5, 7, 3, 2, 6]
            line_2 = [1, 5, 4, 0, 7, 3, 2, 6, 3, 2, 6, 7]
            for p1, p2 in zip(line_1, line_2):
                if vertexes[p1] and vertexes[p2]:
                    p1 = tuple([int(p*Config.stride*ratio) for p in vertexes[p1]])
                    p2 = tuple([int(p*Config.stride*ratio) for p in vertexes[p2]])
                    original_img = cv2.line(original_img, p1, p2, color=(0, 0, 255), thickness=1)
    # print('save output/drv4-2/img_{}.png' .format(id))
    # cv2.imwrite('output/drv4-2/img_{}.png' .format(id), original_img)
    id += 1
    cv2.imshow('img', original_img)
    cv2.waitKey(1)


