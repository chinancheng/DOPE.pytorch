from network.rtpose_vgg import get_model, use_vgg
import cv2
import os
import torch
from utils import *
from config import Config
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from cuboid import Cuboid3d
from pyrr import Quaternion
import json


def convert_rvec_to_quaternion(rvec):
    '''Convert rvec (which is log quaternion) to quaternion'''
    theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
    return Quaternion.from_axis_rotation(raxis, theta)


def draw(img, vertexes, color=(0, 0, 255)):
    for point in vertexes:
        if point:
            point = tuple([int(p * Config.stride * ratio) for p in point])
            img = cv2.circle(img, point, 5, color, -1)
    line_1 = [0, 1, 5, 4, 4, 0, 1, 5, 7, 3, 2, 6]
    line_2 = [1, 5, 4, 0, 7, 3, 2, 6, 3, 2, 6, 7]
    for p1, p2 in zip(line_1, line_2):
        if vertexes[p1] and vertexes[p2]:
            p1 = tuple([int(p * Config.stride * ratio) for p in vertexes[p1]])
            p2 = tuple([int(p * Config.stride * ratio) for p in vertexes[p2]])
            img = cv2.line(img, p1, p2, color=color, thickness=1)

    return img

model = get_model(trunk='vgg19')
model = model.cuda()
use_vgg(model, './model', 'vgg19')
model.load_state_dict(torch.load('./logs-drv3-2/checkpoint.pth'))
model.eval()
path_to_data = '/media/ssd_external/DR-V3'
# path_to_data = './data/JockDM'
# path_to_data = './data/real'
f = open(os.path.join(path_to_data, 'train.txt'))


# parameter of object and camera for pnp solver
object_size = json.load(open(os.path.join(path_to_data, '_object_settings.json')))['exported_objects'][1]['cuboid_dimensions']
intrinsic_settings = json.load(open(os.path.join(path_to_data, '_camera_settings.json')))['camera_settings'][0]['intrinsic_settings']
_cuboid3d = Cuboid3d(object_size)
cuboid3d_points = np.array(_cuboid3d.get_vertices())
matrix_camera = np.zeros((3,3))
matrix_camera[0,0] = intrinsic_settings['fx']
matrix_camera[1,1] = intrinsic_settings['fy']
matrix_camera[0,2] = intrinsic_settings['cx']
matrix_camera[1,2] = intrinsic_settings['cy']
matrix_camera[2,2] = 1
dist_coeffs = np.zeros((4,1))

# path_to_sequences = [os.path.join(path_to_data, '%d.jpg' %(int(line.split('\n')[0]))) for line in f.readlines()]
path_to_sequences = [os.path.join(path_to_data, '%.6d.png' %(int(line.split('\n')[0]))) for line in f.readlines()]
for path in path_to_sequences:
    original_img = cv2.imread(path)
    # original_img = cv2.imread('./data/BottleDataSet/JockMiddle/%.6d.png' %(id))
    # original_img = cv2.imread('./data/real/%d.jpg' % (id))
    original_img = crop(original_img)
    ratio = max(original_img.shape[:2]) / Config.crop_size
    img = cv2.resize(original_img, (Config.crop_size, Config.crop_size))
    img = preprocess(img).float()
    img = torch.unsqueeze(img, 0)
    out, _ = model(img.cuda())
    line = out[0]
    vertex = out[1]
    vertex = vertex.squeeze()
    line = line.squeeze()
    objects, peaks = find_objects(vertex, line)
    if len(objects) > 0:
        for object in objects:
            cuboid2d_points = object[1] + [(object[0][0] * 8, object[0][1] * 8)]
            cuboid3d_points = np.array(cuboid3d_points)
            location = None
            quaternion = None
            obj_2d_points = []
            obj_3d_points = []

            for i in range(8):
                check_point_2d = cuboid2d_points[i]
                # Ignore invalid points
                if (check_point_2d is None):
                    continue
                obj_2d_points.append(check_point_2d)
                obj_3d_points.append(cuboid3d_points[i])
            projected_points = object[1]
            vertexes = projected_points
            draw(original_img, vertexes, (255, 255, 0))
            centroid = tuple([int(point * Config.stride * ratio) for point in object[0]])
            original_img = cv2.circle(original_img, centroid, 5, (255, 255, 0), -1)
            obj_2d_points = np.array(obj_2d_points, dtype=float)
            obj_3d_points = np.array(obj_3d_points, dtype=float)
            valid_point_count = len(obj_2d_points)
            if valid_point_count >= 4:
                ret, rvec, tvec = cv2.solvePnP(obj_3d_points, obj_2d_points, matrix_camera, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if ret:
                    location = list(x[0] for x in tvec)
                    quaternion = convert_rvec_to_quaternion(rvec)

                    projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, matrix_camera,
                                                            dist_coeffs)
                    projected_points = np.squeeze(projected_points)

                    # If the location.Z is negative or object is behind the camera then flip both location and rotation
                    x, y, z = location
                    if z < 0:
                        # Get the opposite location
                        location = [-x, -y, -z]

                        # Change the rotation by 180 degree
                        rotate_angle = np.pi
                        rotate_quaternion = Quaternion.from_axis_rotation(location, rotate_angle)
                        quaternion = rotate_quaternion.cross(quaternion)
                    vertexes = [tuple(p) for p in projected_points]
            draw(original_img, vertexes)
        # print('save output/stage1/img_{}.png' .format(id))
        # cv2.imwrite('output/stage1/img_{}.png' .format(id), original_img)
    cv2.imshow('img', original_img)
    cv2.waitKey(5000)




