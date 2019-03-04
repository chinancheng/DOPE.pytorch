import argparse
import os
from config import Config
from network.rtpose_vgg import get_model, use_vgg
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim
import numpy as np
import json
from utils import *
from cuboid import Cuboid3d
from pyrr import Quaternion


def _eval(class_name, path_to_data_dir, restore):

    # load object config
    object_size = json.load(open(os.path.join(path_to_data_dir, '_object_settings.json')))['exported_objects'][0][
        'cuboid_dimensions']
    _cuboid3d = Cuboid3d(object_size)
    cuboid3d_points = np.array(_cuboid3d.get_vertices())

    # load camera config
    intrinsic_settings = json.load(open(os.path.join(path_to_data_dir, '_camera_settings.json')))['camera_settings'][0]['intrinsic_settings']
    matrix_camera = np.zeros((3, 3))
    matrix_camera[0,0] = intrinsic_settings['fx']
    matrix_camera[1,1] = intrinsic_settings['fy']
    matrix_camera[0,2] = intrinsic_settings['cx']
    matrix_camera[1,2] = intrinsic_settings['cy']
    matrix_camera[2,2] = 1
    dist_coeffs = np.zeros((4, 1))

    # dataloader
    val_dataset = Dataset(path_to_data=path_to_data_dir, class_name=class_name, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # load model
    model = get_model(trunk='vgg19')
    model = model.cuda()
    use_vgg(model, './model', 'vgg19')

    # restore model
    model.load_state_dict(torch.load(restore))
    model.eval()

    for batch_index, (images, _, _, location_target, ratio) in enumerate(val_dataloader):
        images = images.cuda()
        output, _ = model(images)
        line, vertex = output[0], output[1]
        line, vertex = line.squeeze(), vertex.squeeze()
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
                    elif check_point_2d[0] < 0 or check_point_2d[1] < 0 or check_point_2d[
                        0] >= Config.crop_size / Config.stride or check_point_2d[1] >= Config.crop_size / Config.stride:
                        continue
                    else:
                        check_point_2d = (check_point_2d[0] * Config.stride * ratio, check_point_2d[1] * Config.stride * ratio)
                    obj_2d_points.append(check_point_2d)
                    obj_3d_points.append(cuboid3d_points[i])
                projected_points = object[1]
                vertexes = projected_points.copy()
                centroid = tuple([int(point * Config.stride * ratio) for point in object[0]])
                obj_2d_points = np.array(obj_2d_points, dtype=float)
                obj_3d_points = np.array(obj_3d_points, dtype=float)
                valid_point_count = len(obj_2d_points)
                if valid_point_count >= 6:
                    ret, rvec, tvec = cv2.solvePnP(obj_3d_points, obj_2d_points, matrix_camera, dist_coeffs,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
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


def convert_rvec_to_quaternion(rvec):
    '''Convert rvec (which is log quaternion) to quaternion'''
    theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
    return Quaternion.from_axis_rotation(raxis, theta)


if __name__ == '__main__':
    def main(args):
        path_to_data_dir = args.path_to_data_dir
        class_name = args.class_name
        restore = args.restore_checkpoint
        _eval(class_name, path_to_data_dir, restore)


    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--path_to_data_dir', default='/media/external/Bottle_dataset_split', help='path to data directory')
    parser.add_argument('-l', '--path_to_logs_dir', default='./logs/logs-dr-pr-jose-1', help='path to logs directory')
    parser.add_argument('-c', '--class_name', default='Bottle_s_Jose_Cuervo_Bottle_s_Jose_Cuervo_Object0027', type=str,
                        choices=['bottles_Bottle_s_Jack_Daniels_Object0011', 'Bottle_s_Jose_Cuervo_Bottle_s_Jose_Cuervo_Object0027'], help='class name')
    parser.add_argument('-r', '--restore_checkpoint', default=None, required=True, help='path to restore checkpoint file, e.g., ./logs/model-100.pth')
    main(parser.parse_args())
