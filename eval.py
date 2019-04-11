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
from tqdm import tqdm


def _eval(class_name, path_to_data_dir, path_to_checkpoint, img_prefix):

    # load pre-trained model
    model = get_model(trunk='vgg19')
    model = model.cuda()
    use_vgg(model, './model', 'vgg19')
    print("=> Load pre-trained model from {}".format(path_to_checkpoint))
    model.load_state_dict(torch.load(path_to_checkpoint))
    model.eval()

    # parameter of object size for pnp solver
    print("=> Load {} object size".format(class_name))
    path_to_object_seetings = os.path.join(path_to_data_dir, '_object_settings.json')
    if not os.path.exists(path_to_object_seetings):
        raise FileNotFoundError(path_to_object_seetings)
    object_list = json.load(open(path_to_object_seetings))['exported_objects']
    object_size = None
    for obj in object_list:
        if obj['class'].find(class_name) != -1:
            object_size = obj['cuboid_dimensions']
    if not object_size:
        raise ValueError("Object size is none")
    _cuboid3d = Cuboid3d(object_size)
    cuboid3d_points = np.array(_cuboid3d.get_vertices())

    # parameter of camera for pnp solver
    path_to_camera_seetings = os.path.join(path_to_data_dir, '_camera_settings.json')
    if not os.path.exists(path_to_camera_seetings):
        raise FileNotFoundError(path_to_camera_seetings)
    intrinsic_settings = json.load(open(path_to_camera_seetings))['camera_settings'][0]['intrinsic_settings']
    matrix_camera = np.zeros((3, 3))
    matrix_camera[0,0] = intrinsic_settings['fx']
    matrix_camera[1,1] = intrinsic_settings['fy']
    matrix_camera[0,2] = max(intrinsic_settings['cx'], intrinsic_settings['cy'])
    matrix_camera[1,2] = max(intrinsic_settings['cx'], intrinsic_settings['cy'])
    matrix_camera[2,2] = 1

    try:
        dist_coeffs = np.array(json.load(open(path_to_camera_seetings))['camera_settings'][0]["distortion_coefficients"])
    except KeyError:
        dist_coeffs = np.zeros((4, 1))

    # dataloader
    val_dataset = Dataset(path_to_data=path_to_data_dir, class_name=class_name, split='val', img_prefix=img_prefix)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    correct = 0
    wrong = 0
    # set threshold (cm)
    threshold = 3.0

    for batch_index, (images, _, _, location_targets, ratio) in tqdm(enumerate(val_dataloader)):
        images = images.cuda()
        output, _ = model(images)
        line, vertex = output[0], output[1]
        line, vertex = line.squeeze(), vertex.squeeze()
        objects, peaks = find_objects(vertex, line)
        location_predictions = []
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
                obj_2d_points = np.array(obj_2d_points, dtype=np.float32)
                obj_3d_points = np.array(obj_3d_points, dtype=np.float32)
                valid_point_count = len(obj_2d_points)
                if valid_point_count >= 4:
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
                    location_predictions.append(location)
        location_predictions = np.array(location_predictions)
        if len(location_targets) == 0:
            wrong += len(location_predictions)
        else:
            location_targets = location_targets.cpu().data.numpy()[0]
            for location_target in location_targets:
                distances = [np.sqrt(np.sum(np.square(location_target - location_prediction/10.0))) for location_prediction in location_predictions]
                if len(distances) == 0:
                    pass
                    wrong += 1
                elif min(distances) > threshold:
                    wrong += 1
                else:
                    correct += 1

    print('Object: {} Accuracy: {}%'.format(class_name, correct/(wrong+correct)*100.0))


if __name__ == '__main__':

    def main(args):
        path_to_data_dir = args.path_to_data_dir
        if not os.path.exists(path_to_data_dir):
            raise FileNotFoundError(path_to_data_dir)
        path_to_checkpoint = args.checkpoint
        if not os.path.exists(path_to_checkpoint):
            raise FileNotFoundError(path_to_data_dir)
        class_name = args.class_name
        img_prefix = args.img_prefix
        _eval(class_name, path_to_data_dir, path_to_checkpoint, img_prefix)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--path_to_data_dir', default='/media/ssd_external/Bottle_dataset_split', help='path to data directory')
    parser.add_argument('-class', '--class_name', dest="class_name", choices=["Jack_Daniels", "Jose_Cuervo"], default="Jack_Daniels", type=str, help='the class name of object')
    parser.add_argument('-c', '--checkpoint', dest="checkpoint", required=True, type=str,
                        help='the path of model checkpoint')
    parser.add_argument('-prefix', '--img_prefix', dest="img_prefix", default="png", type=str,
                        help='the image prefix')
    main(parser.parse_args())
