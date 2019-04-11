from network.rtpose_vgg import get_model, use_vgg
import cv2
import glob
import os
import torch
from utils import *
from config import Config
from argparse import ArgumentParser
from cuboid import Cuboid3d
from pyrr import Quaternion
import json


def main(args):
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    path_to_data_dir = args.path_to_data_dir
    if not os.path.exists(path_to_data_dir):
        raise FileNotFoundError(path_to_data_dir)
    path_to_checkpoint = args.checkpoint
    if not os.path.exists(path_to_checkpoint):
        raise FileNotFoundError(path_to_data_dir)
    class_name = args.class_name
    fps = args.fps
    img_prefix = args.img_prefix

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
    matrix_camera[0, 0] = intrinsic_settings['fx']
    matrix_camera[1, 1] = intrinsic_settings['fy']
    matrix_camera[0, 2] = max(intrinsic_settings['cx'], intrinsic_settings['cy'])
    matrix_camera[1, 2] = max(intrinsic_settings['cx'], intrinsic_settings['cy'])
    matrix_camera[2, 2] = 1
    try:
        dist_coeffs = np.array(json.load(open(path_to_camera_seetings))['camera_settings'][0]["distortion_coefficients"])
    except KeyError:
        dist_coeffs = np.zeros((4, 1))
    path_to_sequences = sorted(glob.glob(os.path.join(path_to_data_dir, '*.{}'.format(img_prefix))))

    for img_path in path_to_sequences:
        original_img = crop(cv2.imread(img_path))
        ratio = max(original_img.shape[:2]) / Config.crop_size
        img = cv2.resize(original_img, (Config.crop_size, Config.crop_size))
        img = preprocess(img).float()
        img = torch.unsqueeze(img, 0)
        out, _ = model(img.cuda())
        line, vertex = out[0].squeeze(), out[1].squeeze()
        objects, peaks = find_objects(vertex, line)
        original_img = cv2.putText(original_img, "Class name: {}".format(class_name), (50, 50),
                                   cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

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
                    if check_point_2d is None:
                        continue
                    elif check_point_2d[0] < 0 or check_point_2d[1] < 0 or check_point_2d[0] >= Config.crop_size/Config.stride or check_point_2d[1] >= Config.crop_size/Config.stride:
                        continue
                    else:
                        check_point_2d = (check_point_2d[0] * Config.stride * ratio, check_point_2d[1] * Config.stride * ratio)
                    obj_2d_points.append(check_point_2d)
                    obj_3d_points.append(cuboid3d_points[i])
                centroid = tuple([int(point * Config.stride * ratio) for point in object[0]])
                original_img = cv2.circle(original_img, centroid, 5, -1)
                obj_2d_points = np.array(obj_2d_points, dtype=float)
                obj_3d_points = np.array(obj_3d_points, dtype=float)
                valid_point_count = len(obj_2d_points)
                if valid_point_count >= 5:
                    ret, rvec, tvec = cv2.solvePnP(obj_3d_points, obj_2d_points, matrix_camera, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    if ret:
                        location = list(x[0] for x in tvec)
                        quaternion = convert_rvec_to_quaternion(rvec)

                        projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, matrix_camera,
                                                                dist_coeffs)
                        projected_points = np.squeeze(projected_points)
                        # If the location.Z is negative or object is behind the camera then flip both location and rotation
                        x, y, z = location
                        original_img = cv2.putText(original_img, "Location Prediction: x: {:.2f} y: {:.2f} z: {:.2f}".format(x/10, y/10, z/10), (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
                        print("Location Prediction: x: {:.2f} y: {:.2f} z: {:.2f}".format(x/10, y/10, z/10))
                        if z < 0:
                            # Get the opposite location
                            location = [-x, -y, -z]

                            # Change the rotation by 180 degree
                            rotate_angle = np.pi
                            rotate_quaternion = Quaternion.from_axis_rotation(location, rotate_angle)
                            quaternion = rotate_quaternion.cross(quaternion)
                        vertexes = [tuple(p) for p in projected_points]
                        plot(original_img, vertexes)
            if args.save:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, img_path.split('/')[-1])
                print('=> Save {}' .format(output_path))
                cv2.imwrite(output_path, original_img)
            if args.plot:
                original_img = cv2.resize(original_img, (600, 600))
                cv2.imshow('prediction', original_img)
                cv2.waitKey(int(1000/fps))


def plot(img, _vertexes, color=(255, 255, 0), scale=1):
    vertexes = _vertexes.copy()
    # plot
    for point in vertexes:
        if point:
            point = tuple([int(p*scale) for p in point])
            img = cv2.circle(img, point, 7, color, -1)
    line_1 = [0, 1, 5, 4, 4, 0, 1, 5, 7, 3, 2, 6]
    line_2 = [1, 5, 4, 0, 7, 3, 2, 6, 3, 2, 6, 7]
    for p1, p2 in zip(line_1, line_2):
        if vertexes[p1] and vertexes[p2]:
            _p1 = tuple([int(p * scale) for p in vertexes[p1]])
            _p2 = tuple([int(p * scale) for p in vertexes[p2]])
            img = cv2.line(img, _p1, _p2, color=color, thickness=2)
    if vertexes[4] and vertexes[3] and vertexes[8]:
        center = tuple([int(p * scale) for p in vertexes[8]])
        norm = tuple([int(((p1 + p2) // 2 * scale - c) * 10 + c) for p1, p2, c in zip(vertexes[4], vertexes[3], center)])
        img = cv2.circle(img, norm, 7, color, 1)
        img = cv2.line(img, center, norm, color=color, thickness=2)

    return img


if __name__ == "__main__":
    parser = ArgumentParser(description="Semantic Segmentation parser")
    parser.add_argument('-class', '--class_name', dest="class_name", choices=["Jack_Daniels", "Jose_Cuervo"], default="Jack_Daniels", type=str, help='the class name of object')
    parser.add_argument('-d', '--path_to_data_dir', dest="path_to_data_dir", required=True, type=str, help='the path of rgb image')
    parser.add_argument('-c', '--checkpoint', dest="checkpoint", required=True, type=str,
                        help='the path of model checkpoint')
    parser.add_argument('-o', '--output_dir', dest="output_dir", default=None, type=str,
                        help='the path of output folder')
    parser.add_argument('-f', '--fps', dest="fps", default=5, type=int,
                        help='FPS of input sequence')
    parser.add_argument('-s', '--save', dest="save", action='store_true')
    parser.add_argument('-p', '--plot', dest="plot", action='store_true')
    parser.add_argument('-prefix', '--img_prefix', dest="img_prefix", default="jpg", type=str,
                        help='the image prefix')
    main(parser.parse_args())



