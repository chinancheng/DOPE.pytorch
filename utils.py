import numpy as np
from config import Config
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
from pyrr import Quaternion
import cv2


def preprocess(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(img)


def crop(img):
    h, w = img.shape[:2]
    m = max(h, w)
    top = (m - h) // 2
    bottom = (m - h) // 2
    if top + bottom + h < m:
        bottom += 1
    left = (m - w) // 2
    right = (m - w) // 2
    if left + right + w < m:
        right += 1
    pad_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return pad_image


def generate_gaussianmap(center, accumulate_confid_map):
    crop_size = Config.crop_size
    stride = Config.stride
    sigma = Config.sigma

    grid = crop_size / stride
    start = stride / 2.0 - 0.5
    x_range = [i for i in range(int(grid))]
    y_range = [i for i in range(int(grid))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map += cofid_map
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    return accumulate_confid_map


def generate_vecmap(centerA, centerB, accumulate_vec_map, count):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)
    crop_size = Config.crop_size
    stride = Config.stride
    grid = crop_size / stride

    thre = Config.vec_width
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if (norm == 0.0):
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid)
    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]
    mask = np.logical_or.reduce(
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[:, :, np.newaxis])
    accumulate_vec_map += vec_map
    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])
    count[mask == True] = 0

    return accumulate_vec_map, count


def find_objects(vertex2, aff, numvertex=8):
    '''Detects objects given network belief maps and affinities, using heuristic method'''

    all_peaks = []
    peak_counter = 0
    for j in range(vertex2.size()[0]):
        belief = vertex2[j].clone()
        map_ori = belief.cpu().data.numpy()
        map = gaussian_filter(belief.cpu().data.numpy(), sigma=Config.test_sigma)
        p = 1
        map_left = np.zeros(map.shape)
        map_left[p:, :] = map[:-p, :]
        map_right = np.zeros(map.shape)
        map_right[:-p, :] = map[p:, :]
        map_up = np.zeros(map.shape)
        map_up[:, p:] = map[:, :-p]
        map_down = np.zeros(map.shape)
        map_down[:, :-p] = map[:, p:]

        peaks_binary = np.logical_and.reduce(
            (
                map >= map_left,
                map >= map_right,
                map >= map_up,
                map >= map_down,
                map > Config.thresh_map)
        )
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])

        # Computing the weigthed average for localizing the peaks
        peaks = list(peaks)

        win = 5
        ran = win // 2
        peaks_avg = []
        for p_value in range(len(peaks)):
            p = peaks[p_value]
            weights = np.zeros((win, win))
            i_values = np.zeros((win, win))
            j_values = np.zeros((win, win))
            for i in range(-ran, ran + 1):
                for j in range(-ran, ran + 1):
                    if p[1] + i < 0 \
                            or p[1] + i >= map_ori.shape[0] \
                            or p[0] + j < 0 \
                            or p[0] + j >= map_ori.shape[1]:
                        continue

                    i_values[j + ran, i + ran] = p[1] + i
                    j_values[j + ran, i + ran] = p[0] + j

                    weights[j + ran, i + ran] = (map_ori[p[1] + i, p[0] + j])

            # if the weights are all zeros
            # then add the none continuous points
            OFFSET_DUE_TO_UPSAMPLING = 0.4395
            try:
                peaks_avg.append(
                    (np.average(j_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING,
                     np.average(i_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING))
            except:
                peaks_avg.append((p[0] + OFFSET_DUE_TO_UPSAMPLING, p[1] + OFFSET_DUE_TO_UPSAMPLING))
        # Note: Python3 doesn't support len for zip object
        peaks_len = min(len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0]))

        peaks_with_score = [peaks_avg[x_] + (map_ori[peaks[x_][1], peaks[x_][0]],) for x_ in range(len(peaks))]

        id = range(peak_counter, peak_counter + peaks_len)

        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += peaks_len

    objects = []

    # Check object centroid and build the objects if the centroid is found
    for nb_object in range(len(all_peaks[-1])):
        if all_peaks[-1][nb_object][2] > Config.thresh_points:
            objects.append([
                [all_peaks[-1][nb_object][:2][0], all_peaks[-1][nb_object][:2][1]],
                [None for i in range(numvertex)],
                [None for i in range(numvertex)],
                all_peaks[-1][nb_object][2]
            ])

    # Working with an output that only has belief maps
    if aff is None:
        if len(objects) > 0 and len(all_peaks) > 0 and len(all_peaks[0]) > 0:
            for i_points in range(8):
                if len(all_peaks[i_points]) > 0 and all_peaks[i_points][0][2] > Config.threshold:
                    objects[0][1][i_points] = (all_peaks[i_points][0][0], all_peaks[i_points][0][1])
    else:
        # For all points found
        for i_lists in range(len(all_peaks[:-1])):
            lists = all_peaks[i_lists]

            for candidate in lists:
                if candidate[2] < Config.thresh_points:
                    continue

                i_best = -1
                best_dist = 10000
                best_angle = 100
                for i_obj in range(len(objects)):
                    center = [objects[i_obj][0][0], objects[i_obj][0][1]]

                    # integer is used to look into the affinity map,
                    # but the float version is used to run
                    point_int = [int(candidate[0]), int(candidate[1])]
                    point = [candidate[0], candidate[1]]

                    # look at the distance to the vector field.
                    v_aff = np.array([
                        aff[i_lists * 2,
                            point_int[1],
                            point_int[0]].data.item(),
                        aff[i_lists * 2 + 1,
                            point_int[1],
                            point_int[0]].data.item()]) * 10

                    # normalize the vector
                    xvec = v_aff[0]
                    yvec = v_aff[1]

                    norms = np.sqrt(xvec * xvec + yvec * yvec)

                    xvec /= norms
                    yvec /= norms

                    v_aff = np.concatenate([[xvec], [yvec]])

                    v_center = np.array(center) - np.array(point)
                    xvec = v_center[0]
                    yvec = v_center[1]

                    norms = np.sqrt(xvec * xvec + yvec * yvec)

                    xvec /= norms
                    yvec /= norms

                    v_center = np.concatenate([[xvec], [yvec]])

                    # vector affinity
                    dist_angle = np.linalg.norm(v_center - v_aff)

                    # distance between vertexes
                    dist_point = np.linalg.norm(np.array(point) - np.array(center))

                    if dist_angle < Config.thresh_angle \
                            and best_dist > 1000 \
                            or dist_angle < Config.thresh_angle \
                            and best_dist > dist_point:
                        i_best = i_obj
                        best_angle = dist_angle
                        best_dist = dist_point

                if i_best is -1:
                    continue

                if objects[i_best][1][i_lists] is None \
                        or best_angle < Config.thresh_angle \
                        and best_dist < objects[i_best][2][i_lists][1]:
                    objects[i_best][1][i_lists] = (candidate[0], candidate[1])
                    objects[i_best][2][i_lists] = (best_angle, best_dist)

    return objects, all_peaks


def convert_rvec_to_quaternion(rvec):
    '''Convert rvec (which is log quaternion) to quaternion'''
    theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html

    return Quaternion.from_axis_rotation(raxis, theta)
