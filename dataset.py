import torch
import json
from torch.utils.data import dataset
import glob
import os
import cv2
from config import Config
from utils import *
import glob


class Dataset(dataset.Dataset):
    def __init__(self, path_to_data, class_name, split='train'):
        self.split = split
        self._path_to_data = path_to_data
        self._path_to_sequences = glob.glob(os.path.join(path_to_data, split, '*.json'))
        self.class_name = class_name
        self._num_sequences = len(self._path_to_sequences)
        self.crop_size = Config.crop_size
        self.stride = Config.stride
        self.grid = self.crop_size / self.stride

    def __len__(self):
        return self._num_sequences

    def __getitem__(self, index):
        label_file_path = self._path_to_sequences[index]
        img_file_path = label_file_path.replace('json', 'png')
        vertices = []
        locations = []
        img = cv2.imread(img_file_path)
        self.ratio = img.shape[0]/self.crop_size
        img = cv2.resize(img, (self.crop_size, self.crop_size))
        with open(label_file_path) as f:
            labels = json.load(f)
        for label in labels['objects']:
            if label['class'].find(self.class_name) != -1:
                vertex = label['projected_cuboid']
                centroid = label['projected_cuboid_centroid']
                location = label['location']
                vertex.append(centroid)
                vertices.append(vertex)
                locations.append(location)
        heatmaps, pafs = self.get_ground_truth(vertices)
        img = preprocess(img).float()
        heatmaps = heatmaps.transpose((2, 0, 1))
        heatmaps = torch.tensor(heatmaps, dtype=torch.float)
        pafs = pafs.transpose((2, 0, 1))
        pafs = torch.tensor(pafs, dtype=torch.float)
        if len(locations) != 0:
            locations = torch.tensor(locations, dtype=torch.float)

        return img, heatmaps, pafs, locations, self.ratio

    def get_ground_truth(self, vertices):
        heatmaps = np.zeros((int(self.grid), int(self.grid), 9))
        pafs = np.zeros((int(self.grid), int(self.grid), 16))
        for vertex in vertices:
            for idx, point in enumerate(vertex):
                point = tuple([int(i/self.ratio) for i in point])
                center_point = tuple([int(i/self.ratio) for i in vertex[-1]])

                gaussian_map = heatmaps[:, :, idx]
                if idx < 8:
                    count = np.zeros((int(self.grid), int(self.grid)), dtype=np.uint32)
                    if point[0] >= 0 and point[0] < Config.crop_size and point[1] >= 0 and point[1] < Config.crop_size:
                        centerA = np.array(point)
                        centerB = np.array(center_point)
                        pafs[:, :, idx*2:idx*2+2], count = generate_vecmap(centerA, centerB, pafs[:, :, idx*2:idx*2+2], count)
                heatmaps[:, :, idx] = generate_gaussianmap(point, gaussian_map)

        return heatmaps, pafs


if __name__ == '__main__':
    def main():
        path_to_data_dir = '/media/external/Bottle_dataset_split'
        dataset = Dataset(path_to_data_dir)
        print('len(dataset):', len(dataset))
        print(dataset[0])
    main()
