import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import kitti_util


def generate_dispariy_from_velo(pc_velo, height, width, calib):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width)) - 1
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    baseline = 0.54

    disp_map = (calib.f_u * baseline) / depth_map
    disp_map[disp_map < 0] = 0
    return disp_map, depth_map

if __name__ == '__main__':
    data_path = "/content/2011_09_26/2011_09_26_drive_0015_sync"
    lidar_dir = data_path + '/velodyne_points/data/'
    image_dir = data_path + '/image_02/data/'
    disparity_dir = data_path + '/disparity_gt/'
    calib_file = "/content/2011_09_26/2011_09_26_drive_0015_calib.txt"


    if not os.path.isdir(disparity_dir):
        os.makedirs(disparity_dir)

    test_list = "/content/GANet/lists/kitti2015_test.list"
    with open(test_list, 'r') as f:
        file_names = [x.strip() for x in f.readlines()]
        print(file_names)


    for fn in file_names:
        predix = fn[:-4]
        calib = kitti_util.Calibration(calib_file)
        # load point cloud
        lidar_file_name = predix + '.bin'
        print(lidar_file_name)
        lidar = np.fromfile(lidar_dir + '/' + lidar_file_name, dtype=np.float32).reshape((-1, 4))[:, :3]
        print(lidar.shape)
        image_file = '{}/{}.png'.format(image_dir, predix)
        image = cv2.imread(image_file)
        height, width = image.shape[:2]
        disp, depth = generate_dispariy_from_velo(lidar, height, width, calib)
        print(disp.shape, depth[288,100])
        my_dpi = 120
        plt.figure(figsize = (depth.shape[1]/my_dpi, depth.shape[0]/my_dpi), dpi=my_dpi)
        cv2_imshow(image)
        sns.heatmap(depth)
        # np.save(disparity_dir + '/' + predix, disp)
        print(disp, disp * 256)
        skimage.io.imsave(disparity_dir + '/' + fn, (disp * 256).astype('uint16'))
    #     print('Finish Disparity {}'.format(predix))
