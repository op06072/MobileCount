import os
import scipy
import numpy as np
import pandas as pd
import PIL.Image as Image
from scipy import io as sio
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

import multiprocessing as mp
from itertools import repeat

dataRoot = '~/Develope/ML/MobileCount'

dstRoot = os.path.join(os.path.expanduser(dataRoot), 'exp/data/Mall')

if not os.path.exists(dstRoot):
    os.makedirs(dstRoot)


def gaussian_filter_density(pts, dst_size, scale=10000):
    # print gt.shape
    density = np.zeros([dst_size[1], dst_size[0]], dtype=np.float32)

    if pts is None:
        return density

    gt_count = len(pts)

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros([dst_size[1], dst_size[0]], dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        # pt2d[int(pt[0]) - 1][int(pt[1]) - 1] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(dst_size)) / 4  # case: 1 point
        # pdb.set_trace()
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density * scale


def create_density_map(image_shape, head_locations, sigma=5, scale=10000):
    density_map = np.zeros(image_shape[:2])
    for x, y in head_locations:
        density_map[int(y)-1, int(x)-1] += 1
    return gaussian_filter(density_map, sigma=sigma) * scale


def gen_map(img_paths, srcRoot, dstPath):
    mat = sio.loadmat(os.path.join(srcRoot, 'mall_gt.mat'))
    for img_path in img_paths:
        img = Image.open(os.path.join(srcRoot, 'frames/frames', img_path))
        wd, ht = img.size

        dst_wd = wd // 16 * 16
        rate_wd = dst_wd / wd
        dst_ht = ht // 16 * 16
        rate_ht = dst_ht / ht

        img = img.resize((dst_wd, dst_ht), Image.BILINEAR)

        img.save(os.path.join(dstPath, 'img', img_path))

        idx = int(img_path.split('_')[-1].split('.')[0]) - 1
        gt = mat["frame"][0, idx][0, 0][0]

        gt_x = (gt[:, 0] * rate_wd).astype(np.int64)
        gt_y = (gt[:, 1] * rate_ht).astype(np.int64)

        pts = np.vstack((gt_x, gt_y)).transpose()

        filtered_pts = [(pt[0] < dst_wd and pt[1] < dst_ht) for pt in pts]

        pts = pts[filtered_pts, :]

        soft_k = gaussian_filter_density(pts, [dst_wd, dst_ht], 1)
        sio.savemat(os.path.join(dstPath, 'den', img_path.replace('.jpg', '_soft.mat')), {'map': soft_k})
        hard_k = create_density_map([dst_ht, dst_wd], pts)
        sio.savemat(os.path.join(dstPath, 'den', img_path.replace('.jpg', '_hard.mat')), {'map': hard_k})


def generate_den_map(img_paths, srcRoot, dstPath, attr, multi=False):
    srcRoot = os.path.expanduser(srcRoot)
    dstRootAttr = os.path.join(dstPath, attr)
    if not os.path.exists(dstRootAttr):
        os.mkdir(dstRootAttr)
    if not os.path.exists(dstRootAttr + '/img'):
        os.mkdir(dstRootAttr + '/img')
    if not os.path.exists(dstRootAttr + '/den'):
        os.mkdir(dstRootAttr + '/den')

    if multi:
        with mp.Pool(processes=os.cpu_count()) as pool:
            pool.starmap(gen_map, zip(img_paths, repeat(srcRoot), repeat(dstRootAttr)))
    else:
        gen_map(img_paths, srcRoot, dstRootAttr)


def main():
    parallel = True

    data_list = [data for data in os.listdir(os.path.join(dstRoot, 'frames/frames')) if '.jpg' in data]

    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=42)
    train_list.sort()
    test_list.sort()

    if parallel:
        cores = os.cpu_count()
        cores = cores if cores is not None else 1
        train_list = np.array_split(train_list, cores)
        test_list = np.array_split(test_list, cores)

    print('train')
    generate_den_map(train_list, dstRoot, dstRoot, 'train', parallel)

    print('test')
    generate_den_map(test_list, dstRoot, dstRoot, 'test', parallel)


def check_mat_csv():
    data_path = os.path.join(dstRoot, 'train', 'den') + '/'
    data = 1
    data = f"seq_{data:06}"
    soft_den = data_path + f"{data}_soft.mat"
    hard_den = data_path + f"{data}_hard.mat"

    print("\n .mat size")
    print(f"soft mat size: {os.path.getsize(soft_den)} bytes")
    print(f"hard mat size: {os.path.getsize(hard_den)} bytes")

    soft_mat = sio.loadmat(soft_den)
    hard_mat = sio.loadmat(hard_den)

    soft_df = pd.DataFrame(soft_mat['map'])
    hard_df = pd.DataFrame(hard_mat['map'])

    soft_csv = soft_den.replace('.mat', '.csv')
    soft_df.to_csv(soft_csv)

    hard_csv = hard_den.replace('.mat', '.csv')
    hard_df.to_csv(hard_csv)

    print("\n .csv size")
    print(f"soft csv size: {os.path.getsize(soft_csv)} bytes")
    print(f"hard csv size: {os.path.getsize(hard_csv)} bytes")

    os.remove(soft_csv)
    os.remove(hard_csv)


if __name__ == '__main__':
    """gt = loadmat(
        os.path.join(os.path.join(os.path.expanduser(dataRoot), dstRoot), 'mall_gt.mat')
    )
    print(gt)
    sample_pts = gt['frame'][0][0][0][0][0]
    sample_img = cv2.imread(
        os.path.join(
            os.path.join(os.path.join(os.path.expanduser(dataRoot), dstRoot), 'frames/frames'),
            'seq_000001.jpg'
        )
    )
    sample_den1 = gaussian_filter_density(sample_pts, sample_img.shape, 1)

    sample_den2 = create_density_map(sample_img.shape, sample_pts)

    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    axes[0].imshow(sample_img)
    axes[0].set_title('Original Image')

    axes[1].imshow(sample_den1.transpose(), cmap='jet')
    axes[1].set_title('Density Map 1')

    axes[2].imshow(sample_den2, cmap='jet')
    axes[2].set_title('Density Map 2')

    axes[3].imshow(sample_den1.transpose() - sample_den2, cmap='jet')
    axes[3].set_title('Density Map Diff')

    plt.show()
    print(sample_den1)"""

    # main()
    check_mat_csv()
