import os
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
import matplotlib.image as mpimg

predspath = '/data/03-scanpath/datasets_new/SALICON/gt_fixations_one/'
imgspath = '/data/03-scanpath/datasets_new/SALICON/images/val_5000/'

imgspathdir = os.listdir(imgspath)  # 列出文件夹下图片文件名（训练数据集）
predspathdir = os.listdir(predspath)  # 列出文件夹下图片文件名（训练数据集）

img_num = len(imgspathdir)

for img_i in range(img_num):
    # if img_i %100 == 1:
    print(img_i)
    pred_name = imgspathdir[img_i][:-4] + '.mat'
    pred_path = os.path.join(predspath, pred_name)

    pred_fixations = scio.loadmat(pred_path)
    pred_fixations = pred_fixations['gt_fixations']

    image_name = imgspathdir[img_i]
    image_path = os.path.join(imgspath, image_name)
    save_path = os.path.join('/data/qmy/seq_img_salicon/', image_name[:-4], 'GroundTruth.jpg')
    img = mpimg.imread(image_path)
    plt.imshow(img)

    seq = pred_fixations

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for i in range(len(seq)):
        if i < len(seq) - 1:
            plt.plot([seq[i][1], seq[i + 1][1]], [seq[i][0], seq[i + 1][0]], linewidth=2, alpha=0.5, color="royalblue")

    for i in range(0, len(seq)):
        if i == 0:
            color = 'steelblue'
        elif i == len(seq) - 1:
            color = 'brown'
        else:
            color = 'w'
        plt.scatter(seq[i][1], seq[i][0], s=400, c=color, alpha=0.5, linewidths=[1], edgecolors="k")  #
        plt.text(seq[i][1], seq[i][0], i + 1, ha='center', va='center', fontsize=9, color="k")

    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=-0.1)
    plt.cla()

