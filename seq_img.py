import os
import scipy.io as scio
import numpy as np
from measures.ScanMatch import get_score
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import matplotlib.image as mpimg

predspath = '/home/cver/datasets/salicon/prediction'
gtspath = '/home/cver/datasets/salicon/train_gt_ScanMatch'
imgspath = '/home/cver/datasets/salicon/train'

imgspathdir = os.listdir(imgspath)  # 列出文件夹下图片文件名（训练数据集）
gtspathdir = os.listdir(gtspath)
predspathdir = os.listdir(predspath)  # 列出文件夹下图片文件名（训练数据集）


imgspathdir.sort()  # 排序
gtspathdir.sort()
predspathdir.sort()  # 排序

img_num = len(imgspathdir)

for img_i in range (img_num):
    #if img_i %100 == 1:
    print(img_i)
    pred_name = predspathdir[img_i]
    pred_path = os.path.join(predspath, pred_name)
    if img_i == 9673:
        print(pred_name)
        continue
    pred_fixations = scio.loadmat(pred_path)
    pred_fixations = pred_fixations['fixation_sequences']
    pred_fixations = pred_fixations - 1

    # 获取图片真实预测点
    gt_name = gtspathdir[img_i]
    gt_path = os.path.join(gtspath, gt_name)
    gt_fixations = scio.loadmat(gt_path)
    gt_fixations = gt_fixations['gt_fixations']

    gt_fixations = np.array([gt_fixations[:, 1], gt_fixations[:, 0]])
    gt_fixations = gt_fixations.T - 1

    image_name = imgspathdir[img_i]
    image_path = os.path.join(imgspath, image_name)

    img = mpimg.imread(image_path)
    plt.imshow(img)

    seq = pred_fixations
    seq1 = gt_fixations

    color=['b','g','r','c','m','y','k','w']

    for i in range(len(seq)):

        plt.scatter(seq[i][1], seq[i][0], s=500, c='#88c999', alpha=0.8, )     # edgecolors="pink"
        plt.text(seq[i][1], seq[i][0],  i, ha='center', va='center',fontsize=9, color = "b")
        if i < len(seq)-1:
            plt.plot([seq[i][1], seq[i+1][1]],[seq[i][0], seq[i+1][0]] , linewidth=2, color="g")

    for i in range(len(seq1)):

        plt.scatter(seq1[i][1], seq1[i][0], s=500, c='#FFFF80', alpha=0.8, )     # edgecolors="pink"
        plt.text(seq1[i][1], seq1[i][0],  i, ha='center', va='center',fontsize=9, color = "r")
        if i < len(seq1)-1:
            plt.plot([seq1[i][1], seq1[i+1][1]], [seq1[i][0], seq1[i+1][0]], linewidth=2, color="y")

    ScanMatch_score, scanmatch_score1 = get_score(pred_fixations, gt_fixations)

    plt.text(20, 20,  f'score : {scanmatch_score1 :.3f} ', ha='left', va='center', fontsize=9, color="m",bbox=dict(boxstyle='round,pad=0.5', ec='r',lw=1 ,alpha=0.5))

    plt.axis('off')

    plt.savefig(f'imgs/{img_i+1}.png',bbox_inches='tight', pad_inches=0.0)
    # plt.show()
    plt.cla()
