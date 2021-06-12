import os
import shutil

if __name__ == '__main__':
    base = 'data/trainval'
    names = os.listdir(base)
    gt = 'data/gt'
    for name in names:
        gt_path = os.path.join(gt, name + '.txt')
        old = os.path.join(base, name)
        old = os.path.join(old, 'groundtruth.txt')
        shutil.move(old, gt_path)
