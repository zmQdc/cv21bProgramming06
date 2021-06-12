import os

import cv2
from tqdm import tqdm


def tracker(t_type: int, img_dir: str, gt_dir: str):
    if t_type == 0:
        cv2_tracker = cv2.TrackerBoosting_create()
    elif t_type == 1:
        cv2_tracker = cv2.TrackerMIL_create()
    elif t_type == 2:
        cv2_tracker = cv2.TrackerKCF_create()
    elif t_type == 3:
        cv2_tracker = cv2.TrackerTLD_create()
    elif t_type == 4:
        cv2_tracker = cv2.TrackerMedianFlow_create()
    elif t_type == 5:
        cv2_tracker = cv2.TrackerMOSSE_create()
    elif t_type == 6:
        cv2_tracker = cv2.TrackerCSRT_create()
    else:
        print('not correct type!')
        return
    gt = []
    imgs = os.listdir(img_dir)
    gt_txt = os.path.join(img_dir, 'groundtruth.txt')
    with open(gt_txt, 'r') as f:
        for line in f.readlines():
            line = list(map(float, line.strip().split(',')))
            gt.append(list(map(int, line)))
    imgs.sort()
    frame = cv2.imread(os.path.join(img_dir, imgs[0]))
    bbox = gt[0]
    bbox = [bbox[0], bbox[1], bbox[4] - bbox[0], bbox[5] - bbox[1]]
    print(bbox)
    cv2_tracker.init(frame, bbox)
    pr = [gt[0]]
    not_hit_count = fps = 0
    for img in tqdm(imgs):
        if img.endswith('.jpg'):
            frame = cv2.imread(os.path.join(img_dir, img))
            timer = cv2.getTickCount()
            ok, bbox = cv2_tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            if ok:
                x1 = bbox[0]
                y1 = bbox[1]
                w = bbox[2]
                h = bbox[3]
                pr.append([x1, y1, x1 + w, y1, x1 + w, y1 + h, x1, y1 + h])
            else:
                not_hit_count += 1
                pr.append([0, 0, 0, 0, 0, 0, 0, 0])

    print('fps: ' + str(fps))
    print('not hit count ' + str(not_hit_count))
    with open(gt_dir, 'w+') as f:
        for p in pr:
            f.write(','.join(str(x) for x in p))
            f.write('\n')


if __name__ == '__main__':
    img_base = 'data/trainval'
    pr_base = 'data/predict'
    img_dirs = os.listdir(img_base)
    tracker_type = ['Boosting', 'MIL', 'KCF', 'TLD', 'MedianFlow', 'MOSSE', 'CSRT']
    for i in range(len(tracker_type)):
        print('-----------------------\nstart model=', tracker_type[i])
        if not os.path.exists(os.path.join(pr_base, tracker_type[i])):
            os.mkdir(os.path.join(pr_base, tracker_type[i]))
        for img_dir in img_dirs:
            print('start to resolve dir: ', img_dir)
            pr_dir = os.path.join(pr_base, tracker_type[i]) + '/' + img_dir + '.txt'
            try:
                tracker(i, os.path.join(img_base, img_dir), pr_dir)
            except:
                continue
        print('-----------------------end')
