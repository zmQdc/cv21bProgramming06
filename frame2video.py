import cv2
import os
import numpy as np


def frame2video(img_dir: str, video_dir: str, fps: int):
    img_list = os.listdir(img_dir)
    img_list.sort()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    img = cv2.imread(img_dir + '/' + img_list[0])
    img_size = (img.shape[1], img.shape[0])
    video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for i in img_list:
        if i.endswith('.jpg') or i.endswith('.jpeg'):
            path = os.path.join(img_dir, i)
            frame = cv2.imread(path)
            video_writer.write(frame)
    video_writer.release()


def frame_to_video_with_gt(img_dir: str, video_dir: str, fps: int):
    img_list = os.listdir(img_dir)
    img_list.sort()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    img = cv2.imread(img_dir + '/' + img_list[0])
    img_size = (img.shape[1], img.shape[0])
    video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    gt = []
    txt_path = os.path.join(img_dir, img_list[len(img_list) - 1])
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            gt.append(list(map(int, line.strip().split(','))))

    for i in range(len(img_list)):
        if img_list[i].endswith('.jpg') or img_list[i].endswith('.jpeg'):
            path = os.path.join(img_dir, img_list[i])
            frame = cv2.imread(path)
            frame = cv2.rectangle(frame, gt[i][0:2], gt[i][4:6], (255, 0, 0), 2)
            video_writer.write(frame)
    video_writer.release()


if __name__ == '__main__':
    base = 'data/trainval'
    dirs = os.listdir(base)
    for d in dirs:
        try:
            # frame2video(os.path.join(base, d), 'data/video/' + d + '.mp4', 30)
            frame2video(os.path.join(base, d), 'data/video/' + d + '_.mp4', 30)
        except:
            print(d)
            continue
