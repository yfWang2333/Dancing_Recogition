# Dancing_Recognition
Project for dancing recognition
## About data sets
We use the [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) as the whole data set for training and validating feasibility and efficiency of C3D + ConvLSTM network we built. The data set (including training set and validation set) of 4-class classification is a subset of UCF-101 containing 4 classes as GolfSwing, HulaHoop, JumpingJack, JumpRope.
## Reference
- The architecture of network is refered in [Learning Spatiotemporal Features using 3DCNN and Convlutional LSTM for Gensture Recognition](https://ieeexplore.ieee.org/document/8265580).
- The part to convert video to frame sequences (videoclips) and to count the number of frames of each video is forked in `video_jpg_ucf101_hmdb51.py` and `n_frames_ucf101_hmdb51.py` from [two-stream-action-recognition](https://github.com/jeffreyhuang1/two-stream-action-recognition/blob/master/)

