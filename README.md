# Pocheck-V2-Beta

## News
| Date     | Update |
|----------|--------|
| 2019-04-17 | 
 
## Pocheck Server
```
IP : 141.223.108.158
PORT : 2411
ID : intern
PW : 3708
```

## Version
| Requiremnts      | Version |
|-----------------|--------------|
|tensorflow-gpu| 1.8|
|cudatoolkit| 9.0|
|cudnn| 7.3.1|
|python| 3.5.6|


## Webcam - ffmpeg
```
ffmpeg -f alsa -ac 2 -i hw:0 -f v4l2 -s 1920x1080 -i /dev/video0 -t 20 video.mpg
```

## Training Data
- The Asian Face Age Dataset (AFAD)[http://afad-dataset.github.io/]
## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|

## Data Alignment

## Paper
- [1] Research Trends ofr Deep Learning-Based High-Performance Face Recognition Technology;<br>
[https://ettrends.etri.re.kr/ettrends/172/0905172005/33-4_43-53.pdf]<br>
- [2] One-shot Online Learning for Joint Bayesian Model-based
Face Recognition; Hanock Kwak, Chung-Yeon Lee, Beom-Jin Lee, Byoung-Tak Zhang<br>[https://bi.snu.ac.kr/Publications/Conferences/Domestic/KIISE2015W_KwakHN.pdf]<br>
- [3] Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks; Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao<br>[https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html]<br>
- [4] The IEEE Conference on Computer Vision and Pattern Recognition; Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua<br>[https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Niu_Ordinal_Regression_With_CVPR_2016_paper.html]
