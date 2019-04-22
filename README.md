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
- [딥러닝 기반 고성능 얼굴인식 기술 동향](https://ettrends.etri.re.kr/ettrends/172/0905172005/33-4_43-53.pdf)<br>
- [결합 베이지안 모델 기반 얼굴인식을 위한 온라인 순간 학습](https://bi.snu.ac.kr/Publications/Conferences/Domestic/KIISE2015W_KwakHN.pdf)<br>
- [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)<br>
- Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua; The IEEE Conference on Computer Vision and Pattern Recognition[1](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Niu_Ordinal_Regression_With_CVPR_2016_paper.html)
