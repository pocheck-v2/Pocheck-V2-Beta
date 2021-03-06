# Pocheck-V2-Beta

## News
| Date      | Update |
|---------------|--------|
| 2019-04-17 | For PoCheck v1 fork |
| 2019-04-22 | ffserver for streaming service |
| 2019-04-24 | We used the Gender and Race Classification with Face Images model to distinguish Asians data from the Vggface2 dataset. |
 
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
|tensorflow-gpu|1.8|
|cuda-tool-kit|9.0|
|cudnn|7.3.1|
|python|3.5.6|

## Webcam - ffmpeg
- Image Capture and Video Web Streaming using a USB camera<br>[https://twinw.tistory.com/196]
```
ffmpeg -f alsa -ac 2 -i hw:0 -f v4l2 -s 1920x1080 -i /dev/video0 -t 20 video.mpg

```

## RTSP
- Real Time Streaming Protocal
- ffserver.conf
 1. Install the build package
```
sudo apt-get install ffmpeg
```
 2. Web streaming will be tested using ffmpeg, a video recording program, and ffserver, a web streaming server.
 3. Run ffserver.
```
ffserver -f ffserver.conf &
```
 4. Start the stream using ffmpeg.
```
ffmpeg -f v4l2 -s 640x480 -r 30 -i /dev/video0 http://localhost:8090/feed1.ffm
```
 5. Here are the addresses you can check:
```
rtsp://ip_address/test1.mp4
```

## Training Data
- VGGFace2 A large scale image dataset for face recognition<br>
[http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/]<br>
Pirl server address for downloading VggFace dataset<br>
[http://141.223.122.56/vggface2_train.tar.gz]<br>
- The Asian Face Age Dataset (AFAD)<br>
[http://afad-dataset.github.io/]

## Preprocessing
Gender and Race Classification with Face Images<br>
[https://github.com/wondonghyeon/face-classification]<br>

## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|

## Data Alignment

## Paper
- [1] Research Trends ofr Deep Learning-Based High-Performance Face Recognition Technology;<br>
[https://ettrends.etri.re.kr/ettrends/172/0905172005/33-4_43-53.pdf]<br>
- [2] One-shot Online Learning for Joint Bayesian Model-based Face Recognition; Hanock Kwak, Chung-Yeon Lee, Beom-Jin Lee, Byoung-Tak Zhang<br>[https://bi.snu.ac.kr/Publications/Conferences/Domestic/KIISE2015W_KwakHN.pdf]<br>[https://github.com/cyh24/Joint-Bayesian]<br>[https://onedrive.live.com/?authkey=%21AHC4sGY3p8_xNAA&id=D472C693301B900F%21109&cid=D472C693301B900F]
- [3] Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks; Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao<br>[https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html]<br>
- [4] The IEEE Conference on Computer Vision and Pattern Recognition; Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua<br>[https://cv-foundation.org/openaccess/content_cvpr_2016/html/Niu_Ordinal_Regression_With_CVPR_2016_paper.html]
- [5] How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)<br>[https://www.adrianbulat.com/face-alignment]
- [6] Face Alignment Assisted by Head Pose Estimation<br>
[https://arxiv.org/pdf/1507.03148.pdf]
