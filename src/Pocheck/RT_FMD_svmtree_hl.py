from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import sys
import time
import random
import pickle
from PIL import ImageFont, ImageDraw, Image
from liveness import FaceLivenessModels, FaceLiveness

from joint_bayesian import *
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
args = parser.parse_args()

## setting
VIDEO_SAVE = True
LIVENESS = False
HD = False
UNKNOWN_THRESHOLD = 0.10
CHECK_POINT = 4

SET_FPS = 14
SET_WIDTH = 1920
SET_HEIGHT = 1080
# 1920 X 1080


def svmtree_predict(data, tree):
    if tree['sleaf']:
        print('class in this sleaf : ', tree['cls_label'])
        return tree['cls_label']
    elif tree['leaf']:
        print('classes in this leaf :',tree['cls_label'])
        print('class probabilities :',tree['node'].predict_proba(data))
        print('best probability :', np.max(tree['node'].predict_proba(data)[0]))
        print('prediction :',tree['node'].predict(data))
        # print(tree['cls_label'])
        # print(tree['cls_label'][tree['node'].predict(data)[0]])
        return tree['node'].predict(data)[0], np.max(tree['node'].predict_proba(data)[0]), tree['cls_label'] #tree['cls_label'][tree['node'].predict(data)[0]]
    else:
        if tree['node'].predict(data):
            return svmtree_predict(data, tree['left'])
        else:
            return svmtree_predict(data, tree['right'])

# img_t = np.zeros((200,400,3),np.uint8)
fontpath = "../font/NanumGothicBold.ttf"
font = ImageFont.truetype(fontpath, 20)

print(os.getcwd())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
matplotlib.use('agg')

### liveness  ###
if LIVENESS:
    INPUT_DIR_MODEL_LIVENESS = "./"

    def monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close):
        if eyes_close:
            eye_counter += 1
        else:
            if eye_counter >= eye_continuous_close:
                total_eye_blinks += 1
            eye_counter = 0
        return total_eye_blinks, eye_counter


    def monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open):
        if mouth_open:
            mouth_counter += 1
        else:
            if mouth_counter >= mouth_continuous_open:
                total_mouth_opens += 1
            mouth_counter = 0
        return total_mouth_opens, mouth_counter

### liveness end ###

def make_file(table):
    file_list = list()
    for key, val in table.items():
        if val[0] is True:
            file_list.append([key, 1, val[1]])
        else:
            file_list.append([key, 0, val[1]])


    output_name = '../src/test1.txt'
    with open(output_name, 'w') as f:
        for x in file_list:
            f.write(str(x[0])+" "+str(x[1])+ " " + str(x[2]) + '\n')
            print(x)
        print("file saved")

    # os.system('../src/send.sh ' + output_name)



print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options, log_device_placement=True))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, '../parameter/det/')

        minsize = 35  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        humans_dir = '~/datasets/image_actor/actor_100'
        # humans_dir = '~/datasets/actor_male'
        # humans_dir = '~/datasets/image_actor/actor_mtcnn_pur'
        humans_dir = facenet.get_dataset(humans_dir)
        HumanNames = []
        Human_hash = dict()
        Human_count = dict()
        human_len = len(humans_dir)


        for cls in humans_dir:
            HumanNames.append(cls.name)
            Human_hash[cls.name] = [False, 0]
            Human_count[cls.name] = 0
        HumanNames.append('unknown')
        make_file(Human_hash)

        print('Loading feature extractionodel')
        #modeldir = '../parameter/20190423-175316/20190423-175316.pb'
        modeldir = '../parameter/20180402-114759/20180402-114759.pb'
        # modeldir = '../parameter/20170511-185253/20170511-185253.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        print(embeddings)
        # time.sleep(5)
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        # classifier_filename = '../parameter/clf/my_classifier.pkl'
        # classifier_filename = 'actor_tree_hl_2.pkl'
        classifier_filename = 'actor_tree_hl_2.pkl'

        classifier_filename_exp = os.path.expanduser(classifier_filename)


        ### liveness ###


        # liveness Model
        if LIVENESS:
            # face_liveness = FaceLiveness(model=FaceLivenessModels.EYESBLINK_MOUTHOPEN, path=INPUT_DIR_MODEL_LIVENESS)
            face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV, path=INPUT_DIR_MODEL_LIVENESS)

            # liveness Data
            is_fake_count_print = 0
            # eyes_close, eyes_ratio = (False, 0)
            # total_eye_blinks, eye_counter, eye_continuous_close = (
            # 0, 0, 1)  # eye_continuous_close should depend on frame rate
            # mouth_open, mouth_ratio = (False, 0)
            # total_mouth_opens, mouth_counter, mouth_continuous_open = (
            # 0, 0, 1)  # eye_continuous_close should depend on frame rate

        ### liveness end ###

        result_fold = 'jb/'
        with open(result_fold + "mean_emb_result.pkl", "rb") as f:
            emb_result = pickle.load(f)
        with open(result_fold+"A.pkl", "rb") as f:
            A = pickle.load(f)
        with open(result_fold+"G.pkl", "rb") as f:
            G = pickle.load(f)


        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)
        if args.video:
            video_capture = cv2.VideoCapture(args.video)
        elif args.image:
            video_capture = cv2.VideoCapture(args.image)
        else:
            video_capture = cv2.VideoCapture(0)
        # print(args.video)
        c = 0
        if video_capture.isOpened() is False:
            print("camera is not connected")

        wid = 640
        hei = 480

        if HD:
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, SET_WIDTH)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, SET_HEIGHT)
            wid = SET_WIDTH
            hei = SET_HEIGHT


        # #video writer
        if VIDEO_SAVE:
            now = time.localtime()
            s = "%04d%02d%02d_%02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('../video_file/'+ s + '.avi', fourcc, SET_FPS, (int(wid), int(hei)))


        print('Start Recognition!')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            # frame = cv2.resize(frame, (0,0), fx=1.5, fy=1.5)    #resize frame (optional)

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]



                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)
                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_a = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)
                    exp_idx = 0

                    chk_name = []
                    tmp_arr = dict()
                    for cls in humans_dir:
                        tmp_arr[cls.name] = False

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        i -= exp_idx

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            exp_idx += 1
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled_a.append(facenet.prewhiten(scaled[i]))



                        ### liveness ###
                        if LIVENESS:
                            # Detect if frame is a print attack or replay attack based on colorspace
                            face_crop = (bb[i][0], bb[i][1], bb[i][2], bb[i][3])
                            # eyes_close, eyes_ratio = face_liveness.is_eyes_close(frame, face_crop)
                            # mouth_open, mouth_ratio = face_liveness.is_mouth_open(frame, face_crop)
                            is_fake_print = face_liveness2.is_fake(frame, face_crop)
                            #is_fake_replay = face_liveness2.is_fake(frame, face_crop, flag=1)

                            if is_fake_print :
                                is_fake_count_print += 1
                            print("This is Fake Data: {}".format(is_fake_count_print))
                            # total_eye_blinks, eye_counter = monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks,
                            #                                                      eye_counter, eye_continuous_close)
                            # total_mouth_opens, mouth_counter = monitor_mouth_opening(mouth_open, mouth_ratio,
                            #                                                          total_mouth_opens, mouth_counter,
                            #                                                          mouth_continuous_open)
                            #
                            # print("total_eye_blinks        = {}".format(total_eye_blinks))  # fake face if 0
                            # print("total_mouth_opens       = {}".format(total_mouth_opens))  # fake face if 0


                        ### liveness end ###



                        scaled_reshape.append(scaled_a[i].reshape(-1, input_image_size, input_image_size, 3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        # print(type(emb_array))
                        print(emb_array.shape)
                        # res = []
                        # for j in range(512):
                        #     if j%4 == 0: 
                        #         tmp_list = []
                        #         tmp_list.append(emb_array[0, j])
                        #     else:
                        #         tmp_list.append(emb_array[0, j])
                        #     if j%4 == 3:
                        #         res.append(min(tmp_list))
                        # emb_array2 = np.zeros((1, 128))
                        # emb_array2[0, :] = res
                        
                        # predictions = model.predict_proba(emb_array)
                        # print(predictions)
                        # best_class_indices = np.argmax(predictions, axis=1)
                        # print(best_class_indices)
                        # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_class_index, best_class_probabilities, group = svmtree_predict(emb_array, model)
                        
                        # print(HumanNames)
                        print('index :',best_class_index)

                        #JB
                        max_i = 0
                        max_v = -99999999999
                        predictions = []
                        for idx in group:
                            value = Verify(A, G, emb_array, emb_result[idx])
                            predictions.append(value)
                            if max_v < value:
                                max_v = value
                                max_i = idx
                        best_class_index_jb = max_i
                        print(predictions)
                        print(best_class_index, best_class_index_jb)

                        #plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 10
                        try:
                            print('result: ', HumanNames[best_class_index_jb], str(np.round(best_class_probabilities, 2)))
                            # print('result: ', HumanNames[best_class_indices[0]], str(np.round(best_class_probabilities, 2)))
                        except:
                            print('unregistered person')  

                        if best_class_probabilities < UNKNOWN_THRESHOLD:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)  # boxing face


                            img_pil = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((text_x, text_y), "외부인", font=font, fill=(0, 0, 255, 0))
                            frame = np.array(img_pil)

                            # cv2.putText(frame, "unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            #             1, (0, 0, 255), thickness=1, lineType=2)
                            continue
                        # print(Human_hash)
                        # print(Human_count)
                        # print('test')
                        try:
                            result_names = HumanNames[best_class_index_jb]
                            # result_names = HumanNames[best_class_indices[0]]
                        except:
                            result_names = 'unknown'
                            Human_hash['unknown'] = [False, 0]
                            Human_count['unknown'] = 0
                        if (Human_hash[result_names][0] is True) and (best_class_index==best_class_index_jb):
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)  # boxing face

                            frame[0:scaled[i].shape[0], 0:scaled[i].shape[1]] = scaled[i]
                            img_pil = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((0, scaled[i].shape[1] + 10), result_names + " 출석", font=font, fill=(0, 255, 0, 0))
                            frame = np.array(img_pil)



                            img_pil = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((text_x, text_y), result_names + " " + str(np.round(best_class_probabilities, 2)), font=font, fill=(0, 255, 0, 0))
                            frame = np.array(img_pil)

                            # cv2.putText(frame, result_names + " " + str(np.round(best_class_probabilities, 2)),
                            #             (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)
                        else:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255),2)  # boxing face

                            img_pil = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img_pil)
                            # draw.text((text_x, text_y), result_names + " " + str(np.round(best_class_probabilities, 2)), font=font, fill=(0, 0, 255, 0))
                            frame = np.array(img_pil)

                            # cv2.putText(frame, result_names + " " + str(np.round(best_class_probabilities, 2)),
                            #             (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)
                        if args.image:
                            cv2.imwrite('det'+args.image, frame.astype(np.uint8))
                            print('saved')
                        print(result_names)


                        chk_name.append(result_names)
                    for x in chk_name:
                        tmp_arr[x] = True

                    for key, val in tmp_arr.items():
                        if Human_hash[key][0] is True:
                            continue
                        if val is True:
                            Human_count[key] += 1
                            if Human_count[key] == CHECK_POINT:
                                now = time.localtime()
                                s = "%04d/%02d/%02d_%02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
                                Human_hash[key] = [True, s]
                                make_file(Human_hash)
                        else:
                            Human_count[key] = 0


                else:
                    print('Unable to align')

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            str_t = 'FPS: %2.3f' % fps
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            cv2.putText(frame, str_t, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            # c+=1
            now = time.localtime()
            s = "%04d%02d%02d_%02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
            cv2.putText(frame, s, (0, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)

            cv2.imshow('Video', frame)
            
            if VIDEO_SAVE:
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if VIDEO_SAVE:
            out.release()
        video_capture.release()
        # #video writer

        # cv2.destroyAllWindows()
