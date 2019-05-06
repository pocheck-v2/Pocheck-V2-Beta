"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from joint_bayesian import *

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            

                 
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            
            # print(emb_array.shape)
            # print(labels)
            
            # from sklearn.cluster import KMeans

            # temb_array = list(emb_array)
            # tlabels = labels.copy()
            # cent_array = []
            # img_cnts = []
            # for n in range(len(dataset)):
            #     vectors = []
            #     for i in range(len(tlabels)):
            #         print(tlabels[0], n)
            #         if tlabels[0] == n:
            #             # print(type(temb_array.pop()))
            #             vectors.append(temb_array.pop(0))
            #             tlabels.pop(0)
            #             # print('hey')
            #         else:
            #             break
            #     img_cnts.append(len(vectors))
            #     # print(vectors)
            #     cent_array.append(sum(vectors)/len(vectors))
            # print(len(cent_array))
            # # print(cent_array)
            # km_root = KMeans(n_clusters=2, random_state=0).fit(cent_array)
            # print(km_root.labels_)

            # klabels = []
            # for l, imcnt in zip(km_root.labels_, img_cnts):
            #     for _ in range(imcnt):
            #         klabels.append(l)
            
            

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                print("init shape : ",emb_array.shape)
                # node = km_svm_Node(emb_array, np.array(labels), np.array([i for i in range(len(dataset))]))
                tree = km_svm_Tree(emb_array, np.array(labels), np.array([i for i in range(len(dataset))]))
                print(tree.tree)
                # print(node.km_labels)
                # print(emb_array[np.array(node.km_labels)])
                # print(len(emb_array[np.array(node.km_labels, dtype=bool)]))
                # print(len(emb_array[~np.array(node.km_labels, dtype=bool)]))
                
            
                # model = SVC(kernel='linear', probability=True)
                # model.fit(emb_array, labels)
                # print(emb_array.shape)
                # print(len(node.km_labels))
                # model.fit(emb_array, node.km_labels)

                # Create a list of class names
                # class_names = [0, 1]

                # Saving classifier model
                # with open(classifier_filename_exp, 'wb') as outfile:
                #     pickle.dump((node.svm_model, node.class_names), outfile)
                with open(classifier_filename_exp, 'wb') as outfile:
                    # pickle.dump(tree, outfile)
                    pickle.dump((tree.tree, tree.root.classes), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)
                

from sklearn.cluster import KMeans
from sklearn.svm import SVC


class km_svm_Tree(object):

    def __init__(self, data, cls_labels, classes):
        # self.data = data
        # self.cls_labels = cls_labels
        treedic = dict()
        self.root = km_svm_Node(data, cls_labels, classes, treedic)
        self.tree = self.root.treedic


class km_svm_Node(object):

    def __init__(self, data, cls_labels, classes, treedic):
        print(len(classes), self)
        print(treedic)
        self.treedic = treedic
        self.data = data
        self.cls_labels = cls_labels
        self.classes = classes
        print("node data shape : ", data.shape)
        if len(self.classes) > 30:
            self.treedic['leaf'] = False
            self.treedic['sleaf'] = False
            self.treedic['cls_label'] = classes
            self.km_model, self.km_labels, self.cls_left, self.cls_right = self.km_classify(data, cls_labels, classes)
            # print(self.km_labels)
            # print(len(self.km_labels))
            # print(len(self.cls_left))
            # print(len(self.cls_right))
            self.svm_model, self.class_nums = self.svm_classify(data, self.km_labels)
            print(self.svm_model.predict(data))
            self.treedic['node'] = self.svm_model
            self.treedic['left'] = dict()
            self.treedic['right'] = dict()
            # print(len(data[np.array(self.km_labels, dtype=bool)]))
            # print(cls_labels[np.array(self.km_labels, dtype=bool)])
            # print(self.cls_left)
            self.left = km_svm_Node(data[np.array(self.km_labels, dtype=bool)], cls_labels[np.array(self.km_labels, dtype=bool)], self.cls_left, self.treedic['left'])
            self.treedic['left'] = self.left.treedic
            self.right = km_svm_Node(data[~np.array(self.km_labels, dtype=bool)], cls_labels[~np.array(self.km_labels, dtype=bool)], self.cls_right, self.treedic['right'])
            self.treedic['right'] = self.right.treedic
        elif len(self.classes) == 1:
            self.treedic['sleaf'] = True
            self.treedic['cls_label'] = cls_labels[0]
            self.treedic['jb_num'] = self.jb_classify(data, cls_labels)

        else:
            self.treedic['leaf'] = True
            self.treedic['sleaf'] = False
            # self.km_model, self.km_labels, self.cls_left, self.cls_right = self.km_classify(data, cls_labels, classes)
            # print(len(data), len(cls_labels))
            self.svm_model, self.class_nums = self.svm_classify(data, cls_labels)
            self.treedic['node'] = self.svm_model
            self.treedic['cls_label'] = self.class_nums
            self.treedic['jb_num'] = self.jb_classify(data, cls_labels)

    def km_classify(self, data, cls_labels, classes):
        temb_array = list(data)
        tlabels = list(cls_labels)
        # print(tlabels)
        cent_array = []
        img_cnts = []
        # for n in range(len(dataset)):
        for n in classes:#!!!!
            vectors = []
            for i in range(len(tlabels)):
                if tlabels[0] == n:
                    vectors.append(temb_array.pop(0))
                    tlabels.pop(0)
                else:
                    break
            img_cnts.append(len(vectors))
            cent_array.append(sum(vectors)/len(vectors))
        km_model = KMeans(n_clusters=2, random_state=2020).fit(cent_array)

        km_labels = []
        for l, imcnt in zip(km_model.labels_, img_cnts):
            for _ in range(imcnt):
                km_labels.append(l)
        
        # print(km_model.labels_)
        # print(np.array(km_model.labels_, dtype=bool))
        # print(classes)

        return km_model, km_labels, classes[np.array(km_model.labels_, dtype=bool)], classes[~np.array(km_model.labels_, dtype=bool)] # list(km_model.labels_).count(0), list(km_model.labels_).count(1) # km_model.labels_!!!

    def svm_classify(self, data, labels):
        svm_model = SVC(kernel='linear', probability=True).fit(data, labels)
        class_nums = []
        for label in labels:
            if label not in class_nums: class_nums.append(label)
        return svm_model, class_nums
        
    def jb_classify(self, data, labels):
        JointBayesian_Train(data, labels, labels[0])
        return labels[0]

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
