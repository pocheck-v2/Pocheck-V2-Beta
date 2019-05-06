#coding=utf-8
import sys
import numpy as np
from common import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import time

#Before training,the mean must be substract
def JointBayesian_Train(trainingset, label, index, fold = "./JB_result/"):
    if fold[-1] != '/':
        fold += '/'
    print(trainingset.shape)
    # the total num of image
    n_image = len(label)
    # the dim of features
    n_dim   = trainingset.shape[1]
    # filter the complicate label,for count the total people num
    classes, labels = np.unique(label, return_inverse=True)
    # classes, labels = np.unique(labels, return_inverse=True)
    # the total people num
    n_class = len(classes)
    for i in range(n_class):
        labels[i] = i
    print("class : ", classes)
    print("labels : " , labels)
    # save each people items
    cur = {}
    withinCount = 0
    # record the count of each people
    numberBuff = np.zeros(n_image)
    maxNumberInOneClass = 0
    print(trainingset)
    for i in range(n_class):
        # get the item of i
        cur[i] = trainingset[labels==i]
        # get the number of the same label persons
        n_same_label = cur[i].shape[0]
        
        if n_same_label > 1:
            withinCount += n_same_label
        if numberBuff[n_same_label] == 0:
            numberBuff[n_same_label] = 1
            maxNumberInOneClass = max(maxNumberInOneClass, n_same_label)
    print("cur : ", cur)

    print("prepare done, maxNumberInOneClass=", maxNumberInOneClass)

    u  = np.zeros([n_dim, n_class])
    ep = np.zeros([n_dim, withinCount])
    nowp=0
    print("num of class = ", n_class)
    t = time.time()
    for i in range(n_class):
        # the mean of cur[i]
        u[:,i] = np.mean(cur[i], 0)
        b = u[:,i].reshape(n_dim, 1)
        n_same_label = cur[i].shape[0]
        if n_same_label > 1:
            ep[:, nowp:nowp+n_same_label] = cur[i].T-b
            nowp += n_same_label
    print("1 time : ", time.time()- t )
    Su = np.cov(u.T,  rowvar=0)
    Sw = np.cov(ep.T, rowvar=0)
    oldSw = Sw
    SuFG  = {}
    SwG   = {}
    convergence = 1
    min_convergence = 1
    for l in range(1):
        F  = np.linalg.pinv(Sw)
        u  = np.zeros([n_dim, n_class])
        ep = np.zeros([n_dim, n_image])
        nowp = 0
        for mi in range(maxNumberInOneClass + 1):
            if numberBuff[mi] == 1:
		#G = −(mS μ + S ε )−1*Su*Sw−1
                try:
                    G = -np.dot(np.dot(np.linalg.pinv(mi*Su+Sw), Su), F)
                except Exception:
                    break
		#Su*(F+mi*G) for u
                SuFG[mi] = np.dot(Su, (F+mi*G))
		#Sw*G for e
                SwG[mi]  = np.dot(Sw, G)
        for i in range(n_class):
            ##print l, i
            nn_class = cur[i].shape[0]
	    #formula 7 in suppl_760
            u[:,i] = np.sum(np.dot(SuFG[nn_class],cur[i].T), 1)
	    #formula 8 in suppl_760
            ep[:,nowp:nowp+nn_class] = cur[i].T+np.sum(np.dot(SwG[nn_class],cur[i].T),1).reshape(n_dim,1)
            nowp = nowp+nn_class

        Su = np.cov(u.T,  rowvar=0)
        Sw = np.cov(ep.T, rowvar=0)
        convergence = np.linalg.norm(Sw-oldSw)/np.linalg.norm(Sw)
        print_info("Iterations-" + str(l) + ": "+ str(convergence))
        if convergence<1e-6:
            print("Convergence: ", l, convergence)
            break;
        oldSw=Sw

        if convergence < min_convergence:
       	    min_convergence = convergence
            F = np.linalg.pinv(Sw)
            G = -np.dot(np.dot(np.linalg.pinv(2*Su+Sw),Su), F)
            A = np.linalg.pinv(Su+Sw)-(F+G)
            data_to_pkl(G, fold + "G/G_" + str(index) + "_.pkl")
            data_to_pkl(A, fold + "A/A_" + str(index) + "_.pkl")

    F = np.linalg.pinv(Sw)
    try:
        G = -np.dot(np.dot(np.linalg.pinv(2*Su+Sw),Su), F)
        A = np.linalg.pinv(Su+Sw) - (F+G)
    except:
        return None, None
    data_to_pkl(G, fold + "G_con.pkl")
    data_to_pkl(A, fold + "A_con.pkl")

    return A, G

#ratio of similar,the threshold we always choose in (-1,-2)
def predict_joint_bayesian(A, G, emb_array, current_emb):
    max_i = 0
    max_v = -999999999
    for i in range(emb_array.shape[0]):
        value = Verify(A, G, current_emb, emb_array[i])
        if max_v < value:
            max_v = value
            max_i = i
    return max_i, max_v

def Verify(A, G, x1, x2):

    x1.shape = (-1,1)
    x2.shape = (-1,1)
    ratio = np.dot(np.dot(np.transpose(x1),A),x1) + np.dot(np.dot(np.transpose(x2),A),x2) - 2*np.dot(np.dot(np.transpose(x1),G),x2)
    return float(ratio)


def PCA_Train(data, result_fold, n_components=2000):
    print_info("PCA training (n_components=%d)..."%n_components)

    pca = PCA(n_components=n_components)
    pca.fit(data)

    joblib.dump(pca, result_fold+"pca_model.m")
     
    print_info("PCA done.")

    return pca

def data_pre(data):
    data = np.sqrt(data)
    data = np.divide(data, np.repeat(np.sum(data, 1), data.shape[1]).reshape(data.shape[0], data.shape[1]))
    
    return data

def get_ratios(A, G, pair_list, data):
    distance = []
    for pair in pair_list:
        ratio = Verify(A, G, data[pair[0]], data[pair[1]])
        distance.append(ratio)

    return distance

def excute_performance(file_path, t_s=-8, t_e=10, t_step=1):
    with open(file_path, "rb") as f:
        result = pickle.load(f)
        dist = result['distance']
        y    = result['label']
        print(y)
        print("test size: ", y.shape)
        print("negative size: ", y[y==0].shape)
        print("postive size: ",  y[y==1].shape)

        draw_list = []
        while (t_s < t_e):
            pre = dist >= t_s
            y = (y==1)
            report = metrics.classification_report(y_true=y, y_pred=pre)
            print("threshold: ", t_s)
            print(report)

            report_result = report_format(report)
            draw_list.append([report_result, t_s])
            t_s += t_step

        save_draw_file(draw_list)
