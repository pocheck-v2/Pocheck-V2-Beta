import os
import sys
class NAME:
    '''
       DIR name
       '''
    DATASET_DIR = "actor_mtcnn_pur"
    JB_DIR = "jb"
    PARAM_DIR = "parameter"
    DET_DIR = "det"

    '''
    MODEL name
    '''
    FACENET_NAME = '20180402-114759'
    FACENET_PB = FACENET_NAME + '.pb'
    EMB_ARRAY = "mean_emb_result.pkl"
    CLS_NAME = 'actor_pur.pkl'
    A_NAME = "A_con_1078_i5000.pkl"
    G_NAME = "G_con_1078_i5000.pkl"


class PATH:
    '''
    PATH define
    '''
    CONFIG_PATH = os.path.abspath(__file__)
    ROOT_PATH = os.path.dirname(CONFIG_PATH)
    APP_PATH = os.path.join(ROOT_PATH, "Pocheck")
    WEB_PATH = os.path.join(ROOT_PATH, "PoCheck_Web")
    RUN_PATH = os.path.join(ROOT_PATH, "run_file")


    '''
    in APP path
    '''
    DATA_PATH = os.path.join(APP_PATH, NAME.DATASET_DIR)
    JB_PATH = os.path.join(APP_PATH, NAME.JB_DIR)
    A_PATH = os.path.join(JB_PATH, NAME.A_NAME)
    G_PATH = os.path.join(JB_PATH, NAME.G_NAME)
    PARAM_PATH = os.path.join(APP_PATH, NAME.PARAM_DIR)
    DET_PATH = os.path.join(PARAM_PATH, NAME.DET_DIR)
    FACENET_PATH = os.path.join(PARAM_PATH, NAME.FACENET_NAME)
    PB_PATH = os.path.join(FACENET_PATH, NAME.FACENET_PB)
    CLS_PATH = os.path.join(APP_PATH, NAME.CLS_NAME)

class FLAGS:
    minsize = 35  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44
    frame_interval = 3
    batch_size = 1000
    image_size = 182
    input_image_size = 160
    humans_dir = PATH.DATA_PATH
    in_bound = True
    bound_size = 130
