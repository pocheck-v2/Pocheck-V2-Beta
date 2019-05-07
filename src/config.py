import os
import sys
class NAME:
    '''
       DIR name
       '''
    DATASET_DIR = "actor_136"
    JB_DIR = "JB_result"
    PARAM_DIR = "parameter"
    DET_DIR = "det"
    A_DIR = "A"
    G_DIR = "G"

    '''
    MODEL name
    '''
    FACENET_NAME = '20180402-114759'
    FACENET_PB = FACENET_NAME + '.pb'
    EMB_ARRAY = "mean_emb_result.pkl"
    CLS_NAME = '136celeb.pkl'

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
    A_PATH = os.path.join(JB_PATH, NAME.A_DIR)
    G_PATH = os.path.join(JB_PATH, NAME.G_DIR)
    PARAM_PATH = os.path.join(APP_PATH, NAME.PARAM_DIR)
    DET_PATH = os.path.join(PARAM_PATH, NAME.DET_DIR)
    FACENET_PATH = os.path.join(PARAM_PATH, NAME.FACENET_NAME)
    PB_PATH = os.path.join(FACENET_PATH, NAME.FACENET_PB)
    CLS_PATH = os.path.join(APP_PATH, NAME.CLS_NAME)