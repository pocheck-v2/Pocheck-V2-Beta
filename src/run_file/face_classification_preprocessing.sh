export WORK_HOME=../face-classification
source activate tf_c

if [ ! -d ${WORK_HOME}/test ]
then
mkdir ${WORK_HOME}/test
fi

mkdir ${WORK_HOME}/test;
python ${WORK_HOME}/pred.py --img_dir ${WORK_HOME}/test/ --model ${WORK_HOME}/face_model.pkl --output_dir ${WORK_HOME}/result/
