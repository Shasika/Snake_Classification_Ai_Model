import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
#from tensorflow import keras
#from tensorflow.contrib import lite

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 224
LR = 1e-3

MODEL_NAME = 'snakes-{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
    print("\nImage = ",img)
    print("\n",img.split('.')[-2])
    temp_name= img.split('.')[-2]
    #print("\n",temp_name[0:3])
    #temp_name=temp_name[0:3]
    print("\n",temp_name[:1])
    temp_name=temp_name[:1]
    #word_label = img.split('.')[-3]
    word_label = temp_name
    
   # word_label = img[0]
  
    if word_label == 'A': return [0,0,0,0,1]    #A_common_krait
    #if word_label == 'A': return [0,0,0,0,0,1]
    elif word_label == 'B': return [0,0,0,1,0]  #B_hump_nosed_viper
    #elif word_label == 'B': return [0,0,0,0,1,0]
    elif word_label == 'C': return [0,0,1,0,0]  #C_indian_cobra
    #elif word_label == 'C': return [0,0,0,1,0,0]
    elif word_label == 'D': return [0,1,0,0,0]  #D_russels_viper
    #elif word_label == 'D': return [0,0,1,0,0,0]
    elif word_label == 'E' : return [1,0,0,0,0] #E_saw_scaled_viper
    #elif word_label == 'E' : return [0,1,0,0,0,0]
    #else : return [1,0,0,0,0,0]
    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression




import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')



if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')



train = train_data[:-11200]
test = train_data[-11200:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

sess = tf.Session()
tf.summary.image

tf.summary.histogram
file_writer = tf.summary.FileWriter('./log', sess.graph)


model.save(MODEL_NAME)
with open('submission_file.csv','w') as f:
    f.write('id,label\n')
            
with open('submission_file.csv','a') as f:
    for data in tqdm(test):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))

