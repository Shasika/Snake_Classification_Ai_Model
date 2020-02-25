from flask import Flask
from flask_restful import Resource, Api,request, abort
from flask_restful import Resource, Api,request, abort
from sklearn.externals import joblib
import cv2                 #  resizing, images
import numpy as np         # For arrays
import os                  # For directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # Loading bars



app = Flask(__name__)
app.config.from_object('config')
api = Api(app)



# class Hello(Resource):
#     def post(self):
        # value3= request.get_json(force=True)['value_3']
        # decode_str = img_base64.decode("value3")
        # file_like = cStringIO.StringIO(decode_str)
        # img = PIL.Image.open(file_like)
        # # rgb_img[c, r] is the pixel values.
        # rgb_img = img.convert("RGB")
        # return rgb_img


        
      


class Dis(Resource):
    def get(self):

        
        TEST_DIR = 'C:\\xampp\\htdocs\\Android Upload Image\\upload'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'snake-{}-{}.model'.format(LR, '2conv-basic')

        def process_test_data():
            testing_data = []
            for img in tqdm(os.listdir(TEST_DIR)):
                path = os.path.join(TEST_DIR,img)
                img_num = img.split('.')[0]
                img = cv2.imread(path,cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                testing_data.append([np.array(img), img_num])
        
            shuffle(testing_data)
            np.save('test_data.npy', testing_data)
            return testing_data

#train_data = create_train_data()
# If you have already created the dataset:
        train_data = np.load('train_data.npy')


        import tflearn
        from tflearn.layers.conv import conv_2d, max_pool_2d
        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.estimator import regression
        import tensorflow as tf
        tf.reset_default_graph()


        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 4, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')



        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')

        import matplotlib.pyplot as plt

# if you need to create the data:
        test_data = process_test_data()
# if you already have some saved:
        test_data = np.load('test_data.npy')

        fig=plt.figure(figsize=(20,20))

        for num,data in enumerate(test_data[:20]):
    
             img_num = data[1]
             img_data = data[0]
    
             y = fig.add_subplot(6,4,num+1)
             orig = img_data
             data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
    
    
             model_out = model.predict([data])[0]   
             if np.argmax(model_out) == 0: str_label='kunakatuwa'
             elif np.argmax(model_out) == 1: str_label='naya'
             elif np.argmax(model_out) == 2: str_label='thel karawalaya'
             elif np.argmax(model_out) == 3: str_label='Thith polaga'
             elif np.argmax(model_out) == 4: str_label='weli polaga'
        return {'result':str_label}
    #elif np.argmax(model_out) == 1: str_label='1234'
    #elif np.argmax(model_out) == 2: str_label='12345'
  #  elif np.argmax(model_out) == 3: str_label='HJHJKH'
    #if model_out[0] > 0.99 : str_label='Healthy'
    #elif model_out[0] < 0.99 : str_label='lateblight'
    
        



api.add_resource(Dis,'/')
# api.add_resource(Dis,'/dis')
# api.add_resource(HelloWorld, '/')


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')