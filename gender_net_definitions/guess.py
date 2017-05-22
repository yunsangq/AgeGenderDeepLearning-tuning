import caffe
import numpy as np
import utils

GENDER_LIST = ['M', 'F']

caffe.set_mode_gpu()
caffe.set_device(0)

model_file = './deploy.prototxt'
trained = './model_fold_0/caffenet_train_iter_50000.caffemodel'
guess = '../example/example_image0.jpg'

face_list = utils.faceDetector(guess, 'face')

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('../Folds/lmdb/Test_fold_is_0/mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
np.save('./model_fold_0/mean.npy', arr[0])

net = caffe.Classifier(model_file, trained,
                       mean=np.load('./model_fold_0/mean.npy').mean(1).mean(1),
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(227, 227))

if len(face_list) > 0:
    for i in range(len(face_list)):
        img = caffe.io.load_image(face_list[i])
        #plt.imshow(img)
        #plt.show()

        prediction = net.predict([img])
        print('prediction@1: ' + GENDER_LIST[prediction[0].argmax()])
else:
    img = caffe.io.load_image(guess)
    #plt.imshow(img)
    #plt.show()

    prediction = net.predict([img])
    print('prediction@1: ' + GENDER_LIST[prediction[0].argmax()])
