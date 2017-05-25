import caffe
import numpy as np
import utils
import matplotlib.pyplot as plt
import cv2

AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

caffe.set_mode_gpu()
caffe.set_device(0)


def showimage(im):
    if im.ndim == 3:
        im = im[:, :, ::-1]
    plt.set_cmap('jet')
    plt.imshow(im)
    plt.show()


def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    showimage(data)


fold_number = 0
model_file = './resnet-18-deploy.prototxt'
trained = './model_fold_'+str(fold_number)+'/resnet-imagenet_iter_50000.caffemodel'
guess = '../example/example_image5.jpg'

face_list = utils.faceDetector(guess, 'face')

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('../Folds/lmdb/Test_fold_is_'+str(fold_number)+'/mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
np.save('./model_fold_'+str(fold_number)+'/mean.npy', arr[0])

net = caffe.Classifier(model_file, trained,
                       mean=np.load('./model_fold_'+str(fold_number)+'/mean.npy').mean(1).mean(1),
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(227, 227))

oversample = True

if len(face_list) > 0:
    for i in range(len(face_list)):
        img = caffe.io.load_image(face_list[i])
        #plt.imshow(img)
        #plt.show()

        if oversample:
            prediction = net.predict([img], oversample=True)
            argsort = prediction[0].argsort()
            print('prediction@1: ' + AGE_LIST[argsort[-1]])
            print('prediction@2: ' + AGE_LIST[argsort[-2]])
        else:
            prediction = net.predict([img], oversample=False)
            argsort = prediction[0].argsort()
            print('prediction@1: ' + AGE_LIST[argsort[-1]])
            print('prediction@2: ' + AGE_LIST[argsort[-2]])

        filters = net.params['conv1'][0].data[:49]
        #vis_square(filters.transpose(0, 2, 3, 1))
        feat = net.blobs['conv1'].data[0, :49]
        #vis_square(feat, padval=1)

else:
    img = caffe.io.load_image(guess)
    #plt.imshow(img)
    #plt.show()

    if oversample:
        prediction = net.predict([img], oversample=True)
        argsort = prediction[0].argsort()
        print('prediction@1: ' + AGE_LIST[argsort[-1]])
        print('prediction@2: ' + AGE_LIST[argsort[-2]])
    else:
        prediction = net.predict([img], oversample=False)
        argsort = prediction[0].argsort()
        print('prediction@1: ' + AGE_LIST[argsort[-1]])
        print('prediction@2: ' + AGE_LIST[argsort[-2]])

    filters = net.params['conv1'][0].data[:49]
    #vis_square(filters.transpose(0, 2, 3, 1))
    feat = net.blobs['conv1'].data[0, :49]
    #vis_square(feat, padval=1)


