import caffe
import numpy as np
import utils
import matplotlib.pyplot as plt

GENDER_LIST = ['M', 'F']

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

fold_number = 4
model_file = './deploy.prototxt'
trained = './model_fold_'+str(fold_number)+'/caffenet_train_iter_50000.caffemodel'
guess = '../example/example_image0.jpg'

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

if len(face_list) > 0:
    for i in range(len(face_list)):
        img = caffe.io.load_image(face_list[i])
        #plt.imshow(img)
        #plt.show()

        prediction = net.predict([img])
        print('prediction@1: ' + GENDER_LIST[prediction[0].argmax()])

        filters = net.params['conv1'][0].data[:49]
        vis_square(filters.transpose(0, 2, 3, 1))
        feat = net.blobs['conv1'].data[0, :49]
        vis_square(feat, padval=1)

else:
    img = caffe.io.load_image(guess)
    #plt.imshow(img)
    #plt.show()

    prediction = net.predict([img])
    print('prediction@1: ' + GENDER_LIST[prediction[0].argmax()])

    filters = net.params['conv1'][0].data[:49]
    vis_square(filters.transpose(0, 2, 3, 1))
    feat = net.blobs['conv1'].data[0, :49]
    vis_square(feat, padval=1)


