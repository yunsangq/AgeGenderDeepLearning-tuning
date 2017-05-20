import caffe
import numpy as np

AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

caffe.set_mode_gpu()
caffe.set_device(0)

model_file = './deploy.prototxt'
trained = './model_fold_0/caffenet_train_iter_50000.caffemodel'
guess = '../example/example_image8.jpg'

net = caffe.Net(model_file, trained, caffe.TEST)

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('../Folds/lmdb/Test_fold_is_0/mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
np.save('./model_fold_0/mean.npy', arr[0])

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('./model_fold_0/mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0)) # if using RGB instead of BGR
transformer.set_raw_scale('data', 255.0)

net.blobs['data'].reshape(1, 3, 227, 227)

img = caffe.io.load_image(guess)
net.blobs['data'].data[...] = transformer.preprocess('data', img)

output = net.forward()

print(AGE_LIST[output['prob'].argmax()])
