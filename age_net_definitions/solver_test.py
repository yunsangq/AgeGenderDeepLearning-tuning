import caffe
import cv2
from PIL import Image
import matplotlib
import lmdb
import numpy as np

MODEL_FILE = './deploy.prototxt'
TRAIN_FILE = './model_fold_0/caffenet_train_iter_50000.caffemodel'
DB_PATH = '/home/k/PycharmProjects/AgeGenderDeepLearning-master/Folds/lmdb/Test_fold_is_0/age_test_lmdb'
resize_h = 227
resize_w = 227

def test():
    net = caffe.Net(MODEL_FILE, TRAIN_FILE, caffe.TEST)
    caffe.set_mode_gpu()

    lmdb_env = lmdb.open(DB_PATH)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    count = 0
    correct = 0
    for key, value in lmdb_cursor:
        count = count + 1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        #image = image[0, :, :]
        height, width = image.shape
        new_image = cv2.resize(image, (resize_w, resize_h))
        transf = np.zeros((1, resize_h, resize_w), dtype=np.uint8)
        transf[0, :, :] = new_image

        image = image.astype(np.uint8)
        out = net.forward_all(data=np.asarray([image]))
        predicted_label = out['prob'][0].argmax(axis=0)
        if label == predicted_label:
            correct = correct + 1
        # print label
        # print np.shape(image)
        # break
    print 'count: %s, correct: %s, rate: %f' % (count, correct, float(correct) / count)


def main():
    # net = caffe.Classifier(
    #     caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
    #     caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
    #     image_dims=(256, 256))
    # net.set_phase_test()
    # net.set_mode_cpu()
    # net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    # net.set_raw_scale('data', 255)
    # net.set_channel_swap('data', (2, 1, 0))
    # scores = net.predict([caffe.io.load_image("/home/john/Pictures/dog.jpg")])
    # print 'scores: ', scores
    return

def testLMDB():
    lmdb_env = lmdb.open('/home/john/caffe/examples/myimagenet/ilsvrc12_train_lmdb')
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    count = 0
    correct = 0
    for key, value in lmdb_cursor:
        count = count + 1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        # print np.shape(image)
        # break
    print 'end:', count
    return

if __name__ == '__main__':
    print 'start'
    # main()
    #testLMDB()
    test()