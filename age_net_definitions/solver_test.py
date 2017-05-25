import caffe
import cv2
import matplotlib.pyplot as plt
import lmdb
import numpy as np
from collections import defaultdict
import sys
from PIL import Image, ImageOps

MODEL_FILE = './deploy.prototxt'
TRAIN_FILE = './model_fold_0/caffenet_train_iter_50000.caffemodel'
DB_PATH = '/home/k/PycharmProjects/AgeGenderDeepLearning-master/Folds/lmdb/Test_fold_is_0/age_test_lmdb'
fold_number = 0


def flat_shape(x):
    "Returns x without singleton dimension, eg: (1,28,28) -> (28,28)"
    return x.reshape(filter(lambda s: s > 1, x.shape))


def lmdb_reader(fpath):
    lmdb_env = lmdb.open(fpath)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum).astype(np.uint8)
        yield (key, flat_shape(image), label)


if __name__ == "__main__":
    # Extract mean from the mean image file
    mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
    f = open('../Folds/lmdb/Test_fold_is_'+str(fold_number)+'/mean.binaryproto', 'rb')
    mean_blobproto_new.ParseFromString(f.read())
    mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
    f.close()

    count = 0
    correct = 0
    matrix = defaultdict(int)  # (real,pred) -> int
    labels_set = set()

    # CNN reconstruction and loading the trained weights
    net = caffe.Net(MODEL_FILE, TRAIN_FILE, caffe.TEST)
    caffe.set_mode_gpu()

    reader = lmdb_reader(DB_PATH)

    for i, image, label in reader:
        image_caffe = image.reshape(1, *image.shape)
        image_caffe = np.asarray(image_caffe) - mean_image
        img_array = image_caffe[0].transpose((1, 2, 0))  # (c,h,w)->(h,w,c)
        img = Image.fromarray(np.uint8(img_array))
        img = img.resize((227, 227))
        img_array = np.asarray(img)
        plt.imshow(img_array)
        plt.show()
        img_array = img_array.transpose((2, 0, 1))  # (h,w,c) -> (c,h,w)

        img_array = img_array.reshape(1, *img_array.shape)
        out = net.forward_all(data=img_array)
        plabel = int(out['prob'][0].argmax(axis=0))

        count += 1
        iscorrect = label == plabel
        correct += (1 if iscorrect else 0)
        matrix[(label, plabel)] += 1
        labels_set.update([label, plabel])

        if not iscorrect:
            print("\rError: i=%s, expected %i but predicted %i" \
                  % (i, label, plabel))

        sys.stdout.write("\rAccuracy: %.1f%%" % (100. * correct / count))
        sys.stdout.flush()

    print(", %i/%i corrects" % (correct, count))

    print ""
    print "Confusion matrix:"
    print "(r , p) | count"
    for l in labels_set:
        for pl in labels_set:
            print "(%i , %i) | %i" % (l, pl, matrix[(l, pl)])