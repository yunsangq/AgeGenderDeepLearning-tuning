import caffe
import numpy as np
from collections import defaultdict
import lmdb
import sys
import scipy.misc as sm
import cv2

AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

#caffe.set_mode_gpu()
#caffe.set_device(0)
caffe.set_mode_cpu()

model_file = './deploy.prototxt'
trained = './model_fold_0/caffenet_train_iter_50000.caffemodel'
test_lmdb_path = '/home/k/PycharmProjects/AgeGenderDeepLearning-master/Folds/lmdb/Test_fold_is_0/age_test_lmdb'
mean_file_binaryproto = '../Folds/lmdb/Test_fold_is_0/mean.binaryproto'

net = caffe.Net(model_file, trained, caffe.TEST)

mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
f = open(mean_file_binaryproto, 'rb')
mean_blobproto_new.ParseFromString(f.read())
mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
f.close()

count = 0
correct = 0
matrix = defaultdict(int) # (real,pred) -> int
labels_set = set()

lmdb_env = lmdb.open(test_lmdb_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = caffe.io.datum_to_array(datum)
    '''
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0)) # if using RGB instead of BGR
    '''

    image = image.astype(np.uint8)
    out = net.forward_all(data=np.asarray([image]) - mean_image)
    plabel = int(out['prob'][0].argmax(axis=0))
    count += 1
    iscorrect = label == plabel
    correct += (1 if iscorrect else 0)
    matrix[(label, plabel)] += 1
    labels_set.update([label, plabel])

    if not iscorrect:
        print("\rError: key = %s, expected %i but predicted %i" % (key, label, plabel))
        sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
        sys.stdout.flush()

print("\n" + str(correct) + " out of " + str(count) + " were classified correctly")
print ""
print "Confusion matrix:"
print "(r , p) | count"
for l in labels_set:
    for pl in labels_set:
        print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])