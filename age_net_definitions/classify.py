import sys
import caffe
import matplotlib.pyplot as plt
import numpy as np
import lmdb
from collections import defaultdict

AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
model_file = './deploy.prototxt'
trained = './model_fold_0/caffenet_train_iter_50000.caffemodel'
test_lmdb_path = '/home/k/PycharmProjects/AgeGenderDeepLearning-master/Folds/lmdb/Test_fold_is_0/age_test_lmdb'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('../Folds/lmdb/Test_fold_is_0/mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
np.save('./model_fold_0/mean.npy', arr[0])
mu = np.load('./model_fold_0/mean.npy').mean(1).mean(1)

if __name__ == "__main__":
    count = 0
    correct = 0
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    net = caffe.Net(model_file, trained, caffe.TEST)
    caffe.set_mode_gpu()
    caffe.set_device(0)
    lmdb_env = lmdb.open(test_lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    # transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        #plt.imshow(image)
        #plt.show()

        transformed_image = transformer.preprocess('data', image)
        transformed_image = transformed_image.astype(np.uint8)

        net.blobs['data'].data[...] = transformed_image
        #plt.imshow(transformed_image)
        #plt.imshow(np.transpose(transformed_image, (1, 2, 0)))
        #plt.show()

        #image = sm.imresize(image, [227, 227])
        #image = np.transpose(image, (2, 0, 1))
        out = net.forward()
        plabel = int(out['prob'][0].argmax(axis=0))

        count = count + 1
        iscorrect = label == plabel
        correct = correct + (1 if iscorrect else 0)
        matrix[(label, plabel)] += 1
        labels_set.update([label, plabel])


        if not iscorrect:
            print("\rError: key=%s, expected %i but predicted %i" \
                    % (key, label, plabel))

        sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
        sys.stdout.flush()

    print(str(correct) + " out of " + str(count) + " were classified correctly")

    print ""
    print "Confusion matrix:"
    print "(r , p) | count"
    for l in labels_set:
        for pl in labels_set:
            print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])