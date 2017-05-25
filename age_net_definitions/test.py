import caffe
import matplotlib.pyplot as plt
import numpy as np
import lmdb
from collections import defaultdict

GENDER_LIST = ['M', 'F']

caffe.set_mode_gpu()
caffe.set_device(0)

fold_number = 0
model_file = './deploy.prototxt'
trained = './model_fold_'+str(fold_number)+'/caffenet_train_iter_50000.caffemodel'
txt_file = '../Folds/train_val_txt_files_per_fold/test_fold_is_'+str(fold_number)+'/age_test.txt'
aligned = '../../models/age-gender/aligned/'

fname_list = []
label_list = []

with open(txt_file, 'r') as txt:
    for line in txt:
        fname_list.append(line.split(' ')[0])
        label_list.append(int(line.split(' ')[1]))

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

if __name__ == "__main__":
    count = 0.
    correct = 0.
    one_off = 0.
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    for idx in range(len(fname_list)):
        image = caffe.io.load_image(aligned + fname_list[idx])
        # plt.imshow(image)
        # plt.show()
        label = label_list[idx]

        prediction = net.predict([image])
        plabel = prediction[0].argmax()

        count = count + 1
        iscorrect = label == plabel
        correct = correct + (1 if iscorrect else 0)

        if plabel == label or plabel == label-1 or plabel == label+1:
            one_off += 1

        matrix[(label, plabel)] += 1
        labels_set.update([label, plabel])

        if count % 100 == 0:
            print('Exact -> %i/%i, acc: %f' % (correct, count, correct/count))
            print('1-off -> %i/%i, acc: %f' % (one_off, count, one_off/count))

    print("Fold: %i" % (fold_number))
    print("Exact Accuracy: %f%%" % (correct/count))
    print("1-off Accuracy: %f%%" % (one_off / count))
    print(str(correct) + " out of " + str(count) + " were classified correctly")

    print ""
    print "Confusion matrix:"
    print "(r , p) | count"
    for l in labels_set:
        for pl in labels_set:
            print "(%i , %i) | %i" % (l, pl, matrix[(l, pl)])
