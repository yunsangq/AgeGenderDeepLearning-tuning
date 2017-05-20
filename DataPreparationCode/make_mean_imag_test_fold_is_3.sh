TOOLS=/home/k/caffe/build/tools
DATA=/home/k/PycharmProjects/AgeGenderDeepLearning-master/Folds/lmdb/Test_fold_is_3/gender_train_lmdb
OUT=/home/k/PycharmProjects/AgeGenderDeepLearning-master/Folds/lmdb/Test_fold_is_3

$TOOLS/compute_image_mean.bin $DATA $OUT/mean.binaryproto

