TOOLS=/home/k/caffe/build/tools
DATA=/home/k/PycharmProjects/AgeGenderDeepLearning-master/Folds/lmdb/Test_fold_is_2/gender_train_lmdb
OUT=/home/k/PycharmProjects/AgeGenderDeepLearning-master/Folds/lmdb/Test_fold_is_2

$TOOLS/compute_image_mean.bin $DATA $OUT/mean.binaryproto

