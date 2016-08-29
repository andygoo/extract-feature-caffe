
cd $CAFFE_ROOT

find /root/dl-data/zhangli/images -type f -exec echo {} \; | sort  > examples/_temp/temp.txt

sed "s/$/ 0/" examples/_temp/temp.txt > examples/_temp/file_list.txt

rm -fr examples/_temp/featureA

batch_size=$(wc -l examples/_temp/file_list.txt | cut -f1 -d' ')

echo $batch_size

build/tools/extract_features models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel examples/_temp/imagenet_val.prototxt  fc7  examples/_temp/featureA  $batch_size lmdb GPU 0
