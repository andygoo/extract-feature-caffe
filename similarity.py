
import numpy as np
import caffe
import lmdb
from caffe.proto import caffe_pb2
from scipy import spatial


# 3 steps to read form lmdb
fea_lmdb = lmdb.open('/root/caffe/examples/_temp/featureA')
lmdb_txn = fea_lmdb.begin()
lmdb_cursor = lmdb_txn.cursor()
features = []

for key, value in lmdb_cursor:
    datum = caffe_pb2.Datum()
    # Parse from serialized data
    datum.ParseFromString(value)
    data = caffe.io.datum_to_array(datum)
    features.append(data)

out = []
for f in features:
    out.append(f.flatten())

n = len(out)
similarity = np.zeros((n, n), dtype=np.double)

for i in xrange(n):
    for j in xrange(n):
        similarity[i, j] = 1 - spatial.distance.cosine(out[i], out[j])
