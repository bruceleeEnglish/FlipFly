# new_add: pad only the query file to a power-of-two dimension and write Squery.fvecs.
# Use it only when batch.sh has already rotated the base vectors but was run without -q,
# so that the expensive SRHT pass need not be repeated.
#
# Usage:
#   python pad_query.py -d XUltra      -n query.bin
#   python pad_query.py -d OpenAI3072  -n query.bin
# cd /data/dco-benchmark/data/srht
# python pad_query.py -d XUltra     -n query.bin
# python pad_query.py -d OpenAI3072 -n query.bin

import os, struct, argparse
import numpy as np

source = '/data/dataset'


def read_vector_file(filename):
    _, ext = os.path.splitext(filename)
    if ext == '.fvecs':
        fv = np.fromfile(filename, dtype=np.float32)
        if fv.size == 0:
            return np.zeros((0, 0))
        dim = fv.view(np.int32)[0]
        fv = fv.reshape(-1, 1 + dim)
        return fv[:, 1:].copy()
    with open(filename, 'rb') as f:
        n, d = np.fromfile(f, count=2, dtype=np.int32)
        a = np.fromfile(f, count=n * d, dtype=np.float32)
    return a.reshape(n, d)


def next_power_of_two(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def to_fvecs(path, data):
    with open(path, 'wb') as fp:
        for y in data:
            y = np.asarray(y, dtype=np.float32)
            fp.write(struct.pack('I', len(y)))
            fp.write(y.tobytes())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('-n', '--name', required=True, help='query 文件名')
    args = ap.parse_args()

    qpath = os.path.join(source, args.dataset, args.name)
    if not os.path.exists(qpath):
        raise SystemExit(f'query not found: {qpath}')

    Q = read_vector_file(qpath)
    M, D_in = Q.shape
    D = next_power_of_two(D_in)
    if D != D_in:
        print(f'Padding query from D_in={D_in} to D={D}')
        Q = np.pad(Q, ((0, 0), (0, D - D_in)), mode='constant').astype(np.float32)
    else:
        print(f'D_in={D_in} already power of 2, no padding')

    out = os.path.join(source, args.dataset, 'Squery.fvecs')
    to_fvecs(out, Q)
    print(f'Wrote {out}: shape={Q.shape}')
