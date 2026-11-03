# Distance Comparison Operations Are Not Silver Bullets in Vector Similarity Search: A Benchmark Study on Their Merits and Limits

## Abstract

Distance Comparison Operations (DCOs), which determine whether the distance between a data vector and a query is within a given threshold, are a critical performance bottleneck in vector similarity search. 
Recent DCO methods that avoid full-dimensional distance computations promise significant speedups, but their readiness for production vector database systems remains an open question.
To address this, we conduct a comprehensive benchmark of 8 DCO algorithms across 10 datasets (with up to 100M vectors and 12,288 dimensions) and diverse hardware configurations (CPUs with/without SIMD, and GPUs).
Our study reveals that these methods are not silver bullets: their efficiency is highly sensitive to data dimensionality, degrades under out-of-distribution queries, and is unstable across hardware.
Yet, our evaluation also demonstrates often-overlooked merits: they can accelerate index construction and data updates.
Despite these benefits, their unstable performance, which can be slower than a full-dimensional scan, leads us to conclude that no current DCO method is ready for production deployment. 

## Fullpaper

The full version of our paper can be obtained in `fullpaper.pdf`

## Dataset

1. Download and preprocess the datasets. The tested datasets are available at https://drive.google.com/drive/folders/1IDlNI7EO-xK70NaDizrUD7pQqokOAkwx?usp=drive_link or https://pan.baidu.com/s/59NArzpIVuE3X4NsdK5u6Jg.

2. Preprocess the datasets to obtain DCO operator. 

   * ADSampling: `./data/adsampling/batch.sh`
   * DADE: `./data/dade/batch.sh`
   * DDCres: `./data/ddc/batch_res.sh`
   * DDCpca:  `./data/ddc/batch_pca.sh`, after the index is established
   * DDCopq: `./data/ddc/batch_opq.sh`, after the index is established
   * Establish the corresponding index using the preprocessed data

   Note: The IP measurement method can be achieved by normalizing the datasets (`./src/script/hnsw_normalize.sh`), except for FDScanning and DDCres, which can be used dirctly.

3. Index the datasets. 

```sh
./Construction/script/hnsw_batch.sh
./Construction/script/ivf_batch.sh
```

## Environment

OS: Ubuntu 24.04.2 LTS

GCC/G++: >= 13.3.0

CMake: >= 3.28.3

Python: >= 3.10.16

Eigen: >= 3.4.0

## Compile and run our algorithms

1. Execute the following commands to compile relevant code.

```sh
mkdir build
cd build
cmake ..
make
```

2. Test the queries of the datasets. 

```sh
./src/batch/hnsw_search.sh
./src/batch/ivf_seach.sh
./src/batch/hnsw_search_no_simd.sh
./src/batch/ivf_seach_no_simd.sh
./src/batch/OOD_search.sh
./src/batch/IP_search.sh
./GPU/src/batch.sh
```

3. Test the construction time using DCO operator.

* Change `operator` in `./Construction/script/hnsw_batch.sh`

4. Introduction to our DCOs

| DCO     | Experimental number | Dataset preprocessing |
| ----------- | -------- |-----------------------|
| FDScanning  | 0        | data_base.fvecs       |
| PDScanning  | 1        | data_base.fvecs       |
| PDScanning+ | 2        | Pdata_base.fvecs      |
| ADSampling  | 3        | Odata_base.fvecs      |
| DADE        | 4        | Pdata_base.fvecs      |
| DDCres      | 5        | Ldata_base.fvecs      |
| DDCpca      | 6        | Ldata_base.fvecs      |
| DDCopq      | 7        | Qdata_base.fvecs      |



