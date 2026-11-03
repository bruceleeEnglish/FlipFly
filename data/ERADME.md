## Prerequisites

```
faiss: 1.9.0
numpy: 2.2.3
```

## Datesets
The tested datasets are available at https://drive.google.com/drive/folders/1IDlNI7EO-xK70NaDizrUD7pQqokOAkwx?usp=drive_link or https://pan.baidu.com/s/59NArzpIVuE3X4NsdK5u6Jg.

## DCO

* ADSampling: `./data/adsampling/batch.sh`
* DADE: `./data/dade/batch.sh`
* DDCres: `./data/ddc/batch_res.sh`
* DDCpca:  `./data/ddc/batch_pca.sh`, after the index is established
* DDCopq: `./data/ddc/batch_opq.sh`, after the index is established
* Establish the corresponding index using the preprocessed data

Note: The IP measurement method can be achieved by normalizing the datasets (`./src/script/hnsw_normalize.sh`), except for FDScanning and DDCres, which can be used dirctly.



