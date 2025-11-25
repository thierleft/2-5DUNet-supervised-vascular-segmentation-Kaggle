#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=24:00:00
#$ -S /bin/bash
#$ -j y
#$ -N download_FULLDATASET
#$ -o  /home/ID/storage/STORAGESPACE_NAME/LOGS_FOLDER/


hostname
date

cd $HOME/storage/STORAGESPACE_NAME/

# Download and unzip

mkdir -p training_data


#Kidney 6
mkdir -p training_data/Kidney_6
mkdir -p training_data/Kidney_6/LADAF_2022-13_kidney_bottom_dense_label
mkdir -p training_data/Kidney_6/LADAF_2022-13_kidney_bottom_dense_raw

wget -O LADAF_2022-13_kidney_bottom_dense_label.zip "https://www.dropbox.com/scl/fo/rxc2lxobcc2ta61u3kn5g/ANPck7LG6AI1at_4jEKGtTI?rlkey=e6bslfvxxbjf8kl5g68xaux24&st=yelcma4z&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2022-13_kidney_bottom_dense_label.zip -d training_data/Kidney_6/LADAF_2022-13_kidney_bottom_dense_label
rm LADAF_2022-13_kidney_bottom_dense_label.zip

wget -O LADAF_2022-13_kidney_bottom_dense_raw.zip "https://www.dropbox.com/scl/fo/if2nhntn3kg8846g19m03/APHLifqBJTAZvqMsRDAFmVw?rlkey=re3fdpp8bh1da2czqolq8dof7&st=zdf71onh&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2022-13_kidney_bottom_dense_raw.zip -d training_data/Kidney_6/LADAF_2022-13_kidney_bottom_dense_raw
rm LADAF_2022-13_kidney_bottom_dense_raw.zip

# Kidney 5
mkdir -p training_data/Kidney_5
mkdir -p training_data/Kidney_5/LADAF_2021-17_left_half_kidney_dense_label
mkdir -p training_data/Kidney_5/LADAF_2021-17_left_half_kidney_dense_raw

wget -O LADAF_2021-17_left_half_kidney_dense_label.zip "https://www.dropbox.com/scl/fo/dq97jalq8ydm8di04x73s/APOR4RKJCrmIMjAuWUR0s5A?rlkey=dhcet7vuzfob9q48o2ptcz81f&st=o1tu0ydt&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2021-17_left_half_kidney_dense_label.zip -d training_data/Kidney_5/LADAF_2021-17_left_half_kidney_dense_label
rm LADAF_2021-17_left_half_kidney_dense_label.zip

wget -O LADAF_2021-17_left_half_kidney_dense_raw.zip "https://www.dropbox.com/scl/fo/azengm2dzkdfoji5gh570/AONrxQ9FwbgIrg_f6vDOxkc?rlkey=96dk1sh1zz06ugsc38ko708vv&st=gz7cemnh&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2021-17_left_half_kidney_dense_raw.zip -d training_data/Kidney_5/LADAF_2021-17_left_half_kidney_dense_raw
rm LADAF_2021-17_left_half_kidney_dense_raw.zip

# Kidney 4
mkdir -p training_data/Kidney_4
mkdir -p training_data/Kidney_4/LADAF_2022-13_kidney_top_subset_raw

wget -O LADAF_2022-13_kidney_top_subset_raw.zip "https://www.dropbox.com/scl/fo/xbwwb0ihtjwwvanw1qjb4/AAtFFNN1tZpVA8gZMI8Tvd0?rlkey=yp4x03ssxum6bqjxev35yi91p&st=r805oy6n&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2022-13_kidney_top_subset_raw.zip -d training_data/Kidney_4/LADAF_2022-13_kidney_top_subset_raw
rm LADAF_2022-13_kidney_top_subset_raw.zip

# Kidney 3
mkdir -p training_data/Kidney_3
mkdir -p training_data/Kidney_3/LADAF_2020-27_kidney_dense_subset_label
mkdir -p training_data/Kidney_3/LADAF_2020-27_kidney_dense_subset_raw
mkdir -p training_data/Kidney_3/LADAF_2020-27_kidney_whole_sparse_label
mkdir -p training_data/Kidney_3/LADAF_2020-27_kidney_whole_sparse_raw

wget -O LADAF_2020-27_kidney_dense_subset_label.zip "https://www.dropbox.com/scl/fo/exd1a76c9okmap2tq2sq3/ANo_2laeMo6_BTNDLDShtgc?rlkey=gamxo4hbrtfbxva58ovkhswuw&st=lvfa61io&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2020-27_kidney_dense_subset_label.zip -d training_data/Kidney_3/LADAF_2020-27_kidney_dense_subset_label
rm LADAF_2020-27_kidney_dense_subset_label.zip

wget -O LADAF_2020-27_kidney_dense_subset_raw.zip "https://www.dropbox.com/scl/fo/wech2i44u490uhu7xiqi7/AFdaZcCaVhM-jIs_qsj6HNw?rlkey=p65ad8yukfpwfz5klgk0woorl&st=yjteopmo&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2020-27_kidney_dense_subset_raw.zip -d training_data/Kidney_3/LADAF_2020-27_kidney_dense_subset_raw
rm LADAF_2020-27_kidney_dense_subset_raw.zip

wget -O LADAF_2020-27_kidney_whole_sparse_label.zip "https://www.dropbox.com/scl/fo/36m5svjig2ugryjxt8b6j/h?rlkey=2lybc3caxzrl0oshbz0k5ylo8&st=ntrvrcf7&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2020-27_kidney_whole_sparse_label.zip -d training_data/Kidney_3/LADAF_2020-27_kidney_whole_sparse_label
rm LADAF_2020-27_kidney_whole_sparse_label.zip

wget -O LADAF_2020-27_kidney_whole_sparse_raw.zip "https://www.dropbox.com/scl/fo/3tpp5v3wiskukndwu0qi2/AGAYH8wABDdrts78lQJ8lBE?rlkey=zky9j6dtyxb05b7ldxcpzek1k&st=ncli7opa&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2020-27_kidney_whole_sparse_raw.zip -d training_data/Kidney_3/LADAF_2020-27_kidney_whole_sparse_raw
rm LADAF_2020-27_kidney_whole_sparse_raw.zip

# Kidney 2
mkdir -p training_data/Kidney_2
mkdir -p training_data/Kidney_2/S-20-28_kidney_sparse_label
mkdir -p training_data/Kidney_2/S-20-28_kidney_sparse_raw

wget -O S-20-28_kidney_sparse_label.zip "https://www.dropbox.com/scl/fo/ogob4r3fv2fmqgd58o5v5/ANlfngrwju1dMTIjF6HNUIc?rlkey=ictvfp37awc6e0hhnw1e6ypxa&st=j6jiur25&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j S-20-28_kidney_sparse_label.zip -d training_data/Kidney_2/S-20-28_kidney_sparse_label
rm S-20-28_kidney_sparse_label.zip

wget -O S-20-28_kidney_sparse_raw.zip "https://www.dropbox.com/scl/fo/way7zhef0eshbtv75s2zo/AN-kVWZfDsEZq-1CAlDV_B8?rlkey=b76gjwf3wxjh65epsts5u49hq&st=nedg7tbq&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j S-20-28_kidney_sparse_raw.zip -d training_data/Kidney_2/S-20-28_kidney_sparse_raw
rm S-20-28_kidney_sparse_raw.zip

# Kidney 1
mkdir -p training_data/Kidney_1
mkdir -p training_data/Kidney_1/LADAF_2021-17_right_VOI__dense_label
mkdir -p training_data/Kidney_1/LADAF_2021-17_right_VOI_raw
mkdir -p training_data/Kidney_1/LADAF_2021-17_right_whole_kidney_dense_label
mkdir -p training_data/Kidney_1/LADAF_2021-17_right_whole_kidney_raw

wget -O LADAF_2021-17_right_VOI__dense_label.zip "https://www.dropbox.com/scl/fo/isbinop3p4smy2vxduqy9/APYV7jUm8udP22RGFGq3Ov8?rlkey=pxffkipmhsh0z9lb7hkflvvah&st=t5q2lcve&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2021-17_right_VOI__dense_label.zip -d training_data/Kidney_1/LADAF_2021-17_right_VOI__dense_label
rm LADAF_2021-17_right_VOI__dense_label.zip

wget -O LADAF_2021-17_right_VOI_raw.zip "https://www.dropbox.com/scl/fo/o2kqqg6ja8lo2ux0dqs98/AGbxoV-xQl-hGBLjG7I36eI?rlkey=7wc5x8bf58sm8uh9ywswujkva&st=4olekp2j&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2021-17_right_VOI_raw.zip -d training_data/Kidney_1/LADAF_2021-17_right_VOI_raw
rm LADAF_2021-17_right_VOI_raw.zip

wget -O LADAF_2021-17_right_whole_kidney_dense_label.zip "https://www.dropbox.com/scl/fo/t9e0r153vjcmqcnvxvrfb/AIblRzXKhpODZVyfxYHdxV0?rlkey=x7k8achw0g2w7ojdtdydff2al&st=cs7erugk&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2021-17_right_whole_kidney_dense_label.zip -d training_data/Kidney_1/LADAF_2021-17_right_whole_kidney_dense_label
rm LADAF_2021-17_right_whole_kidney_dense_label.zip

wget -O LADAF_2021-17_right_whole_kidney_raw.zip "https://www.dropbox.com/scl/fo/ths8gwj1728zyqsq0no4j/APAtNZ_meuW-Wb3DPKRHYfA?rlkey=b42ndc62x4oirrr0hf41viqte&st=9m0v6p7y&dl=1"
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o -j LADAF_2021-17_right_whole_kidney_raw.zip -d training_data/Kidney_1/LADAF_2021-17_right_whole_kidney_raw
rm LADAF_2021-17_right_whole_kidney_raw.zip

date

