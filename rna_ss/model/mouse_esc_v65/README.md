
 Training using fixed data


mkdir model

CUDA_VISIBLE_DEVICES=0 python train.py  --fold 0 --config tmp_config/config_2.yml

CUDA_VISIBLE_DEVICES=0 python train.py  --fold 1 --config tmp_config/config_2.yml

CUDA_VISIBLE_DEVICES=0 python train.py  --fold 2 --config tmp_config/config_2.yml

CUDA_VISIBLE_DEVICES=0 python train.py  --fold 3 --config tmp_config/config_2.yml

CUDA_VISIBLE_DEVICES=0 python train.py  --fold 4 --config tmp_config/config_2.yml


mkdir prediction

CUDA_VISIBLE_DEVICES=2 python make_prediction_training_data_cv.py --config tmp_config/config_2.yml

gzip prediction/training_data_cv.csv


CUDA_VISIBLE_DEVICES=2 python make_prediction_rna_strand_db.py --config tmp_config/config_2.yml


CUDA_VISIBLE_DEVICES=0 python make_prediction_family_trio.py --config tmp_config/config_2.yml
