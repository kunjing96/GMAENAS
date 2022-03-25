MKL_THREADING_LAYER=GNU MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES=$(($RANDOM % 4)) python pretrain.py --space nasbench101 --save pretrain_nasbench101_nni --finetune_fixed_encoder --param nni
