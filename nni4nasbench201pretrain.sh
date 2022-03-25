MKL_THREADING_LAYER=GNU MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES=$(($RANDOM % 4)) python pretrain.py --space nasbench201 --save pretrain_nasbench201_nni --finetune_fixed_encoder --param nni
