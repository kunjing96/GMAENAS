MKL_THREADING_LAYER=GNU MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES=$(($RANDOM % 4)) python finetune.py --space nasbench101 --save train_nasbench101_nni --param nni
