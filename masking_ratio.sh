for mr in {0.05,0.15,0.25,0.50,0.75}
  do
    for n in {423,}
      do
        if [ ! -d "masking_ratio/$mr/$n/15712" ] ; then
            mkdir -p masking_ratio/$mr/$n/15712
        fi
        python pretrain.py --space nasbench101 --param local:pretrain_nasbench101.json --seed 15712 --save masking_ratio/$mr/$n/15712/ --finetune_fixed_encoder --finetune_train_samples $n --finetune_batch_size $(((n-1) / 10 + 1)) --dropout $mr 2>&1 |tee masking_ratio/$mr/$n/15712/log.txt
      done
  done
