for pt in {graphcl,mvgrl,mae}
  do
    for n in {423,}
      do
        if [ ! -d "pretrain_task/$pt/$n/15712" ] ; then
            mkdir -p pretrain_task/$pt/$n/15712
        fi
        python pretrain.py --space nasbench101 --param local:pretrain_nasbench101.json --seed 15712 --save pretrain_task/$pt/$n/15712/ --finetune_fixed_encoder --finetune_train_samples $n --finetune_batch_size $(((n-1) / 10 + 1)) --model_type $pt 2>&1 |tee pretrain_task/$pt/$n/15712/log.txt 
      done
  done
