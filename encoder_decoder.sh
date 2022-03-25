for ed in {GATConv,GCNConv}
  do
    for n in {423,}
      do
        if [ ! -d "encoder_decoder/$ed/$n/15712" ] ; then
            mkdir -p encoder_decoder/$ed/$n/15712
        fi
        python pretrain.py --space nasbench101 --param local:pretrain_nasbench101.json --seed 15712 --save encoder_decoder/$ed/$n/15712/ --finetune_fixed_encoder --finetune_train_samples $n --finetune_batch_size $(((n-1) / 10 + 1)) --base_model $ed 2>&1 |tee encoder_decoder/$ed/$n/15712/log.txt
      done
  done
