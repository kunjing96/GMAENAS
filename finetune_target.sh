for t in {mse,hinge,bpr}
  do
    for n in {423,846,2115,4230}
      do
        for s in {1..10}
          do
            if [ ! -d "finetune_target/$t/$n/$s" ] ; then
                mkdir -p finetune_target/$t/$n/$s
            fi
            python finetune.py --space nasbench101 --param local:partial_finetune_nasbench101.json --seed $s --save finetune_target/$t/$n/$s/ --finetune_fixed_encoder --model_state_dict ../../../model.pt --finetune_train_samples $n --finetune_batch_size $(((n-1) / 10 + 1)) --finetune_criterion $t 2>&1 |tee finetune_target/$t/$n/$s/log.txt
          done
      done
  done
