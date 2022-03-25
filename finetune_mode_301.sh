# end-to-end fine-tuning
#for n in {100,200,500,1000,10000}
for n in {200,500,1000,10000}
  do
    for s in {1..10}
      do
        if [ ! -d "finetune_mode_301/end2end_finetune/$n/$s" ] ; then
            mkdir -p finetune_mode_301/end2end_finetune/$n/$s
        fi
        python finetune.py --space nasbench301 --param local:end2end_finetune_nasbench301.json --seed $s --save finetune_mode_301/end2end_finetune/$n/$s/ --model_state_dict ../../../model.pt --finetune_train_samples $n --finetune_batch_size $(((n-1) / 10 + 1)) 2>&1 |tee finetune_mode_301/end2end_finetune/$n/$s/log.txt 
      done
  done

# train from scratch
#for n in {100,}
#  do
#    for s in {1..10}
#      do
#        if [ ! -d "finetune_mode_301/train/$n/$s" ] ; then
#            mkdir -p finetune_mode_301/train/$n/$s
#        fi
#        python finetune.py --space nasbench301 --param local:train_nasbench301.json --seed $s --save finetune_mode_301/train/$n/$s/ --finetune_train_samples $n --finetune_batch_size $(((n-1) / 10 + 1)) 2>&1 |tee finetune_mode_301/train/$n/$s/log.txt 
#    done
#  done

# partial fine-tuning
#for n in {100,}
#  do
#    for s in {1..10}
#      do
#        if [ ! -d "finetune_mode_301/partial_finetune/$n/$s" ] ; then
#            mkdir -p finetune_mode_301/partial_finetune/$n/$s
#        fi
#        python finetune.py --space nasbench301 --param local:partial_finetune_nasbench301.json --seed $s --save finetune_mode_301/partial_finetune/$n/$s/ --finetune_fixed_encoder --model_state_dict ../../../model.pt --finetune_train_samples $n --finetune_batch_size $(((n-1) / 10 + 1)) 2>&1 |tee finetune_mode_301/partial_finetune/$n/$s/log.txt 
#      done
#  done
