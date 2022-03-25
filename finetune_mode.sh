# train from scratch
for n in {423,846,2115,4230}
  do
    for s in {1..10}
      do
        if [ ! -d "finetune_mode/train/$n/$s" ] ; then
            mkdir -p finetune_mode/train/$n/$s
        fi
        python finetune.py --space nasbench101 --param local:train_nasbench101.json --seed $s --save finetune_mode/train/$n/$s/ --finetune_train_samples $n --finetune_batch_size $(((n-1) / 10 + 1)) 2>&1 |tee finetune_mode/train/$n/$s/log.txt 
    done
  done

# partial fine-tuning
for n in {423,846,2115,4230}
  do
    for s in {1..10}
      do
        if [ ! -d "finetune_mode/partial_finetune/$n/$s" ] ; then
            mkdir -p finetune_mode/partial_finetune/$n/$s
        fi
        python finetune.py --space nasbench101 --param local:partial_finetune_nasbench101.json --seed $s --save finetune_mode/partial_finetune/$n/$s/ --finetune_fixed_encoder --model_state_dict ../../../model.pt --finetune_train_samples $n --finetune_batch_size $(((n-1) / 10 + 1)) 2>&1 |tee finetune_mode/partial_finetune/$n/$s/log.txt 
      done
  done

# end-to-end fine-tuning
for n in {423,846,2115,4230}
  do
    for s in {1..10}
      do
        if [ ! -d "finetune_mode/end2end_finetune/$n/$s" ] ; then
            mkdir -p finetune_mode/end2end_finetune/$n/$s
        fi
        python finetune.py --space nasbench101 --param local:end2end_finetune_nasbench101.json --seed $s --save finetune_mode/end2end_finetune/$n/$s/ --model_state_dict ../../../model.pt --finetune_train_samples $n --finetune_batch_size $(((n-1) / 10 + 1)) 2>&1 |tee finetune_mode/end2end_finetune/$n/$s/log.txt 
      done
  done
