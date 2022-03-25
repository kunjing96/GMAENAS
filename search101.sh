for sa in {r,ae,rl,bo}
  do
    for s in {1..10}
      do
        if [ ! -d "search101/$sa" ] ; then
            mkdir -p search101/$sa
        fi
        python search.py --search_space nasbench101 --search_algo $sa --seed $s --dataset cifar10 --outdir search101/$sa/ --encoder_model_state_dict pretrain101/best_model_15712.pt 2>&1 |tee search101/$sa/log.txt
      done
  done
