#for sa in {r,ae,rl,bo}
for sa in {rl,bo}
  do
    for s in {1..10}
      do
        if [ ! -d "search301/$sa" ] ; then
            mkdir -p search301/$sa
        fi
        python search.py --search_space nasbench301 --search_algo $sa --seed $s --dataset cifar10 --outdir search301/$sa/ --encoder_model_state_dict pretrain301/best_model_36734.pt --N 100 2>&1 |tee search301/$sa/log.txt
      done
  done
