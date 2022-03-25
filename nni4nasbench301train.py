from pathlib import Path

from nni.experiment import Experiment

search_space = {
    #"finetune_train_samples": {"_type":"choice", "_value": [100]},
    #"finetune_batch_size": {"_type":"choice", "_value": [100, 50, 20, 10, 5]},
    "finetune_epochs": {"_type":"choice", "_value": [25, 50, 75, 100, 125, 150]},
    #"finetune_criterion":{"_type":"choice","_value": ["hinge", "mse", "bpr", "bce"]},
    #"finetune_exp_weighted":{"_type":"choice","_value": [False, True]},
    #"finetune_fixed_encoder":{"_type":"choice","_value": [False, True]},
    #"finetune_graph_encoder_learning_rate":{"_type":"choice","_value": [1.0, 0.1, 0.01, 0.001, 0.0001]},
    "finetune_learning_rate":{"_type":"choice","_value": [1.0, 0.1, 0.01, 0.001, 0.0001]},
    "finetune_weight_decay": {"_type":"choice", "_value": [0, 1e-5, 1e-4, 1e-3, 1e-2]},
    #"finetune_graph_encoder_dropout": {"_type":"choice", "_value": [0.0, 0.1, 0.2, 0.5, 0.8]},
    "finetune_dropout": {"_type":"choice", "_value": [0.0, 0.1, 0.2, 0.5, 0.8]},
    "finetune_opt_reserve_p": {"_type":"choice", "_value": [0.1, 0.2, 0.5, 0.8, 1.0]},
    "finetune_opt_mode": {"_type":"choice", "_value": ["D", "F"]},
    "finetune_optimizer": {"_type":"choice", "_value": ["SGD", "AdamW"]},
    "n_layers": {"_type":"choice", "_value": [3, 6, 9, 12, 16, 20]},
    "d_model": {"_type":"choice", "_value": [32, 64, 128]},
    #"projector_layers": {"_type":"choice", "_value": ["32", "32-32-32", "64", "64-64-64", "128", "128-128-128"]},
}

experiment = Experiment('local')
experiment.config.author_name = 'Jing'
experiment.config.experiment_name = 'nni4nasbench301train'
experiment.config.trial_concurrency = 4
experiment.config.max_exec_duration = '10h'
experiment.config.max_trial_number = 10000
experiment.config.search_space = search_space
#experiment.config.trial_command = 'MKL_THREADING_LAYER=GNU MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES=2 python finetune.py --space nasbench101 --seed 449 --save finetune_nasbench101_nni --param nni'
experiment.config.trial_command = 'bash nni4nasbench301train.sh'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'Anneal' #'GridSearch' #'SMAC' #'Evolution'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.training_service.use_active_gpu = True

experiment.run(54322)

