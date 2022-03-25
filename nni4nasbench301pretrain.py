from pathlib import Path

from nni.experiment import Experiment

search_space = {
    "batch_size": {"_type":"choice", "_value": [1024, 2048, 4096]},
    "num_iterations": {"_type":"choice", "_value": [1000, 5000, 10000]},
    "learning_rate": {"_type":"choice", "_value": [1.0, 0.1, 0.01, 0.001, 0.0001]},
    "weight_decay": {"_type":"choice", "_value": [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
    "graph_encoder_dropout": {"_type":"choice", "_value": [0,0, 0.1, 0.2, 0.5, 0.75]},
    "dropout": {"_type":"choice", "_value": [0.05, 0.15, 0.25, 0.5, 0.75]},
    #"model_type": {"_type":"choice", "_value": ['barlow_twins', 'barlow_twins_l2l', 'grace', 'graphcl', 'bgrl', 'mvgrl', 'mae']},
    #"model_type": {"_type":"choice", "_value": ['barlow_twins', 'graphcl', 'mvgrl', 'mae']},
}

experiment = Experiment('local')
experiment.config.author_name = 'Jing'
experiment.config.experiment_name = 'nni4nasbench301pretrain'
experiment.config.trial_concurrency = 8
experiment.config.max_exec_duration = '100h'
experiment.config.max_trial_number = 10000
experiment.config.search_space = search_space
experiment.config.trial_command = 'bash nni4nasbench301pretrain.sh'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'Anneal' #'GridSearch' #'SMAC' #'Evolution'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.training_service.use_active_gpu = True

experiment.run(54322)

