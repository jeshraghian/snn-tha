import torch 

config = {
        'exp_name' : 'dvs_tha',
        'num_trials' : 5,  # crank it up if Optuna
        'num_epochs' : 600,
        'binarize' : True,
        'data_dir' : "/home/dvs2",
        'batch_size' : 8,
        'seed' : 0,
        'num_workers' : 0,

        # final run sweeps
        'save_csv' : True,
        'save_model' : True,
        'early_stopping': True,
        'patience': 100,

        # final params [usually optuna]
        'grad_clip' : True,
        'weight_clip' : False,
        'batch_norm' : False,
        'dropout1' : 0.43,
        'beta' : 0.9297,
        'lr' : 1.765e-3,
        'slope': 0.24,

        # threshold annealing: note, threshold1 is added to thr_final1 in run.py
        'threshold1' : 10.4,
        'alpha_thr1' : 0.00333,
        'thr_final1' : 1.7565,
        
        'threshold2' : 16.62,
        'alpha_thr2' : 0.0061,
        'thr_final2' : 2.457,

        'threshold3' : 6.81,
        'alpha_thr3' : 0.173,
        'thr_final3' : 9.655,
        
        # fixed params
        'num_steps' : 100,
        'correct_rate': 0.8,
        'incorrect_rate' : 0.2,
        'betas' : (0.9, 0.999),
        't_0' : 735,
        'eta_min' : 0,
        'df_lr' : True, # return learning rate. Useful for scheduling


    }

def optim_func(net, config):
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], betas=config['betas'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_0'], eta_min=config['eta_min'], last_epoch=-1)
    return optimizer, scheduler