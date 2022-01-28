import torch

config = {
        'exp_name' : 'mnist_tha',
        'num_trials' : 5,
        'num_epochs' : 500,
        'binarize' : True,
        'data_dir' : "~/data/mnist",
        'batch_size' : 128,
        'seed' : 0,
        'num_workers' : 0,

        # final run sweeps
        'save_csv' : True,
        'save_model' : True,
        'early_stopping': True,
        'patience': 100,

        # final params
        'grad_clip' : False,
        'weight_clip' : False,
        'batch_norm' : True,
        'dropout1' : 0.02856,
        'beta' : 0.99,
        'lr' : 9.97e-3,
        'slope': 10.22,

        # threshold annealing. note: thr_final = threshold + thr_final
        'threshold1' : 11.666,
        'alpha_thr1' : 0.024,
        'thr_final1' : 4.317,
        
        'threshold2' : 14.105,
        'alpha_thr2' : 0.119,
        'thr_final2' : 16.29,

        'threshold3' : 0.6656,
        'alpha_thr3' : 0.0011,
        'thr_final3' : 3.496,
        
        # fixed params
        'num_steps' : 100,
        'correct_rate': 0.8,
        'incorrect_rate' : 0.2,
        'betas' : (0.9, 0.999),
        't_0' : 4688,
        'eta_min' : 0,
        'df_lr' : True, # return learning rate. Useful for scheduling



    }

def optim_func(net, config):
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], betas=config['betas'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_0'], eta_min=config['eta_min'], last_epoch=-1)
    return optimizer, scheduler
