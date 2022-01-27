import torch

config = {
        'exp_name' : 'fmnist_tha',
        'num_trials' : 5, 
        'num_epochs' : 600,
        'binarize' : True,
        'data_dir' : "~/data/fmnist",
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
        'dropout1' : 0.648,
        'beta' : 0.868,
        'lr' : 8.4e-4,
        'slope': 0.1557,
        'momentum' : 0.855,
        

        # threshold annealing
        'threshold1' : 6.9,
        'alpha_thr1' : 0.0368,
        'thr_final1' : 7.1456,
        
        'threshold2' : 10.25,
        'alpha_thr2' : 0.29687,
        'thr_final2' : 12.826,

        'threshold3' : 17.95,
        'alpha_thr3' : 0.1048,
        'thr_final3' : 9.936668,
        
        # fixed params
        'num_steps' : 100,
        'correct_rate': 0.8,
        'incorrect_rate' : 0.2,
        't_0' : 4688,
        'eta_min' : 0,
        'df_lr' : True, # return learning rate. Useful for scheduling



    }

def optim_func(net, config):
    optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"], momentum=config['momentum'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_0'], eta_min=config['eta_min'], last_epoch=-1)
    return optimizer, scheduler
