import torch 

config = {
        'exp_name' : 'shd_tha',
        'num_trials' : 5,  
        'num_epochs' : 5,
        'binarize' : True,
        'data_dir' : "C:/Users/jeshr/Dropbox/repos/snntorch/dataset/shd", #"/home/shd",
        'batch_size' : 32,
        'seed' : 0,
        'num_workers' : 0,

        # final run sweeps
        'save_csv' : True,
        'save_model' : True,
        'early_stopping': True,
        'patience': 100,

        # final params
        'grad_clip' : True,
        'weight_clip' : True,
        'batch_norm' : True,
        'dropout2' : 0.0176,
        'dropout1' : 0.186,
        'beta' : 0.950,
        'lr' : 6.54e-4,
        'slope': 0.257,


        # threshold annealing. note: thr_final = threshold + thr_final
        'threshold1' : 13.504,
        'alpha_thr1' : 2.78e-5,
        'thr_final1' : 31.767,
        
        'threshold2' : 11.20,
        'alpha_thr2' : 1.36e-5,
        'thr_final2' : 39.92,

        # fixed params
        'num_steps' : 100,
        'correct_rate': 0.8,
        'incorrect_rate' : 0.2,
        'betas1' : 0.9,
        'betas2' : 0.999,
        't_0' : 2604,
        'eta_min' : 0,
        'df_lr' : True, # return learning rate. Useful for scheduling


    }

def optim_func(net, config):
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], betas=(config['betas1'], config['betas2']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_0'], eta_min=config['eta_min'], last_epoch=-1)
    return optimizer, scheduler