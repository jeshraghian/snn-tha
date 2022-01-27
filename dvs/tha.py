# exp relaxation implementation of THA based on Eq (4)

def thr_annealing(config, network):
    alpha_thr1 = config['alpha_thr1']
    alpha_thr2 = config['alpha_thr2']
    alpha_thr3 = config['alpha_thr3']

    thr_final1 = config['thr_final1']
    thr_final2 = config['thr_final2']
    thr_final3 = config['thr_final3']

    network.lif1.threshold += (thr_final1 - network.lif1.threshold) * alpha_thr1
    network.lif2.threshold += (thr_final2 - network.lif2.threshold) * alpha_thr2
    network.lif3.threshold += (thr_final3 - network.lif3.threshold) * alpha_thr3

    return