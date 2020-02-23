Last login: Mon Sep 16 21:12:00 on console
jeremiass-mbp:~ jeremiasknoblauch$ ssh strtfk@orac.csc.warwick.ac.uk
Enter passphrase for key '/Users/jeremiasknoblauch/.ssh/id_rsa': 
Last login: Tue Apr 30 13:17:30 2019 from cpe-24-163-73-30.nc.res.rr.com
------------------------------------------------------------------------------
Welcome to Orac. Documentation on using this system is available on the SCRTP
Wiki at https://wiki.csc.warwick.ac.uk/twiki/bin/view/Main/OracUserGuide.
Please review the documentation in full before attempting to use this system.
------------------------------------------------------------------------------
Please DO NOT run compute jobs on Orac's login node. All jobs should be run
through Slurm. Any jobs run outside Slurm will be terminated without notice to
the user and may result in your account being suspended.
------------------------------------------------------------------------------
Notifications about maintenance, downtime and changes affecting Orac are
provided through the scrtp-cluster-user mailing list. Please ensure you're a
member (http://listserv.warwick.ac.uk/mailman/listinfo/scrtp-cluster-user).
------------------------------------------------------------------------------

Important: Please note that Slurm Workload Manager was upgraded on Tues 30th July.
Please report any issues via https://bugzilla.csc.warwick.ac.uk/

For more information please see:

http://mailman1.csv.warwick.ac.uk/mailman/private/scrtp-cluster-user/2019-July/000078.html
------------------------------------------------------------------------------
[strtfk@orac:login1 ~]$ pwd
/home/stats/strtfk
[strtfk@orac:login1 ~]$ ls
GVI
[strtfk@orac:login1 ~]$ cd GVI
[strtfk@orac:login1 GVI]$ ls
BayesianNN                     DSDGP
BayesianProbit                 kin8mn_0.99_0.001_500_A.slurm
boston_0.5_0.001_500_A.slurm   test_py_packages.py
boston_0.99_0.001_500_A.slurm  test_script.slurm
datasets.py
[strtfk@orac:login1 GVI]$ cd JASA
-bash: cd: JASA: No such file or directory
[strtfk@orac:login1 GVI]$ mkdir JASA
[strtfk@orac:login1 GVI]$ cd JASA
[strtfk@orac:login1 JASA]$ ls
[strtfk@orac:login1 JASA]$ cd ..
[strtfk@orac:login1 GVI]$ ls
BayesianNN                     DSDGP
BayesianProbit                 JASA
boston_0.5_0.001_500_A.slurm   kin8mn_0.99_0.001_500_A.slurm
boston_0.99_0.001_500_A.slurm  test_py_packages.py
datasets.py                    test_script.slurm
[strtfk@orac:login1 GVI]$ cd BayesianNN
[strtfk@orac:login1 BayesianNN]$ ls
AB_boston
AB_boston.sh
AB_concrete
AB_concrete.sh
AB_counterfactual_boston
AB_counterfactual_boston.sh
AB_counterfactual_concrete
AB_counterfactual_concrete.sh
AB_counterfactual_yacht
AB_counterfactual_yacht.sh
AB_yacht
AB_yacht.sh
all_10_0
all_10_0_A-approx
all_10_0_A-approx.sh
all_10_0.sh
all_1_1
all_1_1_A-approx
all_1_1_A-approx.sh
all_1_1.sh
all_AB
all_AB.sh
all_loss_b_g
all_loss_b_g.sh
all_vanilla_A-approx
all_vanilla_A-approx.sh
all_vanilla_sh
all_vanilla.sh
all_vanilla_shrink
all_vanilla_shrink.sh
black_box_alphavi.py
black_box_alphavi.pyc
boston
boston_10_0
boston_10_0.sh
boston_1_1
boston_1_1.sh
boston_loss_b_g
boston_loss_b_g_counterfactual
boston_loss_b_g_counterfactual_finer.sh
boston_loss_b_g_counterfactual.sh
boston_loss_b_g_finer.sh
boston_loss_b_g.sh
boston.sh
boston_shrink2.sh
boston_shrink.sh
boston_slurm
boston_slurm_1_1
boston_slurm_1_1.sh
concrete
concrete_10_0
concrete_10_0.sh
concrete_1_1
concrete_1_1.sh
concrete_loss_b_g
concrete_loss_b_g_counterfactual
concrete_loss_b_g_counterfactual_finer.sh
concrete_loss_b_g_counterfactual.sh
concrete_loss_b_g_finer.sh
concrete_loss_b_g.sh
concrete.sh
concrete_shrink2.sh
concrete_shrink.sh
data
energy
energy_10_0
energy_10_0.sh
energy_1_1
energy_1_1.sh
energy_loss_b_g_counterfactual
energy_loss_b_g_counterfactual.sh
energy.sh
energy_shrink2.sh
energy_shrink.sh
energy_slurm
kin8mn
kin8mn_10_0
kin8mn_10_0.sh
kin8mn_1_1
kin8mn_1_1.sh
kin8mn_loss_b_g_counterfactual
kin8mn_loss_b_g_counterfactual.sh
kin8mn.sh
kin8mn_shrink2.sh
kin8mn_shrink.sh
kin8mn_slurm
kin8mn_slurm_1_1
kin8mn_slurm_1_1.sh
naval
naval_10_0
naval_10_0.sh
naval_1_1
naval_1_1.sh
naval_AR-approx.sh
naval_loss_b_g_counterfactual
naval_loss_b_g_counterfactual.sh
naval.sh
naval_shrink2.sh
naval_shrink.sh
parallel-172748.log
power
power_10_0
power_10_0.sh
power_1_1
power_1_1.sh
power_loss_b_g_counterfactual
power_loss_b_g_counterfactual.sh
power.sh
power_shrink2.sh
power_shrink.sh
power_slurm
prototype_.slurm
prototype.slurm
__pycache__
results
script.sh
settings.txt
slurm-172748.out
test_alpha_AB_counterfactual.py
test_alpha_AB.py
test_alpha_bg_counterfactual.py
test_alpha_bg.py
test_alpha_old.py
test_alpha_on_server.py
test_alpha.py
test_py_packages.py
wine
wine_10_0
wine_10_0.sh
wine_1_1
wine_1_1.sh
wine_loss_b_g_counterfactual
wine_loss_b_g_counterfactual.sh
wine.sh
wine_shrink2.sh
wine_shrink.sh
wine_slurm
yacht
yacht_10_0
yacht_10_0.sh
yacht_1_1
yacht_1_1.sh
yacht_loss_b_g
yacht_loss_b_g_counterfactual
yacht_loss_b_g_counterfactual_finer.sh
yacht_loss_b_g_counterfactual.sh
yacht_loss_b_g_finer.sh
yacht_loss_b_g.sh
yacht.sh
yacht_shrink2.sh
yacht_shrink.sh
[strtfk@orac:login1 BayesianNN]$ vi AB_yacht.sh
[strtfk@orac:login1 BayesianNN]$ vi settings.txt
[strtfk@orac:login1 BayesianNN]$ vi AB_yacht.sh
[strtfk@orac:login1 BayesianNN]$ vi AB_yacht/settings.txt
[strtfk@orac:login1 BayesianNN]$ vi prototype.slurm
[strtfk@orac:login1 BayesianNN]$ vi test_alpha.py

                f.write(repr(neg_test_ll) + '\n')
        with open(file_string + "_test_error.txt", 'a') as f:
                f.write(repr(test_error) + '\n')
        with open(file_string + "_test_time.txt", 'a') as f:
                f.write(repr(running_time) + '\n')

if __name__ == '__main__':
    dataset = str(sys.argv[1])
    alpha = float(sys.argv[2])
    learning_rate = float(sys.argv[3])
    epochs = int(sys.argv[4])
    Dtype = str(sys.argv[5])
    m_prior = float(sys.argv[6])
    v_prior = float(sys.argv[7])
    split_num = int(sys.argv[8])
    subfolder = str(sys.argv[9])
    #beta_D = float(sys.argv[10])

#    dataset = "boston"
#    alpha = 0.75 #0.75
#    learning_rate = 0.001
#    epochs = 500
#    Dtype = "AB-approx"
#    m_prior = 0.0
#    v_prior = 1.0
#    split_num = 0
#    subfolder = ""

#    losstype = str(sys.argv[7])
#    prior_update = bool(sys.argv[8])
#    alpha_update = bool(sys.argv[9])
#    beta = float(sys.argv[10])
#    gamma = float(sys.argv[11])

#    split_num = 2 #computes all splits
#    dataset = "boston"
#    alpha = 0.5
#    learning_rate = 0.001 #0.0001
    #epochs = 20 #00
#    Dtype = "AR-approx"
    losstype = None #"b" #"beta", "gamma"
    prior_update = False
                                                                                   142,1         87%

