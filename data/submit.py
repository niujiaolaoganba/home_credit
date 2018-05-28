import sys
import pandas as pd

# sys.arg[1]

<<<<<<< HEAD
t = pd.read_csv('./output/refit_test_2018-05-23-09-49_stacking_Learner@EnsembleLearner_Id@33_[auc0.758650].csv')
sub = pd.read_csv('./submission/first_20180520.csv')
sub['TARGET'] = t
sub.to_csv('./submission/hyper_20180520_stacking.csv',index = False)
=======
t = pd.read_csv('./output/refit_test_2018-05-26-16-46_single_Learner@clf_lgb_tree_Id@8_[auc0.765725].csv')
sub = pd.read_csv('./submission/third_20180526.csv')
sub['TARGET'] = t
sub.to_csv('./submission/third_hyper_20180526_lgb.csv',index = False)
>>>>>>> tmp
