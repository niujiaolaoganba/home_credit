import sys
import pandas as pd

# sys.arg[1]

t = pd.read_csv('./output/refit_test_2018-05-26-16-46_single_Learner@clf_lgb_tree_Id@8_[auc0.765725].csv')
sub = pd.read_csv('./submission/third_20180526.csv')
sub['TARGET'] = t
sub.to_csv('./submission/third_hyper_20180526_lgb.csv',index = False)
