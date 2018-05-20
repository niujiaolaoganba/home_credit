import sys
import pandas as pd

# sys.arg[1]

t = pd.read_csv('./output/refit_test_2018-05-20-13-15_single_Learner@clf_xgb_tree_Id@11_[auc0.000000].csv')
sub = pd.read_csv('./submission/first_20180520.csv')
sub['TARGET'] = t
sub.to_csv('./submission/hyper_20180520_xgb.csv',index = False)
