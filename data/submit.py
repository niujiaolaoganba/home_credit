import sys
import pandas as pd
import glob
import numpy as np 

t = pd.read_csv('./output/refit_test_2018-06-09-23-37_single_Learner@clf_lgb_tree_Id@3_[auc0.770980].csv')
refer = pd.read_csv('./submission/submission_kernel02.csv')
sub = pd.read_csv('./submission/first_20180520.csv')
sub['TARGET'] = t + refer.TARGET
sub.to_csv('./submission/hyper_20180610_lgb.csv',index = False)

t_fnames = glob.iglob("./output/refit*.7[6,7]*.csv")
stacking_t_fnames = pd.concat([pd.read_csv(f) for f in t_fnames], axis=1)
sub['TARGET'] = stacking_t_fnames.apply(np.mean, axis = 1)
sub.to_csv('./submission/bagging_20180610.csv',index = False)

