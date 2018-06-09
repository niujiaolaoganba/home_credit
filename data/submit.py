import sys
import pandas as pd
import glob
import numpy as np 

t = pd.read_csv('./output/refit_test_2018-06-09-12-58_single_Learner@clf_lgb_tree_Id@3_[auc0.767251].csv')
sub = pd.read_csv('./submission/first_20180520.csv')
sub['TARGET'] = t
sub.to_csv('./submission/hyper_20180609_lgb.csv',index = False)

t_fnames = glob.iglob("./output/refit*07*.csv")
stacking_t_fnames = pd.concat([pd.read_csv(f) for f in t_fnames], axis=1)
sub['TARGET'] = stacking_t_fnames.apply(np.mean, axis = 1)
sub.to_csv('./submission/bagging_20180609.csv',index = False)

