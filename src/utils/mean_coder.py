import pandas as pd
class CatTranf():
    def __init__(self, kind='onehot'):
        self.kind = kind

    def fit(self, train, category_feature):
        df = pd.concat([train,pd.get_dummies(train.category_feature)],axis = 1)
        return df.drop(category_feature, axis = 1)

    def transfrom(self, ):