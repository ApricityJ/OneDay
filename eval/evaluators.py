from pandas import DataFrame
from sklearn.metrics import fbeta_score
from data import loader


class F2Evaluator(object):

    def __init__(self, df_pred: DataFrame):
        self.df_true = loader.to_df_true()
        self.df_pred = df_pred

    def eval(self):
        # 长度判断
        if len(self.df_pred) != len(self.df_true):
            return False, 0

        # 拼接
        df_merged = self.df_pred.merge(self.df_pred, left_on=['CUST_NO'], right_on=['CUST_NO'], how='inner')

        # 计算
        return True, round(fbeta_score(df_merged['FLAG'], df_merged['PRED'], beta=2), 3)


if __name__ == '__main__':
    F2Evaluator().eval()
