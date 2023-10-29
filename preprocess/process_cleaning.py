from pandas import DataFrame

from data import loader, exporter


# 数据清洗
# 对金额四舍五入保留3位小数
def filter_amt_to_round_3():
    amt_dict = {
        'MBANK_TRNFLW_QZ': ['TFT_TRNAMT'],
        'EBANK_CSTLOG_QZ': ['TRNAMT'],
        'APS_QZ': ['APSDTRAMT'],
        'CUST_FA_SUM_QZ': ['DAY_FA_BAL', 'MAVER_FA_BAL', 'SAVER_FA_BAL', 'YAVER_FA_BAL',
                           'DAY_AUM_BAL', 'MAVER_AUM_BAL', 'SAVER_AUM_BAL', 'YAVER_AUM_BAL',
                           'TOT_IVST_BAL', 'MAVER_TOT_IVST_BAL', 'SAVER_TOT_IVST_BAL', 'YAVER_TOT_IVST_BAL'],
        'DP_CUST_SUM_QZ': ['DPSA_BAL', 'MAVER_DPSA_BAL', 'SAVER_DPSA_BAL', 'YAVER_DPSA_BAL',
                           'TD_BAL', 'MAVER_TD_BAL', 'SAVER_TD_BAL', 'YAVER_TD_BAL']
    }

    for k, v in amt_dict.items():
        train, test = loader.to_df_train_test(k)
        train[v] = train[v].round(3)
        exporter.export_df_to_train(k, train)

        test[v] = test[v].round(3)
        exporter.export_df_to_test(f'{k}_A', test)


# 处理重复行
def drop_duplicates():
    ds = loader.to_ds_train_test()

    for k, v in ds.items():
        print(k)
        train, test = v
        train = DataFrame(train).drop_duplicates(keep='first')
        exporter.export_df_to_train(k, train)

        test = DataFrame(test).drop_duplicates(keep='first')
        exporter.export_df_to_test(f'{k}_A', test)


if __name__ == '__main__':
    filter_amt_to_round_3()
    # drop_duplicates()
