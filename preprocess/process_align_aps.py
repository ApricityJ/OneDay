import pandas as pd
from util import exporter, loader

aps = loader.to_concat_df('APS_QZ')
tranflw = loader.to_concat_df('MBANK_TRNFLW_QZ')

# 步骤0
aps['APSDTRDAT_TM'] = pd.to_datetime(aps['APSDTRDAT_TM'], format='%Y%m%d%H%M%S')
tranflw['TFT_DTE_TIME'] = pd.to_datetime(tranflw['TFT_DTE_TIME'], format='%Y%m%d%H%M%S')
aps['APSDTRAMT'] = aps['APSDTRAMT'].round(3)
tranflw['TFT_TRNAMT'] = tranflw['TFT_TRNAMT'].round(3)


def merge_and_fill(aps_df, tranflw_df):
    merged = aps_df.merge(tranflw_df,
                          left_on=['APSDPRDNO', 'APSDTRAMT', 'APSDTRDAT_TM', 'SRC'],
                          right_on=['TFT_CSTACC', 'TFT_TRNAMT', 'TFT_DTE_TIME', 'SRC'],
                          how='left')
    mask = merged['APSDCPTPRDNO'].isna() & merged['TFT_CRPACC'].notna()
    print(f'补充交易对手 -- {len(merged[mask])}')
    merged.loc[mask, 'APSDCPTPRDNO'] = merged.loc[mask, 'TFT_CRPACC']
    return merged[aps_df.columns]


# 步骤1
aps_updated = merge_and_fill(aps, tranflw)

# 步骤2
tranflw['TFT_TRNAMT'] = tranflw['TFT_TRNAMT'] * -1
aps_updated_2 = merge_and_fill(aps_updated, tranflw)

# 步骤3
tranflw_unmatched = tranflw[~tranflw['TFT_CSTNO'].isin(aps_updated_2['APSDPRDNO'])]
tranflw_unmatched['TFT_DTE_TIME'] = tranflw_unmatched['TFT_DTE_TIME'] + pd.Timedelta(seconds=1)
aps_final = merge_and_fill(aps_updated_2, tranflw_unmatched)

aps_final = aps_final[['APSDPRDNO', 'APSDTRDAT_TM', 'APSDTRCOD', 'APSDTRAMT', 'APSDCPTPRDNO']]
aps_final.columns = ['CRD_SRC', 'TRN_DT', 'TRN_COD', 'TRN_AMT', 'CRD_TGT']
aps_final['TRN_DT'] = aps_final['TRN_DT'].dt.strftime('%Y%m%d%H%M%S')

exporter.export_df_to_preprocess('APS', aps_final)
