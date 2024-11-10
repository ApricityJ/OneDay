# 计算客户的交易金额特征
group_by_customer = train_data.groupby("CUST_NO")["TFT_TRNAMT"].agg([
    'mean', 'max', 'min', 'sum', 'count'
]).reset_index()

group_by_customer.columns = ["CUST_NO", "mean_amt", "max_amt", "min_amt", "total_amt", "transaction_count"]


# 提取客户金融资产的均值和波动情况
cust_fa_stats = train_data.groupby("CUST_NO")[["DAY_FA_BAL", "MAVER_FA_BAL"]].agg(['mean', 'std']).reset_index()


# 检测交易金额中的离群值
train_data['is_large_transaction'] = train_data['TFT_TRNAMT'] > train_data['TFT_TRNAMT'].quantile(0.95)


# 合并所有的特征表
combined_data = pd.merge(cust_fa_features, aps_features, on="CUST_NO", how="left")

# 如果还有其他表，继续合并更多特征
# ...

# 最终得到的combined_data就是包含所有特征的训练数据


# 特征选择：可以使用一些工具（如 SelectKBest 或 Lasso) 进行特征选择，去掉冗余特征。

# 使用LightGBM进行特征选择
from sklearn.feature_selection import SelectFromModel

model = lgb.LGBMClassifier(boosting_type='dart', **best_params)
model.fit(X_train, y_train)

# 基于模型的特征选择
selector = SelectFromModel(model, threshold=0.01)
X_selected = X_train.iloc[:, selector.get_support()]










# 构造衍生特征：交易金额的移动平均值和最大值
df['Transaction_Amount_MA7'] = df['Transaction_Amount'].rolling(window=7).mean()  # 7天移动平均
X_train['Transaction_Amount_MA7'] = X_train['Transaction_Amount'].rolling(window=7).mean().fillna(method='bfill')
df['Transaction_Amount_Max7'] = df['Transaction_Amount'].rolling(window=7).max()  # 7天内最大值

# 构造交易频率特征
df['Transaction_Frequency'] = df.groupby('CUST_NO')['Transaction_Amount'].transform('count') / df['Transaction_Days']

# 构造乘积特征和比值特征
df['Amount_Frequency_Product'] = df['Transaction_Amount'] * df['Transaction_Frequency']
df['Amount_Frequency_Ratio'] = df['Transaction_Amount'] / df['Transaction_Frequency']

# 构造对数特征
df['Log_Transaction_Amount'] = np.log1p(df['Transaction_Amount'])  # 使用log(1+x)避免log(0)问题

# 交易金额的等频分桶
df['Transaction_Amount_Binned'] = pd.qcut(df['Transaction_Amount'], q=4, labels=False)  # 分为4个等频区间

# 自定义分桶
bins = [0, 100, 500, 1000, 5000, np.inf]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
df['Transaction_Amount_Custom_Binned'] = pd.cut(df['Transaction_Amount'], bins=bins, labels=labels)

# 构造客户资产总额的多项式特征
df['Customer_Asset_Squared'] = df['Customer_Asset'] ** 2
df['Customer_Asset_Cubed'] = df['Customer_Asset'] ** 3


企业中收_T2=企业中收_T1.groupby(['客户ID']).agg({'近6月平均值':['last'],'近12月平均值':['last']\
		,'本月中收人民币':['std','mean',count_zero],'数据日期':['count']})
        企业中收_T2.reset_index(inplace=True)
        企业中收_T2.columns = ['客户ID','近3月中收平均值','近12月中收平均值','中收_std','中收_mean','无中收期数','中收期数']

企业中收_T1['近6月平均值'] = 企业中收_T1.groupby(['客户ID']).本月中收人民币.apply(
    lambda x: x.rolling(window=6, min_periods=6).mean().round(1))

