import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import lightgbm as lgb
import shap

#=============================================
# データ読み込む
#=============================================
# CSVファイル(sample_car_data.csv)をShift-JISエンコードで読み込みます。
df = pd.read_csv("L用_sample_car_data_数値.csv", encoding='shift-JIS')
# データの先頭3行を表示して、読み込み結果を確認します
df.head(3)

#=============================================
# カテゴリ列のデータをカテゴリ型に変換する
#=============================================
# 数値として扱いたい列（連続値など）
numeric_cols = ["family", "age","children", "income"]

# IDや目的変数は変換対象から除外
exclude_cols = ["customer_id", "manufacturer"]

# 変換対象のカテゴリ列を抽出
categorical_cols = [
    col for col in df.columns
    if col not in exclude_cols + numeric_cols
]

# カテゴリ型にする
df[categorical_cols] = df[categorical_cols].astype("category")


#=============================================
# 説明変数（特徴量）を設定
#=============================================
# 顧客IDと目的変数の「manufacturer」は除外
X_df = df.drop(['customer_id', 'manufacturer'], axis=1)

#=============================================
# 目的変数を設定
#=============================================
y_df = df['manufacturer']

# クラス数（カテゴリーの種類）を確認
classes = np.unique(y_df)
print("クラス:", classes)

#============================================================
# 説明変数(X_df)と目的変数(y_df)を訓練データとテストデータに分割
#============================================================
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=0)

print("訓練データ数:", len(X_train))
print("テストデータ数:", len(X_test))


#=============================================
# LightGBM：目的に応じて設定
#=============================================
if len(classes) == 2:
    objective = 'binary'
    metric = 'binary_error'
else:
    objective = 'multiclass'
    metric = 'multi_error'

params = {
    'objective': objective,
    'metric': metric,
    'verbose': -1,
}
if objective == 'multiclass':
    params['num_class'] = len(classes)



#=============================================
# LightGBM用データセットを作成（特徴量名を指定）
#=============================================
lgb_train = lgb.Dataset(
    X_train.values,
    label=y_train.values,
    feature_name=X_df.columns.tolist()
)
# モデル学習（50回のブースティング）
lgb_model = lgb.train(params, lgb_train, num_boost_round=50)

#=============================================
# テストデータで予測
#=============================================
y_pred_lgb_prob = lgb_model.predict(X_test.values)
# クラス予測（確率→ラベル）
if objective == 'binary':
    y_pred_lgb = (y_pred_lgb_prob > 0.5).astype(int)
else:
    y_pred_lgb = np.argmax(y_pred_lgb_prob, axis=1)
# ==============================
# SHAP値計算
# ==============================
explainer = shap.Explainer(lgb_model, X_df)
shap_values = explainer(X_df)
print("SHAP values shape:", shap_values.values.shape)



#=============================================
# 精度評価
#=============================================
print("【LightGBM】     Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("【LightGBM】     F1 Score:", f1_score(y_test, y_pred_lgb, average='weighted'))

#=============================================
# 混同行列の表示
#=============================================
fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lgb, ax=ax)
ax.set_title("LightGBM Confusion Matrix")

plt.tight_layout()
plt.show()


#=============================================
# LightGBMの特徴量重要度 (特徴量名表示)
#=============================================
ax = lgb.plot_importance(
    lgb_model,
    max_num_features=10
)
plt.title("LightGBM Feature Importance")

plt.savefig("比較用_LightGBM_Feature.png", dpi=300)


# ==============================
# LightGBMパラメータ設定
# ==============================
params = {
    'objective': 'multiclass',
    'metric': 'multi_error',
    'num_class': len(classes),
    'verbose': -1
}

# ==============================
# 学習（100回で学習）
# ==============================
lgb_model = lgb.train(params, lgb_train, num_boost_round=100)

# ==============================
#  全データに対する予測
# ==============================
y_all_pred_prob = lgb_model.predict(X_df.values)  # 予測確率
y_all_pred = np.argmax(y_all_pred_prob, axis=1)  # 予測クラスラベル


# ==============================
# 上位3候補の抽出
# ==============================
top3_classes = np.argsort(y_all_pred_prob, axis=1)[:, ::-1][:, :3]
top3_probs = np.sort(y_all_pred_prob, axis=1)[:, ::-1][:, :3]


# ==============================
# 元データに結果を追加
# ==============================
df["predicted_manufacturer"] = y_all_pred

for i in range(y_all_pred_prob.shape[1]):
    df[f"prob_class_{i}"] = y_all_pred_prob[:, i]

# 上位3クラスと確率の列を追加
for i in range(3):
    df[f"top{i+1}_class"] = top3_classes[:, i]
    df[f"top{i+1}_prob"] = top3_probs[:, i]
    
print(df[[
    "manufacturer",               # 正解ラベル
    "predicted_manufacturer",     # モデルの予測
    "top1_class", "top1_prob",    # 1位予測と確率
    "top2_class", "top2_prob",    # 2位予測と確率
    "top3_class", "top3_prob"     # 3位予測と確率
]].head())

# =======================================
# 　正解割合（Top-3 Accuracy）
# =======================================
top3_accuracy = np.mean([
    y_true in top3 for y_true, top3 in zip(df["manufacturer"], top3_classes)
])
print(f"Top-3 Accuracy: {top3_accuracy:.3f}")


# =======================================
# 正解が topN の何番目
# =======================================
def top_rank(row):
    true_class = row["manufacturer"]
    top_classes = [row["top1_class"], row["top2_class"], row["top3_class"]]
    return top_classes.index(true_class) + 1 if true_class in top_classes else None

df["correct_rank"] = df.apply(top_rank, axis=1)
print(df["correct_rank"].value_counts())



# =======================================
#正解の順位別の件数・割合を可視化
# =======================================
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語対応

rank_counts = df["correct_rank"].value_counts().sort_index()
rank_percent = (rank_counts / rank_counts.sum()) * 100

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(rank_percent.index.astype(str), rank_percent.values, color='skyblue')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1,
            f"{height:.1f}%", ha='center', va='bottom')

ax.set_title("正解クラスの予測順位（Top-k Accuracy）", fontsize=14)
ax.set_xlabel("予測順位")
ax.set_ylabel("割合（%）")
plt.ylim(0, 105)
plt.tight_layout()
plt.show()


# ==============================
# 上位3寄与特徴量抽出
# ==============================
results_top3 = []

for i in range(len(df)):
    pred_class = df.loc[i, "predicted_manufacturer"]
    shap_vals = shap_values.values[i, :].flatten()
    features = list(shap_values.feature_names)

    # 絶対値で上位3特徴量を選ぶ
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_features = [features[j] for j in top_idx]
    top_values = [shap_vals[j] for j in top_idx]

    tmp = {
        "customer_id": df.loc[i, "customer_id"],
        "predicted_class": pred_class,
        "top1_feature": top_features[0],
        "top1_value": top_values[0],
        "top2_feature": top_features[1],
        "top2_value": top_values[1],
        "top3_feature": top_features[2],
        "top3_value": top_values[2]
    }
    results_top3.append(tmp)

top3_df = pd.DataFrame(results_top3)
top3_df.head(3)


# ==============================
# マージ用に列名を合わせる
# ==============================
df = df.rename(columns={"predicted_manufacturer": "predicted_class"})
# ==============================
# マージ
# ==============================
df_merged = pd.merge(df, top3_df, on=["customer_id", "predicted_class"], how="left")


# ==============================
# 列名リネーム　		
# ==============================
rename_dict = {
    "top1_feature": "上位1特徴量",
    "top1_value": "上位1寄与度",
    "top2_feature": "上位2特徴量",
    "top2_value": "上位2寄与度",
    "top3_feature": "上位3特徴量",
    "top3_value": "上位3寄与度",
    "prob_class_0": "メーカー0の確率",
    "prob_class_1": "メーカー1の確率",
    "prob_class_2": "メーカー2の確率",
    "prob_class_3": "メーカー3の確率",
    "prob_class_4": "メーカー4の確率",
    "prob_class_5": "メーカー5の確率",
    "prob_class_6": "メーカー6の確率",
    "top1_class": "トップ1候補メーカー",
    "top1_prob": "トップ1確率",
    "top2_class": "トップ2候補メーカー",
    "top2_prob": "トップ2確率",
    "top3_class": "トップ3候補メーカー",
    "top3_prob": "トップ3確率"
}
df_merged = df_merged.rename(columns=rename_dict)


# ==============================
# CSV出力
# ==============================
df_merged.to_csv("predictions_with_top3.csv", index=False, encoding="utf-8-sig")

#=============================================
#　Topデレクトリとリストファイルを開く
#=============================================
import os
import os.path as osp
current_dpath = os.getcwd()

os.startfile(current_dpath)
os.startfile(current_dpath + "\\predictions_with_top3.csv")

print(" 完了")



# メーカー毎の特徴量重要度ランクを出す
# ==============================
# SHAP値と特徴量名をDataFrameに変換
# ==============================
shap_df = pd.DataFrame(shap_values.values, columns=X_df.columns)
shap_df["predicted_class"] = df["predicted_class"].values  # 予測されたクラスを結合



# クラスごとの SHAP値の平均（特徴量別）を求める
summary_list = []

for cls in sorted(df["predicted_class"].unique()):
    shap_mean = shap_df[shap_df["predicted_class"] == cls].drop(columns="predicted_class").mean().abs()
    shap_mean_sorted = shap_mean.sort_values(ascending=False)

    for feature, value in shap_mean_sorted.items():
        summary_list.append({
            "メーカー": cls,
            "特徴量": feature,
            "平均SHAP値": round(value, 5)
        })

# DataFrameとしてまとめる
shap_summary_df = pd.DataFrame(summary_list)

# メーカーごとに上位5個を表示（必要なら変更可能）
topN = 5
display_df = shap_summary_df.groupby("メーカー").head(topN)


# CSV保存
display_df.to_csv("メーカー別_SHAP_特徴ランキング.csv", index=False, encoding="utf-8-sig")

os.startfile(current_dpath + "\\メーカー別_SHAP_特徴ランキング.csv")
