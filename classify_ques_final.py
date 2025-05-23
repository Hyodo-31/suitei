import sys

#パスの設定
sys.path.append("C:\\Users\\cs22074\\anaconda3\\Lib\\site-packages")

import io
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample
import numpy as np
import json
#import mysql.connector

# 標準出力のエンコーディングをUTF-8に設定
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class Classify:
    def __init__(self, df, testdf, results_csv, metrics_json, cross_val_predictions_csv):
        self.df = df                # 教師データ
        self.testdf = testdf        # 実際の未知のデータ
        self.results_csv = results_csv
        self.metrics_json = metrics_json
        self.cross_val_predictions_csv = cross_val_predictions_csv
        self.Understand1_count = 0
        self.Understand2_count = 0
        self.Understand3_count = 0
        self.Understand4_count = 0
        self.mean_accuracy = 0
        self.precision_y = 0
        self.recall_y = 0
        self.f1score_y = 0
        self.precision_n = 0
        self.recall_n = 0
        self.f1score_n = 0
        self.classifydf = pd.DataFrame()
        self.Featureimportances = []
        self.result = {}
        self.model = None
        self.features = []

    def binary(self):
        # 'Understand' が 2 または 4 の行のみ抽出
        self.classify_df = self.df[self.df['Understand'].isin([2,4])]
        return self.classify_df

    def countUnderstand(self, df):
        self.Understand1_count = df[df['Understand'] == 1].shape[0]
        self.Understand2_count = df[df['Understand'] == 2].shape[0]
        self.Understand3_count = df[df['Understand'] == 3].shape[0]
        self.Understand4_count = df[df['Understand'] == 4].shape[0]

    def makingclassifydf(self):
        # Understand=2 と Understand=4 の数を揃える
        samplerow = min(self.Understand2_count, self.Understand4_count)
        if samplerow == 0:
            print("エラー: Understand=2 または Understand=4 のサンプル数が0です。")
            sys.exit(1)
        
        sampledf_2 = self.classify_df[self.classify_df['Understand'] == 2].sample(n=samplerow, random_state=0)
        sampledf_4 = self.classify_df[self.classify_df['Understand'] == 4].sample(n=samplerow, random_state=0)

        self.classifydf = pd.concat([sampledf_2, sampledf_4])
        print(f"クラスバランス後のデータフレームの行数: {self.classifydf.shape[0]}")

    def train_model(self):
        # データの準備
        tmpdf = self.classifydf     # 教師データ（モデル学習用）
        tmpdf = tmpdf.drop(["UID","WID","Understand","attempt"], axis=1)
        self.features = tmpdf.columns.tolist()
        self.featuresdict = {}
        objective = 'Understand'

        for i in self.features:
            self.featuresdict[i] = 0

        X_data = self.classifydf[self.features]             # モデル作成用の教師データ（特徴量）
        Y_data = self.classifydf[objective]                 # モデル作成用の教師データ（目的変数）
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_data, Y_data)
        print("モデル学習完了")

    def predict_new_data(self):
        """
        新しいデータを予測する関数
        :param new_data: 新規データ(dataframe)
        :return : 予測結果
        """
        print("教師データの個数: " + str(self.classifydf.shape[0]))
        print("予測データの個数: " + str(self.testdf.shape[0]))
        if self.model is None:
            print("モデルが学習されていません。")
            sys.exit(1)
        
        new_data_preprocessed = self.testdf[self.features]
        # 予測の実行
        predictions = self.model.predict(new_data_preprocessed)

        # UID, WID, 予測結果を格納
        results = pd.DataFrame({
            "UID": self.testdf["UID"],
            "WID": self.testdf["WID"],
            "Understand": predictions,
            "attempt": self.testdf["attempt"]
        })
        # 結果をcsvファイルに保存
        results.to_csv(self.results_csv, index=False)

        # 結果の出力完了メッセージ
        print(f"予測結果が '{self.results_csv}' に保存されました。")

    def cross_validate_model(self, n_splits=10):
        """
        10分割交差検定を行い、各クラスの評価指標を計算する関数
        :param n_splits: 分割数（デフォルトは10）
        :return: 評価指標の平均値
        """
        print(f"\n=== {n_splits}分割交差検定を実行中 ===")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 評価指標を格納するリスト
        precision_class0_list = []
        recall_class0_list = []
        f1_class0_list = []
        
        precision_class1_list = []
        recall_class1_list = []
        f1_class1_list = []
        # 予測結果を格納するリスト
        cross_val_predictions = []
        fold = 1
        for train_index, test_index in skf.split(self.classifydf[self.features], self.classifydf['Understand']):
            print(f"  フォールド {fold}: トレーニングデータ {len(train_index)} 行, テストデータ {len(test_index)} 行")
            X_train, X_test = self.classifydf.iloc[train_index][self.features], self.classifydf.iloc[test_index][self.features]
            y_train, y_test = self.classifydf.iloc[train_index]['Understand'], self.classifydf.iloc[test_index]['Understand']
            
            # モデルの訓練
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 予測
            y_pred = model.predict(X_test)
            
            # 各クラスの評価指標を計算
            precision = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall = recall_score(y_test, y_pred, average=None, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
            
            # クラス0の指標（Understand=2と仮定）
            precision_class0_list.append(precision[0])
            recall_class0_list.append(recall[0])
            f1_class0_list.append(f1[0])
            
            # クラス1の指標（Understand=4と仮定）
            precision_class1_list.append(precision[1])
            recall_class1_list.append(recall[1])
            f1_class1_list.append(f1[1])
            
            print(f"    クラス0 - 適合率: {precision[0]:.4f}, 再現率: {recall[0]:.4f}, F値: {f1[0]:.4f}")
            print(f"    クラス1 - 適合率: {precision[1]:.4f}, 再現率: {recall[1]:.4f}, F値: {f1[1]:.4f}")
            fold += 1
            # 予測結果を収集
            test_data = self.classifydf.iloc[test_index][["UID", "WID", "attempt"]].copy()
            test_data["Understand"] = y_pred
            cross_val_predictions.append(test_data)

        # 全フォールドの予測結果を結合
        all_predictions = pd.concat(cross_val_predictions, ignore_index=True)
        # UID順、次にWID順に並べ替え
        all_predictions_sorted = all_predictions.sort_values(by=["UID", "WID"]).reset_index(drop=True)
        # CSVに保存
        all_predictions_sorted.to_csv(self.cross_val_predictions_csv, index=False)
        print(f"交差検定の予測結果が '{self.cross_val_predictions_csv}' に保存されました。")
        
        # 全フォールドの平均を計算
        final_avg_precision_class0 = np.mean(precision_class0_list)
        final_avg_recall_class0 = np.mean(recall_class0_list)
        final_avg_f1_class0 = np.mean(f1_class0_list)
        
        final_avg_precision_class1 = np.mean(precision_class1_list)
        final_avg_recall_class1 = np.mean(recall_class1_list)
        final_avg_f1_class1 = np.mean(f1_class1_list)
        
        # 平均評価指標を表示
        print(f"\n=== {n_splits}分割交差検定の平均評価指標 ===")
        print("クラス0（ラベル0）の平均:")
        print(f"  適合率: {final_avg_precision_class0:.4f}")
        print(f"  再現率: {final_avg_recall_class0:.4f}")
        print(f"  F値: {final_avg_f1_class0:.4f}")
        
        print("クラス1（ラベル1）の平均:")
        print(f"  適合率: {final_avg_precision_class1:.4f}")
        print(f"  再現率: {final_avg_recall_class1:.4f}")
        print(f"  F値: {final_avg_f1_class1:.4f}")
        
        # 評価指標を辞書にまとめる
        metrics = {
            'class0': {
                'precision': final_avg_precision_class0,
                'recall': final_avg_recall_class0,
                'f1_score': final_avg_f1_class0
            },
            'class1': {
                'precision': final_avg_precision_class1,
                'recall': final_avg_recall_class1,
                'f1_score': final_avg_f1_class1
            }
        }
        
        # JSONファイルに保存
        with open(self.metrics_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        print(f"交差検定の評価指標が '{self.metrics_json}' に保存されました。")
        
        return metrics


def main():
    # コマンドライン引数からファイルパスを取得
    # Usage: python script.py <input_csv> <input_csv_test> <results_csv> <metrics_json>
    if len(sys.argv) != 6:
        print("引数の個数は " + str(len(sys.argv)) + " 個です。")
        print("Usage: python script.py <input_csv> <input_csv_test> <results_csv> <metrics_json>")
        sys.exit(1)

    inputfilename = sys.argv[1]
    testfilename = sys.argv[2]
    results_csv = sys.argv[3]
    metrics_json = sys.argv[4]
    cross_val_predictions_csv = sys.argv[5]

    # データの読み込み
    df = pd.read_csv(inputfilename)
    testdf = pd.read_csv(testfilename)
    
    # Classify インスタンスの作成
    datamarge = Classify(df, testdf, results_csv, metrics_json, cross_val_predictions_csv)

    # データを分割
    return_df = datamarge.binary()      # 全データの中から迷い有りと無しのみ抽出
    datamarge.countUnderstand(return_df)    # 迷い有りと無しの数を数える

    # バランスの取れたデータセットの作成
    datamarge.makingclassifydf()        # 迷い無しと有りが1:1のデータセットを作成
    datamarge.train_model()             # モデル学習
    datamarge.predict_new_data()        # 新規データの予測

    # 10分割交差検定の実行
    cross_val_metrics = datamarge.cross_validate_model(n_splits=10)
    
    # データベースへの挿入（必要に応じてコメントを外す）

if __name__ == '__main__':
    main()
