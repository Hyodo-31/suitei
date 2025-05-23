import csv
import random
import collections

# scikit-learnの分類器構築関係のメソッド
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.neural_network import MLPClassifier

# scikit-learnの，正解率とか精度とか出すためのメソッド
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn import model_selection
#from sklearn import cross_validation

#深層学習関連のライブラリ
import numpy as np
#import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from sklearn.model_selection import train_test_split
#from keras.utils import np_utils

# 数値計算のためのライブラリ．実質，numpy.array()というのものを使うためだけにインポートしてます．
# newArray = numpy.array(oldArray)という風に使うと，oldArrayが，numpy形式の配列に変換されて，newArrayに代入されます．
# numpy形式の配列は，[1, 2, 3] + [4, 5, 6]というような形で足し算ができて，この場合，結果は[5, 7, 9]になります．
import numpy

from operator import itemgetter

import time

for i in range(10):
    
    #print(str(i+1)+" times")
    #time.sleep(3)
    #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

    def extractSelectedParameters(row, selectedParameters):
        newRow = []
        selectedIndice = [p[0] for p in selectedParameters]
        for i in range(selectedParameters[0][0], len(row)):
            if i in selectedIndice:
                newRow.append(row[i])
        return newRow


    # 特徴量になる列名
    selectedParameters = [
        #(3 , 'example_for_DL1'),
        #(4 , 'example_for_DL2'),
        (5 , 'time'),
        (6 , 'distance'),
        (7 , 'averageSpeed'),
        (8 , 'maxSpeed'), #
        (9 , 'thinkingTime'), #
        (10 , 'answeringTime'), #
        # (11, 'totalStopTime'), #
        (12, 'maxStopTime '),
        (13, 'totalDDIntervalTime '), #
        (14, 'maxDDIntervalTime '),
        (15, 'maxDDTime'),
        # (16, 'minDDTime'), #
        #(17, 'DDCount'),
        # (18, 'groupingDDCount'),
        # (19, 'groupingCountbool'),
        (20, 'xUTurnCount'),
        (21, 'yUTurnCount'),
        #(22, 'register_move_count1'),
        #(23, 'register_move_count2'),
        #(24, 'register_move_count3'),
        #(25, 'register_move_count4'),
        #(26, 'register01count1'),
        #(27, 'register01count2'),
        #(28, 'register01count3'),
        #(29, 'register01count4'),
        #(30, 'registerDDCount'),
        #(32, 'TF'),
        #(33, '問題毎正解率'),
        #(34, '学習者毎正解率')
    ]


    #2次パラメータの読み込み
    #['uid','wid','自信度','解答日時','チェック',2次パラメータ～]
    outputfilename = 'featurevalue.csv'	#inputdata(パラメータ変換前のファイル)
    fread = open(outputfilename, 'r',encoding='UTF-8-sig')
    #fread = open('outputdata2019su.csv', 'r')
    rows = csv.reader(fread)
    # for od in rows:
    # 	print(od)

    # データベースのデータをもとに，訓練データを作る．
    labels = [] #ラベル（自信度）を入れる配列
    labels2 = []
    labels3 = []
    labels4 = []
    newlabels = []
    features = [] #特徴量を入れる配列
    features2 = []
    features3 = []
    features4 = []
    newfeatures = []
    counter = []
    newcounter = []
    avg_accuracy = []
    avg_p_no = []
    avg_r_no = []
    avg_f_no = []
    avg_p_yes = []
    avg_r_yes = []
    avg_f_yes = []

    avg_accuracy_DL = []


    # 正例，負例の数をカウントしつつ訓練データを作る
    nTrue  = 0
    nFalse = 0
    nFalse2 = 0
    nData  = 0
    nerror = 0
    number = 0

    conditions = str("before_add")
    print(conditions)

    member = [18310001,1818310015,1818310029,1818310030,1818310035,1818310036,1818310041,1818310062,1818310064,1818310067,1818310076,1818310078,1818310081,1818310088,1818310090,1818310091,1818310104,
            1818310105,1818380007,1818380011,1818380026,1818380028,1818380035,1818380047,1818380049,1818380050,1818380056,1818380078,1818380082,1818380096,1818380099,1818320012,1818320013,1818320020,1818320023,
            1818320026,1818320034,1818320043,1818320044,1818320054,1818320057,1818320061,1818320067,1818320085,1818320087,1818320088,1818360005,1818360006,1818360011,1818360015,1818360026,1818360035,1818360066,1818360071,1818320035,1818320089,#2018ku
            
            65,66,67,68,69,70710096,70711031,70810029,70810063,70810072,70810100,70811012,70811036,
            70811039,70811049,70811066,70811071,70812006,70812016,70812049,70812055,30810106,30814218,30814807,
            30814811,30814914,60810015,60810026,60810043,60810092,60710027,60710074,60610008,60811006,60811009,
            60811010,60811028,60811044,60811050,60811057,60811068,60811075,90710037,#2018su

            1518350090,1718310007,1818310009,1818310018,1818310032,1818310035,1818310041,1818310042,1818320067,1818320086,1818380016,1818380092,1818380100,1918310003,
            1918310009,1918310029,1918310050,1918310085,1918310096,1918310099,1918380003,1918380005,1918380010,1918380012,1918380019,1918380033,1918380038,1918380046,1918380047,
            1918380064,1918380087,1918380094,1918320003,1918320005,1918320009,1918320038,1918320042,1918320045,1918320066,1918320079,1918320081,1918320086,1918360003,1918360008,1918360026,1918360036,1918360042,1918360066,37,#2019ku


            30914025,30914026,30914904,60910014,60910020,60910022,60910037,60910044,60910046,60910090,60910103,
            60911041,60911043,60911060,60911065,60911069,60911072,90910001,90910004,90910033,90910034,90910038,#2019su

            102,103,104,105,106,107,108#2023miyazaki

    ]

    question = [389,391,395,357,329,316,336,363,380,387,370,397,355,352,334,315,310,385,348,309,346,373,340,393,308,343,321,349,359,361,#2018ku

                2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,#2019ku
                2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040,2041,2042,2043,2044,2045,2046,2047,2048,2049,2050,2051,2052,2053,2054,2055,2056,2057,2058,2059,#2019ku
                2060,2061,2062,2063,2064,2065,2066,2067,2068,2069,2070,2071,2072,2073,2074,2075,2076,2077,2078,2079,2080,2081,2082,2083,2084,2085,2086,2087,2088,2089,2090,2091,#2019ku
                2092,2093,2094,2095,2096,2097,2098,2099,2100,2101,2102,2103,2104,2105,2106,2107,2108,2109,2110,#2019ku

                22,68,46,191,32,54,67,184,45,89,127,129,141,143,147,160,162,176,181,186,59,60,99,61,139,76,92,58,138,161,#2019su

                36,40,344,350,335,15,64,324,337,359,98,99,104,105,397,9,190,332,10,363,5,199,333,339,394,6,32,33,34,390,
                37,79,86,361,399,13,38,39,44,200,69,71,73,74,309,25,56,58,60,63,23,21,47,51,52,0,18,19,42,44,14,130,301,308,314,26,27,29,35,205,30,88,138,
                143,380,7,17,24,319,328,2,11,12,331,367,59,129,192,333,381,338,393,4,305,306,307,326,50,72,116,318,388#2023miyazaki


    ]

    print(str(i+1)+" times")

    time.sleep(3)

    for row in rows:
        if int(row[0]) in member and int(row[1]) in question:
            if int(row[2]) in [0,1] : # row[2]には自信度が入っている．自信度が，0, 1のいずれかなら，何もせずにスルーする．
                nerror += 1
                pass
            elif int(row[2]) in [2] and int(row[4]) in [0,1]: # 自信度が2の場合のみ，迷いありとみなす． 元は[1,2,3] チェックなしのみを扱うときはrow[4]は[0]，両方は[0,1]にする．
                nTrue += 1
                labels.append(1)
                features.append(extractSelectedParameters(list(row), selectedParameters)) # row[1:]は，1行以降の要素．すなわち，特徴量．
            elif int(row[2]) in [4]: # 自信度が4の場合のみ，迷いなしとみなす．
                nFalse += 1
                labels.append(0)
                features.append(extractSelectedParameters(list(row), selectedParameters))

    print("データ数" + str(len(labels)))
    print("最初のなしの数 " + str(nFalse) + "  割合" + str((1.0*nFalse)/len(labels)))
    print("最初のありの数 " + str(nTrue)  + "  割合" + str((1.0*nTrue)/len(labels)))
    print('nerror: ' + str(nerror))

    if (int(nTrue)<=int(nFalse)):#str→int
        print(str(nTrue) + " < " + str(nFalse) + " is true.")
        for p in range(len(labels)):	#labelsとfeaturesに混ぜて入れたデータをありとなしに分ける
            if labels[p] in [1]:
                labels2.append(1)
                features2.append(features[p])
            elif labels[p] in [0]:
                labels3.append(0)
                features3.append(features[p])
                counter.append(number)
                number += 1
        print("ありの数となしの数は同じ " + str(nTrue) +"個ずつ")
        print(len(labels))
        print(len(labels2))
        print(len(labels3))

        for q in range(10):	#10回繰り返す
            #print('機械学習: ' + str(q+1) + '回目')

            #print("---counterの要素数")
            #print(len(counter))
            for x  in labels3:	#labels3に入れたデータを何度も使うので3は残しておきたい．毎回ループの初めにlabels4にうつす
                labels4.append(x)
            for y in features3:	#同様にfeatures3もfeatures4にうつす
                features4.append(y)
            for z in counter:	#同様にcounterをnewcounterにうつす
                newcounter.append(z)

            #print("---初めのnewcounterの要素数")
            #print(len(newcounter))

            for s in labels2:	#labels4とfeatures4からランダムでありの数とおなじだけなしのデータを取り出しnewlabelsとnewfeaturesに格納
                a = random.choice(newcounter)
                #print('---a')
                #print(a)
                b = labels4.pop(a)
                newlabels.append(b)
                c = features4.pop(a)
                newfeatures.append(c)
                newcounter.pop()
                #print("---popしたあとのnewcounterの要素数")
                #print(len(newcounter))

            #nFalse2 = len(newlabels) #なしのデータ数はこの時点でのnewlabelsの長さ

            for t in labels2:	#なしのデータが入っているnew～にありのデータを付け加える
                newlabels.append(t)
            for u in features2:
                newfeatures.append(u)
        

        # 後で使うメソッドが，ただの配列ではなく，numpy形式の配列を引数に取るので，変換しておく．
            labels = numpy.array(newlabels)
            features = numpy.array(newfeatures)

        #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

        
            #print("なしの数 " + str(nFalse2))
            #print("ありの数 " + str(nTrue))

        #次のループのために空にしておく
            labels4 = []
            features4 = []
            newcounter = []
            newlabels = []
            newfeatures = []

                # n交差できるように標本を分割
            n_folds = 10
            #print(n_folds)
            #skf = cross_validation.StratifiedKFold(labels, n_folds=n_folds, random_state=0) # 後で，このskfというものをfor ... inで辿っていくことで，うまく交差検定できる．
            skf = model_selection.StratifiedKFold(n_splits=n_folds).split(features,labels)

        # 正解率と，迷いなし，迷いありそれぞれの適合率，再現率，F値の総和がここに格納される．
            sum_accuracy = 0
            sum_precision_no = 0
            sum_recall_no    = 0
            sum_f_measure_no = 0
            sum_precision_yes = 0
            sum_recall_yes    = 0
            sum_f_measure_yes = 0

            sum_accuracy_DL = 0
        # 特徴量ごとの重要度の総和が，このnumpy形式の配列に格納される．
        # 特徴量の個数（特徴ベクトルの要素数）だけ，0が並んだ状態で，配列が初期化されている．
            sum_feature_importance = numpy.array([0]*len(features[0]))


        # 交差検定のために分割された学習データ，実験データを使って，実際に，学習，分類していく．
            for train_index, test_index in skf:
            # features_train: 学習データの特徴量
            # features_test : 実験データの特徴量
            # labels_train  : 学習データのラベル
            # labels_test   : 実験データのラベル
                features_train, features_test = features[train_index], features[test_index]
                labels_train,   labels_test   = labels[train_index],   labels[test_index]                
            
            # ランダムフォレストのモデル（分類器生成器）を作る
                rfc = RandomForestClassifier(random_state=0)#svm
            # 学習する．
                rfc.fit(features_train, labels_train)#svm
            # 分類する．
                labels_pred = rfc.predict(features_test)#svm

            # サポートベクターマシンを使いたい時は，上の，ランダムフォレスト関係の部分をコメントアウトして，↓の部分を使う．
                #svc = svm.SVC()
                #svc.fit(features_train, labels_train)
                #labels_pred = svc.predict(features_test)

            
            # 分類正解率を求めて，総和を保存する変数に足し込む．
                accuracy = accuracy_score(labels_test, labels_pred)
                #print(accuracy)

                sum_accuracy += accuracy
            # 適合率，再現率，f値を求めて，総和を保存する変数に足し込む．
            # sには，サンプルの個数が入っているけど，今回は使わない．
            # p, r, fは，3つとも，配列．pは，[迷いなしの場合の適合率, 迷いありの適合率]の2要素からなる配列．再現率，f値も同様．
                p, r, f, s = metrics.precision_recall_fscore_support(labels_test, labels_pred)
                sum_precision_no  += p[0]
                sum_recall_no     += r[0]
                sum_f_measure_no  += f[0]
                sum_precision_yes += p[1]
                sum_recall_yes    += r[1]
                sum_f_measure_yes += f[1]
            # 特徴量の重要度を求めて，総和を保存する配列に足し込む．
                sum_feature_importance = sum_feature_importance + numpy.array(rfc.feature_importances_)

        #10交差検定した結果（正解率，適合率，再現率，F値）を毎回配列に格納．正解率等はそれぞれの値の総和を，交差検定を行った回数(10)で割っている．
            avg_accuracy.append(sum_accuracy / n_folds)
            avg_p_no.append(sum_precision_no / n_folds)
            avg_r_no.append(sum_recall_no / n_folds)
            avg_f_no.append(sum_f_measure_no / n_folds)
            avg_p_yes.append(sum_precision_yes / n_folds)
            avg_r_yes.append(sum_recall_yes / n_folds)
            avg_f_yes.append(sum_f_measure_yes / n_folds)

        #1回ずつの正解率等を表示したかったら以下のコメントアウトをはずす
        # 正解率等を出力．それぞれの値の総和を，交差検定を行った回数で割る．
            #print('----results###')#tab
            #print('avg_accuracy(正解率)　　　　　　:'+str(sum_accuracy     /n_folds))
            #print('avg_p_no(適合率_迷いなし)    :'+str(sum_precision_no /n_folds))
            #print('avg_r_no(再現率_迷いなし)    :'+str(sum_recall_no    /n_folds))
            #print('avg_f_no(F値_迷いなし) 　　    :'+str(sum_f_measure_no /n_folds))
            #print('avg_p_yes(適合率_迷いあり)    :'+str(sum_precision_yes/n_folds))
            #print('avg_r_yes(再現率_迷いあり)   :'+str(sum_recall_yes   /n_folds))
            #print('avg_f_yes(F値_迷いあり)   　　:'+str(sum_f_measure_yes/n_folds))

            #平均値出すやつ、あってるやつ？
            result = []
            result.append([sum_accuracy/n_folds,sum_precision_no/n_folds,sum_recall_no/n_folds,sum_f_measure_no/n_folds,sum_precision_yes/n_folds,sum_recall_yes/n_folds,sum_f_measure_yes/n_folds])
            

    else :
        print(str(nTrue) + " > " + str(nFalse) + " is true.")
        for p in range(len(labels)):	#labelsとfeaturesに混ぜて入れたデータをありとなしに分ける
            if labels[p] in [0]:
                labels2.append(0)
                features2.append(features[p])
            elif labels[p] in [1]:
                labels3.append(1)
                features3.append(features[p])
                counter.append(number)
                number += 1
        print("ありの数となしの数は同じ " + str(nFalse) +"個ずつ")
        print(len(labels))
        print(len(labels2))
        print(len(labels3))

        for q in range(10):	#10回繰り返す
            #print('機械学習: ' + str(q+1) + '回目')

            #print("---counterの要素数")
            #print(len(counter))
            for x  in labels3:	#labels3に入れたデータを何度も使うので3は残しておきたい．毎回ループの初めにlabels4にうつす
                labels4.append(x)
            for y in features3:	#同様にfeatures3もfeatures4にうつす
                features4.append(y)
            for z in counter:	#同様にcounterをnewcounterにうつす
                newcounter.append(z)

            #print("---初めのnewcounterの要素数")
            #print(len(newcounter))

            for s in labels2:	#labels4とfeatures4からランダムでありの数とおなじだけなしのデータを取り出しnewlabelsとnewfeaturesに格納
                a = random.choice(newcounter)
                #print('---a')
                #print(a)
                b = labels4.pop(a)
                newlabels.append(b)
                c = features4.pop(a)
                newfeatures.append(c)
                newcounter.pop()
                #print("---popしたあとのnewcounterの要素数")
                #print(len(newcounter))

            #nFalse2 = len(newlabels) #なしのデータ数はこの時点でのnewlabelsの長さ

            for t in labels2:	#なしのデータが入っているnew～にありのデータを付け加える
                newlabels.append(t)
            for u in features2:
                newfeatures.append(u)
        

        # 後で使うメソッドが，ただの配列ではなく，numpy形式の配列を引数に取るので，変換しておく．
            labels = numpy.array(newlabels)
            features = numpy.array(newfeatures)
        #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


            #print("なしの数 " + str(nFalse2))
            #print("ありの数 " + str(nTrue))

        #次のループのために空にしておく
            labels4 = []
            features4 = []
            newcounter = []
            newlabels = []
            newfeatures = []

        # n交差できるように標本を分割
            n_folds = 10
            print(n_folds)
            #skf = cross_validation.StratifiedKFold(labels, n_folds=n_folds, random_state=0) # 後で，このskfというものをfor ... inで辿っていくことで，うまく交差検定できる．
            skf = model_selection.StratifiedKFold(n_splits=n_folds).split(features,labels)
        # 正解率と，迷いなし，迷いありそれぞれの適合率，再現率，F値の総和がここに格納される．
            sum_accuracy = 0
            sum_precision_no = 0
            sum_recall_no    = 0
            sum_f_measure_no = 0
            sum_precision_yes = 0
            sum_recall_yes    = 0
            sum_f_measure_yes = 0
        # 特徴量ごとの重要度の総和が，このnumpy形式の配列に格納される．
        # 特徴量の個数（特徴ベクトルの要素数）だけ，0が並んだ状態で，配列が初期化されている．
            sum_feature_importance = numpy.array([0]*len(features[0]))


        # 交差検定のために分割された学習データ，実験データを使って，実際に，学習，分類していく．
            for train_index, test_index in skf:
            # features_train: 学習データの特徴量
            # features_test : 実験データの特徴量
            # labels_train  : 学習データのラベル
            # labels_test   : 実験データのラベル
                features_train, features_test = features[train_index], features[test_index]
                labels_train,   labels_test   = labels[train_index],   labels[test_index]

            # ランダムフォレストのモデル（分類器生成器）を作る
                rfc = RandomForestClassifier(random_state=0)#svm
            # 学習する．
                rfc.fit(features_train, labels_train)#svm
            # 分類する．
                labels_pred = rfc.predict(features_test)#svm

            # サポートベクターマシンを使いたい時は，上の，ランダムフォレスト関係の部分をコメントアウトして，↓の部分を使う．
                #svc = svm.SVC()#svm
                #svc.fit(features_train, labels_train)#svm
                #labels_pred = svc.predict(features_test)#svm

            # 分類正解率を求めて，総和を保存する変数に足し込む．
                accuracy = accuracy_score(labels_test, labels_pred)
                sum_accuracy += accuracy
            # 適合率，再現率，f値を求めて，総和を保存する変数に足し込む．
            # sには，サンプルの個数が入っているけど，今回は使わない．
            # p, r, fは，3つとも，配列．pは，[迷いなしの場合の適合率, 迷いありの適合率]の2要素からなる配列．再現率，f値も同様．
                p, r, f, s = metrics.precision_recall_fscore_support(labels_test, labels_pred)
                sum_precision_no  += p[0]
                sum_recall_no     += r[0]
                sum_f_measure_no  += f[0]
                sum_precision_yes += p[1]
                sum_recall_yes    += r[1]
                sum_f_measure_yes += f[1]
            # 特徴量の重要度を求めて，総和を保存する配列に足し込む．
                sum_feature_importance = sum_feature_importance + numpy.array(rfc.feature_importances_)

        #10交差検定した結果（正解率，適合率，再現率，F値）を毎回配列に格納．正解率等はそれぞれの値の総和を，交差検定を行った回数(10)で割っている．
            avg_accuracy.append(sum_accuracy / n_folds)
            avg_p_no.append(sum_precision_no / n_folds)
            avg_r_no.append(sum_recall_no / n_folds)
            avg_f_no.append(sum_f_measure_no / n_folds)
            avg_p_yes.append(sum_precision_yes / n_folds)
            avg_r_yes.append(sum_recall_yes / n_folds)
            avg_f_yes.append(sum_f_measure_yes / n_folds)

            #平均値出すやつ、あってるやつ？
            result = []
            result.append([sum_accuracy/n_folds,sum_precision_no/n_folds,sum_recall_no/n_folds,sum_f_measure_no/n_folds,sum_precision_yes/n_folds,sum_recall_yes/n_folds,sum_f_measure_yes/n_folds])
            

            
            

        #1回ずつの正解率等を表示したかったら以下のコメントアウトをはずす
        # 正解率等を出力．それぞれの値の総和を，交差検定を行った回数で割る．
        #print('----results')
        #print('avg_accuracy(正解率)　　　　　　:'+str(sum_accuracy     /n_folds))
        #print('avg_p_no(適合率_迷いなし)    :'+str(sum_precision_no /n_folds))
        #print('avg_r_no(再現率_迷いなし)    :'+str(sum_recall_no    /n_folds))
        #print('avg_f_no(F値_迷いなし) 　　    :'+str(sum_f_measure_no /n_folds))
        #print('avg_p_yes(適合率_迷いあり)    :'+str(sum_precision_yes/n_folds))
        #print('avg_r_yes(再現率_迷いなし)   :'+str(sum_recall_yes   /n_folds))
        #print('avg_f_yes(F値_迷いなし)   　　:'+str(sum_f_measure_yes/n_folds))
    
    outputfileresult = "all2023su3.csv"
    result.append(str("all2023su"))
    fwrite = open(outputfileresult,'a')
    writer = csv.writer(fwrite, lineterminator='\n')
    writer.writerows(result)
    fwrite.close()
        
        
        # 特徴量の重要度を出力．値が大きいものから順にソートして表示．
    featureIn=[str("feature"),str("importance")]
    print('----features ordered by importance')
    for featureId in sorted( zip(range(len(sum_feature_importance)), sum_feature_importance), key=itemgetter(1), reverse=True):
        print(selectedParameters[featureId[0]][1]+': '+str(featureId[1]))

        featureIn.append([selectedParameters[featureId[0]][1],str(featureId[1])])
    writefile_feature = "all2023su3_fe.csv"
    fwrite = open(writefile_feature,'a')
    writer = csv.writer(fwrite, lineterminator='\n')
    writer.writerows(featureIn)
    fwrite.close()
        #if q in [9]:
        #	print('end')
        # ####


    #10回の正解率等の平均値を算出し，表示
    #print(sum(avg_accuracy_DL)/len(avg_accuracy_DL))
    print('----results(10times)')
    print('avg_accuracy(正解率)　　　　:'+str(sum(avg_accuracy)/len(avg_accuracy)))
    print('avg_p_no(適合率_迷いなし)   :'+str(sum(avg_p_no)/len(avg_p_no)))
    print('avg_r_no(再現率_迷いなし)   :'+str(sum(avg_r_no)/len(avg_r_no)))
    print('avg_f_no(F値_迷いなし) 　　 :'+str(sum(avg_f_no)/len(avg_f_no)))
    print('avg_p_yes(適合率_迷いあり)  :'+str(sum(avg_p_yes)/len(avg_p_yes)))
    print('avg_r_yes(再現率_迷いあり)  :'+str(sum(avg_r_yes)/len(avg_r_yes)))
    print('avg_f_yes(F値_迷いあり)   　:'+str(sum(avg_f_yes)/len(avg_f_yes)))
