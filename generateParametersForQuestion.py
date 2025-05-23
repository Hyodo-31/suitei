import numpy 
import csv
import re

# from operator import itemgetter

#linedatamousesの読み込み
#['uid','wid','time','X','Y','DD','DPos','hLabel','Label','hesitate','understand','date','check']
#データが多すぎるのでファイルを２つに分けてあります．
readers = []
inputfilename1 = 'linedata2.csv'	#inputdata(パラメータ変換前のファイル)
inputfilename2 = 'linedatamouse.csv'	#inputdata(パラメータ変換前のファイル)
fread1 = open(inputfilename1, 'r', encoding = "utf-8")
readers.append(csv.reader(fread1))
outputdatafile = '2025sample.csv'
fread2 = open(inputfilename2, 'r', encoding = "utf-8")
readers.append(csv.reader(fread2))

#fread3 = open('inputdata2019ku3.csv', 'r')
#readers.append(csv.reader(fread3))
#fread4 = open('inputdata2019ku4.csv', 'r')
#readers.append(csv.reader(fread4))
#fread5 = open('inputdata2019ku_37.csv', 'r')
#readers.append(csv.reader(fread5))
#fread6 = open('inputdata2019su1.csv', 'r')
#readers.append(csv.reader(fread6))
#fread7 = open('inputdata2019su2.csv', 'r')
#readers.append(csv.reader(fread7))
#fread8 = open('inputdata2018ku1.csv', 'r')
#readers.append(csv.reader(fread8))
#fread9 = open('inputdata2018ku2.csv', 'r')
#readers.append(csv.reader(fread9))
#fread10 = open('inputdata2018su1.csv', 'r')
#readers.append(csv.reader(fread10))
#fread11 = open('inputdata2018su2.csv', 'r')
#readers.append(csv.reader(fread11))
#f = open('chunk_.csv', 'r')
#reader = csv.reader(f)
#list = []
#for row in reader:
#	chunk = row[1]
#	print(chunk)
#	list.append(chunk)

parametersPerQuestion = []

for i in range(0,2):	#ファイルが一戸ならこれは(0,1)に変更かな
	# uid, widごとにデータを区切る
	# dataは，2次元配列．data[uid][wid]という形で参照する．
	# data[uid][wid]には，配列が格納されている．配列の内容は，1次のパラメタ．timeでソートされている．
	data = {}
	tmpParametersPerQuestion = [] #2次のパラメータを入れる配列

	testCount = 0
	for row in readers[i]:		#ファイルを二個に分けているための処理
		testCount += 1
		row[2] = int(row[2])
		row[3] = int(row[3])
		row[4] = int(row[4])
		row[5] = int(row[5])
		#row[13] = int(row[13])

		if str(row[0]) not in data.keys():
			data[str(row[0])] = {}
		if str(row[1]) not in data[str(row[0])]:
			data[str(row[0])][str(row[1])] = []
		data[str(row[0])][str(row[1])].append({'time': row[2], 'x': row[3], 'y': row[4], 'dd': row[5], 'hLabel': row[7], 'label': row[8], 'hesitate': row[9], 'understand': row[10]})#, 'date': row[11], 'check': row[12]})#,'wordnum': row[13]}) , 2023/06/13変更'tf': row[13]
	print(testCount)
	# パラメタを計算していく
	for user in data.keys():				#ユーザーごとに処理するためのfor文
		for question in data[user].keys():	#上記のユーザーの各問題を処理するためのfor文
			params = data[user][question] # 時間でソートされた1次のパラメタが格納されている配列．
			# 自信度と日付とチェックは最初に取得しておく．
			understand = params[0]['understand']
			#date = params[0]['date']
			#check = params[0]['check']
			date = 0
			check = 0
			#stf = params[0]['tf']			#1なら正解，0なら不正解
			# time(解答にかかった時間)は，最後の要素の時刻から最初の要素の時刻を引けば算出できる．
			time = params[-1]['time'] - params[0]['time']		
			# パラメタの初期値をゼロにしておく．
			distance = 0
			averageSpeed = 0
			maxSpeed = 0
			answeringTime = 0
			thinkingTime = 0
			totalStopTime = 0
			maxStopTime = 0
			stopcount_first = 0#静止回数値初期
			stopcount = 0#静止回数値改善値
			totalDDIntervalTime = 0
			maxDDIntervalTime = 0
			maxDDTime = 0
			minDDTime = 10000000000000000
			DDCount = 0#-params[0]['wordnum']
			xUTurnCount = 0
			yUTurnCount = 0
			groupingDDCount = 0
			chunk_much = 0
			register_drag_flag = 0
			register_move_count1 = 0
			register_move_count2 = 0
			register_move_count3 = 0
			register_move_count4 = 0
			register01count1 = 0
			register01count2 = 0
			register01count3 = 0
			register01count4 = 0
			register1count = 0
			register2count = 0
			register3count = 0
			groupingCountbool = 0

			#2023/03/16追加分
			#マウスドラッグ中のUturn回数
			xUTurnCountDD = 0
			yUTurnCountDD = 0

			#2023/04/17
			FromlastdropToanswerTime = -1


			
			hesiwordnum = int(len(params[0]['hesitate'].split('#')))
			#print('hesiwordnum:' + str(hesiwordnum))

			# ここから下は，パラメタを計算するために補助的に使う変数．
			startTime = -1    # 問題の解答を開始した時刻．（最初のクリック）
			lastDragTime = -1 # 直前にドラッグを開始した時刻
			lastDropTime = -1 # 直前にドロップを行った時刻
			lastXDirection = 0 # 直前のx軸の方向．1なら右方向，-1なら左方向．
			lastYDirection = 0 # 直前のy軸の方向．1なら右方向，-1なら左方向．
			#2023/3/26追加分
			lastXDirectionDD = 0 #マウスをドラッグした際のx軸の方向
			lastYDirectionDD = 0 #マウスをドラッグした際のy軸の方向

			DDstartend = 0 #D&Dの判定ラベル,1ならD&D中

			continuingStopTime = 0 # マウスの静止が継続している時間
			lastDragTime_anspara = 0 # answeringTime計算用の最終単語Dragタイミング
			# 1つ1つデータを見ていって計算していく．

			#len(params) -1を消してみる．
			for i in range(len(params)-1): # 今の行とその次の行の比較によって求めるパラメータがあるので，outofrangeになるのを防ぐためにlen-1にする
				# 距離，速度，最大速度
				currentCoord = numpy.array([params[i]['x'], params[i]['y']])
				nextCoord    = numpy.array([params[i+1]['x'], params[i+1]['y']])
				distance += numpy.sqrt(numpy.power(currentCoord-nextCoord, 2).sum())
				speed = distance*1.0 / (params[i+1]['time']-params[i]['time'])
				if params[i+1]['time']-params[i]['time'] != 0 and speed > maxSpeed:
					maxSpeed = speed
				# 解答開始時刻
				if startTime == -1 and params[i]['dd'] == 2:
					startTime = params[i]['time']
				# 総静止時間，最大静止時間
				#if params[i]['x'] == params[i+1]['x'] and params[i]['y'] == params[i+1]['y']:#最初のやつ
				#if params[i+1]['x'] - params[i]['x'] == 0 and params[i+1]['y'] - params[i]['y'] == 0:
				#if (abs(params[i+1]['x'] - params[i]['x']) < 5) and (abs(params[i+1]['y'] - params[i]['y']) < 5):
				stopdistance = float(numpy.sqrt((params[i+1]['x'] - params[i]['x'])**2 + (params[i+1]['y'] - params[i]['y'])**2))
				#if numpy.sqrt((params[i+1]['x'] - params[i]['x'])**2 + (params[i+1]['y'] - params[i]['y'])**2) < 5:
				if stopdistance < 5:
					#print(stopdistance)
					#stopcount_first += 1
					stopTime = params[i+1]['time']-params[i]['time']
					totalStopTime += stopTime
					continuingStopTime += stopTime
					if continuingStopTime > 500:
						stopcount += 1
					if continuingStopTime > maxStopTime:
						maxStopTime = continuingStopTime
				else:
					continuingStopTime = 0
				
				# DD間時間
				if params[i]['dd'] == 2: # drag時
					#ラベル数カウント
					labelC = len(re.findall(r'[0-9]+', params[i]['label']))
					lastDragTime = params[i]['time']
					if lastDropTime != -1:
						DDIntervalTime = params[i]['time'] - lastDropTime
						if DDIntervalTime > maxDDIntervalTime:
							maxDDIntervalTime = DDIntervalTime
						totalDDIntervalTime += DDIntervalTime
					#レジスタ使用判断用
					#if params[i]['y'] > 215 and params[i]['y'] <= 295:
					#	register_flag = 1
					#elif params[i]['y'] > 295 and params[i]['y'] <= 375:
					#	register_flag = 2
					#elif params[i]['y'] > 375:
					#	register_flag = 3
					#else:register_flag = 0
					if (params[i]['y'] > 215 and params[i]['y'] <= 295) or (params[i]['y'] > 295 and params[i]['y'] <= 375) or (params[i]['y'] > 375):
						register_drag_flag = 1
					else:register_drag_flag = 0

				# DD時間
				if params[i]['dd'] == 1: # drop時
					lastDropTime = params[i]['time']
					DDTime = params[i]['time'] - lastDragTime
					if DDTime > maxDDTime:
						maxDDTime = DDTime
					if DDTime < minDDTime:
						minDDTime = DDTime
					DDCount += 1

					#レジスタ使用判断用
					#if register_flag != 0:#レジスタ内移動のときのみ
					#	if params[i]['y'] > 215 and params[i]['y'] <= 295:
					#		register_flag = 1
					#		register1count += 1
					#	elif params[i]['y'] > 295 and params[i]['y'] <= 375:
					#		register_flag = 2
					#		register2count += 1
					#	elif params[i]['y'] > 375:
					#		register_flag = 3
					#		register3count += 1
					#else:
					#	register_flag = 0
					#if register_flag != 0:
					#	register_count += 1
					if register_drag_flag == 1:
						if (params[i]['y'] > 215 and params[i]['y'] <= 295) or (params[i]['y'] > 295 and params[i]['y'] <= 375) or (params[i]['y'] > 375):
							register_move_count1 += 1
						else:register_move_count2 += 1
					else:
						if (params[i]['y'] > 215 and params[i]['y'] <= 295) or (params[i]['y'] > 295 and params[i]['y'] <= 375) or (params[i]['y'] > 375):
							register_move_count3 += 1
						else:register_move_count4 += 1
					#print(params[i]['label'])
					#labelC = len(re.findall(r'[0-9]+', params[i]['label']))
					#print(labelC)
					#ラベル考慮
					#DDCount += labelC - 1
				#グルーピング回数 矩形選択して動かしたやつ（違うところに移動させてないのも含む）
				if params[i]['dd'] == 2 and '#' in params[i]['label']:
					groupingDDCount += 1
				# 差の定義（Uターンのフラグと閾値の判断に使用）
				xUTurnDist = params[i+1]['x']-params[i]['x']
				yUTurnDist = params[i+1]['y']-params[i]['y']

				#マウスをドラッグさせている間のUturnを判別
				
				if params[i]['dd'] == 2: # drag時
					DDstartend = 1
				elif params[i]['dd'] == 1: #drop時
					DDstartend = 0
				else :					#単語をつかんでいるor単語を話している
					pass

				
				
				if params[i+1]['x']-params[i]['x'] < 0:
					xDirection = 1
				elif params[i+1]['x']-params[i]['x'] > 0:
					xDirection = -1
				else:
					xDirection = 0
				if ((xDirection == 1 and lastXDirection == -1) or (xDirection == -1 and lastXDirection == 1))and ((xUTurnDist < -5) or (xUTurnDist > 5)) and (DDstartend == 0):
					xUTurnCount += 1
					lastXDirection = xDirection
				elif xDirection != 0:
					lastXDirection = xDirection
				# y方向の単語ドラッグ中のUターン．（y方向のUターンとほぼおなじなので，関数化した方がキレイ）
				if params[i+1]['y']-params[i]['y'] < 0:
					yDirection = 1
				elif params[i+1]['y']-params[i]['y'] > 0:
					yDirection = -1
				else:
					yDirection = 0
				if ((yDirection == 1 and lastYDirection == -1) or (yDirection == -1 and lastYDirection == 1))and ((yUTurnDist < -5) or (yUTurnDist > 5)) and (DDstartend == 0): 
					yUTurnCount += 1
					lastYDirection = yDirection
				elif yDirection != 0:
					lastYDirection = yDirection

				#D&D中のUターンX
				if DDstartend == 1:
					if params[i+1]['x']-params[i]['x'] < 0:
						xDirectionDD = 1
					elif params[i+1]['x']-params[i]['x'] > 0:
						xDirectionDD = -1
					else:
						xDirectionDD = 0
					if ((xDirectionDD == 1 and lastXDirectionDD == -1) or (xDirectionDD == -1 and lastXDirectionDD == 1))and ((xUTurnDist < -5) or (xUTurnDist > 5)):
						xUTurnCountDD += 1
						lastXDirectionDD = xDirectionDD
					elif xDirectionDD != 0:
						lastXDirectionDD = xDirectionDD
				#D&D中のUターンY
					if params[i+1]['y']-params[i]['y'] < 0:
						yDirectionDD = 1
					elif params[i+1]['y']-params[i]['y'] > 0:
						yDirectionDD = -1
					else:
						yDirectionDD = 0
					if ((yDirectionDD == 1 and lastYDirectionDD == -1) or (yDirectionDD == -1 and lastYDirectionDD == 1))and ((yUTurnDist < -5) or (yUTurnDist > 5)):
						yUTurnCountDD += 1
						lastYDirectionDD = yDirectionDD
					elif yDirectionDD != 0:
						lastYDirectionDD = yDirectionDD

				#D&D中以外のUターンX
				elif DDstartend == 0:
					if params[i+1]['x']-params[i]['x'] < 0:
						xDirection = 1
					elif params[i+1]['x']-params[i]['x'] > 0:
						xDirection = -1
					else:
						xDirectionDD = 0
					if ((xDirection == 1 and lastXDirection == -1) or (xDirection == -1 and lastXDirection == 1))and ((xUTurnDist < -5) or (xUTurnDist > 5)):
						xUTurnCount += 1
						lastXDirection = xDirection
					elif xDirection != 0:
						lastXDirection = xDirection
				#D&D中以外のUターンY
					if params[i+1]['y']-params[i]['y'] < 0:
						yDirection = 1
					elif params[i+1]['y']-params[i]['y'] > 0:
						yDirection = -1
					else:
						yDirection = 0
					if ((yDirection == 1 and lastYDirection == -1) or (yDirection == -1 and lastYDirection == 1))and ((yUTurnDist < -5) or (yUTurnDist > 5)):
						yUTurnCount += 1
						lastYDirection = yDirection
					elif yDirection != 0:
						lastYDirection = yDirection

			# 上のループはlen-1しているので解答の最後の一行には入らない
			# そのため解答の最後の一行分は別にここで計算（が，これを行うことはないと思う）
			# DD間時間
			lastDragTime_anspara = lastDropTime
			if params[-1]['dd'] == 2: # drag時
				labelC = len(re.findall(r'[0-9]+', params[-1]['label']))
				lastDragTime = params[-1]['time']
				if lastDropTime != -1:
					DDIntervalTime = params[-1]['time'] - lastDropTime
					if DDIntervalTime > maxDDIntervalTime:
						maxDDIntervalTime = DDIntervalTime
					totalDDIntervalTime += DDIntervalTime
				#レジスタ使用判断用
				#if params[i]['y'] > 215 and params[i]['y'] <= 295:
				#	register_flag = 1
				#elif params[i]['y'] > 295 and params[i]['y'] <= 375:
				#	register_flag = 2
				#elif params[i]['y'] > 375:
				#	register_flag = 3
				#else:register_flag = 0
				if (params[i]['y'] > 215 and params[i]['y'] <= 295) or (params[i]['y'] > 295 and params[i]['y'] <= 375) or (params[i]['y'] > 375):
					register_drag_flag = 1
				else:register_drag_flag = 0				

			# DD時間
			if params[-1]['dd'] == 1: # drop時
				lastDropTime = params[-1]['time']
				DDTime = params[-1]['time'] - lastDragTime
				if DDTime > maxDDTime:
					maxDDTime = DDTime
				if DDTime < minDDTime:
					minDDTime = DDTime
				DDCount += 1

				#レジスタ使用判断用
				#if register_flag != 0:
				#	if params[i]['y'] > 215 and params[i]['y'] <= 295:
				#		register_flag = 1
				#		register1count += 1
				#	elif params[i]['y'] > 295 and params[i]['y'] <= 375:
				#		register_flag = 2
				#		register2count += 1
				#	elif params[i]['y'] > 375:
				#		register_flag = 3
				#		register3count += 1
				#else:
				#	register_flag = 0
				#if register_flag != 0:
				#	register_count += 1
				if register_drag_flag == 1:
					if (params[i]['y'] > 215 and params[i]['y'] <= 295) or (params[i]['y'] > 295 and params[i]['y'] <= 375) or (params[i]['y'] > 375):
						register_move_count1 += 1
					else:register_move_count2 += 1
				else:
					if (params[i]['y'] > 215 and params[i]['y'] <= 295) or (params[i]['y'] > 295 and params[i]['y'] <= 375) or (params[i]['y'] > 375):
						register_move_count3 += 1
					else:register_move_count4 += 1
				#print(params[-1]['label'])
				#labelC = len(re.findall(r'[0-9]+', params[-1]['label']))
				#print(labelC)
				#DDCount += labelC - 1
			#グルーピング回数 矩形選択して動かしたやつ（違うところに移動させてないのも含む）
			if params[-1]['dd'] == 2 and '#' in params[-1]['label']:
				groupingDDCount += 1
			
			if groupingDDCount != 0:
				groupingCountbool = 1
			else:
				groupingCountbool = 0

			#DDCountマイナス値の処理
			if DDCount < 0:
				DDCount = 0
			# 平均速度
			if time != 0:
				averageSpeed = distance*1.0 / time
			else:
				averageSpeed = 0
			# 解答時間
			thinkingTime = startTime - params[0]['time'] #最初の単語をクリックするまでの時間
			answeringTime = params[-1]['time'] - startTime #最初の単語をクリックしてから決定までの時間
			#print(lastDragTime_anspara)
			if register_move_count1 != 0:
				register01count1 = 1
			if register_move_count2 != 0:
				register01count2 = 1
			if register_move_count3 != 0:
				register01count3 = 1
			if register_move_count4 != 0:
				register01count4 = 1
			
			registerDDCount = register_move_count1 + register_move_count2 + register_move_count3

			#最後の単語を掴んでから決定ボタンを押した時間
			FromlastdropToanswerTime = params[-1]['time'] - lastDropTime

			tmpParametersPerQuestion.append([user,question,understand,date,check,time,distance,averageSpeed,maxSpeed,thinkingTime,answeringTime,totalStopTime,
			maxStopTime,totalDDIntervalTime,maxDDIntervalTime,maxDDTime,minDDTime,DDCount,groupingDDCount,groupingCountbool,xUTurnCount,yUTurnCount,
			register_move_count1,register_move_count2,register_move_count3,register_move_count4,register01count1,register01count2,register01count3,register01count4,registerDDCount,
			stopcount,xUTurnCountDD,yUTurnCountDD,FromlastdropToanswerTime])

	parametersPerQuestion.extend(tmpParametersPerQuestion)

	# for x in parametersPerQuestion:
	# 	print(x)

print('end')

# csvファイルで出力

print(inputfilename1)

fwrite = open(outputdatafile,'w', encoding = "utf-8")
writer = csv.writer(fwrite, lineterminator='\n')
writer.writerows(parametersPerQuestion)
fwrite.close()