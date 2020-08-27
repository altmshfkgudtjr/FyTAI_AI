from pymongo import MongoClient
import MeCab
import re
import math
from bson.objectid import ObjectId
from tqdm import tqdm
from except_nouns import except_nouns

# 한글 Regex
REG = re.compile('[^ 가-힣]+')
# 필요 태그
TAGS = ['NNG', 'NNP']



# DB Connector 가져오기
def get_db():
	client = MongoClient('localhost', 27017)
	db = client['fytai']
	return (client, db)

# DB Connector 닫기
def close_db(client):
	return client.close()

# 데이터 가져오기
def get_videos(db):
	return db.video.find({})

# 한글만 뽑음
def get_korean(text):
	return REG.sub('', text)

# 미캡 필요 태그만 추출
def parse_tags(model, text):
	output = []
	result = model.parse(text)
	lines = result.split('\n')
	for line in lines:
		if line == 'EOS' or len(line.split('\t')) <= 1:
			continue
		# 한 글자일 경우 Skip
		if len(line.split('\t')[0]) <= 1:
			continue
		token, tag = line.split('\t')[0], line.split('\t')[1].split(',')[0]
		if tag in TAGS:
			output.append(token)
	return output

# 잘못된 Token 수정
def except_token(tokens):
	for ex_noun in except_nouns:
		ch = 0
		items = ex_noun.values()
		for item in list(items)[0]:
			if item in tokens:
				ch += 1
		if ch == len(list(items)[0]):
			for item in list(items)[0]:
				tokens.remove(item)
			tokens.append(list(ex_noun.keys())[0])
	return tokens

# TF
def tf(token, doc):
	return doc.count(token)

# IDF
def idf(token, docs):
	df = 0
	for doc in docs:
		df += token in doc
	return math.log(len(docs)/(df + 1))

# TF-IDF
def tf_idf(token, doc, docs):
	return tf(token, doc) * idf(token, docs)

# TF-IDF를 이용한 불용어 추출기
def stopword_checker(n=10000):
	print("< 불용어일 수도 있는 Token들 추출 >\n\n")

	# DB 가져오기
	client, db = get_db()

	# 영상정보 가져오기
	videos = get_videos(db)

	comment_list = []

	# Koren 댓글만 잘라서 모으기
	for video in videos:
		comments = video['comments']
		
		for comment in comments:
			replies = comment['replies']
			
			for reply in replies:
				result = get_korean(reply['content'])
				comment_list.append(result)
		
			result = get_korean(comment['content'])
			comment_list.append(result)

	docs = []

	# MeCab 모델 선언
	mecab = MeCab.Tagger()

	# Mecab 형태소 분석
	print(":::: 형태소 분석 중...\n")
	for comment in comment_list:
		output = parse_tags(mecab, comment)
		docs.append(output)

	# 잘못된 Token 수정
	for doc in docs:
		filtered = []
		filtered = except_token(doc)
		doc = filtered

	# 불용어 파일 읽어오기
	print(":::: 불용어 처리 중...\n")
	with open('korean_stopwords.txt', 'r', encoding='UTF8') as f:
		stopwords = f.read().split()

	# 불용어 처리
	for doc in docs:
		filtered = []
		for token in doc:
			if token not in stopwords:
				filtered.append(token)
		doc = filtered

	# TF-IDF 적용
	print(":::: TF-IDF 계산 중...\n")
	N = len(docs)

	tfidf_list = []
	dup_list = []

	for doc in docs:
		for token in doc:
			if token in dup_list:
				continue

			dup_list.append(token)
			tfidf_list.append(
				{
					'token': token,
					'tf_idf': tf_idf(token, doc, docs)
				}
			)

	# tf_idf 값으로 오름차순 정렬
	output = sorted(tfidf_list, key=lambda x: x['tf_idf'])
	output.reverse()

	print("총 토큰 수:", len(output))
	print(":::: Token", n, "개 추출 중...")
	with open('check_please.txt', 'w', encoding='UTF8') as f:
		for token in output[:n]:
			f.write(token['token']+'\n')

	return True
			

# DB에 Token화
def Tokenizer_DB():
	# MeCab 모델 선언
	mecab = MeCab.Tagger()

	# 불용어 파일 읽어오기
	with open('korean_stopwords.txt', 'r', encoding='UTF8') as f:
		stopwords = f.read().split()

	# DB 가져오기
	client, db = get_db()

	# 영상정보 가져오기
	videos = get_videos(db)

	# 현재 영상 갯수
	videos_count = db.video.count_documents({})

	comment_list = []

	# Token Column 생성
	for idx in tqdm(range(videos_count), desc="Video Tokenizing", mininterval=0.1):
		video = videos[idx]

		comments = video['comments']
		
		for comment in comments:
			replies = comment['replies']
			
			for reply in replies:
				result = get_korean(reply['content'])
				result = parse_tags(mecab, result)
				result = except_token(result)

				tokens = []

				for token in result:
					if token not in stopwords:
						tokens.append(token)

				reply['tokens'] = tokens

			result = get_korean(comment['content'])
			result = parse_tags(mecab, result)
			result = except_token(result)

			tokens = []

			for token in result:
				if token not in stopwords:
					tokens.append(token)

			comment['tokens'] = tokens

		video_tokens = []

		if 'tags' in video:
			for token in video['tags']:
				result = get_korean(token)
				result = parse_tags(mecab, result)
				result = except_token(result)
				video_tokens += result
		else:
			db.video.update_one(
				{
					"_id": ObjectId(video['_id'])
				},
				{
					"$set": {
						'tags': []
					}
				}
			)

		db.video.update_one(
			{
				"_id": ObjectId(video['_id'])
			}, 
			{
				'$set': {
					'comments': comments,
					'tokens': video_tokens
				}
			}
		)

	print("\nVideo의 Tokenizing이 완료되었습니다!\n")

	close_db(client)
	
	return True

# String to Token
def str2token(string):
	# MeCab 모델 선언
	mecab = MeCab.Tagger()

	# 불용어 파일 읽어오기
	with open('./tokenizer/korean_stopwords.txt', 'r', encoding='UTF8') as f:
		stopwords = f.read().split()

	result = get_korean(string)
	result = parse_tags(mecab, result)
	result = except_token(result)

	tokens = []

	for token in result:
		if token not in stopwords:
			tokens.append(token)

	return tokens

# Tokenizer_DB()

# Tokenizer_1(1500, 1501)

# Tokenizer_2(1500, 1501)

# stopword_checker()