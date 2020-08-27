import sys
sys.path.insert(0, "./tokenizer")
from pymongo import MongoClient
import gensim
from gensim.matutils import cossim
from bson.objectid import ObjectId
from lda_learner import load_model, load_lda, get_data
from tokenizer import str2token
from tqdm import tqdm
import numpy as np

# DB lda_vec 칼럼 갱신 함수
def update_similarity():
	print("DB cursor connecting...")
	try:
		client, db = get_db()
	except:
		print("\nDB cursor connecting failed!")
		return False
	print("\nDB cursor connected")

	videos = get_data()
	videos_count = db.video.count_documents({})

	print("\nDB에 LDA Vector 값 적용을 시작합니다.")

	for idx in tqdm(range(videos_count), desc="Video Tokenizing", mininterval=0.2):
		video = videos[idx]

		comments = video['comments']
		
		for comment in comments:
			replies = comment['replies']
			
			try:
				for reply in replies:
					vec = lda[dictionary.doc2bow(reply['tokens'])]
					vec = list(map(lambda x: (x[0], float(x[1])), vec))
					reply['lda_vec'] = vec
			except:
				pass

			vec = lda[dictionary.doc2bow(comment['tokens'])]
			vec = list(map(lambda x: (x[0], float(x[1])), vec))
			comment['lda_vec'] = vec

		vec = lda[dictionary.doc2bow(video['tokens'])]
		vec = list(map(lambda x: (x[0], float(x[1])), vec))
		video['lda_vec'] = vec

		db.video.update_one(
			{
				"_id": ObjectId(video['_id'])
			},
			{
				"$set": {
					'comments': comments,
					'lda_vec': video['lda_vec']
				}
			}
		)
	print("\nVideo의 Vector 적용이 완료되었습니다!\n")

	print("DB cursor unconnecting...")
	close_db(client)

	print("Done!")

	return True

# 2개의 문장 유사도 계산 함수
def docs_similarity(doc1="", doc2=""):
	if doc1 == "" or doc2 == "": 
		print("Input 2 arguments, doc1<string>, doc2<string>")
		return False

	tokens_1 = str2token(doc1)
	tokens_2 = str2token(doc2)

	vec_1 = lda[dictionary.doc2bow(tokens_1)]
	vec_2 = lda[dictionary.doc2bow(tokens_2)]

	sim = cossim(vec_1, vec_2)
	return sim

# 연관도 정렬 함수
def video_sim_sort(video_id="VckKZ4et3Lc", show=True):
	print("VIDEO ID :", video_id, " sorting...\n")

	client, db = get_db()
	video = db.video.find_one({"hash": video_id})

	target = video['lda_vec']
	target = list(map(tuple, target))
	
	docs = []
	
	comments = video['comments']

	for comment in comments:
		replies = comment['replies']

		for reply in replies:
			vec = reply['lda_vec']
			vec = list(map(lambda x: (x[0], np.float32(x[1])), vec))
			reply['lda_vec'] = vec
			docs.append(reply)

		vec = comment['lda_vec']
		vec = list(map(lambda x: (x[0], np.float32(x[1])), vec))
		comment['lda_vec'] = vec
		docs.append(comment)

	for doc in docs:
		doc['sim'] = cossim(target, doc['lda_vec'])

	docs = sorted(docs, key=lambda x: x['sim'])
	docs.reverse()

	if show:
		print("영상 제목 :", video['title'])
		print("영상 Token :", video['tokens'], "\n")

		for idx, doc in enumerate(docs[:20]):
			print("<",idx+1,"::::",doc['sim'],">",doc['content'])

	return docs

# FyTAI 정렬을 이용한 댓글 추출 함수
def video_sim_comment(video_id="VckKZ4et3Lc"):
	comments = video_sim_sort(video_id=video_id, show=False)
	output = []

	for comment in comments:
		if comment['sim'] < 0.6:
			break
		if len(comment['content']) > 30:
			output.append(comment)

	for idx, doc in enumerate(output[:20]):
		print("<",idx+1,"::::",doc['sim'],">",doc['content'])

# DB Connector 가져오기
def get_db():
	client = MongoClient('localhost', 27017)
	db = client['fytai']
	return (client, db)

# DB Connector 닫기
def close_db(client):
	return client.close()

# Model Loading
if __name__ == "__main__":
	print("\nModel Loading...")
	try:
		dictionary, corpus = load_model()
		lda = load_lda()
	except:
		print("\nModel loading failed")
		exit()