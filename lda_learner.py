import os
from pymongo import MongoClient
import gensim
from gensim.test.utils import datapath
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim
import platform

''' HyperParameter '''
WORKERS = 1 			# CPU 갯수
NUM_TOPICS = 10 		# 토픽 갯수
PASSES = 30				# 전체 코퍼스에서 모델을 학습시키는 빈도를 제어
ITERATION = 70			# 각각 문서에 대해서 루프를 얼마나 돌리는지를 제어
MIN_COUNT = 10 			# 등장 빈도수 및 길이로 딕셔너리 필터링
IS_REPLY = True			# 대댓글 포함
''' ============== '''

os_platform = platform.platform()
if os_platform.startswith("Windows"):
	save_model_path = os.getcwd() + "\\lda_output\\model"
	save_dict_path = os.getcwd() + "\\lda_output\\dict"
else:
	# 여기서부터
	print("Mac 또는 Linux는 경로를 직접 입력하고, 해당 print문을 지워주세요.")
	exit()
	# 여기까지
	save_model_path = "/home/ubuntu/lda_output/model"
	save_dict_path = "/home/ubuntu/lda_output/dict"


# 모델 저장하기
def save_model(model, dictionary, model_path=save_model_path, dict_path=save_dict_path):
	model.save(datapath(model_path))
	dictionary.save(dict_path)

	print("\nmodel saved.\n")

#=============================================================================================

# Video 데이터 가져오는 함수
def get_data(N=0):
	client = MongoClient('localhost', 27017)
	db = client['fytai']
	
	if N == 0:
		posts = db.video.find({})
	else:
		posts = db.video.find({}).skip(N)

	client.close()
	
	return posts

# Corpus|Dictionary 생성 함수
def make_cor_dict(tf_idf=True, is_reply=IS_REPLY):
	print("Corpus | Dictionary 생성 중...")

	videos = get_data()
	
	corpus = []
	dictionary = corpora.Dictionary()

	for video in videos:
		for comment in video['comments']:
			if is_reply:
				for reply in comment['replies']:
					dictionary.add_documents([reply['tokens']])
					corpus += [reply['tokens']]
			dictionary.add_documents([comment['tokens']])
			corpus += [comment['tokens']]

	# 등장 빈도수 및 길이로 딕셔너리 최종 필터링 => 고려해볼 것. 왜냐? 댓글당 Token 개수가 너무 적다.
	dictionary.filter_extremes(no_below=MIN_COUNT)

	# 딕셔너리 기반으로 모든 토큰을 정수로 인코딩
	corpus = [dictionary.doc2bow(tokens) for tokens in corpus]

	# 코퍼스 TF-IDF 수식 적용
	if tf_idf:
		tfidf = TfidfModel(corpus)
		corpus = tfidf[corpus]

	return corpus, dictionary

# LDA 학습 함수
def learn_lda(corpus=None, dictionary=None, num_topics = NUM_TOPICS, passes = PASSES, 
				iterations = ITERATION):
	print("\nLDA Training...\n")
	ldamodel = LdaMulticore(
					corpus,
					num_topics = num_topics,
					id2word = dictionary,
					passes = passes,
					workers = WORKERS,
					iterations = iterations
				)
	print("\nLDA Training Done!\n")

	print("\nCoherence | Perplexity computing...\n")
	cm = CoherenceModel(model=ldamodel, corpus=corpus, coherence='u_mass')
	coherence = cm.get_coherence()
	perplexity = ldamodel.log_perplexity(corpus)

	return ldamodel, coherence, perplexity

# LDA 모델 시각화
def visualization(ldamodel, corpus, dictionary, name=""):
	print("\nVisualizing LDA Model...\n")
	vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

	print("\nsaving...\n")
	pyLDAvis.save_html(vis, name + "gensim_output.html")
	
	print("\nDone!\n")

#==============================================================================

# LDA 모델 생성 함수
def LDA():
	corpus, dictionary = make_cor_dict(tf_idf=False)
	ldamodel, cohorence, perflexity =  learn_lda(
											corpus = corpus, 
											dictionary = dictionary
										)
	print("cohorence:",cohorence)
	print("perflexity:",perflexity)

	# 모델 저장
	save_model(ldamodel, dictionary)

	# 모델 시각화
	visualization(ldamodel, corpus, dictionary)