# FyTAI AI



### 🚧 Contents

- Tokenizer
- LDA learner
- Similarity


### 🖨 Tokenizer

- **형태소분석기** : MeCab
  - NNG(명사), NNP(고유명사) 만 추출
    **[다른 형태소를 추가 추출할 경우, 상단 전역변수 "TAGS" 를 수정할 수 있습니다.]**
  - 본 Tokenizer에서 사용한 MeCab은 [Github : mecab-ko-msvc](https://github.com/Pusnow/mecab-ko-msvc) 를 기반으로  사용되었습니다.
    **[추가 Module :** [Github : mecab-python-msvc](https://github.com/Pusnow/mecab-python-msvc) **]**
  
  ```python
  import MeCab
  mecab = MeCab.Tagger()
  result = mecab.parse("아버지가방에들어가셨다.")
  print(result)
  
  #===============================================#
  아버지  NNG,*,F,아버지,*,*,*,*
  가      JKS,*,F,가,*,*,*,*
  방      NNG,장소,T,방,*,*,*,*
  에      JKB,*,F,에,*,*,*,*
  들어가  VV,*,F,들어가,*,*,*,*
  셨      EP+EP,*,T,셨,Inflect,EP,EP,시/EP/*+었/EP/*
  다      EF,*,F,다,*,*,*,*
  .       SF,*,*,*,*,*,*,*
  EOS
  ```
  
- **언어** : Korean

이 외에도 한 글자일 경우, **금**, **돈**과 같이 유의미한 Token이 추출될 수 있으나, 무의미한 Token 갯수가 현저히 많으므로 제외하였습니다.

현재 모델링 대상은 Korean 대상을 전제하므로, Korean을 제외한 타 언어는 제외하였습니다. 



### 🧬LDA

**Gensim**을 이용한 **LDA** 모델 학습기입니다. 

- Youtube Comments 대상으로 코드가 구성되어 있으며, 댓글 포함여부를 설정할 수 있습니다.

- **TF-IDF** 수식 적용여부를 설정할 수 있습니다.
- Visualization 적용이 되어있습니다.



> **How to run?**

```python
python -i lda_learner.py

>>> LDA()
```



### 🔗Similarity

**Viedo**와 **Comment**간의 유사도 계산기입니다.

> **How to run?**
```python
python -i similarity.py

# lda_vector Column 갱신
>>> update_similarity()

# 영상 댓글 연관도순 정렬(상위20개만 출력)
>>> video_sim_comment(video_id="abasdfkekd")
```
