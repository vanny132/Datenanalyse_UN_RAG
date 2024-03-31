# Interact with Sportnews Articles
## Abstract
This repository shows an approach to interact with scraped sportnews. The dataset includes about 60,000 articles. www.reviersport.de published the articles.

The main goal is to pass queries to a large language model. In particual, the queries focus on details related to events and dates and they have to answers excatly. Therefore, serveral retrieving methods would test and combine with eachother to reach the goal. The inference will verdict by the developer and the use of the RAGAS framework is in testing. 

As generation model we used `Llama-2-13-b`, `Sauerkraut Mixtral 8x7B` and `Sauerkraut SOLAR` and as embeddings model we use `all-mpnet-base-v2`. But `Sauerkraut SOLAR` wasn't taken into account for the test, because the inference time of over 67 seconds is too long for a production-ready application.

A NER-model extracts entities out of the query to use the values for pre-filtering the documents by date, season and club. This approach take into account to concentrate to the best fitting articles. Further, it would combine the following ranking/retriever methods: ranking by metadata, time-weighted retriever, similarity search and BM25 algorithm. 

Today's result shows the siginificant impact of pre-filtering by extracted entity values from the query. Also the similarity search shows the best scores, if there isn't an extracted entity value. Furthermore the smaller chunks with a token length of 128 and an overlap of 42 achieves a 53% higher qunatity of correct answers than the other tested chunk strategies. 

## Inference Process
<p align="center">
  <img src="https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/9da0f59c-4b31-4fec-8e5b-17b3f0631474", width="80%">
</p>

### Extract entities with NER model
The motivation to extract entities is to filter the documents (articles) by entities. So it's possible to filter the documents by date, club, and season if the query contains them and this reduce the scope to a closed set of data. 

There is a post process for date-entity to parse it to a `datetime` object. If the result of the parser isn't `None`, then the data will filter by the date with the condition `query_date <= published_date` with `published_date` as the released date of the document.

For the entities `club` and `season` it's an `in` condition, because the `news_keywords` are comma-seperated keywords. The process splits them and it's possible to search for `club` and `season` values in the list of keywords.
  
### Ranking by Metadata
The crawled articles contains an attribute `news_keywords`. This attribute is filled with comma-separated keywords. In the ranking by metadata, the process take the extracted entity values from the query and calculate a score as follows:
```math
score = \frac{1}{quantity\ of\ keywords}
```
The more keywords are in the field `news_keywords`, the more irrelevant the individual extracted keyword will be. 
### Time-weighted ranking
The weigthed ranking is based on the different between the extracted and parsed date from the query:
```math
difference\ [hours] = \frac{|query\ datetime - published\ datetime|}{3600}
```
This means that the higher the `decay_rate`, the more irrelevant are the articles that lie more in the past or in the future:
```math
score = (1 - decay\_rate)^{difference\ [hours]}
```
A `decay_rate` of zero means, that the time has no impact on the ranking. The behaivor of the ranking of this case wasn't part of the development until now. The `decay_rate` is set to 0.1 in this project.
### Similarity Search
For similarity search the query and the articles are transformed into embedding vectors. Further, the cosine similarity will calculate:
```math
score = cos(\theta) = \frac{A \cdot B}{||A|| \cdot ||B||}
```
where `A` and `B` represent the embeddings of the query and document.
In the project `all-mpnet-base-v2`is used to generate the embeddings for both, the query and the articels.

### BM25
The Best Match 25 (BM25) is a ranking function, which often used in information retrieval. BM25 extends the principles of TF-IDF, while both compute the relevance of a document, but BM25 consider other factors like document length and the average length of documents in collection. BM25 calculates for each document $d$ in a collection of $D$ the score as follows:

```math
\text{score}(d,q) = \epsilon \cdot \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{l(d)}{\text{avg(l(D))}}\right)}
```
There is an $\epsilon$ to prevent negative $IDF$ scores. For more information take look into the[BM25 implementation](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py).
$IDF$ is the inverse term-frequency of $q_i$. A less term-frequency results in a higher score. Thereby, the score is more less for words with a high frequnency like *the*, *and*, *or* and so on.
$f(q_i, d)$ is the term frequency of the term in the document $d$.
The constants $k_1$ and $b$ are set to $k1=1.5$ and $b=0.75$.
Further $l(d)$ is the length of the document and $avg(l(d))$ is the average length of all documents in the collection. Therefore, longer documents are more penalized than short documents.

### Reciprocal Rank Fusion (RRF)
Reciprocal Rank Fusion is a simple method for combining several difference ranking methods in field of information retrieval. RRF simply sorts the documents according to the rank of each document in all retriever sets:
```math
score (d\in D) = \sum_{r \in R}{} \frac{1}{k+r(d)}
```
with $k = 60$. A higher $k$ relativizes the rank-distances. 
Further,  the spatial distance between the scores of the first and second rank doesn't cosider by RRF. As instance, in the similarity search the first rank has a score of 0.9 and the second rank a score of 0.3. If we leave $k$ out of consideration and calculate the RRF-score for the first and second document, we see that the difference between $\frac{1}{1}= 1$ for the first rank and $\frac{1}{2} = 0.5$ for the second rank is smaller than the difference of the cosine similarity between the first and second rank, as shown in the following graph:
<p align="center">
  <img src="https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/3ce21449-4252-4064-a7a8-2aa38874d55f", width="80%" title = "Left: Score by method (like cosine-similarity) of datapoints - Rigth: The RRF-score for each datapoint/value">
</p>
That presents, that we loss information in RRF. 

A solution is to normalize the scores of the ranking/retrieving methods between zero and one. Then, the scores can sum.
### Long-context re-ranking
The long-context re-ranking is based on the paper [lost in middle](https://arxiv.org/abs/2307.03172). The paper shows well, that models are better at using relevant information that occurs at the very beginning or end of its input context, and performance degrades significantly when models must access and use information located in the middle of its input context. That means for the re-ranking, that retrieved documents have to be in a specific order according to the following graph:
<p align="center">
  <img src="https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/c2c2cd86-5b8a-44ea-9fd4-6b44bd3a0891", width="40%">
</p>
This graph shows, that the fourth relevant document has to be on the last position. Based on the graph the long-context re-ranker was implemented in the project. 

# RAG Evaluation
After we passed different queries to the inference process, we evaluate the RAG pipeline by RAGAS and human. Additional we add *langchain criterea*. 

## RAGAS
RAGAS is a framework for the following named metrics in this section. They are model-based metrics. You need the query, answer, ground truth and context to compute the metrics. The following based on the [RAGAS paper](https://arxiv.org/abs/2309.15217).
### Faithfulness
For the faithfulness, RAGAS prompt the evaluation model to generate statements based on the query. In a further prompt, the model verdicts, if statement is supported or not. The ratio is the faitfulness score 
```math
F = \frac{Number\ of\ statements\ that\ were\ supported}{Total\ number\ of\ statements}
```
<ins>Idea:</ins> The idea of faithfulness is, that the answer should be ground in the given context. This is important to avoid hallucinations and that the retrieved context can act as a justification for the generated answer. The score is between $0$ and $1$ and higher score is better.

### Answer Relevance
To compute the answer relevance the model generates questions for the given answer. Next, we calculate the cosine-similarity between the embeddings of each question $q_i$ and the original query:
```math
AR = \frac{1}{n} \cdot \sum_{i=1}^{n}sim(q,q_i)
```
<ins>Idea:</ins> Does the generated answer adress the actual question/query, that was provided? The more higher the score, the better the answer adresses to the question/query. The score is between $0$ and $1$ and higher score is better.

### Context Relevance
The first step to calculate the context relevance is to split the provided context into sentences. Next, the evaluation model picks the relevant sentences and in last step the score will calculate as follows:
```math
CR = \frac{Number\ of\ extracted\ sentences}{Total\ number\ of\ sentences\ in\ context}
```
<ins>Idea:</ins> The retrieved context should be focused and containing as little irrelevant information as possible. This is important, if long context passages are retrieved to the LLM. The score is between $0$ and $1$ and higher score is better.


### Limitations
The project shows the limitations in the model-based approach, because RAGAS is very prone for wrong generations. If the model isn't complex enough the original [RAGAS library](https://github.com/explodinggradients/ragas) fails and the prompts and the extraction of information out of the prompt have to adapt. Further, less complex models often vary in the model output and then the approach is very inrobust and unreliable.
The paper experiments with GPT-3.5.

## Langchain Criteria
Langchain provides an amount of different evaluation criterias, see [Criteria Evaluation](https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain).

Out of this basket of metrics, we use the following criterias:
- Conciseness: Is the submission concise and on point?
- Relevance: Is the submission reffering to a real quote from the text?
- Correctness: Is the submission correct, accurate and factual?
- Helpfulness: Is the submission helpful, insightful and appropriate?
- Detail: Does the submission demonstrate attention to detail?

## Human evaluation
The human evaluation process awarded points to each answer:
- `5` Points for the right answers in the desired language or if the model answers with the expected behavior, if it isn't know the answer.
- `3` Points for the right answers in the wrong language or if we observe hallucation. Also, if the facts in answer is right, but there is a missing fact or if one of two expected facts is wrong.
- `0`Points if the answer is wrong.
  
We have to note, that is a subjectiv evaluation. In NLP common human-annotaded datasets are annotaded by more than five different people. To get a more reliable result, it has to be evaluate by other people, also.
# Experiments
The goal is to find the retrieving method combination, which have the best inference result. 
There are three setting points in the execution:
- Retrieving methods (four options and thus sixteen combinations)
- Chunk-size (size/overlap) in tokens: (128/42), (256/85), (508/170)
- Generation model: [Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [Sauerkraut Mixtral 8x7b](https://huggingface.co/VAGOsolutions/SauerkrautLM-Mixtral-8x7B-Instruct), [Sauerkraut SOLAR](https://huggingface.co/VAGOsolutions/SauerkrautLM-SOLAR-Instruct)

All of them would combine and test and then evaluate by human and RAGAS. There are eleven different questions/prompt in the test. All-in-all there are 528 tests per generation model.

## Results
The first thing you notice is the inference time of the models, shown in the following figure: 
<p align="center">
  <img src="https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/8e34df1a-eeb9-4922-be42-3db684c0ec61", width="60%">
</p>
As a result, the SOLAR model is no longer taken into account in the following tests, because it isn't user friendly,  because the inference time is of more than 67 seconds.

\
We investigated several aspects like the influence of different combined retrieving methods and chunk-sizes, also different queries. In the one hand the NER-model is able to extract the entities from the queries and can process them and on the other hand there are queries, that aren't able to handle by the NER-model. With this approach we wanted to determine if there is an impact of pre-filtering the data by specific entities/keywords. All of this is scored by human and RAGAS, but we use the RAGAS library, which has a lot development potential, because the model outputs aren't reliable. Based on this reason, we adapted the library to get valid results for _context_precision_ and _context_recall_. The RAGAS developer work on a better and more reliable solution actually.

### Human Evaluation
In the following plot, there is the mean score of each retrieving combinations scored by human. The plot shows, that the combination of time-weigthed and metadata retrieving is the best combination.
<figure>
  <img src="https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/61e8b292-f55a-4b18-82a6-2ecde7ce2b9a" alt="Mean score per retrieving combination">
</figure>

The graphic also shows that the metadata ranking method combined with other ranking methods, achieves the highest scores. This is explained by the fact, that the metadata ranking method works with the extracted entities from the query and the documents are filtered by the entities before. The filtering by entities achieves the best result in human evaluation as also shown in the following graphic _Comparison of Llama-2-13b and Sauerkraut Mixtral 8x7B by Retrieving Methods and Scores_. The best results that excepted metadata ranking is time-weighted retrieving and the combination of time-weighted and similarity search. 

The above plot presents, that the bahavior of the approach isn't match with the expected bahavior of the prompter. However, this depends on the query as the follwing graphic shows:

![image](https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/d34a0665-d21a-4fb8-97da-28d5797d59ce)

The plot shows the best result for the last three queries. That's because, that the NER model can extract the right values for the entities to search with them through the `news_keywords` in the dataset. That results in the good scores for the questions for the last three questions. To get a better understanding for the extraction process, please feel free to check the notebook _GLiNER/GLiNER_demo.ipynb_.

At last, the chunk-strategie has also an impact:
![image](https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/f603fb9e-ec36-4f14-a32f-ca7894d05e57)
The chunk-size of 128 and an overlap of 42 tokens has the highest density of right answers and match better the expected bahaviour of the prompter than the other chunk-strategies. Compared to the other chunk-strategie it achieves 53% more right answers than the chunk-size of 256 and an overlap of 85 tokens and 33% more right answers than the chunk-size of 508 and an overlap of 170 tokens.

As conclusion for the human evaluation we can note the following points:
- Filtering the data by entities like club, season and the published date improves the RAG performace significant.
- The right prompt to the model has also impact on the quality, because from a good prompt we can extract the right entities values to use it in the pre-filtering.
- If it isn't possible to extract any entity value, than the similarity search reaches the best mean score.
- Smaller chunks achieve a higher density of the expected behaviour
  
# Dataset
The dataset was crawled from www.reviersport.de and includes 59,938 articles. For each articles there are the following attributes crawled:
- URL
- Artilce's content
- Teaser
- Headline
- Description: A short description about the article of the authors.
- News-keywords: Keywords about the article selected by the authors.
- Keywords: Keywords about the article selected by the authors.
- Publishing date
- Author
