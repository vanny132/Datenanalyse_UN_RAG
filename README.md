# Interact with Sportnews Articles
## Abstract
This repository shows an approach to interact with scraped sportnews. The dataset includes about 60,000 articles. www.reviersport.de published the articles.

The main goal is to pass queries to a large language model. In particual, the queries focus on details related to events and dates and they have to answers excatly. Therefore, serveral retrieving methods would test and combine with eachother to reach the goal. The inference will verdict by the developer and the use of the RAGAS framework is in testing. 

In the first try `Llama-2-13B` is used as generation model and `all-mpnet-base-v2` as embeddings model. Addtional tests with `Mistral-7-B` and other models are planned.

A NER-model extracts entities out of the query to use the values for pre-filtering the documents by date, leauge and club. This approach take into account to concentrate to the best fitting articles. Further, it would combine the follwoing ranking/retriever methods: ranking by metadata, time-weighted retriever, similarity search and BM25 algorithm. 

Today's result shows the best answer is given by ***pre-filtering of documents by entities and BM25 algorithm***. To the query `Wie hat Leverkusen am 2023-11-25 in der 1.Bundesliga gespielt?` is the answers of Llama-2-13B `Leverkusen hat am 2023-11-25 in der 1.Bundesliga gegen Werder Bremen gewonnen mit 3:0 (2:0). Es war ein hervorragendes Spiel und alle haben draufgefreut. Danke für die Frage!`. The answers correct and the approach shows succesful, that's able to answers questions to specific dates. But it have to investigate in further tests with more different queries. 




## Inference Process
<p align="center">
  <img src="https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/9da0f59c-4b31-4fec-8e5b-17b3f0631474", width="80%">
</p>

### Extract entities with NER model
The idea to extract entities is to filter the documents (articles) by entities. So it's possible to filter the documents by date, club, and season if the query contains them. This reduce the scope to a closed set of data. 

There is a post process for date-entity to parse it to a `datetime` object. If the result of the parser isn't `None`, then the data will filter by the date with the condition `query_date <= published_date` with `published_date` as the released date of the document.

For the entities `club` and `season` it's an `in` condition, because the `news_keywords` are comma-seperated keywords. They can split and we can search for the `club` and `season` values in the list of keywords.
  
### Ranking by Metadata
The crawled articles contains an attribute `news_keywords`. This attribute is filled with keywords comma-separated. In the ranking by metadata, the process take the extracted entity-values from the query and calculate a score as follows:
```math
score = \frac{1}{quantity\ of\ keywords}
```
The more keywords are in the field `news_keywords`, the more irrelevant the individual extracted keyword will be. 
### Time-weighted ranking
The weigthed ranking is based on the different between the extracted and parsed date from the query:
```math
difference\ [hours] = \frac{|query\ datetime - published\ datetime|}{3600}
```
Further the higher the `decay_rate` is, the more irrelevant are the articles that lie more in the past or in the future:
```math
score = (1 - decay\_rate)^{difference\ [hours]}
```
A `decay_rate` of zero means, that the time has no impact on the ranking. The behaivor of the ranking of this case wasn't part of the development until now. 
### Similarity Search
For similarity search the query and the articles are transform in embeddings vector. Further, the cosine similarity will calculate:
```math
score = cos(\theta) = \frac{A \cdot B}{||A|| \cdot ||B||}
```
where `A` and `B` represent the embeddings of the query and document.
In the project `all-mpnet-base-v2`is used to generate the embeddings for both, the query and the articels. Several studies shows, that removing stopwords can improve the similarity search. Therefore there is step to remove stopwords, if it's desired.

### BM25
The Best Match 25 (BM25) is a ranking function, which used in information retrieval. BM25 extends the principles of TF-IDF, while both compute the relevance of a document, but BM25 consider other factors like document length and the average length of documents in collection. BM25 calculates for each document $d$ in a collection of $D$ the score as follows:

```math
\text{score}(d,q) = \epsilon \cdot \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{l(d)}{\text{avg(l(D))}}\right)}
```
There is an $\epsilon$ to prevent negative $IDF$ scores. [See BM25 implementation](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py).
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
That presents, that we loss some information in RRF. 

A solution is to normalize the scores of the ranking/retrieving methods between zero and one. Then, the scores can sum.
### Long-context re-ranking
The long-context re-ranking is based on the paper [lost in middle](https://arxiv.org/abs/2307.03172). The paper shows well, that models are better at using relevant information that occurs at the very beginning or end of its input context, and performance degrades significantly when models must access and use information located in the middle of its input context. That means for the re-ranking, that retrieved documents have to be in a specific order as the following graph shows:
<p align="center">
  <img src="https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/c2c2cd86-5b8a-44ea-9fd4-6b44bd3a0891", width="40%">
</p>
This graph shows, that the fourth relevant document has to be on the last position. Based on the graph the long-context re-ranker was implemented in the project. 

# RAG Evaluation
After we passed different queries to the inference process, we evaluate the RAG pipeline by RAGAS and human. Additional we add *langchain criterea*. 

## RAGAS
RAGAS is a framework for the following metrics. They are model-based metrics. You need query, answer and context to compute the metrics. The following based on the [RAGAS paper](https://arxiv.org/abs/2309.15217)
### Faithfulness
First, RAGAS prompt the evaluation model to generate statements based on the query. In a further prompt, the model verdicts, if statement is supoorted or not. The ratio is the faitfulness score 
```math
F = \frac{Number\ of\ statements\ that\ were\ supported}{Total\ number\ of\ statements}
```
<ins>Idea:</ins> The idea is, that the answer should be ground in the given context. This is important to avoid hallucinations and that the retrieved context can act as a justification for the generated answer. The score is between $0$ and $1$ and higher score is better.

### Answer Relevance
To compute the answer relevance the model generates questions for the given answer. Next, we calculate the cosine-similarity between the embeddings of each question $q_i$ and the original query:
```math
AR = \frac{1}{n} \cdot \sum_{i=1}^{n}sim(q,q_i)
```
<ins>Idea:</ins>Does the generated answer adress the actual question/query, that was provided? The more higher the score, the better the answer adresses to the question/query. The score is between $0$ and $1$ and higher score is better.

### Context Relevance
The first step to calculate the context relevance is to split the provided context into sentences. Next, the evaluation model picks the relevant sentences and in last step the score will calculate as follows:
```math
CR = \frac{Number\ of\ extracted\ sentences}{Total\ number\ of\ sentences\ in\ context}
```
<ins>Idea:</ins> The retrieved context should be focused and containing as little irrelevant information as possible. This is important, if long context passages are retrieved to the LLM. The score is between $0$ and $1$ and higher score is better.


### Limitations
The project shows the limitations in the model-based approach, because RAGAS is very prone for wrong generations. If the model isn't complex enough the original [RAGAS library](https://github.com/explodinggradients/ragas) fails and the prompts and the extraction of information out of the prompt have to adapt. Further, less complex models oft vary in the model output and then the approach is very inrobust and unreliable.
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
Therefore we have to note, that is a subjectiv evaluation. In NLP common human-annotaded datasets are annotaded by more than five different people.
# Experiments
The goal is to find the methods, which have the best inference result. 
There are three setting points in the execution:
- Retrieving methods (four options and sixteen combinations)
- Chunk-size (size/overlap): (128/42), (256/85), (508/170)
- Generation model: [Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [Sauerkraut Mixtral 8x7b](https://huggingface.co/VAGOsolutions/SauerkrautLM-Mixtral-8x7B-Instruct), [Sauerkraut SOLAR](https://huggingface.co/VAGOsolutions/SauerkrautLM-SOLAR-Instruct)
All of them would combine and test and then evaluate by human and RAGAS. There are eleveen different questions/prompt in the test.

## Results
The result shows different aspect. 

In the following plot, there is the mean score of each retrieving methods scored by human. The plot shows, if no method is specified that closely matches the expected behaviour.
![image](https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/08a27839-8ef3-447a-8550-6db7401ef56b)

The graphic shows that the metadata ranking method, when combined with other ranking methods, achieves the highest scores. This is explained by the fact, that the metadata ranking method works with the extracted entities from the query and the documents are filtered by the entities before. The filtering by entities achieves the best result in human evaluation as also shown in the following graphic _Comparison of Llama-2-13b and Sauerkraut Mixtral 8x7B by Retrieving Methods and Scores_. The best results that excepted metadata ranking is similarity and similarity combined with time-weigthed method. 

The following barchart shows the amount of scores per retrieving combinations. 
![image](https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/a4eb7d05-b085-4bbc-b81f-cc173c044e04)

The above two plots shows, that the bahavior of the approach isn't match with the expected bahavior of the prompter. However, this depends on the query as the follwing graphic shows.

![image](https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/238b2bb6-0fa7-4ecd-8934-23600b7c1367)


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
