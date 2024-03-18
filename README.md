# Datenanalyse_UN_RAG
## Abstract
This repository shows an approach to interact with scraped sportnews. The dataset includes about 60,000 articles. www.reviersport.de published the articles.

The main goal is to pass queries to a large language model. In particual, the queries focus on details related to events and dates and they have to answers excatly. Therefore, serveral retrieving methods would test and combine with eachother to reach the goal. The inference will verdict by the developer and the use of the RAGAS framework is in testing. 

In the first try `Llama-2-13B` is used as generation model and `all-mpnet-base-v2` as embeddings model. Addtional tests with `Mistral-7-B` and other models are planned.

A NER-model extracts entities out of the query to use the values for pre-filtering the documents by date, leauge and club. This approach take into account to concentrate to the best fitting articles. Further, it would combine the follwoing ranking/retriever methods: ranking by metadata, time-weighted retriever, similarity search and BM25 algorithm. 

Today's result shows the best answer is given by ***pre-filtering of documents by entities and BM25 algorithm***. To the query `Wie hat Leverkusen am 2023-11-25 in der 1.Bundesliga gespielt?` is the answers of Llama-2-13B `Leverkusen hat am 2023-11-25 in der 1.Bundesliga gegen Werder Bremen gewonnen mit 3:0 (2:0). Es war ein hervorragendes Spiel und alle haben draufgefreut. Danke für die Frage!`. The answers correct and the approach shows succesful, that's able to answers questions to specific dates. But it have to investigate in further tests with more different queries. 




## Inference Process
![image](https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/9da0f59c-4b31-4fec-8e5b-17b3f0631474)

## Ranking by Metadata
The crawled articles contains an attribute `news_keywords`. This attribute is filled with keywords comma-separated. In the ranking by metadata, the process take the extracted entity-values from the query and calculate a score as follows:
```math
score = \frac{1}{quantity\ of\ keywords}
```
The more keywords are in the field `news_keywords`, the more irrelevant the individual extracted keyword will be. 
## Time-weighted ranking
The weigthed ranking is based on the different between the extracted and parsed date from the query:
```math
difference\ [hours] = \frac{|query\ datetime - published\ datetime|}{3600}
```
Further the higher the `decay_rate` is, the more irrelevant are the articles that lie more in the past or in the future:
```math
score = (1 - decay_rate)^{difference\ [hours]}
```
A `decay_rate` of zero means, that the time has no impact on the ranking. The behaivor of the ranking of this case wasn't part of the development until now. 
## Similarity Search
For similarity search the query and the articles are transform in embeddings vector. Further, the cosine similarity will calculate:
```math
score = cos(\theta) = \frac{A \cdot B}{||A|| \cdot ||B||}
```
where `A` and `B` represent the embeddings of the query and document.
In the project `all-mpnet-base-v2`is used to generate the embeddings for both, the query and the articels. 
## BM25

## Reciprocal Rank Fusion (RRF)
Reciprocal Rank Fusion is a simple method for combining several difference ranking methods in field of information retrieval. RRF simply sorts the documents according to the rank of each document in all retriever sets:
```math
score (d\in D) = \sum_{r \in R}{} \frac{1}{k+r(d)}
```
with $k = 60$. A higher $k$ relativizes the rank-distances. Further,  the spatial distance between the scores of the first and second rank doesn't cosider by RRF. As instance, in the similarity search the first rank has a score of 0.9 and the second rank a score of 0.3. If we leave $k$ out of consideration and calculate the RRF-score for the first and second document, we see that the difference between $\frac{1}{1}= 1$ for the first rank and $\frac{1}{2} = 0.5$ for the second rank is smaller than the difference of the cosine similarity between the first and second rank. That shows, we loss some information in RRF. 

A solution is to normalize the scores of the ranking/retrieving mehtods between zero and one. Then, the scores can sum.
## Long-context re-ranking
The long-context re-ranking is based on the paper [lost in middle](https://arxiv.org/abs/2307.03172). The paper shows well, that models are better at using relevant information that occurs at the very beginning or end of its input context, and performance degrades significantly when models must access and use information located in the middle of its input context. That means for the re-ranking, that retrieved documents have to be in a specific order as the following graph shows:
<p align="center">
  <img src="https://github.com/vanny132/Datenanalyse_UN_RAG/assets/102876328/c2c2cd86-5b8a-44ea-9fd4-6b44bd3a0891", width="40%">
</p>
This graph shows, that the fourth relevant document has to be on the last position. Based on the graph the long-context re-ranker was implemented in the project. 
