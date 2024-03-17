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

## Time-weighted ranking

## Similarity Search

## BM25

## Reciprocal Rank Fusion (RRF)

## Long-context re-ranking
