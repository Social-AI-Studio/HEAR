# Hate Speech Diffusion and Influence in Online Social Networks 

### Tasks Remain
1. Annotation using lexicon sets (done) and PerpectiveAPI (in progress) then test annotation agreement among different methods
2. Read more related papers to find available methods
3. Build propagation dataset from half-prepared data
4. Test known methods (RvNN, Cascade-LSTM etc.)
---
### Proposed Model
Proposed model framework diagram. Briefly describe the novelty in this proposed model in point form.

### Data Collection and Annotation
Currently using COVID-HATE dataset provided by CLAWS lab, GeorgiaTech
#### Dataset Description
- After filtering (filter out non-English tweets), about 10 millions tweets
- Around 2000 annotated by human, other classified (using a classifier trained with the annotated data)
- We use the data in the range from Jan 15 to April 17
#### Dataset Characteristic
![](https://i.imgur.com/akRRd1M.png)
![](https://i.imgur.com/nTlkrKq.png)
![](https://i.imgur.com/aqgu7H7.png)
![](https://i.imgur.com/1pwBhp7.png)
- Generally speaking, the propagation graph is more complete than the original one
- Long propagation path mostly resulted by self-retweets
- Category ratio (classified)
	- Neutral: 8598572 
	- Other: 957474
	- Hate: 287938
	- Counterhate: 64235
### Empirical Analysis
#### Dataset Temporal Distribution
![](https://i.imgur.com/GBFl7jD.png)
![](https://i.imgur.com/xGtBAIU.png)
#### Annotation Agreement
- The annotation agreement of classified result and lexicon-based annotation
	- Negative agreement is kind of high (more than 90% both annotate as neutral)
	- Positive agreement is low (less than 1% both annotate as hateful)
- Annotation agreement of PerspectiveAPI
	- Annotating	
#### Hateful Posts Origin Distributions
**Origin Definitions**
- Retweet: refers to the hate tweet is retweeted from another hateful/non-hateful tweet.
- Quote: refers to the hate tweet is quoted from another hateful/non-hateful tweet.
- Reply: refers to the hate tweet is a reply to another hateful/non-hateful tweet.
- Source: refers to the hate tweet is an original tweet posted by the user.

|Origin | #Hateful_Posts | #Posts | Hateful Ratio | 
|:-----:|:------:|:---------------:|:------:|

#### Viral Hate-Induced Tweets and Hateful Influencers.
List the top 10 viral tweets that induced the most hateful tweets.
1. The top 10 components contain most hateful tweets
2. Find the tweet with highest centrality in the components

|Rank| TweetID | Tweet_Owner | Tweet_Text | #Hate_Tweets_Induced | 
|:--:|:-------:|:-----------:|:----------:|:--------------------:|
|1|1239685852093169664|25073877|The United States will be powerfully supporting those industries, like Airlines and others, that are particularly affected by the Chinese Virus. We will be stronger than ever before!|3513|
|2|1240243188708839424|25073877| I always treated the Chinese Virus very seriously, and have done a very good job from the beginning, including my very early decision to close the “borders” from China - against the wishes of almost all. Many lives were saved. The Fake News new narrative is disgraceful &amp; false!| 472|
|3|1241897485779468288|25073877| My friend (always there when I’ve needed him!), Senator @RandPaul, was just tested “positive” from the Chinese Virus. That is not good! He is strong and will get better. Just spoke to him and he was in good spirits.|246|
|4|1242989308736196608| 51310666|China has asked India not to use the term #ChineseVirus .. let’s not use chinese virus although it came China but we should not use the term Chinese virus .. if you agree with me pls don’t use the word Chinese virus| 219|
|5|1236821135964004352| 872148729184362497| 1. I am announcing that I, along with 3 of my senior staff, are officially under self-quarantine after sustained contact at CPAC with a person who has since been hospitalized with the Wuhan Virus.  My office will be closed for the week.| 182|
|6|1239954823178567680| 970207298| I've said it once &amp; I'll say it again loud enough for the @WhiteHouse, @FoxNews, &amp; everyone else to hear: coronavirus does not discriminate. Bigotry against people of Asian descent is unacceptable, un-American, &amp; harmful to our COVID-19 response efforts| 175| 
|7|1241770465820938240| 179525357| #BREAKING: Corona virus finally died today in india due to shock. https://t.co/M4kC1xCVqm| 174| 
|8|1241030189309726720| 3995778614| Once again president Trump starts the Coronavirus press conference calling it the Chinese virus.| 165|
|9|1240385571039539200| 429227921| My wife is from Taiwan, so my kids are half-Chinese. Because of racist assholes like @JohnCornyn &amp; @realDonaldTrump, their classmates are already blaming them for the virus &amp; asking if they eat bats. My 9 year old son even came home &amp; asked me if it was true. It's heart-breaking| 141|
|10|1241504277048238080| 29501253| We need to bring people together to fight Coronavirus.  Blaming China may seem like good politics, but it doesn’t solve anything, or mitigate the Trump Administration’s failures.  Calling it the “Chinese virus” only breeds disunity, discrimination and division.  Enough already.| 132|

#### Propagation Feature Statistics
- Follow the proposed prapagation analysis in *Hierarchical Propagation Networks for Fake News Detection: Investigation and Exploitation* (Shu et al, ICWSM 2020) 
- Some features cannot be computed in our case since the data is different
| Feature Code | Description |
|:------------:|:-----------:|
|S1|Propagation tree depth|
|S2|Number of nodes in propagation|
|S3|Max outdegree in the propagation|
|S5|Depth of node with max outdegree|
|T1|Average time diff between nodes (in mins)|
|T2|Time diff between first and last node (in mins)|
|T3|Time diff between source node and node with max outdegree (in mins)|
|T8|Time diff between source node and the first child (in mins)|
- In order to compare propagation feature statistics, we need to divide propagations into groups
	- Based on hatefulness of the source tweet, hate severity in the graph etc.
- In the original paper (Shu et al, 2020), the feature statistics of each group are reported in their min, max and mean values
	- As in our observation, the distribution of these statistics are distributed more similar to a log uniform distribution
	- Report mean values may not be meaningful, need to find other stats to report
##### Divided by label of source node

##### Experiments - Predicting Hate Speech
Designing some features based on our empirical findings to improve hate speech detection.



