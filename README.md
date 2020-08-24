# Hate Speech Diffusion and Influence in Online Social Networks 

### Tasks from 04.08.2020 Meeting
Work on the following tasks for next meeting:
1. Collect the retweets/quotes/reply for our COVID-19 dataset.
2. Update the hateful posts origin distributions.
3. Perform analysis on Viral Hate-Induced Tweets and Hateful Influencers.

---
### Proposed Model
Proposed model framework diagram. Briefly describe the novelty in this proposed model in point form.

### Data Collection and Annotation
Collect own covid-19 dataset and perform annotation on the collected dataset.

##### Dataset
* Total number of tweets: 7.5 millions
* Total number of edges (relation): 3.9 millions
	* Number of retweets: 1.6 millions
	* Number of quoted tweets: 1.5 millions
	* Number of reply tweets: 0.8 million

### Empirical Analysis
#### Hateful Ratio of Each Day
![](https://i.imgur.com/aQ2j4jf.png)
##### Hateful Posts Origin Distributions
**Origin Definitions**
- Retweet: refers to the hate tweet is retweeted from another hateful/non-hateful tweet.
- Quote: refers to the hate tweet is quoted from another hateful/non-hateful tweet.
- Reply: refers to the hate tweet is a reply to another hateful/non-hateful tweet.
- Source: refers to the hate tweet is an original tweet posted by the user.

|Origin | #Hateful_Posts | #Posts | Hateful Ratio | 
|:-----:|:------:|:---------------:|:------:|
| All | 958590  | 7530747 | 0.1273|
| Retweet | 189923  | 1610543 | 0.1179|
| Quote | 258634  | 1504289 | 0.1719|
| Reply | 137877  | 814307 | 0.1693|
| Source | 450457  | 4090634 | 0.1101|

##### Viral Hate-Induced Tweets and Hateful Influencers.
List the top 10 viral tweets that induced the most hateful tweets.

|Rank| TweetID | Tweet_Owner | Tweet_Text | #Hate_Tweets_Induced | Induced_Hate_Tweet_Text |
|:--:|:-------:|:-----------:|:----------:|:--------------------:|:-----------------------:|
|1   |         |             |            |                      |                         |
|2   |         |             |            |                      |                         |
|3   |         |             |            |                      |                         |
|4   |         |             |            |                      |                         |
|5   |         |             |            |                      |                         |
|6   |         |             |            |                      |                         |
|7   |         |             |            |                      |                         |
|8   |         |             |            |                      |                         |
|9   |         |             |            |                      |                         |
|10  |         |             |            |                      |                         |

List the top 10 user accounts that induced the most hateful tweets.

|Rank| Username | #Hate_Tweets_Induced | Induced_Hate_Tweet_Text |
|:--:|:--------:|:--------------------:|:-----------------------:|
|1   |          |                      |                         |
|2   |          |                      |                         |
|3   |          |                      |                         |
|4   |          |                      |                         |
|5   |          |                      |                         |
|6   |          |                      |                         |
|7   |          |                      |                         |
|8   |          |                      |                         |
|9   |          |                      |                         |
|10  |          |                      |                         |

##### Experiments - Predicting Hate Speech
Designing some features based on our empirical findings to improve hate speech detection.



