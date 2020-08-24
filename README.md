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
1. The top 10 components contain most hateful tweets
2. Find the tweet with highest centrality in the components

|Rank| TweetID | Tweet_Owner | Tweet_Text | #Hate_Tweets_Induced | 
|:--:|:-------:|:-----------:|:----------:|:--------------------:|
|1   |1240371160078000128|15012486|“If I get corona, I get corona. At the end of the day, I'm not gonna let it stop me from partying”: Spring breakers are still flocking to Miami, despite coronavirus warnings. https://t.co/KoYKI8zNDH https://t.co/rfPfea1LrC|59301|
|2   |1246520412894199808|7587032|Are people ignoring social distancing advice?  Britons have been spotted enjoying the warm weather over the weekend, despite government advice to stay at home amid the #coronavirus pandemic 👇  For the latest updates on #COVID19, visit: https://t.co/Gk97JUNAKD https://t.co/YDvEEm9hA0|2226|
|3   |1245112522748899329|254789059|In case you were wondering what the Corona Virus test its like. #CoronavirusUSA #coronavirustesting https://t.co/zdKUgMi1cm|2179|
|4   |1242894756415332352|21148293|He’s lying. I was sent to the #COVID19 isolation ward room in a major hospital ER from a separate urgent care facility after showing UNBEARABLY PAINFUL symptoms. The hospital couldn’t test me for #coronavirus because of CDC (Pence task force) restrictions. #TESTTESTTEST https://t.co/18fRiOBsdN https://t.co/0sU9fHu4r0|1131|
|5   |1244508599038009344|1141991448|Who r u trying to kill, Corona or humans? Migrant labourers and their families were forced to take bath in chemical solution upon their entry in Bareilly.  @Uppolice @bareillytraffic  @Benarasiyaa @shaileshNBT https://t.co/JVGSvGqONm|988|
|6   |1240388848552738826|1651522832|😱#Florida spring-breakers:  “If I get #corona, I get corona....I’m not going to let it stop me from parting.”  #Coronavirus “is really messing up my spring break.”  “This virus ain’t that serious.”  We’re “trying to get drunk before everything closes.”  https://t.co/NzHTdy03gs|925|
|7   |1251218655888592896|1546101560|Non-English|735|
|8   |1248684161055023106|252751061|Surgeon Gen. Jerome Adams calls on communities of color to adhere to #coronavirus advice from Trump’s task force: “If not for yourself, then for your abuela. Do it for your grand daddy. Do it for your big mama. Do it for your pop-pop.|685|
|9   |1248631268549394432|254098258|I’ve just been fitted for PPE and we’re about to go into an intensive care unit at Milton Keynes Hospital to witness the incredible efforts of medics treating Covid 19 patients. Please note we won’t be depleting the hospital’s PPE stock #COVID19 #coronavirus #nhs https://t.co/L20jDvjc8i|679|
|10  |1248329497897775105|457060718|The boss is in a better place. Such a relief. The country can breathe again  #COVID  #NHS https://t.co/Km9ER9GYxs|569|

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



