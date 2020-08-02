# Analysis on COVID19 Dataset

## Public Data
* tweet text
* tweet created time
* label
    * perspective api
    * lexicon based

## Raw Data
* All tweet status
* Not yet filtered
    * some non-English tweets

## Graph-based Analysis
### Tweet-based graph
#### Setting
* node: tweet
* edge: tweet relation
    * reply
    * retweet
    * quote
    
#### Analysis
* degree centrality
    * find nodes with highest degree centrality
    * few hateful tweets contains in these big component
* hate component
    * components with most hateful tweets

#### Problem
* component too small
* graph shape - radial
    * may be normal

### User-based graph
#### Setting
* node: user
* edge: follow

#### Problem
* no user information


## Time-based Analysis

### Hate through time
* relation between hate and time
* critical time point
    * may indicate some event that trigger lots of hate speech
* method
    * plot hate ratio with time
    * time scale is the key problem

### Topic evolvement
* topic evolve with time as well
* some topics may have higher relation to hate speech
* use time-hate relation to find critical topics

#### Analaysis
* first use Twitter-LDA to model topics in the data
* topic-time relation and hate-time relation

#### Problem
* Twitter-LDA model each user's interest in different topics (distribution)
    * most of users in data post one or two tweets
* the data is intentionally collected to have strong relation to COVID19


```python

```
