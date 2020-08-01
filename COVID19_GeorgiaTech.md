# Note on GeorgiaTech COVID19 Dataset

## Dataset Structure
    ├── classifier              # High-level classifier
    │   └── models              # Pre-trained Logistic Regression models (pickled)
    ├── tweet_data              # Human and machine labeled COVID-HATE data
    │   ├── all_tweet_ids.csv   # IDs of all 30,929,269 English tweets
    │   ├── annotated           # Human annotations for 2,319 English tweets
    │   ├── classified          # Machine classifications for 30,929,269 tweets 
    │   └── geolocated          # Location information at different levels of granularity
    ├── social_network          # This has the Twitter social network
    │   ├── egonet_nodes.csv    # List of nodes for which the ego-network was collected
    │   ├── edges.csv           # Edges in the network
    └── README.md

## Dataset Description
### Tweets
* Total 3000438 tweets
* Features
    * created time
    * user_id
    * ***NO TWEET TEXT PROVIDED***
        * need to be crawled via Twitter API manually

### Annotations
#### Annotation Process
1. 2319 tweets annotated manually
2. A classfier  wtrainedith the annotated tweets
    * hashtag + BERT embedding
3. All tweets classified with the classifier

#### Labels

* For manually annotated tweets
    * 4 different categories
        * hate, counterhate, neutral, non-asian aggression
* For classified tweets
    * 3 dimensional score
        * hate, counterhate, neutral
    * aggregated into 4 categories
        * hate, counterhate, neutral, other

### Social Network
* ego-network of random sampled user
    * followers and followees

### Geoinfo
* geological information of tweets (if available)

## Analysis Deisgn
### Time-based
### Relation-based
### User-based

## TODO
* ~~understand dataset~~
* crawl tweets text
* analysis on annotated tweets
* analysis on all tweets

## Problem
* deleted tweets (can't access via api)
* crawling problem (api limit)
* the tweet ids of annotated tweets are not provided
    * unless we crawl all tweets we don't know their tids