# Propagation Features Analysis
## Features
- Follow the proposed prapagation analysis in *Hierarchical Propagation Networks for Fake News Detection: Investigation and Exploitation* (Shu et al, ICWSM 2020) 
- Some features cannot be computed in our case since the data is different

| Feature Code | Description |
|:------------:|:-----------|
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
## Dataset
- GeorgiaTech
- COVID-Hate
## Propagation Division
- In order to compare features between hateful and non-hateful propagations, we need to firt define hateful and non-hateful propagations 
- Two strategies for division
    - Based on the label of source node, i.e. the label of the propagation is dicided by the label of its source node
    - Based on the ratio of hateful nodes, i.e. the label of the propagation is dicided by the ratio of hateful nodes in it
## Analysis Result
### GeorgiaTech Dataset
#### Divided by label of source node
- The number of positive(hateful source tweet) and negative(non-hateful source tweet) propagations
	- Positive: 6538
	- Negative: 212252

| Feature Code | P.Mean | P.Max | P.Min | N.Mean | N.Max | N.Min |
|:------------:|:------:|:-----:|:-----:|:------:|:-----:|:-----:|
|S1| 1.08 | 7 | 1 | 1.07| 10| 1|
|S2| 5.13 | 4825 | 2 | 3.89| 20014 | 2 |
|S3| 3.60 | 4692 | 1 | 2.67 | 14819 | 1 |
|S5| 0.05| 10 | 0 | 0.04| 9 | 0 |
|T1| 826.41 | 76045.45 | 0 | 645.65 | 26253.45 | 0 |
|T2| 1316.63 | 115670.22 | 0 | 990.83 | 126253.45 | 0 |
|T3| 26.63 | 60531.15 | 0 | 9.16 | 98136.23 | 0 |
|T8| 615.574674 | 76045.45 | 0 | 513.38 | 126253.45 | 0 |

#### Divided by ratio of hateful tweets
- Threshold: 0.3
- The number of positive(hateful ratio higher than threshold) and negative(hateful ratio lower than threshold) propagations
	- Positive: 12804
	- Negative: 205986


| Feature Code | P.Mean | P.Max | P.Min | N.Mean | N.Max | N.Min |
|:------------:|:------:|:-----:|:-----:|:------:|:-----:|:-----:|
|S1| 1.06 | 7 | 1 | 1.07| 10| 1|
|S2| 2.92 | 557 | 2 | 3.99 | 20014 | 2 |
|S3| 1.83 | 551 | 1 | 2.75 | 14819 | 1 |
|S5| 0.05| 3 | 0 | 0.04| 10 | 0 |
|T1| 769.80 | 110938.25 | 0 | 643.36 | 126253.45 | 0 |
|T2| 1074.53 | 115670.22 | 0 | 995.97 | 126253.45 | 0 |
|T3| 28.69 | 98136.23 | 0 | 8.50 | 75870.12 | 0 |
|T8| 613.88 | 110938.25 | 0 | 510.37 | 126253.45 | 0 |

### COVID-Hate Dataset
#### Divided by label of source node
- The number of positive(hateful source tweet) and negative(non-hateful source tweet) propagations
	- Positive: 77430
	- Negative: 590652

| Feature Code | P.Mean | P.Max | P.Min | N.Mean | N.Max | N.Min |
|:------------:|:------:|:-----:|:-----:|:------:|:-----:|:-----:|
|S1| 1.05 | 9 | 1 | 1.06| 34| 1|
|S2| 2.74 | 1020 | 2 | 2.92| 2822 | 2 |
|S3| 1.71 | 1010 | 1 | 1.87 | 2752 | 1 |
|S5| 0.005| 5 | 0 | 0.008| 24 | 0 |
|T1| 312.65 | 43085.52 | 0 | 391.75 | 44402.12 | 0 |
|T2| 1316.63 | 115670.22 | 0 | 990.83 | 126253.45 | 0 |
|T3| 0.23 | 26358.97 | 0 | 0.20 | 31332.08 | 0 |
|T8| 252.08 | 43085.51 | 0 | 308.19 | 44402.11 | 0 |

#### Divided by ratio of hateful tweets
- Threshold: 0.3
- The number of positive(hateful ratio higher than threshold) and negative(hateful ratio lower than threshold) propagations
	- Positive: 105487
	- Negative: 562595


| Feature Code | P.Mean | P.Max | P.Min | N.Mean | N.Max | N.Min |
|:------------:|:------:|:-----:|:-----:|:------:|:-----:|:-----:|
|S1| 1.06 | 10 | 1 | 1.05| 34| 1|
|S2| 2.88 | 1536 | 2 | 2.91 | 2822 | 2 |
|S3| 1.83 | 1211 | 1 | 1.86 | 2752 | 1 |
|S5| 0.006| 5 | 0 | 0.008 | 24 | 0 |
|T1| 316.31 | 43085.51 | 0 | 395.00 | 44402.12 | 0 |
|T2| 446.27 | 43102.85 | 0 | 561.66 | 45533.93 | 0 |
|T3| 0.25 | 26358.96 | 0 | 0.20 | 31332.08 | 0 |
|T8| 248.83 | 43085.51 | 0 | 311.60 | 44402.11 | 0 |
