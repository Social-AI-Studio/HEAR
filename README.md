# Early Prediction of Hate Speech Propagation
Code for IAAA 2021 Paper: "Early Prediction of Hate Speech Propagation" <p>
<img src="figure/framework.jpg" width="1200">
</p>

## Data
The detail preprocessing of data is in `implmentation/build_propagation_level_data.py` and `implementation/generate_fold_data.py`.
The input data are two files `nodes.json` and `edges.json` stored in one folder. The required data format is as below:
```
# nodes.json
{
  TWEET_ID:
  {
    "user_id": USER_ID,
    "text": TWEET_CONTENT,
     "timestamp": TIMESTAMP,
     "label": HATEFUL_LABEL
   },
   ......
}

```
```
# edges.json
[
  [
    TWEET_ID_from, TWEET_ID_to
  ],
  ......
]
```
Follow the preprocessing pipeline the generated data can be loaded by PyTorch dataset implemented in `implementation/dataset.py`.
## Our model:HEAR
The PyTorch implementation of our model is provided in `implementation/model.py`, once you prepared the required `nodes.json` and `edges.json`, you can modify `implementation/run.sh` to change the dataset path and run the whole pipeline:
```
sh run.sh
```
or if you have already processed the data:
```
python main.py -d YOUR_DATASET_PATH -e TRAIN_EPOCH_NUM -k FOLD_NUM
```
## Citation
If you find this paper useful, please cite following reference:
```
  @article{hear,
  title={Early Prediction of Hate Speech Propagation},
  author={Lin, Ken-Yu and Lee, Roy Ka-Wei and Gao, Wei and Peng, Wen-Chih}
  journal={International Workshop on Intelligence-Augmented Anomaly Analytics},
  year={2021}
}
```
