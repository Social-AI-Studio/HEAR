# modify configs here
fold_num=5
label_days=7
epochs=20
dataset='test_dataset/' # modify here
magnitude_file='wordembed.magnitude' # modify here

python build_propagation_level_data.py -k $label_days -d $dataset
python generate_fold_data.py -k $fold_num -d $dataset -m $magnitude_file
python generate_node_classification_data.py -k $fold_num -d $dataset -m $magnitude_file
python main.py -k $fold_num -d $dataset -e $epochs
