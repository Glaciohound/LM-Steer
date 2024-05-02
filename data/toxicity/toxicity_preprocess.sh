#download the data from https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data (it requires creating an account on Kaggle)
# echo "make sure you have downloaded the data from https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data (all_data.csv) and placed it in data/toxicity/jigsaw-unintended-bias-in-toxicity-classification"

DATADIR=$1
#processing the data
echo "preprocessing the data"
python -u data/toxicity/create_jigsaw_toxicity_data.py $DATADIR

N=$(wc -l ${DATADIR}/toxicity_eq0.jsonl | cut -d ' ' -f1)
n=$(wc -l ${DATADIR}/toxicity_gte0.5.jsonl | cut -d ' ' -f1)
python -u data/toxicity/random_sample.py ${DATADIR}/toxicity_eq0.jsonl ${DATADIR}/toxicity_eq0_subsample.jsonl $N $n


TEST=2000
DEV=2000
TRAIN=$(($n-$DEV-$TEST))
python -u data/toxicity/split_train_dev_test.py ${DATADIR}/toxicity_eq0_subsample.jsonl 0 $TRAIN $DEV $TEST $DATADIR
python -u data/toxicity/split_train_dev_test.py ${DATADIR}/toxicity_gte0.5.jsonl 1 $TRAIN $DEV $TEST $DATADIR
