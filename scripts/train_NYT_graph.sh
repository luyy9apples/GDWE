set -e

EMB_SIZE=50
WIN_SIZE=5
NEGATIVE=10
ALPHA=0.75
SUBSAMPLE=0.0001
UNI_SIZE=100000000
MAX_VOCAB_SIZE=100000
LR=0.2
SEED=191731

GRAPH_METHOD=0 # 3:not use graph
BN=1000
CN=5

# new params
KNOW_ALG=0 # 0:long+short 1:long 2:short
E_THRESH=0.2
N_THRESH=4
J_THRESH=0.05
AL=1.25

START_YEAR=1990
END_YEAR=2016

DN_DIR=./corpus/nyt/cond_data/
MODEL_DIR=./model/e${LR}_c${CN}_N${N_THRESH}_i1_b${BN}_E${E_THRESH}_J${J_THRESH}_A${AL}/
if [ ! -d $MODEL_DIR ]; then
	mkdir $MODEL_DIR
fi

./src/yskip -d $EMB_SIZE -w $WIN_SIZE -n $NEGATIVE -a $ALPHA -s $SUBSAMPLE -u $UNI_SIZE -m $MAX_VOCAB_SIZE -e $LR -r $SEED -g $GRAPH_METHOD -i 1 -T 1 -b $BN -k $KNOW_ALG -c $CN -E $E_THRESH -N $N_THRESH -J $J_THRESH -A $AL -y $START_YEAR -Y $END_YEAR $DN_DIR $MODEL_DIR