DATADIR=./corpus/nyt/forClassification
EMBDIR=./model/e0.2_c5_N4_i1_b1000_E0.2_J0.05_A1.25
RESULT=classification_result/kg 
START_YEAR=1990
END_YEAR=2016
CLASSES_NUM=7
DROPOUT=0.1
EMBSIZE=50
HIDDEN_DIM=50
LAYERS_NUM=2
ETA=0.01
EPOCH=10
BATCH_SIZE=16
LEN_SEQ=128
CUDA=0

python -u classifier_m.py --datadir $DATADIR --embdir $EMBDIR --result $RESULT --start $START_YEAR --end $END_YEAR --classes $CLASSES_NUM --dropout $DROPOUT --embsize $EMBSIZE --hidden $HIDDEN_DIM --layers $LAYERS_NUM --eta $ETA --epoch $EPOCH --batch $BATCH_SIZE --seq $LEN_SEQ --cuda $CUDA --test