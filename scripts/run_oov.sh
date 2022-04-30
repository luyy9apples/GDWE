DATADIR=../corpus/nyt/forClassification
EMBDIR=../g_yskip_i/model/e0.2_c5_N4_i1_b1000_E0.2_J0.05_A1.25
THRESH=100
RESULT=result_oov/kg-t-${THRESH} 
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
LEN_SEQ=256
CUDA=0

python -u classifier_oov.py --datadir $DATADIR --text t_sep_class_text_oov_${THRESH} --label t_sep_class_label_oov_${THRESH} --embdir $EMBDIR --result $RESULT --start $START_YEAR --end $END_YEAR --classes $CLASSES_NUM --dropout $DROPOUT --embsize $EMBSIZE --hidden $HIDDEN_DIM --layers $LAYERS_NUM --eta $ETA --epoch $EPOCH --batch $BATCH_SIZE --seq $LEN_SEQ --cuda $CUDA --test
