#!bin/bash
MODEL_TYPE=$1
DATASET=$2
MODEL=$3
PROF_MODE=$4
OPTIMIZER=$5
INSTANCE_TYPE=$6

# JOB_DIR="/home/ubuntu/Deep-Cloud/job/$MODEL_TYPE/$DATASET-$MODEL.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"

# # Get profile result at half of 2epoch
# sudo -i -u root bash << EOF

# python3.6 $JOB_DIR --batch_size 16 --prof_start_batch 500 --prof_end_batch 520
# sleep 3

# python3.6 $JOB_DIR --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
# sleep 3

# python3.6 $JOB_DIR --batch_size 64 --prof_start_batch 500 --prof_end_batch 520
# sleep 3

# python3.6 $JOB_DIR --batch_size 128 --prof_start_batch 500 --prof_end_batch 520
# sleep 3

# python3.6 $JOB_DIR --batch_size 256 --prof_start_batch 500 --prof_end_batch 520
# sleep 3

# EOF

JOB_DIR1="/home/ubuntu/Deep-Cloud/job/rnn/babi-rnn.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"
JOB_DIR2="/home/ubuntu/Deep-Cloud/job/rnn/babi-lstm.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"
JOB_DIR3="/home/ubuntu/Deep-Cloud/job/rnn/babi-bilstm.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"
JOB_DIR4="/home/ubuntu/Deep-Cloud/job/rnn/imdb-rnn.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"
JOB_DIR5="/home/ubuntu/Deep-Cloud/job/rnn/imdb-lstm.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"
JOB_DIR6="/home/ubuntu/Deep-Cloud/job/rnn/imdb-bilstm.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"
JOB_DIR7="/home/ubuntu/Deep-Cloud/job/rnn/reuter-rnn.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"
JOB_DIR8="/home/ubuntu/Deep-Cloud/job/rnn/reuter-lstm.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"
JOB_DIR9="/home/ubuntu/Deep-Cloud/job/rnn/reuter-bilstm.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"

sudo -i -u root bash << EOF

python3.6 $JOB_DIR1 --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR2 --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR3 --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR4 --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR5 --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR6 --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR7 --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR8 --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR9 --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
sleep 3

EOF
