#!bin/bash
MODEL_TYPE=$1
DATASET=$2
MODEL=$3
PROF_MODE=$4
OPTIMIZER=$5
INSTANCE_TYPE=$6

PROF_POINT=1.5

# RNN Job Test
JOB_DIR3="/home/ubuntu/Deep-Cloud/job/rnn/babi-bilstm.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"
JOB_DIR6="/home/ubuntu/Deep-Cloud/job/rnn/imdb-bilstm.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"
JOB_DIR9="/home/ubuntu/Deep-Cloud/job/rnn/reuters-bilstm.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"

# Get profile result
sudo -i -u root bash << EOF

python3.6 $JOB_DIR3 --batch_size 16
sleep 3
python3.6 $JOB_DIR3 --batch_size 32
sleep 3
python3.6 $JOB_DIR3 --batch_size 64
sleep 3
python3.6 $JOB_DIR3 --batch_size 128
sleep 3
python3.6 $JOB_DIR3 --batch_size 256
sleep 3

python3.6 $JOB_DIR6 --batch_size 16
sleep 3
python3.6 $JOB_DIR6 --batch_size 32
sleep 3
python3.6 $JOB_DIR6 --batch_size 64
sleep 3
python3.6 $JOB_DIR6 --batch_size 128
sleep 3
python3.6 $JOB_DIR6 --batch_size 256
sleep 3

python3.6 $JOB_DIR9 --batch_size 16
sleep 3
python3.6 $JOB_DIR9 --batch_size 32
sleep 3
python3.6 $JOB_DIR9 --batch_size 64
sleep 3
python3.6 $JOB_DIR9 --batch_size 128
sleep 3
python3.6 $JOB_DIR9 --batch_size 256
sleep 3

EOF
