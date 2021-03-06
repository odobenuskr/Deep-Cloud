#!bin/bash
MODEL_TYPE=$1
DATASET=$2
MODEL=$3
PROF_MODE=$4
OPTIMIZER=$5
INSTANCE_TYPE=$6

PROF_POINT=1.5

JOB_DIR="/home/ubuntu/Deep-Cloud/job/$MODEL_TYPE/$DATASET-$MODEL.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"

# Get profile result at half of 2epoch
sudo -i -u root bash << EOF

python3.6 $JOB_DIR --batch_size 16 
sleep 3

python3.6 $JOB_DIR --batch_size 32
sleep 3

python3.6 $JOB_DIR --batch_size 64
sleep 3

python3.6 $JOB_DIR --batch_size 128
sleep 3

python3.6 $JOB_DIR --batch_size 256
sleep 3

EOF
