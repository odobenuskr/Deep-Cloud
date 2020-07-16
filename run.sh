#!bin/bash
MODEL_TYPE=$1
DATASET=$2
MODEL=$3
PROF_MODE=$4
OPTIMIZER=$5
INSTANCE_TYPE=$6

JOB_DIR="/home/ubuntu/Deep-Cloud/job/$MODEL_TYPE/$DATASET-$MODEL.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"

# Get profile result at half of 2epoch
sudo -i -u root bash << EOF

python3.6 $JOB_DIR --batch_size 16 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR --batch_size 32 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR --batch_size 64 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR --batch_size 128 --prof_start_batch 500 --prof_end_batch 520
sleep 3

python3.6 $JOB_DIR --batch_size 256 --prof_start_batch 500 --prof_end_batch 520
sleep 3

EOF

# python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 16 --prof_start_batch 5625 --prof_end_batch 5626
# sleep 3

# python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 32 --prof_start_batch 2812 --prof_end_batch 2813
# sleep 3

# python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 64 --prof_start_batch 1405 --prof_end_batch 1406
# sleep 3

# python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 128 --prof_start_batch 702 --prof_end_batch 703
# sleep 3

# python3.6 /home/ubuntu/Deep-Cloud/job/$JOB_NAME --batch_size 256 --prof_start_batch 351 --prof_end_batch 352
# sleep 3