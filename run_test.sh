#!bin/bash
INSTANCE_TYPE=$1
PROF_MODE=$2
BATCH_SIZE=$3

# VGG16
JOB_DIR1="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model VGG16 --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# Get profile result
sudo -i -u root bash << EOF
python3.6 $JOB_DIR1 --batch_size $BATCH_SIZE
sleep 2
EOF
