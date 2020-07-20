#!bin/bash
MODEL_TYPE=$1
DATASET=$2
MODEL=$3
PROF_MODE=$4
OPTIMIZER=$5
INSTANCE_TYPE=$6

PROF_POINT=1.5

# JOB_DIR="/home/ubuntu/Deep-Cloud/job/$MODEL_TYPE/$DATASET-$MODEL.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE"

# # Get profile result at half of 2epoch
# sudo -i -u root bash << EOF

# python3.6 $JOB_DIR --batch_size 16 --prof_point PROF_POINT
# sleep 3

# python3.6 $JOB_DIR --batch_size 32 --prof_point PROF_POINT
# sleep 3

# python3.6 $JOB_DIR --batch_size 64 --prof_point PROF_POINT
# sleep 3

# python3.6 $JOB_DIR --batch_size 128 --prof_point PROF_POINT
# sleep 3

# python3.6 $JOB_DIR --batch_size 256 --prof_point PROF_POINT
# sleep 3

# EOF

JOB_DIR1="/home/ubuntu/Deep-Cloud/job/cnn/mnist-lenet5.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"
JOB_DIR2="/home/ubuntu/Deep-Cloud/job/cnn/mnist-resnetsmall.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"
JOB_DIR3="/home/ubuntu/Deep-Cloud/job/cnn/mnist-vggsmall.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"
JOB_DIR4="/home/ubuntu/Deep-Cloud/job/cnn/fmnist-lenet5.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"
JOB_DIR5="/home/ubuntu/Deep-Cloud/job/cnn/fmnist-resnetsmall.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"
JOB_DIR6="/home/ubuntu/Deep-Cloud/job/cnn/fmnist-vggsmall.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"
JOB_DIR7="/home/ubuntu/Deep-Cloud/job/cnn/cifar10-lenet5.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"
JOB_DIR8="/home/ubuntu/Deep-Cloud/job/cnn/cifar10-resnetsmall.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"
JOB_DIR9="/home/ubuntu/Deep-Cloud/job/cnn/cifar10-vggsmall.py --optimizer $OPTIMIZER --prof_or_latency $PROF_MODE --prof_point $PROF_POINT"

# Get profile result at half of 2epoch
sudo -i -u root bash << EOF

python3.6 $JOB_DIR1 --batch_size 64
sleep 3
python3.6 $JOB_DIR2 --batch_size 64
sleep 3
python3.6 $JOB_DIR3 --batch_size 64
sleep 3
python3.6 $JOB_DIR4 --batch_size 64
sleep 3
python3.6 $JOB_DIR5 --batch_size 64
sleep 3
python3.6 $JOB_DIR6 --batch_size 64
sleep 3
python3.6 $JOB_DIR7 --batch_size 64
sleep 3
python3.6 $JOB_DIR8 --batch_size 64
sleep 3
python3.6 $JOB_DIR9 --batch_size 64
sleep 3

EOF
