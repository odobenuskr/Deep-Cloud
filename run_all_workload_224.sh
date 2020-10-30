#!bin/bash
INSTANCE_TYPE=$1
PROF_MODE=$2
BATCH_SIZE=$3

# Define 46 Different DL Jobs
# LeNet5
JOB_DIR1="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model LeNet5 --dataset 224 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# VGGSmall
JOB_DIR2="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model VGGSmall --dataset 224 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# VGG11
JOB_DIR3="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model VGG11 --dataset 224 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# VGG13
JOB_DIR4="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model VGG13 --dataset 224 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# VGG16
JOB_DIR5="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model VGG16 --dataset 224 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# VGG19
JOB_DIR6="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model VGG19 --dataset 224 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# ResNetSmall
JOB_DIR7="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model ResNetSmall --dataset 224 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# ResNet18
JOB_DIR8="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model ResNet18 --dataset 224 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# ResNet34
JOB_DIR9="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model ResNet34 --dataset 224 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# ResNet50
JOB_DIR10="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py --model ResNet50 --dataset 224 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE"

# Get profile result
sudo -i -u root bash << EOF

python3.6 $JOB_DIR1 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR2 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR3 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR4 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR5 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR6 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR7 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR8 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR9 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR10 --batch_size $BATCH_SIZE
sleep 2

EOF

