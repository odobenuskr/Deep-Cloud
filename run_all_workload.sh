#!bin/bash
INSTANCE_TYPE=$1
PROF_MODE=$2
BATCH_SIZE=$3

# Define 46 Different DL Jobs
# LeNet5
JOB_DIR1="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model LeNet5 --dataset 16 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR2="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model LeNet5 --dataset 32 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR3="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model LeNet5 --dataset 64 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR4="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model LeNet5 --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR5="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model LeNet5 --dataset 256 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE

# VGGSmall
JOB_DIR6="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGGSmall --dataset 16 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR7="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGGSmall --dataset 32 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR8="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGGSmall --dataset 64 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR9="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGGSmall --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR10="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGGSmall --dataset 256 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE

# VGG11
JOB_DIR11="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG11 --dataset 32 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR12="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG11 --dataset 64 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR13="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG11 --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR14="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG11 --dataset 256 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE

# VGG13
JOB_DIR15="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG13 --dataset 32 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR16="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG13 --dataset 64 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR17="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG13 --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR18="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG13 --dataset 256 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE

# VGG16
JOB_DIR19="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG16 --dataset 32 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR20="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG16 --dataset 64 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR21="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG16 --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR22="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG16 --dataset 256 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE

# VGG19
JOB_DIR23="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG19 --dataset 32 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR24="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG19 --dataset 64 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR25="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG19 --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR26="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model VGG19 --dataset 256 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE

# ResNetSmall
JOB_DIR27="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNetSmall --dataset 16 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR28="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNetSmall --dataset 32 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR29="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNetSmall --dataset 64 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR30="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNetSmall --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR31="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNetSmall --dataset 256 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE

# ResNet18
JOB_DIR32="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet18 --dataset 16 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR33="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet18 --dataset 32 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR34="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet18 --dataset 64 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR35="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet18 --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR36="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet18 --dataset 256 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE

# ResNet34
JOB_DIR37="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet34 --dataset 16 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR38="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet34 --dataset 32 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR39="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet34 --dataset 64 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR40="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet34 --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR41="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet34 --dataset 256 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE

# ResNet50
JOB_DIR42="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet50 --dataset 16 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR43="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet50 --dataset 32 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR44="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet50 --dataset 64 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR45="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet50 --dataset 128 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE
JOB_DIR46="/home/ubuntu/Deep-Cloud/job/workload/train_workload.py" --model ResNet50 --dataset 256 --prof_or_latency $PROF_MODE --instance_type $INSTANCE_TYPE

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
python3.6 $JOB_DIR11 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR12 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR13 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR14 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR15 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR16 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR17 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR18 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR19 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR20 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR21 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR22 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR23 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR24 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR25 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR26 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR27 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR28 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR29 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR30 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR31 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR32 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR33 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR34 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR35 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR36 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR37 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR38 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR39 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR40 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR41 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR42 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR43 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR44 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR45 --batch_size $BATCH_SIZE
sleep 2
python3.6 $JOB_DIR46 --batch_size $BATCH_SIZE
sleep 2

EOF
