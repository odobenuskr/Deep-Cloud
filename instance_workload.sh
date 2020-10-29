#!/bin/bash

# Type of ec2 instance to run a deep learning job
INSTANCE_TYPE=$1

# Run with profiling or just measure latency(profiling/latency)
PROF_MODE=$2

LAUNCH_INFO=$(aws ec2 run-instances --image-id ami-abcd1234 --count 1 --instance-type $INSTANCE_TYPE \
--key-name my-key-pair --subnet-id subnet-abcd1234 --security-group-ids sg-abcd1234)

# Instance ID and Public DNS Parsing
sleep 60
INSTANCE_ID=$(echo $LAUNCH_INFO | jq -r '. | .Instances[0].InstanceId')
INSTANCE_PUB_DNS=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID | jq -r '. | .Reservations[0].Instances[0].PublicDnsName')
echo $INSTANCE_PUB_DNS

# Setting for Deep Learning
sleep 60
echo 'clone start'
ssh -o "StrictHostKeyChecking no" -i awspwd.pem ubuntu@$INSTANCE_PUB_DNS 'git clone https://github.com/odobenuskr/Deep-Cloud.git'
echo 'setting start'
ssh -i awspwd.pem -t ubuntu@$INSTANCE_PUB_DNS 'cd /home/ubuntu/Deep-Cloud/;sudo bash ./settings.sh'

# Run Experiments
sleep 10
echo 'run start'
EXP_COMMAND="cd /home/ubuntu/Deep-Cloud/;sudo bash ./run_all_workload.sh $INSTANCE_TYPE $PROF_MODE 16"
ssh -i awspwd.pem -t ubuntu@$INSTANCE_PUB_DNS $EXP_COMMAND

# Run Tensorboard backgroound
sleep 10
TB_COMMAND="cd /home/ubuntu/Deep-Cloud/;sudo bash ./tensorboard_result.sh $INSTANCE_TYPE"
ssh -i awspwd.pem -t ubuntu@$INSTANCE_PUB_DNS $TB_COMMAND
# BASE_COMMAND2="cd /home/ubuntu/Deep-Cloud/;sudo bash ./tensorboard_result.sh "
# RUN_COMMAND2="$BASE_COMMAND2$INSTANCE_TYPE"
# ssh -i awspwd.pem -t ubuntu@$INSTANCE_PUB_DNS $RUN_COMMAND2

# Get csv files from instance
sleep 10
mkdir $INSTANCE_TYPE
scp -i awspwd.pem \
ubuntu@$INSTANCE_PUB_DNS:~/Deep-Cloud/tensorstats/* ./$INSTANCE_TYPE/
 
# Terminate Instance
sleep 10
TERMINATE_INFO=$(aws ec2 terminate-instances --instance-ids $INSTANCE_ID)
echo $TERMINATE_INFO
