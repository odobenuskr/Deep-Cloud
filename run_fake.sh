# CNN Job Test
JOB_DIR1="/home/ubuntu/Deep-Cloud/job/cnn/fake/fake-mnist-lenet5.py"
JOB_DIR2="/home/ubuntu/Deep-Cloud/job/cnn/fake/fake-fmnist-resnetsmall.py"
JOB_DIR3="/home/ubuntu/Deep-Cloud/job/cnn/fake/fake-cifar10-vggsmall.py"

JOB_DIR4="/home/ubuntu/Deep-Cloud/job/cnn/real/real-mnist-lenet5.py"
JOB_DIR5="/home/ubuntu/Deep-Cloud/job/cnn/real/real-fmnist-resnetsmall.py"
JOB_DIR6="/home/ubuntu/Deep-Cloud/job/cnn/real/real-cifar10-vggsmall.py"

# Get profile result
sudo -i -u root bash << EOF

python3.6 $JOB_DIR1
sleep 3

python3.6 $JOB_DIR2
sleep 3

python3.6 $JOB_DIR3
sleep 3

python3.6 $JOB_DIR4
sleep 3

python3.6 $JOB_DIR5
sleep 3

python3.6 $JOB_DIR6
sleep 3

EOF