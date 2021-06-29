## ROS2 End-to-End Lane Following Model with SVL Simulator

See detailed instructions and tutorial at: https://www.svlsimulator.com/docs/tutorials/lane-following

If running without docker-compose installed:

docker run -it -v /home/sean/lanefollowing:/lanefollowing --runtime=nvidia lgsvl/lanefollowing:openpilot 

python -u ros2_ws/src/lane_following/train/train.py