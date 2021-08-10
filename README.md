## ROS2 End-to-End Lane Following Model with SVL Simulator

See detailed instructions and tutorial at: https://www.svlsimulator.com/docs/tutorials/lane-following

Use this command to build repo with openpilot implemented
docker build . -t lgsvl/lanefollowing:openpilot

If running without docker-compose installed:

docker run -it -v /home/sean/lanefollowing:/lanefollowing --runtime=nvidia lgsvl/lanefollowing:openpilot 

python -u ros2_ws/src/lane_following/train/train.py
python -u ros2_ws/src/lane_following/drive.py

Tasks:

Move perception module to repo and try and visualize results
    Load and move model
    Move preprocess function
    Visualize output to see lane lines

