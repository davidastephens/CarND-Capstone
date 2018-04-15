PYTHONPATH=/home/ubuntu/CarND-Capstone/CarND-Capstone/ros/src/tl_detector/light_classification/models
echo $PYTHONPATH
python3 train.py --logtostderr --train_dir=../training/ --pipeline_config_path=../training/ssd_mobilenet_v1_coco.config
