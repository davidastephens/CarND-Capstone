export PYTHONPATH=$PYTHONPATH:.:$PWD/../slim:..
python3 ../object_detection/train.py --logtostderr --train_dir=model/ --pipeline_config_path=model/ssd_mobilenet_v1_coco.config
