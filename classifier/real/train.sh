export PYTHONPATH=$PYTHONPATH:.:$PWD/../slim:..
python3 ../object_detection/train.py --logtostderr --train_dir=model_rcnn/ --pipeline_config_path=model_rcnn/faster_rcnn_resnet101_coco.config
