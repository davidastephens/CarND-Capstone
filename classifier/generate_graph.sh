export PYTHONPATH=$PYTHONPATH:.:slim
rm -rf traffic_light_graph
rm -rf traffic_light_graph_real

python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path sim/model_rcnn/faster_rcnn_resnet101_coco.config  --trained_checkpoint_prefix sim/model_rcnn/model.ckpt-13728 --output_directory traffic_light_graph
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path real/model_rcnn/faster_rcnn_resnet101_coco.config  --trained_checkpoint_prefix real/model_rcnn/model.ckpt-7016 --output_directory traffic_light_graph_real





