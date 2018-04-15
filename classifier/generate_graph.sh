export PYTHONPATH=$PYTHONPATH:.:slim
rm -rf traffic_light_graph

python3 object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path sim/model/ssd_mobilenet_v1_coco.config  --trained_checkpoint_prefix sim/model/model.ckpt-8356 --output_directory traffic_light_graph





