export PYTHONPATH=$PYTHONPATH:.:..
python generate_tfrecord.py --csv_input=data/train.csv  --output_path=data/train.record
python generate_tfrecord.py --csv_input=data/test.csv  --output_path=data/test.record
