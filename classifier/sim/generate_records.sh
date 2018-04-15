export PYTHONPATH=$PYTHONPATH:.:..
python3 generate_tfrecord.py --csv_input=data/train.csv  --output_path=data/train.record
python3 generate_tfrecord.py --csv_input=data/test.csv  --output_path=data/test.record
