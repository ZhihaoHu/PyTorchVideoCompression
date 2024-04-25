ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
CUDA_VISIBLE_DEVICES=3 python -u $ROOT/main.py --snapshot_dir snapshot --log log.txt --config config.json

