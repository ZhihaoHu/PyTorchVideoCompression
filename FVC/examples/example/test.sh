ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
CUDA_VISIBLE_DEVICES=$1  python -u $ROOT/main.py --log log.txt --test --pretrain snapshot/iter2261420.model --config config.json
