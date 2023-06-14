ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
CUDA_VISIBLE_DEVICES=$1  python -u $ROOT/main.py --log log.txt --testuvg --pretrain snapshot/iter2261420.model --config config.json
