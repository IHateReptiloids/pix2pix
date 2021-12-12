# pix2pix
My implementation of [pix2pix](https://arxiv.org/abs/1611.07004)
Supports only facades and edges2shoes datasets.

## Installation
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Training
Train pix2pix on facades:
```
python train.py --logging
```
Train pix2pix on edges2shoes:
```
python train.py --dataset edges2shoes --logging --num-epochs 15 --checkpointing-freq 5 --epoch-num-iters 12450 --train-batch-size 4 --train-log-freq 1000 --val-batch-size 4 --val-log-freq 10
```

## Testing
```
python test.py --dataset <facades|edges2shoes> --wandb-file-name <WANDB_TRAINER_STATE_FILE_NAME> --wandb-run-path <WANDB_RUN_PATH>
```
