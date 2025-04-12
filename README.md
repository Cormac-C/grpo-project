# Group Relative Policy Optimization (GRPO) on The Countdown Reasoning Task

## Requirements

To install requirements:

```setup env
conda env create -f environment.yml

conda activate grpo-proj
```

To run the scripts you will need to set your WANDB key to monitor training progress and results:

```setup tokens
export WANDB_KEY= <your token>
```

## Generate Dataset

To generate a custom countdown you can use the following script:

```dataset script
python scripts/create_dataset.py --save-dir=./data/countdown.json --num-samples=10000
```

If you want to change the configuration of the countdown task you can use the following flags:

```dataset script flags
python scripts/create_dataset.py --save-dir=./data/countdown.json --num-samples=10000 --num-operands=6 --max-target=100 --min-number=1 --max-number=100
```

## Training

To train the model you can use the following script with flags to set the base model, hyperparameters, the dataset location, and save dir:

```train script
python scripts/train.py --basemodel=Qwen/Qwen2.5-1.5B --dataset=./data/countdown.json --output-dir=./output --num-epochs=1 --batch-size=8 --learning-rate=1e-5 --num-outputs=5 --epsilon=0.1 --beta=0.05 --mu=1
```
