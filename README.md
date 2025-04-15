# Group Relative Policy Optimization (GRPO) on The Countdown Reasoning Task

## Requirements

To install requirements:

```setup env
conda env create -f environment.yml

conda activate grpo-proj
```

To run the scripts you will need to set your WANDB key and project details to monitor training progress and results:

```setup tokens
export WANDB_KEY= <your token>

export WANDB_ENTITY= <your entity/team name>

export WANDB_PROJECT= <your project name>
```

## Training

To train the model you can use the following script to use the countdown dataset on HuggingFace:

```train script
python scripts/train.py --base-model=Qwen/Qwen2.5-1.5B --dataset-type=HF --output-dir=./output
```

To run a custom training run with a custom data, hyperparameters, etc. you can use a script with flags like the following example.

```
python scripts/train.py --base-model=Qwen/Qwen2.5-0.5B --dataset-type=JSON --dataset=./data/small-scale/countdown.json --output-dir=./output --num-epochs=1 --batch-size=2 --learning-rate=1e-5 --num-outputs=2 --epsilon=0.1 --beta=0.05 --mu=1
```

## Generate Dataset

To generate a custom countdown you can use the following script:

```dataset script
python scripts/create_dataset.py --save-dir=./data/main --num-samples=10000
```

If you want to change the configuration of the countdown task you can use the following flags:

```dataset script flags
python scripts/create_dataset.py --save-dir=./data/small-scale --num-samples=1000 --num-operands=3 --max-target=25 --min-number=1 --max-number=25
```
