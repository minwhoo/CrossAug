# CrossAug
This is the code for the CIKM 2021 paper [CrossAug: A Contrastive Data Augmentation Method for Debiasing Fact Verification Models](https://arxiv.org/abs/2109.15107).

In this work, we propose a data augmentation method for debiasing fact verification models by generating contrastive samples.

## Setup

### Install dependencies

Our code is based on Python 3.7, and experiments are run on a single GPU.

```
pip install -r requirements.txt
```

### Download the data

Download FEVER[^1], FEVER Symmetric[^2], Adversarial FEVER[^3], and Fool Me Twice[^4] datasets using the bash script below:

```
./download_data.sh
```

You can either download the FEVER train set augmented with CrossAug [here](http://milabfile.snu.ac.kr:15000/sharing/zdhCfoLzP) or manually generate augmented data from the next section.


## Data Augmentation
### Augment FEVER train dataset with CrossAug

```
python run_crossaug.py \
  --in_file fever_data/fever.train.jsonl \
  --out_file fever_data/fever+crossaug.train.jsonl
```

### Use the fine-tuned negative claim generation model

We have uploaded the negative claim generation model fine-tuned with WikiFactCheck-English[^5] dataset on the Huggingface repository.
An example code for using the fine-tuned model is provided below:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = 'minwhoo/bart-base-negative-claim-generation'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

examples = [
    "Little Miss Sunshine was filmed over 30 days.",
    "Magic Johnson did not play for the Lakers.",
    "Claire Danes is wedded to an actor from England."
]

batch = tokenizer(examples, max_length=1024, padding=True, truncation=True, return_tensors="pt")
out = model.generate(batch['input_ids'].to(model.device), num_beams=5)
negative_examples = tokenizer.batch_decode(out, skip_special_tokens=True)
print(negative_examples)
# ['Little Miss Sunshine was filmed less than 3 days.', 'Magic Johnson played for the Lakers.', 'Claire Danes is married to an actor from France.']
```

## Train and test the model

For training and evaluation, we slightly modified the code from [this repo](https://github.com/TalSchuster/pytorch-transformers), which was in turn modified from an older version of Huggingface transformers library.

- Train with CrossAug-augmented dataset and evaluate on fact verification dev sets
```bash
TRAIN_SEED=177697310
python run_fever.py \
    --task_name fever \
    --do_train \
    --train_task_name fever+crossaug \
    --do_eval \
    --eval_task_names fever symmetric adversarial fm2 \
    --data_dir ./fever_data/ \
    --do_lower_case \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --save_steps 100000 \
    --output_dir ./crossaug_trained_models_seed=$TRAIN_SEED/ \
    --output_preds \
    --seed $TRAIN_SEED
```

- Train baseline (no augmentation)
```bash
TRAIN_SEED=177697310
python run_fever.py \
    --task_name fever \
    --do_train \
    --train_task_name fever \
    --do_eval \
    --eval_task_names fever symmetric adversarial fm2 \
    --data_dir ./fever_data/ \
    --do_lower_case \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --save_steps 100000 \
    --output_dir ./baseline_trained_models_seed=$TRAIN_SEED/ \
    --output_preds \
    --seed $TRAIN_SEED
```

## Results

Training and evaluation with the above commands should result in the following accuracies.

|          | FEVER dev | Symmetric | Adversarial | FM2 dev   |
|----------|-----------|-----------|-------------|-----------|
| No aug   | **86.43** | 59.14     | 50.00       | 41.15     |
| PoE      | 86.14     | 63.88     | 51.31       | **47.39** |
| CrossAug | 85.05     | **68.20** | **52.48**   | 45.17     |


## Citation
```
@inproceedings{lee2021crossaug,
  title={CrossAug: A Contrastive Data Augmentation Method for Debiasing Fact Verification Models},
  author={Minwoo Lee and Seungpil Won and Juae Kim and Hwanhee Lee and Cheoneum Park and Kyomin Jung},
  booktitle={Proceedings of the 30th ACM International Conference on Information & Knowledge Management},
  publisher={Association for Computing Machinery},
  series={CIKM '21},
  year={2021}
}
```

[^1]: [FEVER: a large-scale dataset for Fact Extraction and VERification](https://arxiv.org/abs/1803.05355)
[^2]: [Towards Debiasing Fact Verification Models](https://arxiv.org/abs/1908.05267)
[^3]: [Adversarial attacks against Fact Extraction and VERification](https://arxiv.org/abs/1903.05543)
[^4]: [Fool Me Twice: Entailment from Wikipedia Gamification](https://arxiv.org/abs/2104.04725)
[^5]: [Automated Fact-Checking of Claims from Wikipedia](https://aclanthology.org/2020.lrec-1.849/)
