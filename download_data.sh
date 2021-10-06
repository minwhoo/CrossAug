# mkdir fever_data
cd fever_data
wget https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl
wget https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.jsonl
wget http://milabfile.snu.ac.kr:15000/sharing/zdhCfoLzP
wget -O fm2.dev.jsonl https://github.com/google-research/fool-me-twice/raw/main/dataset/dev.jsonl
wget -O symmetric.dev.jsonl https://github.com/TalSchuster/FeverSymmetric/raw/master/symmetric_v0.1/fever_symmetric_generated.jsonl
wget https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc_baselines/fever_adversarial.zip
unzip fever_adversarial.zip
mv fever_adversarial/test.jsonl adversarial.dev.jsonl
rm -rf fever_adversarial*
cd ..