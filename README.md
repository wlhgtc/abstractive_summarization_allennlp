# Train:
- make-vocab:
`
python -m allennlp.run train experiments/baseline_vocab.json  -s tmp/baseline/ --include-package abstractive_summarization
`
- replace:
`
cat experiments/baseline.json > tmp/baseline/config.json
`
- train-recover:
`
python -m allennlp.run train experiments/baseline.json  -r -s tmp/baseline/ --include-package abstractive_summarization
`
- vis:
`
cd tmp/baseline/log/train
tensorboard --logdir=./
`