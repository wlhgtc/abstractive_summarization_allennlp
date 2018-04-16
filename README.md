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
# predcit
`
python -m allennlp.run predict tmp/baseline/model.tar.gz tmp/test.jsonl --batch-size 32 --cuda-device 0 --include-package abstractive_summarization --predictor abstract-generator
`