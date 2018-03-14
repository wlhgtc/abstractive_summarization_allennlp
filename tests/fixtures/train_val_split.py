import pandas as pd
import fire


def main(train_data_path,split_ratio):
    train = pd.read_json(train_data_path,lines=True,encoding='utf-8')
    dataset_len = len(train)
    #str_lens = train.article.str.len()
    #str_lens.describe()
    train_index = train.index.tolist()
    validation_index = train.sample(dataset_len*split_ratio).index.tolist()
    [train_index.remove(i) for i in validation_index]
    #train.article = train.article.str.encode('utf-8')
    #train.summarization = train.summarization.str.encode('utf-8')
    train.ix[train_index].to_json('train.jsonl',lines=True,orient='records')
    train.ix[validation_index].to_json('val.jsonl',lines=True,orient='records')

if __name__ == '__main__':
    fire.Fire(main)