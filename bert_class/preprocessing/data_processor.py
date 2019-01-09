"""从文件读取数据，构造成features
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
"""
import os
import csv
import random
from collections import Counter
from tqdm import tqdm
from util.Logginger import init_logger
import config.args as args
import operator

logger = init_logger("bert_class", logging_path=args.log_path)


def train_val_split(X, y, valid_size=0.3, random_state=2018, shuffle=True):
    """
    训练集验证集分割
    :param X: sentences
    :param y: labels
    :param random_state: 随机种子
    """
    logger.info('train val split')

    train, valid = [], []
    bucket = [[] for _ in args.labels]

    for data_x, data_y in tqdm(zip(X, y), desc='bucket'):
        bucket[int(data_y)].append((data_x, data_y))

    del X, y

    for bt in tqdm(bucket, desc='split'):
        N = len(bt)
        if N == 0:
            continue
        test_size = int(N * valid_size)

        if shuffle:
            random.seed(random_state)
            random.shuffle(bt)

        valid.extend(bt[:test_size])
        train.extend(bt[test_size:])

    if shuffle:
        random.seed(random_state)
        random.shuffle(valid)
        random.shuffle(train)

    return train, valid


def sent_label_split(line):
    """
    句子处理成单词
    :param line: 原始行
    :return: 单词， 标签
    """
    line = line.strip('\n').split('@')
    label = line[0]
    sent = line[1].split(' ')
    return sent, label


def bulid_vocab(vocab_size, min_freq=1, stop_word_list=None):
    """
    建立词典
    :param vocab_size: 词典大小
    :param min_freq: 最小词频限制
    :param stop_list: 停用词 @type：file_path
    :return: vocab
    """
    count = Counter()

    with open(os.path.join(args.ROOT_DIR, args.RAW_DATA), 'r') as fr:
        logger.info('Building vocab')
        for line in tqdm(fr, desc='Build vocab'):
            words, label = sent_label_split(line)
            count.update(words)

    if stop_word_list:
        stop_list = {}
        with open(os.path.join(args.ROOT_DIR, args.STOP_WORD_LIST), 'r') as fr:
                for i, line in enumerate(fr):
                    word = line.strip('\n')
                    if stop_list.get(word) is None:
                        stop_list[word] = i
        count = {k: v for k, v in count.items() if k not in stop_list}
    count = sorted(count.items(), key=operator.itemgetter(1))
    # 词典
    vocab = [w[0] for w in count if w[1] >= min_freq]
    if vocab_size-3 < len(vocab):
        vocab = vocab[:vocab_size-3]
    vocab = args.flag_words + vocab
    assert vocab[0] == "[PAD]", ("[PAD] is not at the first position of vocab")
    logger.info('Vocab_size is %d'%len(vocab))

    with open(args.VOCAB_FILE, 'w') as fw:
        for w in vocab:
            fw.write(w + '\n')
    logger.info("Vocab.txt write down at {}".format(args.VOCAB_FILE))


def produce_data(custom_vocab=False, stop_word_list=None, vocab_size=None):
    targets, sentences = [],[]
    with open(args.RAW_DATA,'r') as fr:
        for line in fr:
            lines = line.strip().split('@')
            target = lines[0]
            sent_list = lines[1].split()
            sent = "".join(sent_list)
            targets.append(target)
            sentences.append(sent)
            if custom_vocab:
                bulid_vocab(vocab_size, stop_word_list)
    train, valid = train_val_split(sentences, targets)


    with open(args.TRAIN, 'w') as fw:
        for sent, label in train:
                sent = [str(s) for s in sent]
                line = "\t".join([str(label), " ".join(sent)])
                fw.write(line + '\n')
        logger.info("Train set write down")

    with open(args.VALID, 'w') as fw:
        for sent, label in valid:
                sent = [str(s) for s in sent]
                line = "\t".join([str(label), " ".join(sent)])
                fw.write(line + '\n')
        logger.info("Dev set write down")


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """数据预处理的基类，自定义的MyPro继承该类"""
    def get_train_examples(self, data_dir):
        """读取训练集 Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """读取验证集 Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """读取标签 Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """读tsv文件"""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MyPro(DataProcessor):
    def _create_example(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            text_a = line[1]
            label = line[0]
            example = InputExample(guid=guid, text_a=text_a, label=label)
            examples.append(example)
        return examples

    def get_train_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = self._create_example(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
        examples = self._create_example(lines, "dev")
        return examples

    def get_labels(self):
        return args.labels


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    # 标签转换为数字
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length-2)]
        # 句子首尾加入标示符
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        # 词转换成数字
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        # --------------看结果是否合理-------------------------
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        # ----------------------------------------------------

        feature = InputFeature(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_id=label_id)
        features.append(feature)

    return features


