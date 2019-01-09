"""Note:
pytorch BERT 模型包含三个文件：模型、vocab.txt, bert_config.json, 有两种加载方式：
（1）在线下载。这种方式下，模型和vocab会通过url的方式下载，只需将bert_model设置为 "bert_model=bert-base-chinese"
     另外，还需要设置cache_dir路径，用来存储下载的文件。
（2）先下载好文件。下载好的文件是tensorflow的ckpt格式的，首先要利用convert_tf_checkpoint_to_pytorch转换成pytorch格式存储
     这种方式是通过本地文件夹直接加载的，要注意这时的文件命名方式。首先指定bert_model=存储模型的文件夹
     第二，将vocab.txt和bert_config.json放入该目录下，并在配置文件中指定VOCAB_FILE路径。当然vocab.txt可以不和模型放在一起，
     但是bert_config.json文件必须和模型文件在一起。具体可见源代码file_utils
"""
# -----------ARGS---------------------
ROOT_DIR = "/home/daizelin/bert_class/"
RAW_DATA = "data/police_train.csv"
STOP_WORD_LIST = None
CUSTOM_VOCAB_FILE = None
VOCAB_FILE = "model/vocab.txt"
TRAIN = "data/train.tsv"
VALID = "data/dev.tsv"
log_path = "output/logs"
plot_path = "output/images/loss_acc.png"
data_dir = "data/"                            # 原始数据文件夹，应包括tsv文件
cache_dir = "model/"
output_dir = "output/checkpoint"              # checkpoint和预测输出文件夹

bert_model = "model/pytorch_pretrained_model" # BERT 预训练模型种类 bert-base-chinese
task_name = "bert_class"                      # 训练任务名称


flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
max_seq_length = 128
do_train = True
do_eval = False
do_lower_case = True
train_batch_size = 32
eval_batch_size = 32
learning_rate = 1e-3
num_train_epochs = 4
warmup_proportion = 0.1
no_cuda = False
local_rank = -1
seed = 42
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
labels = [str(i) for i in range(9)]