from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch


if __name__ == "__main__":
    tf_checkpoint_path = "/home/daizelin/bert_class/model/chinese_L-12_H-768_A-12/bert_model.ckpt"
    bert_config_file = "/home/daizelin/bert_class/model/chinese_L-12_H-768_A-12/bert_config.json"
    pytorch_dump_path = "/home/daizelin/bert_class/model/chinese_L-12_H-768_A-12.ckpt"
    convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path)
