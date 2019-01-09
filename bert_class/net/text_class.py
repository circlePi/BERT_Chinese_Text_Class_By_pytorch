import torch.nn as nn
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel

class TextClass(PreTrainedBertModel):
    """自定义分类模型类，这里可以很方便地将Bert encoder
       模型和我们自定义的如CNN、RNN一起使用"""
    def __init__(self,
                 config,
                 num_labels):
        super(TextClass, self).__init__(config)
        self.bert = BertModel(config)
        # 默认情况下，bert encoder模型所有的参数都是参与训练的，32的batch_size大概8.7G显存
        # 可以通过以下设置为将其设为不训练，只将classifier这一层进行反响传播，32的batch_size大概显存1.1G
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

        self.num_labels = num_labels

    def forward(self, input_ids, token_type_ids, attention_mask, label_ids=None, output_all_encoded_layers=False):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=output_all_encoded_layers)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


