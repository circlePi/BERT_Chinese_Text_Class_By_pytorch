# BERT_Chinese_Text_Class_By_pytorch
A implements of Chinese text class based on BERT_Pretrained_Model in the framework of pytorch 

## How to start
- Download the Chinese BERT Pretrained Model from google search and place it into the model directory
- python convert_tf_checkpoint_to_pytorch.py to transfer the Pretrained Model into pytorch form 
- prepare Chinese raw data, you can modify the preprocessing.data_processor to adapt your data
- implement the code by "python run_bert_class"

## Tips
- When converting the tensorflow checkpoint into the pytorch, it's expected to choice the "bert_model.ckpt", instead of "bert_model.ckpt.index", as the input file. Otherwise, you will see that the model can learn nothing and give almost same random outputs for any inputs. This means, in fact, you have not loaded the true ckpt for your model
- When using multiple GPUs, the non-tensor calculations, such as accuracy and f1_score, are not supported by DataParallel instance
- As recommanded by Jocob in his paper <url>https://arxiv.org/pdf/1810.04805.pdf<url/>, in fine-tuning tasks, the hyperparameters are expected to set as following: **Batch_size**: 16 or 32, **learning_rate**: 5e-5 or 2e-5 or 3e-5, **num_train_epoch**: 3 or 4
- The pretrained model has a limit for the sentence of input that its length should is not larger than 512, the max position embedding dim. The data flows into the model as: Raw_data -> WordPieces -> Model. Note that the length of wordPieces is generally larger than that of raw_data, so a safe max length of raw_data is at ~128 - 256 
- Upon testing, we found that fine-tuning all layers could get much better results than those of only fine-tuning the last classfier layer. The latter is actually a feature-based way 
