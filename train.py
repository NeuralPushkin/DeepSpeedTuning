import os


"""
MASTER port should be open if train with ddp
RAnk - main gpu

"""
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"# for ddp
os.environ['WORLD_SIZE'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false" #uncoment for large files

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
from transformers import TextDataset,DataCollatorForLanguageModeling

torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2", 
                                          bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2").cuda()


model.resize_token_embeddings(len(tokenizer))








train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator



train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)

training_args = TrainingArguments(output_dir='deepspeed',
                                  num_train_epochs=5, 
                                  logging_steps=300, 
                                  save_steps=3000,
                                  per_device_train_batch_size=15,
                                  per_device_eval_batch_size=15,
                                  warmup_steps=100,
                                  weight_decay=0.01, 
                                  
                                  fp16=True,
                                  #warmup_steps=10,
                                  #weight_decay=0.01,  
                                  #fp16=True, 
                                  #fp16_opt_level='O1', not useful beacuse deepspeed
                                  report_to="wandb",
                                  deepspeed='ds_config_gpt_j.json')
trainer = Trainer(model=model, args=training_args, 
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset)
trainer.train()
