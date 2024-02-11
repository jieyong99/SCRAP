# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
from ast import literal_eval
import numpy as np
from collections import Counter
import random
import math

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from peft import PeftModelForSeq2SeqLM,get_peft_config, LoraConfig, get_peft_model

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset, read_line_examples_from_file
from eval_utils import compute_scores, extract_spans_para, compute_f1_scores, evaluate
import json

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor


os.environ["TOKENIZERS_PARALLELISM"] = "false"



def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str, required=False,
                        help="The name of the task, selected from: [asqp]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True,
                        help="The name of the dataset, selected from: [rest15_top3, rest15_top5, rest16_top3, rest16_top5]")                      
    parser.add_argument("--num_reasonings", default='16', type=int, required=True,
                        help="The number of the reasonings, selected from: [1, 2, 4, 8, 16]")                      
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the val/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")
    parser.add_argument("--do_self_consistency", action='store_true', 
                        help="Whether to run self_consistency with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # lora parameters 
    parser.add_argument("--use_peft", action='store_true', help="Whether to use PEFT or not to train adapters")
    parser.add_argument("--peft_lora_r", type=int, default=64, help="the r parameter of the LoRA adapters")
    parser.add_argument("--peft_lora_alpha", type=int, default=16, help="the alpha parameter of the LoRA adapters")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    # self-consistency
    parser.add_argument("--num_return_sequences", default=15, type=int)

    # result filepath
    args = parser.parse_args()
    
    if '/' in args.model_name_or_path:
        save_model_name = args.model_name_or_path.split('/')[-1]
    else:
        save_model_name = args.model_name_or_path

    # output_dir
    output_dir = f"./outputs/{save_model_name}/{args.task}/{args.dataset}/x{args.num_reasonings}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.output_dir = output_dir

    return args

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=f'./data/{args.task}/{args.dataset}/x{args.num_reasonings}', 
                       data_type=type_path, max_len=args.max_seq_length)


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, hparams, tfm_model, tokenizer):
        super().__init__()
        self.hparams.update(vars(hparams))
        self.model = tfm_model
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids=batch["source_ids"],
                       attention_mask=batch["source_mask"],
                       labels=lm_labels,
                       decoder_attention_mask=batch['target_mask'])

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)

        return loss

    def evaluate(self, batch, data_type, stage=None):
        # get f1
        outs = self.model.generate(input_ids=batch['source_ids'],
                                   attention_mask=batch['source_mask'],
                                   max_length=self.hparams.max_seq_length,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   num_beams=1)

        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]

        pred_quads = []
        for i in range(len(dec)):
            pred_quads.append(extract_spans_para(dec[i], 'pred', args.task, args.dataset, data_type, self_con=False))

        
        gold_quads = []
        for t in target:
            tt = literal_eval(t)
            gold_quads.append(tt)
            
        scores = compute_f1_scores(pred_quads, gold_quads, 'test')
        f1 = torch.tensor(scores['f1'], dtype=torch.float64)

        # get loss
        loss = self._step(batch)

        if stage:
            self.log(f"{stage}_loss",
                     loss,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

            self.log(f"{stage}_f1",
                     f1,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val", "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test", "test")

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def train_dataloader(self):
        print("load training data.")
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                drop_last=True, shuffle=True, num_workers=4)

        t_total = (
            (len(dataloader.dataset) // (self.hparams.batch_size * max(1, len(self.hparams.n_gpu))))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=4)

def self_consistency(path, num_path, threshold, gold_quads):
    with open(path) as f:
        outputs = f.readlines()
    
    length = outputs[::num_path]

    pred_quads = []

    for i in range(len(length)):
        o_idx = i * num_path
        multi_outputs_ = outputs[o_idx:o_idx + num_path]

        multi_outputs = []

        for j in range(len(multi_outputs_)):
            multi_outputs.extend(
                extract_spans_para(multi_outputs_[j], 'pred', 'asqp', 'rest15', 'test',self_con=True))

        output_quads = []
        counter = dict(Counter(multi_outputs))
        #print(i, counter)
        for quad, count in counter.items():
            if count >= threshold:
                output_quads.append(quad)
        
        output = []
        for q in output_quads:
            at, ac, sp, ot = q
            output.append([at, ac, sp, ot])
        
        pred_quads.append(output)

    # Compute model performance
    assert len(pred_quads) == len(gold_quads)
    num_samples = len(gold_quads)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = gold_quads[i]
        pred_list = pred_quads[i]

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    #print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels,'test')
    #print(scores)

    return scores


# initialization
args = init_args()
set_seed(args.seed)

print("\n", "="*30, f"{args.dataset} with {args.num_reasonings} reasonings by {args.model_name_or_path}", "="*30, "\n")

# sanity check
# show one sample to check the code and the expected output
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

peft_config = {
    "peft_type": "LORA",
    "task_type": "SEQ_2_SEQ_LM",
    "inference_mode": False,
    "r": args.peft_lora_r,
    "target_modules": ["q", "v"],
    "lora_alpha": args.peft_lora_alpha,
    "lora_dropout": 0.1,
    "fan_in_fan_out": False,
    "bias": "none"}

peft_config = get_peft_config(peft_config)

# training process
if args.do_train:
    print("\n****** Conduct Training ******")

    # initialize the T5 model
    if args.use_peft:
        base_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        tfm_model = PeftModelForSeq2SeqLM(base_model,peft_config)
    else:
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    model = T5FineTuner(args, tfm_model, tokenizer)

    model.configure_optimizers() 
    train_loader = model.train_dataloader()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f'ckpt_model_seed_{args.seed}',
        monitor='val_f1',
        mode='max',
        save_top_k=1,
        # save_last=True
        )

    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_f1",
                                        min_delta=0.00,
                                        patience=5,
                                        verbose=True,
                                        mode="max")

    # prepare for trainer
    train_params = dict(
        accelerator="gpu",
        devices=1,
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    
    # Load the checkpoint
    path = f'{args.output_dir}/ckpt_model_seed_{args.seed}.ckpt' 

    # initialize the T5 model
    if args.use_peft:
        base_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        tfm_model = PeftModelForSeq2SeqLM(base_model,peft_config)
    else:        
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    model = T5FineTuner(args, tfm_model, tokenizer)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.model.save_pretrained(f"{args.output_dir}/best_model")
    tokenizer.save_pretrained(f"{args.output_dir}/best_model")

    print("Finish training and saving the model!")


# evaluation
if args.do_inference:
    print("\n****** Conduct inference on trained checkpoint ******")

    # initialize the T5 model from previous checkpoint
    print(f"Load trained model from {args.output_dir}/best_model")
    print('Note that a pretrained model is required and `do_true` should be False')

    if args.use_peft:
        base_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)    
        model = PeftModelForSeq2SeqLM.from_pretrained(base_model,f"{args.output_dir}/best_model")
    else:        
        model = T5ForConditionalGeneration.from_pretrained(f"{args.output_dir}/best_model")

    tokenizer = T5Tokenizer.from_pretrained(f"{args.output_dir}/best_model")

    print()
    test_dataset = ABSADataset(tokenizer=tokenizer, data_dir=f'./data/{args.task}/{args.dataset}/x{args.num_reasonings}', 
                       data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4)

    # compute the performance scores
    scores = evaluate(test_loader, model, tokenizer, args.task, args.dataset, 'test', args.output_dir)

    log_file_path = os.path.join(args.output_dir, f"SFT_score_result.txt")
    local_time = time.asctime(time.localtime(time.time()))
    exp_results = "Precision: {:.2f} Recall: {:.2f} F1 = {:.2f}".format(scores['precision'], scores['recall'], scores['f1'])

    log_str = f'============================================================\n'
    log_str += f"{local_time} \n{exp_results}\n\n"

    with open(log_file_path, "a+") as f:
        print(exp_results)
        f.write(log_str)

if args.do_self_consistency:
    print("\n****** Generate outputs for Self-Consistency ******")

    if args.use_peft:
        base_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)    
        model = PeftModelForSeq2SeqLM.from_pretrained(base_model,f"{args.output_dir}/best_model")
    else:        
        model = T5ForConditionalGeneration.from_pretrained(f"{args.output_dir}/best_model")

    tokenizer = T5Tokenizer.from_pretrained(f"{args.output_dir}/best_model")

    print()
    test_dataset = ABSADataset(tokenizer=tokenizer, data_dir=f'./data/{args.task}/{args.dataset}/x{args.num_reasonings}', 
                    data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4)

    # compute the performance scores

    device = torch.device(f'cuda:0')
    model.to(device)

    model.eval()

    outputs, targets = [], []

    for batch in tqdm(test_loader):
        # need to push the data to device
        
        outs = model.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=512, do_sample=True, early_stopping=True, num_return_sequences=args.num_return_sequences,
                                    temperature=1.3, top_k=50)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)
    
    dir_path = f'{args.output_dir}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    rst_path = f'{args.output_dir}/{args.model_name_or_path}_path{args.num_return_sequences}_results.txt'

    with open(rst_path, 'w') as f:
        for item in outputs:
            f.write("%s\n" % item)

    print("\n****** Finish Generating outputs ******")
    print()
    print("\n****** Self-Consistency Evaluation ******")

    gold_quads = []
    with open(f'./original_data/asqp/rest15/test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.lower()
            q = line.split('####')[1].strip()
            qq = literal_eval(q)
            gold_quads.append(qq)

    log_file_path = os.path.join(args.output_dir, f"sc_{args.num_return_sequences}_seqs_score_result.txt")
    local_time = time.asctime(time.localtime(time.time()))

    with open(log_file_path, "a+") as f:
        f.write(f"============================================================\n")
        f.write(f"{local_time} \nNum Sequences: {args.num_return_sequences}\n") # Order Top
        f.write(f"============================================================\n")

        threshold = math.ceil(args.num_return_sequences/2)
        scores = self_consistency(rst_path, args.num_return_sequences, threshold, gold_quads)
        scores['precision'] = float(scores['precision'])
        scores['recall'] = float(scores['recall'])
        scores['f1'] = float(scores['f1'])
        exp_results = "Precision: {:.2f} Recall: {:.2f} F1 = {:.2f}".format(scores['precision'], scores['recall'], scores['f1'])
        log_str = f"threshold: {threshold} {exp_results}\n\n"
        print(log_str)
        f.write(log_str)

    print()
    print("****** Finish Self-Consistency Evaluation ******")