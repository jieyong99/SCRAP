
import re
import numpy as np
import json
import torch
import os
from tqdm import tqdm
from ast import literal_eval
from datasets import Dataset, load_dataset, DatasetDict


def extract_spans_para(seq, seq_type, task, dataset, data_type, self_con=False):
    quads = []
    if seq_type == 'pred':
        if(len(seq.split('therefore, the quadruplets are:'))>1):
            quadruplets = seq.split('therefore, the quadruplets are:')[1].strip()
            sents = [q.strip() for q in quadruplets.split('[SSEP]')]
            for s in sents:
                    # food quality is bad because pizza is over cooked.
                    try:
                        index_ac = s.index("[AC]")
                        index_sp = s.index("[SP]")
                        index_at = s.index("[AT]")
                        index_ot = s.index("[OT]")

                        combined_list = [index_ac, index_sp, index_at, index_ot]
                        arg_index_list = list(np.argsort(combined_list))  # .tolist()

                        result = []
                        for i in range(len(combined_list)):
                            start = combined_list[i] + 4
                            sort_index = arg_index_list.index(i)
                            if sort_index < 3:
                                next_ = arg_index_list[sort_index + 1]
                                re = s[start: combined_list[next_]]
                            else:
                                re = s[start:]
                            result.append(re.strip())

                        ac, sp, at, ot = result             
                    except ValueError:
                        try:
                            print(f'In {seq_type} seq, cannot decode: {s}')
                            pass
                        except UnicodeEncodeError:
                            print(f'In {seq_type} seq, a string cannot be decoded')
                            pass
                        at, ac, sp, ot = '', '', '', ''
                    if [at, ac, sp, ot] not in quads:
                        if self_con:
                            quads.append((at, ac, sp, ot))
                        else:
                            quads.append([at, ac, sp, ot])
        else:
            if self_con:
                quads.append(('','','',''))
            else:
                quads.append(['','','',''])
    elif seq_type == 'gold':
        dataset = dataset.split('_')[0]
        with open(f'original_data/{task}/{dataset}/{data_type}.txt', 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.lower()
                q = line.split('####')[1].strip()
                qq = literal_eval(q)
                quads.append(qq)
    return quads


def compute_f1_scores(pred_pt, gold_pt, data_type, verbose=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    if data_type == "test":
        for i in range(len(pred_pt)):
            n_gold += len(gold_pt[i])
            n_pred += len(pred_pt[i])

            for t in pred_pt[i]:
                if t in gold_pt[i]:
                    n_tp += 1

    if verbose and data_type == "test":
        print(
            f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}"
        )

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (
        precision + recall) if precision != 0 or recall != 0 else 0
    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

    return scores


def compute_scores(pred_seqs, gold_seqs, task, dataset, data_type):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs), (len(pred_seqs), len(gold_seqs))
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        pred_list = extract_spans_para(pred_seqs[i], 'pred', task, dataset, data_type, self_con=False)
        all_preds.append(pred_list)
   
    if data_type == 'test' or data_type == 'val':     
        all_labels = extract_spans_para(str(all_labels), 'gold', task, dataset, data_type, self_con=False)
    scores = compute_f1_scores(all_preds, all_labels, data_type)

    return scores, all_labels, all_preds



def evaluate(data_loader, model, tokenizer, task, dataset, data_type, output_dir, verbose=True, use_peft=False):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device('cuda')
    

    model.to(device)
    model.eval()
    
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=512, num_beams=5, early_stopping=True)
        
        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        
        outputs.extend(dec)
        targets.extend(target)
    
    dir_path = f'{output_dir}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    rst_path = f'{output_dir}/{data_type}_results.txt'

    with open(rst_path, 'w') as f:
        for item in outputs:
            f.write("%s\n" % item)
            

    scores, all_labels, all_preds = compute_scores(outputs, targets, task, dataset, data_type)
    results = {'scores': scores, 'labels': all_labels, 'preds': all_preds}
    if verbose:
        print(f"Results: ")
        print(scores)

    return scores