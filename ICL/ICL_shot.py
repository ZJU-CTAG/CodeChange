import imp
import json
import pickle
import re
from numpy import pad
import torch
import transformers
from peft import LoraConfig,prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer,logging, set_seed,T5ForConditionalGeneration, T5Tokenizer,RobertaTokenizer,AutoModel
from peft import PeftModel
import argparse
from datasets import load_dataset,Dataset
import os
from tqdm import tqdm
import sys
from accelerate import Accelerator
from incontext_process import bm25_preprocess
sys.path.append("..") 
sys.path.append("../evaluate") 
from utils import *
import bleu
from compute_gleu_cup import calcGleu
import multiprocessing as mp
import nltk
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from rq1utils import *


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
EOF_STRINGS = ["<|endoftext|>", "</s>", "\n"]
accelerator = Accelerator(mixed_precision='bf16')

@dataclass
class Arguments():
    model_path: Optional[str] = field(default="facebook/opt-125m")
    peft_path: Optional[str] = field(default="", metadata={"help": "The path to the peft model, lora or prefix."})
    num_shot: Optional[int] = field(default=0,
                                    metadata={"help": "The number of shots for the ICL(0,1,2,3,4)."})
    task: Optional[str] = field(
        default='',metadata={"help": "The target task to test on."}
    )
    lang: Optional[str] = field(
        default='java',metadata={"help": "The target language to test on."}
    )
    input_form: Optional[str] = field(
        default='', metadata={"help": "code,diff"}
    )
    model_max_length: int = field(
        default=1024, metadata={"help": "Maximum length of the sequence for target data"}
    )
    icl_max_length: int = field(
        default=1024, metadata={"help": "Maximum length of the sequence for similarity datas"}
    )
    batch_size: int = field(
        default=1, metadata={"help": "Batch size for the model"}
    )
    # fp16: bool = field(
    #     default=False, metadata={"help": "Whether to use fp16."}
    # )
    seed: int = field(
        default=42, metadata={"help": "Random seed"}
    )
    output_dir: Optional[str] = field(
        default='output_dir', metadata={"help": "The output directory where the model predictions will be written."}
    )
    num_workers: int = field(
        default=16, metadata={"help": "Number of workers for the dataloader"}
    )
    


def get_input_output_pattern():
    input = "### Instruction:\n"
    output = "### Answer:\n"
    return input, output


def format_instance(example,args,tokenizer):
    INPUT, OUTPUT = get_input_output_pattern()
    format_length = len(tokenizer.tokenize(INPUT+' '+OUTPUT))
    task = args.task
    if task == 'CodeReview':
        if args.input_form == 'diff':
            instruction = f'Please write a code review according to the diff hunk.'
            example['input_code'] = f'diff hunk:\n{example["patch"]}'
            example['input_code'] = format_diff(example['input_code'], tokenizer, args.model_max_length, format_length, instruction, task, label=example['msg'])
        else:
            instruction = f'Please write a code review according to the code before and after the diff hunk.'
            code_before, code_after = from_CMG_diff_get_code(example['patch'])
            code_before_token = f"code change before:\n{code_before}"
            code_after_token = f"code change after:\n{code_after}"
            code_before_token, code_after_token = format_code(code_before_token, code_after_token, tokenizer, args.model_max_length, format_length, instruction, task, label=example['msg'])
            example['input_code'] = f'{code_before_token}\n{code_after_token}'
        example['context'] = example['input_code']
        example['response'] = example['msg']
    elif task == 'CommitMsgGeneration':
        if args.input_form == 'diff':
            instruction = f'Please write a commit message according to the diff hunk.'
            example['input_code'] = f'diff hunk:\n{example["diff"]}' 
            example['input_code'] = format_diff(example['input_code'], tokenizer, args.model_max_length,format_length, instruction, task, label=example['nl'])
        else:
            instruction = f'Please write a commit message according to the code before and after the diff hunk.'
            code_before, code_after = from_codisum_diff_get_code(example['diff'])
            code_before_token = f"code change before:\n{code_before}"
            code_after_token = f"code change after:\n{code_after}"
            code_before_token, code_after_token = format_code(code_before_token, code_after_token, tokenizer, args.model_max_length, format_length, instruction, task, label=example['nl'])
            example['input_code'] = f'{code_before_token}\n{code_after_token}'
        example['context'] = example['input_code']
        example['response'] = example['nl']
    elif task == 'JITCommentUpdate':
        if args.input_form == 'diff':
            instruction = f'Please write a new comment according to the original comment and the diff hunk.'
            ori_comment_token = f"original comment message:\n{example['old_nl']}"
            ori_comment_token_len = len(tokenizer.tokenize(ori_comment_token)) 
            example['input_code'] = f'diff hunk:\n{example["diff"]}' 
            example['input_code'] = format_diff(example['input_code'], tokenizer, args.model_max_length-ori_comment_token_len, format_length, instruction, task, label=example['nl'])
            example['input_code'] = example['input_code'] + f'\n{ori_comment_token}'
        else:
            instruction = f'Please write a new comment according to the original comment and the code before and after the diff hunk.'
            code_before, code_after = from_JITCUP_diff_get_code(example['diff'])
            code_before_token = f"code change before:\n{code_before}"
            code_after_token = f"code change after:\n{code_after}"
            ori_comment_token = f"original comment message:\n{example['old_nl']}"
            ori_comment_token_len = len(tokenizer.tokenize(ori_comment_token))            
            code_before_token, code_after_token = format_code(code_before_token, code_after_token, tokenizer, args.model_max_length-ori_comment_token_len, format_length, instruction, task, label=example['nl'])
            example['input_code'] = f'{code_before_token}\n{code_after_token}\n{ori_comment_token}'
        example['context'] = example['input_code']
        example['response'] = example['nl']  

    def instance_stack():
        instances = []
        
        instance_length = args.icl_max_length // args.num_shot
        if task == 'CommitMsgGeneration':
            example['diff_candidates_tokens'] = example['diff_candidates_tokens'][:args.num_shot][::-1]
            if args.input_form=='diff':
                for i in range(args.num_shot):
                    temp_instance = example['diff_candidates_tokens'][i]
                    input_token = f"diff hunk:\n{temp_instance['diff']}"
                    input_token = format_diff(input_token,tokenizer,instance_length,format_length,instruction,task,label=temp_instance['nl'])
                    
                    instance = (f"{INPUT}{instruction}\n"
                            f"{input_token}\n"
                            f"{OUTPUT}"
                            f"{temp_instance['nl'].strip()}\n")
                    instances.append(instance)
            else:
                for i in range(args.num_shot):
                    temp_instance = example['diff_candidates_tokens'][i]
                    code_before,code_after = from_codisum_diff_get_code(temp_instance['diff'])
                    code_before_token = f"code change before:\n{code_before}"
                    code_after_token = f"code change after:\n{code_after}"
                    code_before_token,code_after_token = format_code(code_before_token,code_after_token,tokenizer,instance_length,format_length,instruction,task,label=temp_instance['nl'])
                    instance = (f"{INPUT}{instruction}\n"
                            f"{code_before_token}\n{code_after_token}\n"
                            f"{OUTPUT}"
                            f"{temp_instance['nl'].strip()}\n")
                    instances.append(instance)
        elif task == 'JITCommentUpdate':
            example['diff_candidates_tokens'] = example['diff_candidates_tokens'][:args.num_shot][::-1]
            if args.input_form=='diff':
                for i in range(args.num_shot):
                    temp_instance = example['diff_candidates_tokens'][i]
                    input_token = f"diff hunk:{temp_instance['diff']}\noriginal comment message{temp_instance['old_nl']}"
                    input_token = format_diff(input_token,tokenizer,instance_length,format_length,instruction,task,label=temp_instance['nl'])
                    instance = (f"{INPUT}{instruction}\n"
                            f"{input_token}\n"
                            f"{OUTPUT}"
                            f"{temp_instance['nl'].strip()}\n")
                    instances.append(instance)
            else:
                for i in range(args.num_shot):
                    temp_instance = example['diff_candidates_tokens'][i]
                    code_before,code_after = from_JITCUP_diff_get_code(temp_instance['diff'])
                    code_before_token = f"code change before:\n{code_before}"
                    code_after_token = f"code change after:\n{code_after}"
                    ori_comment_token = f"original comment message:\n{example['old_nl']}"
                    ori_comment_token_len = len(tokenizer.tokenize(ori_comment_token))
                    code_before_token,code_after_token = format_code(code_before_token,code_after_token,tokenizer,instance_length-ori_comment_token_len,format_length,instruction,task,label=temp_instance['nl'])
                    instance = (f"{INPUT}{instruction}\n"
                            f"{code_before_token}\n{code_after_token}\n"
                            f"{ori_comment_token}\n"
                            f"{OUTPUT}"
                            f"{temp_instance['nl'].strip()}\n")
                    instances.append(instance)
        elif task == 'CodeReview':
            example['patch_candidates_tokens'] = example['patch_candidates_tokens'][:args.num_shot][::-1]
            if args.input_form=='diff':
                for i in range(args.num_shot):
                    temp_instance = example['patch_candidates_tokens'][i]
                    input_token = f"diff hunk\n:{temp_instance['patch']}"
                    input_token = format_diff(input_token,tokenizer,instance_length,format_length,instruction,task,label=temp_instance['msg'])
                    instance = (f"{INPUT}{instruction}\n"
                            f"{input_token}\n"
                            f"{OUTPUT}"
                            f"{temp_instance['msg'].strip()}\n")
                    instances.append(instance)
            else:
                for i in range(args.num_shot):
                    temp_instance = example['patch_candidates_tokens'][i]
                    code_before,code_after = from_CMG_diff_get_code(temp_instance['patch'])
                    code_before_token = f"code change before:\n{code_before}"
                    code_after_token = f"code change after:\n{code_after}"
                    code_before_token,code_after_token = format_code(code_before_token,code_after_token,tokenizer,instance_length,format_length,instruction,task,label=temp_instance['msg'])
                    instance = (f"{INPUT}{instruction}\n"
                            f"{code_before_token}\n{code_after_token}\n"
                            f"{OUTPUT}"
                            f"{temp_instance['msg'].strip()}\n")
                    instances.append(instance)
        return instances
    if args.num_shot > 0:
        instances = instance_stack()
        input_prompt = (f"{''.join(instances)}"
                        f"{INPUT}{instruction}\n"
                        f"{example['context']}\n"
                        f"{OUTPUT}")
    else:
        input_prompt = (f"{INPUT}{instruction}\n"
                        f"{example['context']}\n"
                        f"{OUTPUT}")
    return {'text':input_prompt}


def create_dataset(tokenizer,args):
    assert args.task in ['CodeReview','CommitMsgGeneration','JITCommentUpdate']
    assert args.lang in ['java', 'python','cpp', 'csharp', 'javascript']
    print(f'get {args.lang} dataset for {args.task}...')
    train_path,test_path = get_task_train_test_dataset_path(args.task, args.lang)
    cache_path = test_path.replace('test','test_cache_32')
    if os.path.exists(cache_path):
        print(f'Loading {args.task} dataset from {train_path} and {cache_path}...')
        test_dataset = load_dataset('json',data_files=cache_path,split='train')
    else:
        print('get test dataset...')
        test_dataset = load_dataset('json',data_files=test_path,split='train')
        test_dataset = test_dataset.select(range(2000))
        
        print('process train dataset...')
        train_dataset = load_dataset('json',data_files=train_path,split='train')
        train_dataset = train_dataset.select(range(16000))
        text_field = get_text_field(args.task)
        print(f'bm25 process for {len(test_dataset)} test samples in {len(train_dataset)} trainng samples...')
        test_dataset = bm25_preprocess(train_dataset,test_dataset,text_field)
        print(f'bm25 result test path cached to {cache_path}')
        test_dataset.to_json(cache_path)
        
    print('incontext format...')
    test_dataset = test_dataset.map(format_instance,num_proc=args.num_workers, fn_kwargs={"args":args,"tokenizer":tokenizer})
    print(f'test_dataset length:{len(test_dataset)}')

    return test_dataset

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_model_tokenizer(args):
    if 'CCT5' in args.model_path or 'codet5' in args.model_path:
        base_model = T5ForConditionalGeneration.from_pretrained( 
            args.model_path,
            device_map='auto',
            trust_remote_code=True,
        )

    else:
        base_model = AutoModelForCausalLM.from_pretrained( 
            args.model_path,
            device_map='auto',
            trust_remote_code=True,
            # load_in_8bit=True,
            torch_dtype=torch.bfloat16,
        )

    if args.peft_path != '':
        tokenizer = AutoTokenizer.from_pretrained(args.peft_path,model_max_length=args.model_max_length,use_fast=True)
        base_model = PeftModel.from_pretrained(base_model,args.peft_path)
        # base_model.merge_and_unload()
        print(len(tokenizer))
        # print embedding length
        print(len(base_model.get_input_embeddings().weight))
        base_model = prepare_model_for_kbit_training(base_model)
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=base_model,
        )
    else:

        if 'CCT5' in args.model_path:
            tokenizer = AutoTokenizer.from_pretrained('//nasdata/Model/codet5-base',model_max_length=args.model_max_length+args.icl_max_length)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path,model_max_length=args.model_max_length+args.icl_max_length,use_fast=True)
            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            tokenizer.padding_side='left'
    return base_model, tokenizer


def evaluate():
    parser = transformers.HfArgumentParser((Arguments))
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    set_seed(args.seed)
    model, tokenizer = get_model_tokenizer(args)
    model.eval()
    test_dataset = create_dataset(tokenizer,args)
    print(test_dataset[0]['text'])
    
    rq = 'rq1' if args.peft_path =='' else 'rq2'
    model_name = args.model_path.split('/')[3]


    step_dir = args.peft_path.split('-')[-1].replace('/','') if args.peft_path != '' else 'base'
    if rq == 'rq1':
        rq1_output_dir = os.path.join(
            args.output_dir,rq,model_name,step_dir
        )
        os.makedirs(rq1_output_dir,exist_ok=True)
        rq_output_path = os.path.join(
            rq1_output_dir,
            f'{args.task}_{args.input_form}_{args.lang}_shot{args.num_shot}.output'
        )
    else:
        if 'lora' in args.peft_path:
            peft_method = 'lora'
        elif 'prefix' in args.peft_path:
            peft_method = 'prefix'
        elif 'prompt' in args.peft_path:
            peft_method = 'prompt'
        elif 'pt' in args.peft_path and 'prom' not in args.peft_path:
            peft_method = 'pt'

        rq2_output_dir = os.path.join(
            args.output_dir,rq,model_name,peft_method,step_dir
        )
        os.makedirs(rq2_output_dir,exist_ok=True)
        rq_output_path = os.path.join(
            rq2_output_dir,
            f'{args.task}_{args.input_form}_{args.lang}.output'
        )
    rq_gold_path = rq_output_path.replace('.output','.gold')
    print(f'result will be saved in rq_output_path:{rq_output_path}...')
    if args.task == "CodeReview":
        test_dataset = test_dataset.rename_column("response", "labels")
        test_dataset = test_dataset.map(lambda x: tokenizer(x['text'],padding='max_length',truncation=True,max_length=tokenizer.model_max_length,),batched=True,remove_columns=['patch','msg'])
        test_dataset.set_format(type='torch',columns=['input_ids','attention_mask','labels'])
        eval_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        preds = []
        golds = []
        # print(tokenizer.model_max_length)
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu-4 for CodeReview set"):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            # print(input_ids.shape)
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids,max_new_tokens=100,max_time=10)

                outputs = outputs.detach().cpu().numpy()[:,batch["input_ids"].shape[1]:]
                all_predictions = tokenizer.batch_decode(outputs,skip_special_tokens=True,clean_up_tokenization_spaces=True)
                all_predictions = [re.split("(%s)" % "|".join(EOF_STRINGS), i.strip())[0] for i in all_predictions]
                all_answers = [i.replace("\n", " ").replace("\t", " ") for i in all_predictions]
                labels = [i.replace("\n", " ").replace("\t", " ") for i in labels]

            preds.extend(all_answers)
            golds.extend(labels)
            # print(golds)
            # print(preds)
            # break

        correct = 0.0
        idx_rec = []
        assert len(golds)==len(preds)
        for i in range(len(golds)):
            if formatString_1(preds[i]) == formatString_1(golds[i]):
                correct += 1
                idx_rec.append(i)

        acc = 100 * correct/float(len(golds))
        print(f'Accuracy: {acc}')
        predictions = []
        with open(rq_output_path, 'w') as f, open(rq_gold_path, 'w') as f1:
            index = 0
            for ref, gold in zip(preds, golds):
                predictions.append(str(index) + '\t' + ref)
                f.write(str(index) + '\t' + ref + '\n') 
                f1.write(str(index) + '\t' + gold + '\n')
                index+=1
        
        (goldMap, predictionMap) = bleu.computeMaps(predictions,rq_gold_path)
        this_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

        os.rename(rq_output_path, rq_output_path.replace('.output', f'_bleu{this_bleu}.output'))
        
        print('bleu score: {}'.format(this_bleu))
        
    elif args.task  == 'CommitMsgGeneration':

        eval_dataset_base = test_dataset.rename_column("response", "labels")
        eval_dataset_prepare = eval_dataset_base.map(lambda x: tokenizer(x['text'],padding='max_length',truncation=True,max_length=tokenizer.model_max_length),batched=True,remove_columns=['diff'])
        eval_dataset_prepare.set_format(type='torch',columns=['input_ids','attention_mask','labels'])
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset_prepare,batch_size=args.batch_size,shuffle=False)
        eval_dataloader = accelerator.prepare(eval_dataloader)

        golds = []
        preds = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for CommitMsgGeneration set"):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            with torch.no_grad():
                # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    outputs = model.generate(input_ids=input_ids,max_new_tokens=100,max_time=10)

                    outputs = outputs.detach().cpu().numpy()[:,batch["input_ids"].shape[1]:]
                    all_predictions = tokenizer.batch_decode(outputs,skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    all_predictions = [re.split("(%s)" % "|".join(EOF_STRINGS), i.strip())[0] for i in all_predictions]
                    all_answers = [i.replace("\n", " ").replace("\t", " ") for i in all_predictions]
                    labels = [i.replace("\n", " ").replace("\t", " ") for i in labels]

            preds.extend(all_answers)
            golds.extend(labels)

        assert len(golds)==len(preds)
        correct = 0.0
        idx_rec = []
        for i in range(len(golds)):
            if formatString_1(preds[i]) == formatString_1(golds[i]):
                correct += 1
                idx_rec.append(i)

        acc = 100 * correct/float(len(golds))

        print(f'Accuracy: {acc}')

        predictions = []

        with open(rq_output_path, 'w') as f, open(rq_gold_path, 'w') as f1:
            index = 0
            for ref, gold in zip(preds, golds):
                predictions.append(str(index) + '\t' + ref)
                f.write(str(index) + '\t' + ref + '\n')
                f1.write(str(index) + '\t' + gold + '\n')
                index+=1
        
        (goldMap, predictionMap) = bleu.computeMaps(predictions,os.path.join(rq_gold_path))
        this_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

        os.rename(rq_output_path, rq_output_path.replace('.output', f'_bleu{this_bleu}.output'))
        
        print('bleu score: {}'.format(this_bleu))

    elif args.task == "JITCommentUpdate":

        eval_dataset_base = test_dataset.rename_column("response", "labels")
        eval_dataset_prepare = eval_dataset_base.map(lambda x: tokenizer(x['text'],padding='max_length',truncation=True,max_length=tokenizer.model_max_length),batched=True)
        eval_dataset_prepare.set_format(type='torch',columns=['input_ids','attention_mask','labels','old_nl'])
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset_prepare,batch_size=args.batch_size,shuffle=False)
        eval_dataloader = accelerator.prepare(eval_dataloader)

        golds = []
        preds = []
        srcs = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval GLEU for JIT Comment Update set"):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            old_msgs = batch['old_nl']

            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids,max_new_tokens=100,max_time=10)
                outputs = outputs.detach().cpu().numpy()[:,batch["input_ids"].shape[1]:]
                all_predictions = tokenizer.batch_decode(outputs,skip_special_tokens=True,clean_up_tokenization_spaces=True)
                all_predictions = [re.split("(%s)" % "|".join(EOF_STRINGS), i.strip())[0] for i in all_predictions]
                all_answers = [i.replace("\n", " ").replace("\t", " ") for i in all_predictions]
                labels = [i.replace("\n", " ").replace("\t", " ") for i in labels]
                old_msgs = [i.replace("\n", " ").replace("\t", " ") for i in old_msgs]

            preds.extend(all_answers)
            golds.extend(labels)
            srcs.extend(old_msgs)


        
        with open(rq_output_path, 'w') as f, open(rq_gold_path, 'w') as f1:
            index = 1
            for ref, gold in zip(preds, golds):
                f.write(str(index) + '\t' + ref + '\n')
                f1.write(str(index) + '\t' + gold + '\n')
                index+=1
            print(f'write to {args.task} file done')
            
        assert(len(golds) == len(preds))

        src_tokens, gold_tokens, pred_tokens = [], [], []
        for src, pred, gold in zip(srcs, preds, golds):
            src_tokens.append([x.lower().replace("``", "\"") for x in tokenizer.tokenize(src)])
            gold_tokens.append([x.lower().replace("``", "\"") for x in tokenizer.tokenize(gold)])
            pred_tokens.append([x.lower().replace("``", "\"") for x in tokenizer.tokenize(pred)])
        gleu = calcGleu(src_tokens, gold_tokens, pred_tokens, lowercase=True)
        
        correct = 0.0
        idx_rec = []
        for i in range(len(golds)):
            if formatString_1(preds[i]) == formatString_1(golds[i]):
                correct += 1
                idx_rec.append(i)

        acc = 100 * correct/float(len(golds))

        os.rename(rq_output_path, rq_output_path.replace('.output', f'_gleu{gleu}_acc{acc}.output'))
        
        print(f'Accuracy: {acc}')
        print(f'Gleu score: {gleu}')
        
    print(f'saved to {rq_output_path}')
    
if __name__ == "__main__":
    
    logging.set_verbosity_error()
    evaluate()