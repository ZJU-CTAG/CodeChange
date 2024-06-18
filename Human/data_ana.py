import json
import re
import sys

from regex import D
from torch import rand
from utils import *
sys.path.append("..") 
sys.path.append("../evaluate") 
import bleu
from t5bleu import _bleu_RAM
from compute_gleu_cup import calcGleu
from transformers import AutoTokenizer,RobertaTokenizer
from collections import defaultdict
import numpy as np 
import scipy.stats as st 
import csv
from tqdm import tqdm
from cliffs_delta import cliffs_delta
import random
from  compare_mt import sign_utils
from compare_mt import scorers

EOF_STRINGS = ["<|endoftext|>", "</s>", "\n"]
def preprocess(i,flag=False):
    if flag:
        i = re.split("(%s)" % "|".join(EOF_STRINGS), i.strip())[0]
    return i.replace("\n", " ")

def random_sample(par=0.2):
    list1 = range(200)
    list2 = range(200,2000)
    re_sample = int(len(list1)*par)
    list2 = random.sample(list2,re_sample)
    list1 = random.sample(list1,len(list1)-re_sample)
    list1.extend(list2)
    return list1

def lang_sample():
    intervals = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000)]

    # 在每个区间内随机抽样40个索引
    sampled_indices = []
    for start, end in intervals:
        interval_indices = random.sample(range(start, end), 40)
        sampled_indices.extend(interval_indices)
    return sampled_indices

def get_path(task):
    if task == 'CodeReview':
        return 'CRG_196.jsonl'
    elif task == 'CommitMsgGeneration':
        return 'CMG_196.jsonl'

    

def cal(refs,hypos1,hypos2,srcs):
    direcs = [(0,1)]
    bs_name = 'bleu' 
    bs = scorers.create_scorer_from_profile(bs_name, case_insensitive=False)

    wins, sys_stats = sign_utils.eval_with_paired_bootstrap(refs, [hypos1, hypos2], srcs, bs,
                                                            num_samples=1000,
                                                    compare_directions=direcs,)
    res = str(wins)
    return res

def _cal(dis1,dis2):
    st1 = str(st.mannwhitneyu(dis1,dis2))
    st2 = str(cliffs_delta(dis1,dis2))
    return st1+'\n'+st2

def get_tokenizer(model):
    if model == 'slm':
        tokenizer = RobertaTokenizer.from_pretrained("/codet5-base")
    else:
        tokenizer =  AutoTokenizer.from_pretrained("codellama-7b",model_max_length=2048,use_fast=True)
    return tokenizer


def _cal_bleu(out,gold):
    item = out[0]
    if 'codet5' in item.split('\t')[0]:
        out = [item.split('\t')[-1] for item in out]
        gold = [item.split('\t')[-1] for item in gold]
        gold = [gold]
        out = [i.strip().split() for i in out]
        this_bleu = _bleu_RAM(gold, out)
    else:
        (goldMap, predictionMap) = bleu.computeMapsFromRAM(out, gold)
        this_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    return this_bleu


def other_cal(refs,hypos1,hypos2,srcs):
    direcs = [(0,1)]
    bs_name = 'bleu' 
    bs = scorers.create_scorer_from_profile(bs_name, case_insensitive=False)

    wins, sys_stats = sign_utils.eval_with_paired_bootstrap(refs, [hypos1, hypos2], srcs, bs,
                                                    compare_directions=direcs,)



def cacl_bleu(task,par):
    random_list = lang_sample()
    # random_list = range(2000)
    output_path = get_path(task)
    gold_res = []
    peft_res,icl_res,slm_res = [],[],[]
    with open(output_path, 'r') as file:
        datas = []
        for line in file:
            data = json.loads(line)
            datas.append(data)
        # read output
        for i in random_list:
            temp_sample = datas[i]
            data_index = int(temp_sample['peft'].split('\t')[0])
            gold_res.append(str(data_index)+'\t'+temp_sample['nl'])
            peft_res.append(temp_sample['peft'])
            icl_res.append(temp_sample['icl'])
            slm_res.append(str(data_index)+'\t'+temp_sample['slm'])
    # print(datas[3494])
    bleu_peft = _cal_bleu(peft_res,gold_res)
    peft_lines,icl_lines,slm_lines = [],[],[]
    for i,j in zip(peft_res,gold_res):
        this_bleu = _cal_bleu([i], [j])
        peft_lines.append(this_bleu)

    bleu_icl = _cal_bleu(icl_res,gold_res)
    for i,j in zip(icl_res,gold_res):
        this_bleu = _cal_bleu([i], [j])
        icl_lines.append(this_bleu)
    # bleu_icl_code = _cal_bleu(icl_code_res,gold_res)
    bleu_slm = _cal_bleu(slm_res,gold_res)
    for i,j in zip(slm_res,gold_res):
        this_bleu = _cal_bleu([i], [j])
        slm_lines.append(this_bleu)


    return bleu_peft,bleu_icl,bleu_slm,random_list



def calc_gleu_random(task):
    random_list = sorted(random_sample())
    output_path = get_path(task)
    gold_res = []
    src_res = []
    peft_res,icl_res,icl_code_res,slm_res = [],[],[],[]
    with open(output_path, 'r') as file:
        datas = []
        for line in file:
            data = json.loads(line)
            datas.append(data)
        # read output
        for i in random_list:
            temp_sample = datas[i]
            # print(temp_sample['icl-diff'])
            gold_res.append(preprocess(temp_sample['target']))
            src_res.append(preprocess(temp_sample['old']))
            peft_res.append(preprocess(temp_sample['peft'].split('\t')[-1],True))
            icl_res.append(preprocess(temp_sample['icl-diff'].split('\t')[-1],True))
            icl_code_res.append(preprocess(temp_sample['icl-code'].split('\t')[-1],True))
            slm_res.append(preprocess(temp_sample['slm'],True))
    tokenizer = {
        'slm':get_tokenizer('slm'),
        'llm':get_tokenizer('llm'),
    }
    src_tokens, gold_tokens = [], []
    peft_tokens, icl_tokens, icl_code_tokens, slm_tokens = [], [], [], []

    for gold,src, peft,icl,icl_code in zip(gold_res,src_res,peft_res, icl_res, icl_code_res):
        src_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(src)])
        gold_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(gold)])
        peft_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(peft)])
        icl_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(icl)])
        icl_code_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(icl_code)])
    peft_gleu = calcGleu(src_tokens,gold_tokens,peft_tokens,lowercase=True)
    icl_gleu = calcGleu(src_tokens,gold_tokens,icl_tokens,lowercase=True)
    icl_code_gleu = calcGleu(src_tokens,gold_tokens,icl_code_tokens,lowercase=True)

    src_tokens.clear()
    gold_tokens.clear()
    # print()
    for gold,src, slm in zip(gold_res,src_res,slm_res):
        src_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['slm'].tokenize(src)])
        gold_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['slm'].tokenize(gold)])
        slm_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['slm'].tokenize(slm)])
    slm_gleu = calcGleu(src_tokens,gold_tokens,slm_tokens,lowercase=True)
    return peft_gleu,icl_gleu,icl_code_gleu,slm_gleu,random_list


def ana_p_value(xianzhu):
    res = ''
    for item in xianzhu:
        p_value = float(item.split('\n')[0].split('pvalue=')[1].split(')')[0])
        print(p_value)
        if p_value<0.05:
            res+='True'
        else:
            res+='False'
    return res

def calc_gleu_for_all(path):
    gold_res = []
    src_res = []
    peft_res,icl_res,slm_res = [],[],[]
    xianzhu = []
    peft_lines, icl_lines, slm_lines = [], [], []

    with open(path, 'r') as file:
        datas = []
        for line in file:
            data = json.loads(line)
            datas.append(data)
        for i in range(len(datas)):  
            temp_sample = datas[i]
            gold_res.append(preprocess(temp_sample['nl']))
            src_res.append(preprocess(temp_sample['old_nl']))
            peft_res.append(preprocess(temp_sample['peft'].split('\t')[-1],True))
            icl_res.append(preprocess(temp_sample['icl'].split('\t')[-1],True))
            slm_res.append(preprocess(temp_sample['slm'],True))
    tokenizer = {
        'slm':get_tokenizer('llm'),
        'llm':get_tokenizer('llm'),
    }
    src_tokens, gold_tokens = [], []
    peft_tokens, icl_tokens, slm_tokens = [], [], []

    icl_gleu_res = [preprocess(i) for i in icl_res]

    for gold,src, peft,icl in zip(gold_res,src_res,peft_res, icl_gleu_res):
        src_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(src)])
        gold_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(gold)])
        peft_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(peft)])
        icl_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(icl)])
    peft_gleu = calcGleu(src_tokens,gold_tokens,peft_tokens,lowercase=True)
    icl_gleu = calcGleu(src_tokens,gold_tokens,icl_tokens,lowercase=True)
    
    for i, j, k in zip(src_tokens,gold_tokens, peft_tokens):
        this_bleu = calcGleu([i], [j], [k],lowercase=True)
        peft_lines.append(this_bleu)

    for i, j,k in zip(src_tokens,gold_tokens,icl_tokens):
        this_bleu = calcGleu([i], [j], [k],lowercase=True)
        icl_lines.append(this_bleu)

    src_tokens.clear()
    gold_tokens.clear()
    # print()
    for gold,src, slm in zip(gold_res,src_res,slm_res):
        src_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['slm'].tokenize(src)])
        gold_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['slm'].tokenize(gold)])
        slm_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['slm'].tokenize(slm)])
    slm_gleu = calcGleu(src_tokens,gold_tokens,slm_tokens,lowercase=True)

    for i, j,k in zip(src_tokens,gold_tokens,slm_tokens):
        this_bleu = calcGleu([i], [j], [k],lowercase=True)
        slm_lines.append(this_bleu)



    return peft_gleu,icl_gleu,slm_gleu

def calc_gleu_for_catgory(category,path):
    gold_res = []
    src_res = []
    peft_res,icl_res,slm_res = [],[],[]
    with open(path, 'r') as file:
        datas = []
        for line in file:
            data = json.loads(line)
            if  category == data['label'][0]  : 
                datas.append(data)
        print(f"Category {category} has {len(datas)} samples.")
        for i in range(len(datas)): 
            temp_sample = datas[i]
            gold_res.append(preprocess(temp_sample['nl']))
            src_res.append(preprocess(temp_sample['old_nl']))
            peft_res.append(preprocess(temp_sample['peft'].split('\t')[-1],True))
            icl_res.append(preprocess(temp_sample['icl'].split('\t')[-1],True))
            slm_res.append(preprocess(temp_sample['slm'],True))
    tokenizer = {
        'slm':get_tokenizer('slm'),
        'llm':get_tokenizer('llm'),
    }
    src_tokens, gold_tokens = [], []
    peft_tokens, icl_tokens, slm_tokens = [], [], []

    for gold,src, peft,icl in zip(gold_res,src_res,peft_res, icl_res):
        src_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(src)])
        gold_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(gold)])
        peft_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(peft)])
        icl_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['llm'].tokenize(icl)])
    peft_gleu = calcGleu(src_tokens,gold_tokens,peft_tokens,lowercase=True)
    icl_gleu = calcGleu(src_tokens,gold_tokens,icl_tokens,lowercase=True)

    src_tokens.clear()
    gold_tokens.clear()
    # print()
    for gold,src, slm in zip(gold_res,src_res,slm_res):
        src_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['slm'].tokenize(src)])
        gold_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['slm'].tokenize(gold)])
        slm_tokens.append([x.lower().replace("``", "\"") for x in tokenizer['slm'].tokenize(slm)])
    slm_gleu = calcGleu(src_tokens,gold_tokens,slm_tokens,lowercase=True)
    return peft_gleu,icl_gleu,slm_gleu



def cacl_bleu_for_all(task):
    output_path = get_path(task)
    gold_res = []
    peft_res, icl_res, slm_res = [], [], []
    codet5_res = []
    with open(output_path, 'r') as file:
        datas = []
        for line in file:
            data = json.loads(line)
            datas.append(data)
        for i in range(len(datas)):  
            temp_sample = datas[i]
            data_index = int(temp_sample['peft'].split('\t')[0])
            gold_res.append(str(data_index) + '\t' + temp_sample['nl'])
            peft_res.append(temp_sample['peft'])
            icl_res.append(temp_sample['icl'])
            slm_res.append(str(data_index) + '\t' + temp_sample['slm'])
            codet5_res.append('codet5:'+ '\t'+str(data_index)+' ' + temp_sample['codet5'])
    
    bleu_peft = _cal_bleu(peft_res, gold_res)
    peft_lines, icl_lines, slm_lines = [], [], []
    codet5_lines = []
    for i, j in zip(peft_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        peft_lines.append(this_bleu)

    bleu_icl = _cal_bleu(icl_res, gold_res)
    for i, j in zip(icl_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        icl_lines.append(this_bleu)

    bleu_slm = _cal_bleu(slm_res, gold_res)
    for i, j in zip(slm_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        slm_lines.append(this_bleu)
    
    bleu_codet5 = _cal_bleu(codet5_res, gold_res)
    for i, j in zip(codet5_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        codet5_lines.append(this_bleu)
    
    print('bleu_codet5:',bleu_codet5)


    return bleu_peft, bleu_icl, bleu_slm



def cacl_bleu_for_category(category, task):
    output_path = get_path(task)
    gold_res = []
    peft_res, icl_res, slm_res = [], [], []
    with open(output_path, 'r') as file:
        datas = []
        for line in file:
            data = json.loads(line)
            if  category == data['label'][0]  :  
                datas.append(data)
        print(f"Category {category} has {len(datas)} samples.")
        for i in range(len(datas)):  
            temp_sample = datas[i]
            data_index = int(temp_sample['peft'].split('\t')[0])
            gold_res.append(str(data_index) + '\t' + temp_sample['target'])
            peft_res.append(temp_sample['peft'])
            icl_res.append(temp_sample['icl-diff'])
            slm_res.append(str(data_index) + '\t' + temp_sample['slm'])
    
    bleu_peft = _cal_bleu(peft_res, gold_res)
    peft_lines, icl_lines, slm_lines = [], [], []
    for i, j in zip(peft_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        peft_lines.append(this_bleu)

    bleu_icl = _cal_bleu(icl_res, gold_res)
    for i, j in zip(icl_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        icl_lines.append(this_bleu)

    bleu_slm = _cal_bleu(slm_res, gold_res)
    for i, j in zip(slm_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        slm_lines.append(this_bleu)


    return bleu_peft, bleu_icl, bleu_slm

def cacl_bleu_for_language_and_category(language, category, task):
    output_path = get_path(task)
    gold_res = []
    peft_res, icl_res, slm_res = [], [], []
    with open(output_path, 'r') as file:
        datas = []
        for line in file:
            data = json.loads(line)
            if data['lang'] == language and data['label'][0] == category:  
                datas.append(data)
        if len(datas) == 0:
            return 0, 0, 0, []
        for i in range(len(datas)):  
            temp_sample = datas[i]
            data_index = int(temp_sample['peft'].split('\t')[0])
            gold_res.append(str(data_index) + '\t' + temp_sample['nl'])
            peft_res.append(temp_sample['peft'])
            icl_res.append(temp_sample['icl'])
            slm_res.append(str(data_index) + '\t' + temp_sample['slm'])
    
    bleu_peft = _cal_bleu(peft_res, gold_res)
    peft_lines, icl_lines, slm_lines = [], [], []
    for i, j in zip(peft_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        peft_lines.append(this_bleu)

    bleu_icl = _cal_bleu(icl_res, gold_res)
    for i, j in zip(icl_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        icl_lines.append(this_bleu)

    bleu_slm = _cal_bleu(slm_res, gold_res)
    for i, j in zip(slm_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        slm_lines.append(this_bleu)



    return bleu_peft, bleu_icl, bleu_slm


def cacl_bleu_for_language(language, task):
    output_path = get_path(task)
    gold_res = []
    peft_res, icl_res, slm_res = [], [], []
    codet5_res = []
    src_res = []
    with open(output_path, 'r') as file:
        datas = []
        for line in file:
            data = json.loads(line)
            if data['lang'] == language: 
                datas.append(data)
        if len(datas) == 0:
            return 0, 0, 0, []
        
        for temp_sample in datas:
            data_index = int(temp_sample['peft'].split('\t')[0])
            diff = temp_sample['diff']
            src_res.append(diff)
            gold_res.append(str(data_index) + '\t' + temp_sample['nl'])
            peft_res.append(temp_sample['peft'])
            icl_res.append(temp_sample['icl'])
            slm_res.append(str(data_index) + '\t' + temp_sample['slm'])
            codet5_res.append('codet5:' + '\t'+ temp_sample['codet5'])
    
    bleu_peft = _cal_bleu(peft_res, gold_res)
    bleu_icl = _cal_bleu(icl_res, gold_res)
    bleu_slm = _cal_bleu(slm_res, gold_res)
    bleu_codet5 = _cal_bleu(codet5_res, gold_res)
    
    peft_lines, icl_lines, slm_lines = [], [], []
    codet5_lines = []
    for i, j in zip(peft_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        peft_lines.append(this_bleu)

    for i, j in zip(icl_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        icl_lines.append(this_bleu)

    for i, j in zip(slm_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        slm_lines.append(this_bleu)
    
    for i, j in zip(codet5_res, gold_res):
        this_bleu = _cal_bleu([i], [j])
        codet5_lines.append(this_bleu)

    print('bleu_codet5:',bleu_codet5)


    return bleu_peft, bleu_icl, bleu_slm


def calc_bleu_for_language_and_tech(language, tech, task):
    output_path = get_path(task)
    gold_res = []
    tech_res = []
    src_res = []
    datas = []
    with open(output_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['lang'] == language:
                datas.append(data)
    for data in datas:
        data_index = int(data['peft'].split('\t')[0])
        gold_res.append(str(data_index) + '\t' + data['nl'])
        # gold_res.append(data['nl'])
        if tech=='slm':
            tech_res.append(str(data_index) + '\t' + data['slm'])
            # tech_res.append(data['slm'])
        elif tech=='codet5':
            tech_res.append('codet5:' + '\t' + data[tech])
            # tech_res.append(data[tech])
        else:
            tech_res.append(data[tech])
            # tech_res.append(data[tech].split('\t')[-1])
        src_res.append(data['diff'])
    if len(gold_res) == 0:
        return []
    
    this_bleu = _cal_bleu(tech_res, gold_res)
    bleu_scores = []
    for i, j in zip(tech_res, gold_res):
        bleu_score = _cal_bleu([i], [j])
        bleu_scores.append(bleu_score)
    
    return this_bleu,bleu_scores
    # return tech_res, gold_res,src_res


def compare_languages_for_tech(lang1, lang2, tech, task):
    this_bleu1,bleu_scores1 = calc_bleu_for_language_and_tech(lang1, tech, task)
    this_bleu2,bleu_scores2 = calc_bleu_for_language_and_tech(lang2, tech, task)
    # tech_res1, gold_res1,src_res1 = calc_bleu_for_language_and_tech(lang1, tech, task)
    # tech_res2, gold_res2,src_res2 = calc_bleu_for_language_and_tech(lang2, tech, task)
    
    if len(bleu_scores1) == 0 or len(bleu_scores2) == 0:
        return f"无法比较 {lang1} 和 {lang2} 在 {tech} 下的性能。"
    print(f"{lang1}:{bleu_scores1[:10]} \n {lang2}:{bleu_scores2[:10]} ")
    # result = cal([gold_res1,gold_res2],tech_res1, tech_res2,[src_res1,src_res2])
    result = _cal(bleu_scores1, bleu_scores2)
    return f"{lang1}:{this_bleu1} 和 {lang2}:{this_bleu2} 在 {tech} 下的性能比较: {result}"



if __name__ == '__main__':


    categories = ['Doc-mod', 'Feat-mod', 'Ref-con', 'Doc&Code']
    for category in categories:
        bleu_peft, bleu_icl, bleu_slm = cacl_bleu_for_category(category, 'CommitMsgGeneration')
        print(f"BLEU scores: PEFT={bleu_peft}, ICL={bleu_icl}, SLM={bleu_slm}")


    
    categories = ['Doc', 'Feat', 'Ref', 'Code&Doc']
    path = 'JITCU_196.jsonl'
    for category in categories:
        peft_gleu,icl_gleu,bleu_slm = calc_gleu_for_catgory(category,path)
        print(f'slm_gleu:{bleu_slm},icl_gleu:{icl_gleu}, peft_gleu:{peft_gleu}')


    categories = ['Doc', 'Feat', 'Ref', 'Code&Doc']
    path = 'CRG_196.jsonl'
    for category in categories:
        bleu_peft, bleu_icl, bleu_slm = cacl_bleu_for_category(category, 'CodeReview')
