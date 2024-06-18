

from functools import partial
import jsonlines
from gensim import corpora
from gensim.summarization import bm25
from sklearn.inspection import partial_dependence
from tqdm import tqdm
import time
from datasets import Dataset

def process_item(obj,train,text_field,BM25model,average_idf):
    query = obj[text_field]
    query = query.split()
    # query = " ".join(query.split()[:256])
    score = BM25model.get_scores(query,average_idf)
    # score_end_time = time.time()
    rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:32] 
    # rtn_time = time.time()
    # print(f'score_time:{score_end_time-start_time},rtn_time:{rtn_time-score_end_time}')
    code_candidates_tokens = []
    for i in range(len(rtn)):
        # code_candidates_tokens.append({'code_tokens': train[rtn[i][0]]['code_tokens'], 'docstring_tokens': train[rtn[i][0]]['docstring_tokens'], 'score': rtn[i][1], 'idx':i+1})            
        temp_sample = train[rtn[i][0]]
        # temp_sample['diff'] = ' '.join(temp_sample['diff'].split()[:256])
        # temp_sample['score'] = rtn[i][1]
        # temp_sample['idx'] = i+1
        code_candidates_tokens.append(temp_sample)
    obj[f'{text_field}_candidates_tokens'] = code_candidates_tokens

    return obj

def bm25_preprocess(train,test,text_field):
    """
    """
    code = train[text_field]
    code = [' '.join(obj.split()) for obj in code ]
    # code = [obj.split()[:1024] for obj in code ]
    bm25_model = bm25.BM25(code)
    average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
    
    #multi-processing
    partial_process_item = partial(process_item, train=train, text_field=text_field, BM25model=bm25_model, average_idf=average_idf)
    res = run_imap_mp(partial_process_item, test,num_processes=50, is_tqdm=True)
    test = Dataset.from_list(res)
    return test    


def run_imap_mp(func, argument_list, num_processes='', is_tqdm=True):

    result_list_tqdm = []
    try:
        import multiprocessing
        if num_processes == '':
            num_processes = 50
        pool = multiprocessing.Pool(processes=num_processes)
        if is_tqdm:
            from tqdm import tqdm
            for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
                result_list_tqdm.append(result)
        else:
            for result in pool.imap(func=func, iterable=argument_list):
                result_list_tqdm.append(result)
        pool.close()
    except:
        result_list_tqdm = list(map(func,argument_list))   
    return result_list_tqdm