


def get_model_name(string):
    if 'codellama' in string:
        return 'codellama'
    elif 'polycoder' in string:
        return 'polycoder'
    elif 'starcoder' in string:
        return 'starcoder'
    elif 'llama-2' in string:
        return 'llama-2'

def get_task_dataset_path(task,do_train=True,input_form='code',langs=''):
    if task == 'CodeReview':
        path = '../Dataset/CodeReview/train.jsonl'
    elif task =='CommitMsgGeneration':
        path = f'../Dataset/CommitMsgGeneration/train_{langs}_random_16000.jsonl'
    elif task =='JITCommentUpdate':
        path = '../Dataset/JITCommentUpdate/train.jsonl'
    elif task == 'CodeRefinement':
        path = '../Dataset/CodeRefinement/ref-train.jsonl'
    elif task == 'JITDefectPrediction':
        path = '../Dataset/JITDefectPrediction/changes_train_fixed.jsonl'
    elif task == 'QualityEstimation':
        path = '../Dataset/QualityEstimation/train.jsonl'
    elif task =='CoDiSum':
        path = '../Dataset/CoDiSum/train.jsonl'
    elif task =='CoREC':
        path ='../Dataset/CoREC/cleaned_train.jsonl'
    return [task,path]
