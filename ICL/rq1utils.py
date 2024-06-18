def get_task_train_test_dataset_path(task, lang='java'):
    if task == 'CodeReview':
        train_path = '../Dataset/CodeReview/train.jsonl'
        test_path = '../Dataset/CodeReview/test.jsonl'
    elif task =='JITCommentUpdate':
        train_path = '../Dataset/JITCommentUpdate/train.jsonl'
        test_path = '../Dataset/JITCommentUpdate/test.jsonl'
    elif task =='CoDiSum':
        train_path = '../Dataset/CoDiSum/train.jsonl'
        test_path = '../Dataset/CoDiSum/test.jsonl'
    elif task =='CommitMsgGeneration':
        train_path = f'../Dataset/CommitMsgGeneration/train_{lang}_random_16000.jsonl'
        test_path = f'../Dataset/CommitMsgGeneration/test_{lang}_random_2000.jsonl'
    return train_path, test_path
