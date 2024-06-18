import copy
import logging
from dataclasses import dataclass, field
import pickle
from typing import Dict, Optional, Sequence
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    PeftModel,
    get_peft_config,
    PrefixTuningConfig,
    TaskType,
    PeftType,
    PromptTuningConfig,
    PromptEncoderConfig
)
import os
import torch
import transformers
from rq2utils import get_task_dataset_path
from torch.utils.data import Dataset
from transformers import Trainer,BitsAndBytesConfig
from datasets import load_dataset
import datasets
import sys
from accelerate import Accelerator


sys.path.append('../')
from utils import *

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

os.environ['TOKENIZERS_PARALLELISM']='true'

# accelerator = Accelerator()
@dataclass
class HyperArguments():
    model_path: Optional[str] = field(default="facebook/opt-125m")
    peft_method: Optional[str] = field(default="lora",
                                       metadata={"help": "peft method to use."})
    task: Optional[str] = field(
        default='',metadata={"help": "The target task to test on."}
    )
    input_form: Optional[str] = field(
        default='', metadata={"help": "code,diff,nlp. nlp for the alpaca dataset, code and diff for the other dataset"}
    )
    langs: Optional[str] = field(
        default='', metadata={"help": "langs for finetuning in CMG"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    learning_rate: float = field(default=2e-5)
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    output_dir:Optional[str] = field(
        default='./RQ2-PEFT-model',
        metadata={'help': "model save dir"}     
    )
    warmup_steps:Optional[int] = field(
        default=5,
        metadata={'help': "warm up steps"}
    )
    logging_steps:Optional[int] = field(
        default=10,
        metadata={'help': "logging steps"}     
    )
    save_strategy:Optional[str] = field(
        default='epoch',
        metadata={'help': "save strategy"}     
    )
    # save_steps: Optional[float] = field(
    #     default=1000,
    #     metadata={'help': "save steps"}     
    # )
    # max_steps: Optional[int] = field(
    #     default=3000,
    #     metadata={'help': "max steps"}     
    # )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={'help': "save total limit"}     
    )
    weight_decay:Optional[float] = field(
        default=0.05
    )
    report_to:Optional[str] = field(
        default='wandb',
        metadata={'help': "report to"}
    )
    num_train_epochs:Optional[int] = field(
        default=3,
        metadata={'help': "epochs to train"}
    )
    per_device_train_batch_size:Optional[int] = field(
        default=1
    )
    resume_from_checkpoint: Optional[str] = field(default='',metadata={'help':'either training checkpoint or final adapter'})
    lr_scheduler_type: Optional[str] = field(default='cosine',metadata={'help':'lr scheduler type'})
    gradient_accumulation_steps: Optional[int] = field(default=4,metadata={'help':'gradient accumulation steps'})
    gradient_checkpointing: Optional[bool] = field(default=True,metadata={'help':'gradient checkpointing'})
    # fp16: Optional[bool] = field(default=True,metadata={'help':'fp16'})
    bf16: Optional[bool] = field(default=True,metadata={'help':'bf16'})
    torch_compiler: Optional[bool] = field(default=True,metadata={'help':'torch compiler'})
    max_grad_norm: Optional[float] = field(default=2.0,metadata={'help':'max grad norm'})
    
def get_input_output_pattern():
    input = "### Instruction:\n"
    output = "### Answer:\n"
    return input, output


def format_instance(example, task, args, tokenizer):
    INPUT, OUTPUT = get_input_output_pattern()
    format_length = len(tokenizer.tokenize(INPUT+' '+OUTPUT))
    if task == 'CodeReview':
        if args.input_form =='diff':
            instruction = f'Please write a code review according to the diff hunk.'
            example['input_code'] = f'diff hunk:\n{example["patch"]}'
            example['input_code'] = format_diff(example['input_code'], tokenizer, tokenizer.model_max_length, format_length, instruction, task, label=example['msg'])
        else:
            instruction = f'Please write a code review according to the code before and after the diff hunk.'
            code_before, code_after = from_CMG_diff_get_code(example['patch'])
            code_before_token = f"code change before:\n{code_before}"
            code_after_token = f"code change after:\n{code_after}"
            code_before_token, code_after_token = format_code(code_before_token, code_after_token, tokenizer, tokenizer.model_max_length, format_length, instruction, task, label=example['msg'])
            example['input_code'] = f'{code_before_token}\n{code_after_token}'
        example['context'] = example['input_code']
        example['response'] = example['msg']
    elif task == 'CommitMsgGeneration':
        if args.input_form=='diff':
            instruction = f'Please write a commit message according to the diff hunk.'
            example['input_code'] = f'diff hunk:\n{example["diff"]}' 
            example['input_code'] = format_diff(example['input_code'], tokenizer, tokenizer.model_max_length,format_length, instruction, task, label=example['nl'])
        else:
            instruction = f'Please write a commit message according to the code before and after the diff hunk.'
            code_before, code_after = from_codisum_diff_get_code(example['diff'])
            code_before_token = f"code change before:\n{code_before}"
            code_after_token = f"code change after:\n{code_after}"
            code_before_token, code_after_token = format_code(code_before_token, code_after_token, tokenizer, tokenizer.model_max_length, format_length, instruction, task, label=example['nl'])
            example['input_code'] = f'{code_before_token}\n{code_after_token}'
        example['context'] = example['input_code']
        example['response'] = example['nl']
    elif task == 'JITCommentUpdate':
        if args.input_form =='diff':
            instruction = f'Please write a new comment according to the original comment and the diff hunk.'
            ori_comment_token = f"original comment message:\n{example['old_nl']}"
            ori_comment_token_len = len(tokenizer.tokenize(ori_comment_token)) 
            example['input_code'] = f'diff hunk:\n{example["diff"]}' 
            example['input_code'] = format_diff(example['input_code'], tokenizer, tokenizer.model_max_length-ori_comment_token_len, format_length, instruction, task, label=example['nl'])
            example['input_code'] = example['input_code'] + f'\n{ori_comment_token}'
        else:
            instruction = f'Please write a new comment according to the original comment and the code before and after the diff hunk.'
            code_before, code_after = from_JITCUP_diff_get_code(example['diff'])
            code_before_token = f"code change before:\n{code_before}"
            code_after_token = f"code change after:\n{code_after}"
            ori_comment_token = f"original comment message:\n{example['old_nl']}"
            ori_comment_token_len = len(tokenizer.tokenize(ori_comment_token))            
            code_before_token, code_after_token = format_code(code_before_token, code_after_token, tokenizer, tokenizer.model_max_length-ori_comment_token_len, format_length, instruction, task, label=example['nl'])
            example['input_code'] = f'{code_before_token}\n{code_after_token}\n{ori_comment_token}'
        example['context'] = example['input_code']
        example['response'] = example['nl']  

    input_prompt = (f"{INPUT}{instruction}\n"
                    f"{example['context']}\n"
                    f"{OUTPUT}")

    return {'text':input_prompt}


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer,
    model: transformers.PreTrainedModel,
):

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    # logging.warning(examples[0])
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    return dict(input_ids=input_ids, labels=labels)


def get_train_dataset(args,tokenizer):
    dataset_paths = get_task_dataset_path(args.task, do_train=True,langs=args.langs) 
    logging.warning(f"Loading {dataset_paths[0]} data from {dataset_paths[1]}...")
    dataset = load_dataset('json', data_files=dataset_paths[1], split='train', num_proc=4)
    dataset = dataset.select(list(range(16000))).shuffle(seed=42)
    logging.warning(f"formatting {len(dataset)} examples...")
    dataset_temp = dataset.map(lambda x:format_instance(x,args.task,args,tokenizer), num_proc=1)
    logging.warning(f'formatting {len(dataset_temp)} examples finished...')
    dataset_res = datasets.Dataset.from_dict({'text':dataset_temp['text'],'input_ids':dataset_temp['response']})
    return dataset_res


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        logging.warning("Formatting inputs...")

        list_data_dict = get_train_dataset(data_args,tokenizer)
        sources = list_data_dict['text']
        targets = [f"{example['input_ids']}{tokenizer.eos_token}" for example in list_data_dict]
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer, data_args)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    # print(train_dataset[0])
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((HyperArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    
    device_map = 'auto'
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    # gradient_accumulation_steps = training_args.gradient_accumulation_steps

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=training_args.model_max_length,
        use_fast=True,
    )
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if args.peft_method == 'lora':
        if 'llama' in args.model_path.lower():
            lora_target_modules = ['q_proj','v_proj','k_proj','o_proj'] 
        elif 'codegen' in args.model_path.lower():
            lora_target_modules = ["qkv_proj"]
        elif 'incoder' in args.model_path.lower():
            lora_target_modules = ["q_proj","k_proj","v_proj","out_proj"]
        logging.warning(f"Target modules: {lora_target_modules}")
        config = LoraConfig(
            r=8,                    # follow lora paper
            lora_alpha=16,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
    elif args.peft_method == 'prefix':
        config = PrefixTuningConfig(
            num_virtual_tokens=10,  # follow prefix paper setting
            inference_mode=False,
            task_type="CAUSAL_LM",
        )
    elif args.peft_method == 'prompt':
        config = PromptTuningConfig(
            task_type='CAUSAL_LM',
            num_virtual_tokens=100,
        )
    elif args.peft_method == 'pt':
        config = PromptEncoderConfig(
            task_type='CAUSAL_LM',
            num_virtual_tokens=20,
            encoder_hidden_size=128
        )
        
    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    args.model_name = args.model_path.lower().split('/')[3]
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=args)
    langs_prefix = f'_{args.langs}' if args.langs!='' else ''
    training_args.run_name=f'RQ2_{args.peft_method}_{args.model_name}_{args.task}{langs_prefix}_{args.input_form}'
    training_args.ddp_find_unused_parameters = False 
    print(data_module['train_dataset'][0])
    print(tokenizer.decode(list(data_module['train_dataset'][0]['input_ids'])))
    training_args.output_dir = training_args.output_dir.replace('PEFT',args.peft_method)
    output_dir = os.path.join(training_args.output_dir, args.model_name, args.task,args.langs, args.input_form)
    print('output_dir:',output_dir)
    os.makedirs(output_dir, exist_ok=True)
    training_args.output_dir = output_dir
    
    # model,data_module = accelerator.prepare(model, **data_module)
    
    trainer = Trainer(model=model, 
                      tokenizer=tokenizer, 
                      args=training_args, 
                      **data_module)
    trainer.train()
    logging.warning(f"Saving model checkpoint to {output_dir}")
    trainer.save_state()

    return args, training_args


# def evaluate(args):
    
    

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    args, training_args = train()
    
    
