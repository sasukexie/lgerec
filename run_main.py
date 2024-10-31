import argparse

from recbole.quick_start import run
from datetime import datetime
import common.tool as tool
import platform
import torch

RUNNING_FLAG = None


def main(model_name, dataset_name, parameter_dict, config_file=None):
    # 1.set param
    parser = argparse.ArgumentParser()
    # set model
    parser.add_argument('--model', '-m', type=str, default=model_name, help='name of models')
    # set datasets # ml-1m,ml-20m,amazon-books,lfm1b-tracks
    parser.add_argument('--dataset', '-d', type=str, default=dataset_name, help='name of datasets')
    # set config
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    # get param
    args, _ = parser.parse_known_args()
    # config list
    config_file_list = ['zone/common.yaml']

    if config_file:
        config_file_list.append(f'zone/{config_file}.yaml')

    global RUNNING_FLAG
    RUNNING_FLAG = f'RF{datetime.now().strftime("%Y%m%d%H%M%S")}' if RUNNING_FLAG == None else RUNNING_FLAG
    parameter_dict['running_flag'] = RUNNING_FLAG
    system_name = platform.system()
    if system_name == 'Windows':
        parameter_dict['gpu_id'] = '0'
    elif system_name == 'Linux':
        pass

    # set multi-proc
    nproc = 1
    world_size = -1
    # nproc = torch.cuda.device_count()
    # gpu_id = ''
    # if nproc>1:
    #     world_size = nproc*2
    #     for i in range(nproc):
    #         gpu_id += f'{i},'
    # parameter_dict['gpu_id'] = gpu_id

    # 2.call recbole_trm: config,dataset,model,trainer,training,evaluation
    run(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict, nproc=nproc, world_size=world_size)


def process_0(parameter_dict):
    # param
    # set model # MODEL,SimDCL,SASRec,BERT4Rec,BPR,GRU4RecF
    # set datasets # ml-1m,ml-20m,Amazon_Books,Amazon_Sports_and_Outdoors,Amazon_All_Beauty,amazon-books,lfm1b-tracks
    model_name_arr = ['LGERec']  # GRU4RecF,BPR
    dataset_name_arr = ['ml-100k']  # ] #
    for model_name in model_name_arr:
        for dataset_name in dataset_name_arr:
            main(model_name, dataset_name, parameter_dict)


def process_1(parameter_dict, dataset_name_arr):
    # param
    # set model
    model_name = 'LGERec'
    parameter_dict1 = {
        'embedding_size': 64,
        'num_heads': 6, # ml 8,16, yelp 8
        'n_layers': 2, # ml 0.1, yelp 0.1
        'dropout': 0.1, # ml 0.1, yelp 0.1
        'neg_topk': 5,
        'llm_embed_weight': 0.01, # not #ml 0.01,yelp 0.01,lfm1b 0.01
        'stopping_step': 20,
        'cl_weight': 0., # 0.1,0.01
        'sem_weight': 0.,  # 0.01
        'a_cl_weight': 0., # 0.1,0.01
        'reg_weight': 1e-05,
        'cl_temperature': 1.0, # ml 0.1,1.0, yelp 0.1
        'is_all_embed': True, # ml True, yelp False
        'is_open_llm': True,
        'is_open_attention': True,
        'flag': '#all_llm',
        # 'train_neg_sample_args': None,
    }

    tool.tranfer_dict(parameter_dict, parameter_dict1)
    pre_model = 'Llama-2-7b-hf'  # Llama-2-7b-hf,Qwen2.5-14B-Instruct
    system_name = platform.system()
    if system_name == 'Windows':
        print("This is a Windows System")
        parameter_dict['pre_model'] = f'E:/data/dataset/rs/llm/{pre_model}'
    elif system_name == 'Linux':
        print("This is a Linux System")
        parameter_dict['pre_model'] = f'/media/data/dataset/llm/{pre_model}'

    for dataset_name in dataset_name_arr:
        main(model_name, dataset_name, parameter_dict)


if __name__ == '__main__':
    parameter_dict = {
        'epochs': 500,
        'train_batch_size': 4096,
        'eval_batch_size': 4096,
        'gpu_id': '0',  # (str) The id of GPU device(s).
    }
    # param
    # set model # MODEL,SimDCL,SASRec,BERT4Rec,BPR,GRU4RecF
    # set datasets # ['steam','lfm1b-tracks','ml-1m']
    # process_base()

    # model & dataset
    dataset_name_arr = ['ml-100k']  # ['yelp2022','lfm1b-tracks','ml-1m','ml-100k']
    process_1(parameter_dict, dataset_name_arr)
