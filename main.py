import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml
import json
import gradio as gr

from asdl.grammar import GrammarRule, Grammar, ReduceAction
from dataset.dataset import Dataset
from config.config import init_arg_parser
from test import test, test_sim
from train import train
from user import user
from utils import GridSearch

if __name__ == '__main__':

    # load config file
    args = init_arg_parser()
    parameters = yaml.load(open(args.config_file).read(), Loader=yaml.FullLoader)
    params = parameters['experiment_env']

    # Fix seed for reproducing results
    SEED = params['seed']

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    is_cuda = torch.cuda.is_available()

    print("Cuda Status on system is {}".format(is_cuda))

    if params['dataset'] is 'conala' or 'codesearchnet':
        asdl_text = open('./asdl/grammar.txt').read()
    if params['dataset'] is 'django':
        asdl_text = open('./asdl/grammar2.txt').read()
    all_productions, grammar, primitives_type = Grammar.from_text(asdl_text)
    act_list = [GrammarRule('<pad>', None, []), GrammarRule('<s>', None, []),
                GrammarRule('</s>', None, []), GrammarRule('<unk>', None, [])]

    act_list += [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in all_productions]

    Reduce = ReduceAction('Reduce')
    act_dict = dict(
        [(act.label, act) if isinstance(act, GrammarRule) or isinstance(act, ReduceAction) else (act, act) for act in
         act_list])
    act_dict[Reduce.label] = Reduce
    
    # Depends on the graphic card and the NVCC version. -> (10.1, 9.2, ...) or we can do cpu
    device = torch.device('cuda:%s' % (params['GPU']) if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda(params['GPU'])
    else:
        map_location = 'cpu'

    gridsearch = GridSearch(params)

    if params['train']:
        # Load train set -> CoNaLa or Django 
        if params['dataset'] == 'conala':
            train_set = Dataset(pd.read_csv(args.train_path_conala + 'conala-train.csv'))
            dev_set = Dataset(pd.read_csv(args.train_path_conala + 'conala-val.csv'))
        elif params['dataset'] == 'django':
            train_set = Dataset(pd.read_csv(args.train_path_django + 'train.csv'))
            dev_set = Dataset(pd.read_csv(args.dev_path_django + 'dev.csv'))
        else:
            sys.exit('Wrong train path for' + params['dataset'])

        train(train_set, dev_set, args, gridsearch, act_dict, grammar, primitives_type, device, map_location, is_cuda)

    if params['test']:
        # Load test set -> CoNaLa or Django 
        if params['dataset'] == 'conala':
            test_set = Dataset(pd.read_csv(args.test_path_conala + 'conala-test.csv'))
            out = test(test_set, gridsearch, args, map_location, act_dict, grammar, primitives_type, device, is_cuda)
            with open("conala_test_output.json", "w") as final:
                json.dump(out, final, indent=4)

        elif params['dataset'] == 'django':
            test_set = Dataset(pd.read_csv(args.test_path_django + 'test.csv'))
            out = test(test_set, gridsearch, args, map_location, act_dict, grammar, primitives_type, device, is_cuda)
            with open("django_test_output.json", "w") as final:
                json.dump(out, final, indent=4)

        elif params['dataset'] == 'user':
            user(gridsearch, args, map_location, act_dict, grammar, primitives_type, device, is_cuda)
        
        elif params['dataset'] == 'sim':
            test_set = Dataset(pd.read_csv(args.test_path_django + 'test.csv'))
            out = test_sim(test_set, gridsearch, args, map_location, act_dict, grammar, primitives_type, device, is_cuda)
            with open("sim_test_output.json", "w") as final:
                json.dump(out, final, indent=4)
        else:
            sys.exit('Wrong test path for' + params['dataset'])
