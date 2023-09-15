import pickle

import torch

from model.nl2code import nl2code
from utils import decode_action_user
from playground import user_intent_call
import gradio as gr


def user(gridsearch, args, map_location, act_dict, grammar, primitives_type, device, is_cuda):

    for params in gridsearch.generate_setup():

        params['dataset'] = "conala"

        path_folder_config = './outputs/{0}.dataset_{1}._word_freq_{2}.nl_embed_{3}.action_embed_{4}.att_size_{5}.hidden_size_{6}.epochs_{7}.dropout_enc_{8}.dropout_dec_{9}.batch_size{10}.parent_feeding_type_{11}.parent_feeding_field_{12}.change_term_name_{13}.seed_{14}/'.format(
                    params['model'],
                    params['dataset'],
                    params['word_freq'],
                    params['nl_embed_size'],
                    params['action_embed_size'],
                    params['att_size'],
                    params['hidden_size'],
                    params['epochs'],
                    params['dropout_encoder'],
                    params['dropout_decoder'],
                    params['batch_size'],
                    params['parent_feeding_type'],
                    params['parent_feeding_field'],
                    params['change_term_name'],
                    params['seed']
                )

        vocab = pickle.load(open(path_folder_config + 'vocab', 'rb'))

        model = nl2code(params, act_dict, vocab, grammar, primitives_type, device, path_folder_config)

        print(path_folder_config)

        model.to(device)

        model.load_state_dict(
            torch.load(
                path_folder_config + 'model.pt',
                map_location=map_location
            ),
        )

        model.eval()

        def user_call(user_input):

            user_intent = user_input
            user_intent, slot_map = user_intent_call(user_intent)
            # user_intent = str(user_intent)
            out = decode_action_user(model, act_dict, is_cuda, user_intent, slot_map)
            return out
        
        demo = gr.Interface(fn=user_call, inputs="text", outputs="text", title="Code Generation From Natural Language Using Transformers")

        demo.launch()   