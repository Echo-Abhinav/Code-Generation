import pickle
import sys
import time

import torch
from torch import optim

from model.nl2code import nl2code
from utils import evaluate_action, epoch_time


def train(train_set, dev_set, args, gridsearch, act_dict, grammar, primitives_type, device, map_location,
          is_cuda):

    for params in gridsearch.generate_setup():

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

        epoch = train_iter = 0
        report_loss = report_examples = 0.
        patience = num_trial = 0

        model = nl2code(params, act_dict, vocab, grammar, primitives_type, device, path_folder_config)

        print(model)
        print(vocab)

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=params['lr'])

        best_metric = -0.1

        while epoch <= params['epochs']:
            is_better = False
            epoch += 1
            start_time = time.time()
            model.train()
            for batch_examples in train_set.batch_iter(params['batch_size'], shuffle=True):

                batch_examples = [e for e in batch_examples if len(eval(e.snippet_actions)) <= params['len_max']]

                train_iter += 1
                optimizer.zero_grad()

                ret_val = model.score(batch_examples)
                loss = -ret_val[0]

                loss_val = torch.sum(loss).data.item()
                report_loss += loss_val
                report_examples += len(batch_examples)
                loss = torch.mean(loss)

                loss.backward()

                optimizer.step()

                torch.cuda.empty_cache()

            log_str = '[EPOCH %d] loss_train=%.5f' % (epoch, report_loss / report_examples)

            print(log_str, file=sys.stderr)

            report_loss = report_examples = report_examples_dev = 0.

            print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - start_time), file=sys.stderr)

            if args.dev_path_conala:
                report_examples_dev = report_loss_val = 0.

                model.eval()

                metric, metric_2 = evaluate_action(dev_set.examples, model, act_dict, params['metric'], is_cuda)

                end_time = time.time()

                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                if params['metric'] == 'BLEU':
                    print('Epoch: {0} | Time: {1}m {2}s, {3}={4}, accuracy={5} '.format(epoch, epoch_mins, epoch_secs, params['metric'], metric, metric_2))
                elif params['metric'] == 'accuracy':
                    print('Epoch: {0} | Time: {1}m {2}s, {3}={4}, BLEU={5} '.format(epoch, epoch_mins, epoch_secs, params['metric'], metric, metric_2))

                if metric > best_metric:
                    best_metric = metric
                    torch.save(model.state_dict(),
                               path_folder_config + 'model.pt'
                               )

                    is_better = True

                    torch.save(optimizer.state_dict(), path_folder_config + 'optim.bin')

                if is_better:
                    patience = 0
                elif patience < params['patience'] and epoch >= params['lr_decay_after_epoch']:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                if patience >= params['patience'] and epoch >= params['lr_decay_after_epoch']:
                    num_trial += 1
                    print('hit #%d trial' % num_trial, file=sys.stderr)
                    if num_trial == params['max_num_trial']:
                        print('early stop!', file=sys.stderr)
                        break

                    # decay lr, and restore from previously best checkpoint
                    lr = optimizer.param_groups[0]['lr'] * params['lr_decay']
                    print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                    # load model
                    model.load_state_dict(
                        torch.load(
                            path_folder_config + 'model.pt',
                            map_location=map_location
                        ),
                    )

                    model.to(device)

                    print('restore parameters of the optimizers', file=sys.stderr)

                    optimizer.load_state_dict(torch.load(
                        path_folder_config + 'optim.bin'))

                    # set new lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    patience = 0
