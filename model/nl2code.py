import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformers import BertModel

from asdl.grammar import GrammarRule
from asdl.hypothesis import DecodeHypothesis
from dataset.dataset import Batch
from model import nn_utils
from model.nn_utils import LabelSmoothing
from model.pointernet import PointerNet
import ast


class nl2code(nn.Module):
    def __init__(self, parameters, act_dict, vocab, grammar, primitives_type, device, path_folder_config):
        super(nl2code, self).__init__()

        self.args = parameters
        self.vocab = vocab
        self.act_dict = act_dict
        self.primitives_type = primitives_type
        self.device = device
        self.grammar = grammar

        # source token embedding
        self.nl_embed = nn.Embedding(len(vocab.source), parameters['nl_embed_size'])

        # embedding table of ASDL rules
        # the last entry is the embedding for Reduce action
        self.action_embed = nn.Embedding(len(act_dict), parameters['action_embed_size'])

        # embedding table for target primitive tokens
        self.primitive_embed = nn.Embedding(len(vocab.primitive), parameters['action_embed_size'])

        # embedding table for fields in constructors
        self.field_embed = nn.Embedding(len(grammar.field2id), parameters['field_embed_size'])
        self.type_embed = nn.Embedding(len(grammar.type2id), parameters['type_embed_size'])
        self.cardinalities = nn.Embedding(len(grammar.cardinalities), parameters['cardinality_embed_size'])

        # Chose your architecture
        if self.args['model'] == 'bert':
            self.tokenizer = pickle.load(open(path_folder_config + 'bert_tokenizer', 'rb'))

            self.encoder = BertModel.from_pretrained('bert-base-uncased')

            self.encoder.resize_token_embeddings(len(self.tokenizer))
        else:
            self.encoder = nn.LSTM(self.args['nl_embed_size'], int(parameters['hidden_size'] / 2), bidirectional=True)

        input_dim = self.args['action_embed_size']  # previous action
        input_dim += self.args['att_size']

        if self.args['parent_feeding_field']:
            input_dim += self.args['field_embed_size']
        if self.args['parent_feeding_type']:
            input_dim += self.args['type_embed_size']

        if self.args['copy'] == False:
            self.label_smoothing = LabelSmoothing(self.args['primitive_token_label_smoothing'], len(self.vocab.primitive),
                                                  ignore_indices=[0, 1, 2])

        self.decoder_lstm = nn.LSTMCell(input_dim, self.args['hidden_size'])

        if self.args['copy'] is True:
            # pointer net for copying tokens from source side
            self.src_pointer_net = PointerNet(query_vec_size=self.args['att_size'],
                                              src_encoding_size=self.args['hidden_size'])

            self.primitive_predictor = nn.Linear(self.args['att_size'], 2)

        # Reshape BERT embeddings to hidden_size
        if self.args['model'] == 'bert':
            self.src_enc_linear = nn.Linear(768, self.args['hidden_size'], bias=False)

        # attention: dot product attention
        # project source encoding to decoder rnn's hidden space
        self.att_src_linear = nn.Linear(self.args['hidden_size'], self.args['hidden_size'], bias=False)

        self.att_vec_linear = nn.Linear(self.args['hidden_size'] + self.args['hidden_size'], self.args['att_size'],
                                        bias=False)

        # set dropout parameters
        self.dropout_encoder = nn.Dropout(parameters['dropout_encoder'])
        self.dropout_decoder = nn.Dropout(parameters['dropout_decoder'])

        # bias for predicting ApplyConstructor and GenToken actions
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(vocab.code)).to(self.device).zero_())
        self.tgt_token_readout_b = nn.Parameter(torch.FloatTensor(len(vocab.primitive)).to(self.device).zero_())

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(parameters['hidden_size'], parameters['hidden_size'])

        self.query_vec_to_action_embed = nn.Linear(self.args['att_size'], self.args['action_embed_size'],
                                                   bias=self.args['readout'] == 'non_linear')
        if self.args['query_vec_to_action_diff_map']:
            self.query_vec_to_primitive_embed = nn.Linear(self.args['att_size'], self.args['nl_embed_size'],
                                                          bias=self.args['readout'] == 'non_linear')
        else:
            self.query_vec_to_primitive_embed = self.query_vec_to_action_embed

        self.read_out_act = torch.tanh if self.args['readout'] == 'non_linear' else nn_utils.identity

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.action_embed.weight, self.production_readout_b)
        self.tgt_token_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_primitive_embed(q)),
                                                    self.primitive_embed.weight, self.tgt_token_readout_b)

        self.new_long_tensor = torch.LongTensor
        self.new_tensor = torch.FloatTensor

    def score(self, examples):
        """Given a list of examples, compute the log-likelihood of generating the target AST
        """
        batch = Batch(examples, self.args, self.act_dict, self.vocab, self.device, self.grammar)

        # hidden = [n layers * n directions, batch size, hid dim]

        src_encodings, last_cell = self.encode(batch.src_sents_var, batch.src_sents_len)

        # take last state of encoder as input
        dec_init_vec = self.init_decoder_state(last_cell)

        query_vectors = self.decode(batch, src_encodings, dec_init_vec)

        apply_rule_prob = F.softmax(self.production_readout(query_vectors), dim=-1)

        tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2,
                                           index=batch.apply_rule_idx_matrix.unsqueeze(2)).squeeze(2)
        
        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)

        # (tgt_action_len, batch_size)
        tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=batch.primitive_idx_matrix.unsqueeze(2)).squeeze(2)


        if not (self.args['copy']):
            if self.training:
                # Label smoothing configured to 0 <=> tgt_primitive_gen_from_vocab_prob.log()
                tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                    gen_from_vocab_prob.log(),
                    batch.primitive_idx_matrix)

            else:
                tgt_primitive_gen_from_vocab_log_prob = tgt_primitive_gen_from_vocab_prob.log()

            # (tgt_action_len, batch_size)
            action_prob = tgt_apply_rule_prob.log() * batch.apply_rule_mask + \
                          tgt_primitive_gen_from_vocab_log_prob * batch.gen_token_mask


        else:

            primitive_predictor = F.softmax(self.primitive_predictor(query_vectors), dim=-1)

            primitive_copy_prob = self.src_pointer_net(src_encodings, batch.src_token_mask, query_vectors)

            tgt_primitive_copy_prob = torch.sum(primitive_copy_prob * batch.primitive_copy_token_idx_mask, dim=-1)

            action_mask_pad = torch.eq(batch.apply_rule_mask + batch.gen_token_mask + batch.primitive_copy_mask, 0.)
            action_mask = 1. - action_mask_pad.float()
            action_prob = tgt_apply_rule_prob * batch.apply_rule_mask + \
                          primitive_predictor[:, :, 0] * tgt_primitive_gen_from_vocab_prob * batch.gen_token_mask + \
                          primitive_predictor[:, :, 1] * tgt_primitive_copy_prob * batch.primitive_copy_mask

            # print('prob_actions', action_prob)

            # avoid nan in log
            action_prob.data.masked_fill_(action_mask_pad.data, 1.e-7)
            action_prob = action_prob.log() * action_mask

        scores = torch.sum(action_prob, dim=0)

        returns = [scores]

        return returns

    def encode(self, src_sents_var, src_sents_len):
        """Encode the input natural language utterance

        Args:
            src_sents_var: a variable of shape (src_sent_len, batch_size), representing word ids of the input
            src_sents_len: a list of lengths of input source sentences, sorted by descending order

        Returns:
            src_encodings: source encodings of shape (batch_size, src_sent_len, hidden_size)
            last_state, last_cell: the last hidden state and cell state of the encoder,
                                   of shape (batch_size, hidden_size)
        """
        if self.args['model'] == 'bert':
            # Mark each of the tokens as belonging to sentence "1".
            src_sents_var = src_sents_var.permute(1, 0)
            embeddings = self.encoder(src_sents_var)

            last_hidden_states = self.dropout_encoder(embeddings[0])

            #print(last_hidden_states)

            src_encodings = self.dropout_encoder(embeddings.last_hidden_state)

            src_encodings = self.src_enc_linear(src_encodings)

            return src_encodings, Variable(self.new_tensor(src_encodings.size(0), src_encodings.size(2)).to(self.device).zero_())

        else:

            src_token_embed = self.dropout_encoder(self.nl_embed(src_sents_var))

            packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len)

            # src_encodings: (tgt_query_len, batch_size, hidden_size)
            src_encodings, (last_state, last_cell) = self.encoder(packed_src_token_embed)
            src_encodings, _ = pad_packed_sequence(src_encodings)
            # src_encodings: (batch_size, tgt_query_len, hidden_size)
            src_encodings = src_encodings.permute(1, 0, 2)

            # (batch_size, hidden_size * 2)
            last_cell = torch.cat([last_cell[0], last_cell[1]], 1)

            return src_encodings, last_cell

    def init_decoder_state(self, enc_last_cell):
        """Compute the initial decoder hidden state and cell state"""

        if self.args['model'] == 'bert':
            return enc_last_cell, enc_last_cell
        else:
            h_0 = self.decoder_cell_init(enc_last_cell)
            h_0 = torch.tanh(h_0)
            return h_0, Variable(self.new_tensor(h_0.size()).to(self.device).zero_())

    def decode(self, batch, src_encodings, hidden):
        batch_size = len(batch)

        h_tm1 = hidden

        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        zero_action_embed = Variable(self.new_tensor(self.args['action_embed_size']).to(self.device).zero_())

        att_vecs = []
        history_states = []
        att_weights = []

        # print(batch.max_action_num)

        for t in range(batch.max_action_num):

            if t == 0:
                x = Variable(self.new_tensor(batch_size, self.decoder_lstm.input_size).to(self.device).zero_())
            else:
                a_tm1_embeds = []
                for example in batch.examples:
                    # action t - 1
                    if t < len(eval(example.snippet_actions)):
                        a_tm1 = eval(example.snippet_actions)[t - 1]
                        # print(a_tm1)
                        # print(type(a_tm1))
                        if a_tm1 in self.act_dict:
                            a_tm1_embed = self.action_embed.weight[self.vocab.code[a_tm1]]
                        else:
                            a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1]]
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds, att_tm1]

                # if self.args['parent_feeding_type'] is True:
                #     parent_type_embeds = torch.stack(parent_type_embeds)
                #     inputs.append(parent_type_embeds)
                # elif self.args['parent_feeding_field']:
                #     parent_field_embeds = torch.stack(parent_field_embeds)
                #     inputs.append(parent_field_embeds)
                if self.args['parent_feeding_type'] is True:
                    parent_field_type_embed = self.type_embed(batch.get_frontier_type_idx(t))
                    inputs.append(parent_field_type_embed)

                if self.args['parent_feeding_field'] is True:
                    parent_field_embed = self.field_embed(batch.get_frontier_field_idx(t))
                    inputs.append(parent_field_embed)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, src_encodings,
                                                         src_encodings_att_linear,
                                                         src_token_mask=batch.src_token_mask,
                                                         return_att_weight=True)

            history_states.append((h_t, cell_t))
            att_vecs.append(att_t)
            att_weights.append(att_weight)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        att_vecs = torch.stack(att_vecs, dim=0)
        return att_vecs

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_token_mask=None, return_att_weight=False):
        """Perform a single time-step of computation in decoder LSTM

            Args:
                x: variable of shape (batch_size, hidden_size), input
                h_tm1: Tuple[Variable(batch_size, hidden_size), Variable(batch_size, hidden_size)], previous
                       hidden and cell states
                src_encodings: variable of shape (batch_size, src_sent_len, hidden_size * 2), encodings of source utterances
                src_encodings_att_linear: linearly transformed source encodings
                src_token_mask: mask over source tokens (Note: unused entries are masked to **one**)
                return_att_weight: return attention weights

            Returns:
                The new LSTM hidden state and cell state
            """

        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. p.4 haut
        att_t = self.dropout_decoder(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def parse(self, src_sent):

        primitive_vocab = self.vocab.primitive
        # T = torch.cuda if self.cuda else torch

        src_sent_var = nn_utils.to_input_variable([src_sent], self.vocab.source, device=self.device, training=False)

        # Variable(1, src_sent_len, hidden_size * 2)
        src_encodings, last_cell = self.encode(src_sent_var, [len(src_sent)])
        # (1, src_sent_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        h_tm1 = self.init_decoder_state(last_cell)

        with torch.no_grad():
            hyp_scores = Variable(self.new_tensor([0.]).to(self.device))

        # For computing copy probabilities, we marginalize over tokens with the same surface form
        # `aggregated_primitive_tokens` stores the position of occurrence of each source token
        aggregated_primitive_tokens = OrderedDict()
        for token_pos, token in enumerate(src_sent):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

        t = 0

        hypotheses = [DecodeHypothesis()]
        completed_hypotheses = []

        while len(completed_hypotheses) < self.args['beam_size'] and t < self.args['len_max']:

            hyp_num = len(hypotheses)

            # (hyp_num, src_sent_len, hidden_size * 2)
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            # (hyp_num, src_sent_len, hidden_size)
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            if t == 0:

                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.decoder_lstm.input_size).to(self.device).zero_())

            else:
                actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]

                a_tm1_embeds = []

                for a_tm1 in actions_tm1:
                    # print(a_tm1)
                    # print(type(a_tm1[0]))
                    if a_tm1[0] in self.act_dict:
                        a_tm1_embed = self.action_embed.weight[self.vocab.code[a_tm1[0]]]
                    else:
                        # print(self.vocab.primitive[str(a_tm1[0])])
                        a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[str(a_tm1[0])]]

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds, att_tm1]

                # parents = [hyp.stack[-1][hyp.pointer[-1]] for hyp in hypotheses]

                if self.args['parent_feeding_type'] is True:
                    parent_type_embeds = self.type_embed(Variable(self.new_long_tensor([
                        self.grammar.type2id[parent_type] for parent_type, _, _ in parents]).to(self.device)))
                    inputs.append(parent_type_embeds)

                if self.args['parent_feeding_field'] is True:
                    parent_field_embeds = self.field_embed(Variable(self.new_long_tensor([
                        self.grammar.field2id[parent_field] for _, parent_field, _ in parents]).to(self.device)))
                    inputs.append(parent_field_embeds)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_src_encodings_att_linear,
                                             src_token_mask=None)

            # Variable(batch_size, grammar_size)
            # apply_rule_log_prob = torch.log(F.softmax(self.production_readout(att_t), dim=-1))
            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            # Variable(batch_size, primitive_vocab_size)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

            if not (self.args['copy']):
                primitive_prob = gen_from_vocab_prob
            else:
                # Variable(batch_size, src_sent_len)
                primitive_copy_prob = self.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)
                primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)
                primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob

            unk_copy_token = []

            gentoken_prev_hyp_ids = []
            gentoken_new_hyp_unks = []
            applyrule_new_hyp_scores = []
            applyrule_new_hyp_prod_ids = []
            applyrule_prev_hyp_ids = []

            for hyp_id, hyp in enumerate(hypotheses):
                action_type = self.get_valid_continuation_types(hyp.rules)
                if action_type == 'ActionRule':
                    mask = self.create_mask_action(hyp.rules)
                    productions = [i for (i, bool) in enumerate(mask) if bool]
                    for prod_id in productions:
                        prod_score = apply_rule_log_prob[hyp_id, prod_id].data.item()
                        new_hyp_score = hyp.score + prod_score
                        applyrule_new_hyp_prod_ids.append(prod_id)
                        applyrule_new_hyp_scores.append(new_hyp_score)
                        applyrule_prev_hyp_ids.append(hyp_id)
                else:
                    # Primitives
                    gentoken_prev_hyp_ids.append(hyp_id)
                    if self.args['copy'] is True:
                        # last_rule = rules.pop(0)
                        for token, token_pos_list in aggregated_primitive_tokens.items():
                            # Get probability token number k (sum to get value)
                            sum_copy_prob = torch.gather(primitive_copy_prob[hyp_id], 0,
                                                         Variable(
                                                             torch.LongTensor(token_pos_list).to(self.device))).sum()

                            # Get global probability copying token number k
                            gated_copy_prob = primitive_predictor_prob[hyp_id, 1] * sum_copy_prob

                            if token in primitive_vocab:
                                # For dev_set, always True
                                token_id = primitive_vocab[token]
                                primitive_prob[hyp_id, token_id] = primitive_prob[hyp_id, token_id] + gated_copy_prob
                            else:
                                # Token unknown
                                unk_copy_token.append({'token': token, 'token_pos_list': token_pos_list,
                                                       'copy_prob': gated_copy_prob.data.item()})

                        if self.args['copy'] is True and len(unk_copy_token) > 0:
                            unk_i = np.array([x['copy_prob'] for x in unk_copy_token]).argmax()
                            token = unk_copy_token[unk_i]['token']
                            primitive_prob[hyp_id, primitive_vocab.unk_id] = unk_copy_token[unk_i]['copy_prob']
                            gentoken_new_hyp_unks.append(token)

            new_hyp_scores = None
            if applyrule_new_hyp_scores:  
                new_hyp_scores = Variable(self.new_tensor(applyrule_new_hyp_scores).to(self.device))
            if gentoken_prev_hyp_ids:  
                primitive_log_prob = torch.log(primitive_prob)
                gen_token_new_hyp_scores = (
                        hyp_scores[gentoken_prev_hyp_ids].unsqueeze(1) + primitive_log_prob[gentoken_prev_hyp_ids,
                                                                         :]).view(-1)

                if new_hyp_scores is None:  
                    new_hyp_scores = gen_token_new_hyp_scores
                else:
                    new_hyp_scores = torch.cat([new_hyp_scores, gen_token_new_hyp_scores])

            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(new_hyp_scores.size(0),
                                                                   self.args['beam_size'] - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                if new_hyp_pos < len(applyrule_new_hyp_scores):
                    prev_hyp_id = applyrule_prev_hyp_ids[new_hyp_pos]
                    prev_hyp = hypotheses[prev_hyp_id].copy()
                    prod_id = applyrule_new_hyp_prod_ids[new_hyp_pos]
                    production = self.vocab.code.id2word[prod_id]
                    action = self.act_dict[production]
                    prev_hyp.actions.append((action.label, prev_hyp.rules[0][1]))
                    prev_hyp.rules = action.apply(prev_hyp.rules, self.primitives_type)
                    if action.rhs == []:
                        action_label = action.label
                        prev_hyp.pointer, prev_hyp.stack = self.completion(action_label, prev_hyp.pointer,
                                                                           prev_hyp.stack)
                    else:
                        prev_hyp.stack.append([*zip(action.rhs, action.rhs_names, action.iter_flags)])
                        prev_hyp.pointer.append(0)
                    prev_hyp.score = new_hyp_score
                else:
                    token_id = (new_hyp_pos - len(applyrule_new_hyp_scores)) % primitive_prob.size(1)
                    numer = (new_hyp_pos - len(applyrule_new_hyp_scores))
                    deno = primitive_prob.size(1)

                    k = torch.div(numer, deno, rounding_mode='trunc')

                    prev_hyp_id = gentoken_prev_hyp_ids[k]
                    prev_hyp = hypotheses[prev_hyp_id].copy()
                    last_rule = prev_hyp.rules[0]

                    if last_rule[1] == '-' or last_rule[1] == '?':
                        prev_hyp.rules.pop(0)

                        if token_id == primitive_vocab.unk_id:
                            if gentoken_new_hyp_unks:
                                token = self.terminal_type(gentoken_new_hyp_unks[k])
                            else:
                                token = self.terminal_type(primitive_vocab.id2word[primitive_vocab.unk_id])
                        else:
                            token = self.terminal_type(primitive_vocab.id2word[token_id.item()])

                        if token == 'Reduce_primitif':
                            prev_hyp.actions.append((token, '?'))
                        else:
                            prev_hyp.actions.append((token, last_rule[1]))

                        action = token

                        prev_hyp.pointer, prev_hyp.stack = self.completion(token, prev_hyp.pointer, prev_hyp.stack)
                        prev_hyp.score = new_hyp_score

                    else:

                        last_rule = prev_hyp.rules[0]
                        if token_id == primitive_vocab.unk_id:
                            if gentoken_new_hyp_unks:
                                token = self.terminal_type(gentoken_new_hyp_unks[k])
                            else:
                                token = self.terminal_type(primitive_vocab.id2word[primitive_vocab.unk_id])
                        else:
                            token = self.terminal_type(primitive_vocab.id2word[token_id.item()])

                        if token == 'Reduce_primitif':
                            prev_hyp.actions.append((token, '?'))
                            prev_hyp.rules.pop(0)
                        else:
                            prev_hyp.actions.append((token, last_rule[1]))

                        action = token

                        prev_hyp.pointer, prev_hyp.stack = self.completion(token, prev_hyp.pointer, prev_hyp.stack)
                        prev_hyp.score = new_hyp_score

                new_hyp = prev_hyp

                if new_hyp.rules == []:
                    new_hyp.score /= (t + 1)
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                # hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                hypotheses = new_hypotheses
                hyp_scores = Variable(self.new_tensor([hyp.score for hyp in hypotheses]).to(self.device))
                t += 1
            else:
                break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses

    def create_mask_action(self, rule):
        next_rule, iterflag = rule[0]
        if next_rule == None:
            return [act.is_applicable([('stmt', '-')]) for act in self.act_dict.values() if
                    isinstance(act, GrammarRule)]
        else:
            if iterflag == '-':
                return [act.is_applicable(rule) for act in self.act_dict.values() if isinstance(act, GrammarRule)]
            else:
                return [act.is_applicable(rule) if isinstance(act, GrammarRule) else True for act in
                        self.act_dict.values()]

    def get_valid_continuation_types(self, rules):
        next_rule, iterflag = rules[0]
        if next_rule in self.primitives_type:
            return 'GenToken'
        else:
            return 'ActionRule'


    @staticmethod
    def terminal_type(terminal):
        try:
            a = float(terminal)
        except (TypeError, ValueError, OverflowError):
            try:
                a = ast.literal_eval(terminal)
                return eval(a)
            except:
                return str(terminal)
        else:
            try:
                b = int(a)
            except (TypeError, ValueError, OverflowError):
                return a
            else:
                return b


    def shift(self, pointer):
        if len(pointer) == 0:
            return pointer
        else:
            pointer[-1] += 1
            return pointer

    def completion(self, action, pointer, stack):
        if len(pointer) == 0:
            return pointer, stack
        elif stack[-1][pointer[-1]][2] == '-' or stack[-1][pointer[-1]][2] == '?' or action == 'Reduce' or action == 'Reduce_primitif':
            pointer = self.shift(pointer)
            if pointer[-1] == len(stack[-1]):
                pointer = pointer[:-1]
                stack = stack[:-1]
                pointer, stack = self.completion(None, pointer, stack)
                return pointer, stack
            else:
                return pointer, stack
        else:
            return pointer, stack
