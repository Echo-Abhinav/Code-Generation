from transformers import BertTokenizer
import re

QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['var_0', 'str_0', 'var_1', 'str_1', 'var_2', 'str_2', 'var_3', 'str_3', 'var_4', 'str_4'])

def canonicalize_intent_user(intent, tokenizer='None'):
    marked_token_matches = QUOTED_TOKEN_RE.findall(intent)

    slot_map = dict()
    var_id = 0
    str_id = 0
    for match in marked_token_matches:
        quote = match[0]
        value = match[1]
        quoted_value = quote + value + quote

        slot_type = infer_slot_type(quote, value)

        if slot_type == 'var':
            slot_name = 'var_%d' % var_id
            var_id += 1
            slot_type = 'var'
        else:

            slot_name = 'str_%d' % str_id
            str_id += 1
            slot_type = 'str'

        intent = intent.replace(quoted_value, slot_name)

        slot_map[slot_name] = {'value': value.strip().encode().decode('unicode_escape', 'ignore'),
                                'quote': quote,
                                'type': slot_type}

    intent = tokenizer.tokenize(intent.lower())
    intent = ['[CLS]'] + intent + ['[SEP]']
    return intent, slot_map
        
def infer_slot_type(quote, value):
    if quote == '`' and value.isidentifier():
        return 'var'
    return 'str'

def user_intent_call(intent):
    intent, slot_map = canonicalize_intent_user(intent, tokenizer)
    print(intent)
    print("+++++++")
    print(slot_map)

    return intent, slot_map