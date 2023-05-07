from ast import literal_eval
from modules import shared
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length

REPLY_ATTRIBUTES = ' (2 paragraphs, engaging, natural, authentic, descriptive, creative)'

# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text

# add "\n##" as a custom stopping string
def state_modifier(state):
    strings = literal_eval(f"[{state['custom_stopping_strings']}]")
    strings.append(r'\n##')
    state['custom_stopping_strings'] = ', '.join(f'"{s}"' for s in strings)
    return state

def custom_generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs['impersonate'] if 'impersonate' in kwargs else False
    _continue = kwargs['_continue'] if '_continue' in kwargs else False
    also_return_rows = kwargs['also_return_rows'] if 'also_return_rows' in kwargs else False
    is_instruct = state['mode'] == 'instruct'

    replacements = {
        '<|user|>': state['name1'].strip(),
        '<|bot|>': state['name2'].strip(),
    }

    provided_context = state['context'].strip()
    context = '''## <|bot|>\nYou're "<|bot|>" in this never-ending roleplay with "<|user|>".\n### Input:\n'''
    context += provided_context
    context += '''\n### Response: (OOC) Understood. I will take this info into account for the roleplay. (end OOC)\n### New Roleplay:\n'''
    context = replace_all(context, replacements)

    rows = [context]
    min_rows = 3

    # Finding the maximum prompt size
    chat_prompt_size = state['chat_prompt_size']
    if shared.soft_prompt:
        chat_prompt_size -= shared.soft_prompt_tensor.shape[1]

    max_length = min(get_max_prompt_length(state), chat_prompt_size)

    template = '### Instruction:\n#### <|user|>: <|user-message|>\n<|sep|>### Response<|reply-attributes|>:\n#### <|bot|>: <|bot-message|>\n'

    user_turn = replace_all(template.split('<|sep|>')[0], replacements)
    bot_turn = replace_all(template.split('<|sep|>')[1], replacements)
    user_turn_stripped = replace_all(user_turn.split('<|user-message|>')[0], replacements)
    bot_turn_stripped = replace_all(bot_turn.split('<|bot-message|>')[0], replacements)
    # add reply attributes
    bot_turn = bot_turn.replace('<|reply-attributes|>', '')
    bot_turn_stripped = bot_turn_stripped.replace('<|reply-attributes|>', REPLY_ATTRIBUTES)

    # Building the prompt
    i = len(shared.history['internal']) - 1
    while i >= 0 and len(encode(''.join(rows))[0]) < max_length:
        if _continue and i == len(shared.history['internal']) - 1:
            rows.insert(1, bot_turn_stripped + shared.history['internal'][i][1].strip())
        else:
            rows.insert(1, bot_turn.replace('<|bot-message|>', shared.history['internal'][i][1].strip()))

        string = shared.history['internal'][i][0]
        if string not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            rows.insert(1, replace_all(user_turn, {'<|user-message|>': string.strip(), '<|round|>': str(i)}))

        i -= 1

    if impersonate:
        min_rows = 2
        rows.append(user_turn_stripped.rstrip(' '))
    elif not _continue:
        # Adding the user message
        if len(user_input) > 0:
            rows.append(replace_all(user_turn, {'<|user-message|>': user_input.strip(), '<|round|>': str(len(shared.history["internal"]))}))

        # Adding the Character prefix
        rows.append(apply_extensions("bot_prefix", bot_turn_stripped.rstrip(' ')))

    while len(rows) > min_rows and len(encode(''.join(rows))[0]) >= max_length:
        rows.pop(1)

    prompt = ''.join(rows)

    # save the prompt to a text file for inspection and debugging
    with open('chat_prompt.txt', 'w') as f:
        f.write(prompt)

    if also_return_rows:
        return prompt, rows
    else:
        return prompt