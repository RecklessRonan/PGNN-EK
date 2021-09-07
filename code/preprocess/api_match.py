import javalang
from javalang.ast import Node


def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def get_child(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                yield from expand(item)
            elif item:
                yield item

    return list(expand(children))


def get_sequence(node, sequence, api_sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    # find api
    if token == 'MethodInvocation':
        api = [get_token(child) for child in children if not get_child(child)]
        # api_sequence.append(' '.join(api))
        if len(api) > 1:
            api_sequence.append(api[-1])
    for child in children:
        get_sequence(child, sequence, api_sequence)


def api_match(api_sequence, java_api):
    description_sequence = []
    for api in api_sequence:
        loc = java_api.loc[java_api['index_name'].str.contains(api, case=True)]
        if not loc.empty:
            description = loc['method_description'].iloc[0]
            if description != 'None':
                description_sequence.append(description)
    return description_sequence


def parse_program(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    return parser.parse_member_declaration()
