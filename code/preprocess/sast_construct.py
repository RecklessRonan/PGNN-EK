import javalang
from javalang.ast import Node
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, DataCollatorWithPadding
from anytree import AnyNode
from tqdm import tqdm


# use javalang to generate ASTs and depth-first traverse to generate ast nodes corpus
def get_token(node):
    token = 'None'
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
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))


def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    for child in children:
        get_sequence(child, sequence)


def parse_program(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree


# checkpoint = 'microsoft/codebert-base'
checkpoint = '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/zhurenyu/huggingface-models/codebert'
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
ast_tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
roberta = RobertaModel.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
config = RobertaConfig.from_pretrained(checkpoint)
javalang_special_tokens = ['CompilationUnit', 'Import', 'Documented', 'Declaration', 'TypeDeclaration', 'PackageDeclaration',
                           'ClassDeclaration', 'EnumDeclaration', 'InterfaceDeclaration', 'AnnotationDeclaration', 'Type',
                           'BasicType', 'ReferenceType', 'TypeArgument', 'TypeParameter', 'Annotation', 'ElementValuePair',
                           'ElementArrayValue', 'Member', 'MethodDeclaration', 'FieldDeclaration', 'ConstructorDeclaration',
                           'ConstantDeclaration', 'ArrayInitializer', 'VariableDeclaration', 'LocalVariableDeclaration',
                           'VariableDeclarator', 'FormalParameter', 'InferredFormalParameter', 'Statement', 'IfStatement',
                           'WhileStatement', 'DoStatement', 'ForStatement', 'AssertStatement', 'BreakStatement', 'ContinueStatement',
                           'ReturnStatement', 'ThrowStatement', 'SynchronizedStatement', 'TryStatement', 'SwitchStatement',
                           'BlockStatement', 'StatementExpression', 'TryResource', 'CatchClause', 'CatchClauseParameter',
                           'SwitchStatementCase', 'ForControl', 'EnhancedForControl', 'Expression', 'Assignment', 'TernaryExpression',
                           'BinaryOperation', 'Cast', 'MethodReference', 'LambdaExpression', 'Primary', 'Literal', 'This',
                           'MemberReference', 'Invocation', 'ExplicitConstructorInvocation', 'SuperConstructorInvocation',
                           'MethodInvocation', 'SuperMethodInvocation', 'SuperMemberReference', 'ArraySelector', 'ClassReference',
                           'VoidClassReference', 'Creator', 'ArrayCreator', 'ClassCreator', 'InnerClassCreator', 'EnumBody',
                           'EnumConstantDeclaration', 'AnnotationMethod', 'Modifier']
special_tokens_dict = {'additional_special_tokens': javalang_special_tokens}
num_added_toks = ast_tokenizer.add_special_tokens(special_tokens_dict)


#  generate tree for AST Node
def create_tree(root, node, node_list, sub_id_list, leave_list, tokenizer, parent=None):
    id = len(node_list)
    node_list.append(node)
    token, children = get_token(node), get_child(node)

    if children == []:
        # print('this is a leaf:', token, id)
        leave_list.append(id)

    # Use roberta.tokenizer to generate subtokens
    # If a token can be divided into multiple(>1) subtokens, the first subtoken will be set as the previous node,
    # and the other subtokens will be set as its new children
    token = token.encode('utf-8', 'ignore').decode("utf-8")
    sub_token_list = tokenizer.tokenize(token)

    if id == 0:
        # the root node is one of the tokenizer's special tokens
        root.token = sub_token_list[0]
        root.data = node
        # record the num of nodes for every children of root
        root_children_node_num = []
        for child in children:
            node_num = len(node_list)
            create_tree(root, child, node_list, sub_id_list,
                        leave_list, tokenizer, parent=root)
            root_children_node_num.append(len(node_list) - node_num)
        return root_children_node_num
    else:
        # print(sub_token_list)
        new_node = AnyNode(
            id=id, token=sub_token_list[0], data=node, parent=parent)
        if len(sub_token_list) > 1:
            sub_id_list.append(id)
            for sub_token in sub_token_list[1:]:
                id += 1
                AnyNode(id=id, token=sub_token, data=node, parent=new_node)
                node_list.append(sub_token)
                sub_id_list.append(id)

        for child in children:
            create_tree(root, child, node_list, sub_id_list,
                        leave_list, tokenizer, parent=new_node)
    # print(token, id)


# traverse the AST tree to get all the nodes and edges
def get_node_and_edge(node, node_index_list, tokenizer, src, tgt, variable_token_list, variable_id_list):
    token = node.token
    node_index_list.append(tokenizer.convert_tokens_to_ids(token))
    # node_index_list.append([vocab_dict.word2id.get(token, UNK)])
    # find out all variables
    if token in ['VariableDeclarator', 'MemberReference']:
        if node.children:  # some chidren are comprised by non-utf8 and will be removed
            variable_token_list.append(node.children[0].token)
            variable_id_list.append(node.children[0].id)

    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        get_node_and_edge(child, node_index_list, tokenizer,
                          src, tgt, variable_token_list, variable_id_list)


# generate pytorch_geometric input format data from ast
def get_pyg_data_from_ast(ast, tokenizer=ast_tokenizer):
    node_list = []
    sub_id_list = []  # record the ids of node that can be divide into multple subtokens
    leave_list = []  # record the ids of leave
    new_tree = AnyNode(id=0, token=None, data=None)
    root_children_node_num = create_tree(
        new_tree, ast, node_list, sub_id_list, leave_list, tokenizer)
    # print('root_children_node_num', root_children_node_num)
    x = []
    edge_src = []
    edge_tgt = []
    # record variable tokens and ids to add data flow edge in AST graph
    variable_token_list = []
    variable_id_list = []
    get_node_and_edge(new_tree, x, tokenizer, edge_src,
                      edge_tgt, variable_token_list, variable_id_list)

    ast_edge_num = len(edge_src)
    edge_attr = [[0] for _ in range(ast_edge_num)]
    # set subtoken edge type to 2
    for i in range(len(edge_attr)):
        if edge_src[i] in sub_id_list and edge_tgt[i] in sub_id_list:
            edge_attr[i] = [2]
    # add data flow edge
    variable_dict = {}
    for i in range(len(variable_token_list)):
        # print('variable_dict', variable_dict)
        if variable_token_list[i] not in variable_dict:
            variable_dict.setdefault(
                variable_token_list[i], variable_id_list[i])
        else:
            # print('edge', variable_dict.get(variable_token_list[i]), variable_id_list[i])
            edge_src.append(variable_dict.get(variable_token_list[i]))
            edge_tgt.append(variable_id_list[i])
            edge_src.append(variable_id_list[i])
            edge_tgt.append(variable_dict.get(variable_token_list[i]))
            variable_dict[variable_token_list[i]] = variable_id_list[i]
    dataflow_edge_num = len(edge_src) - ast_edge_num

    # add next-token edge
    nexttoken_edge_num = len(leave_list)-1
    for i in range(nexttoken_edge_num):
        edge_src.append(leave_list[i])
        edge_tgt.append(leave_list[i+1])
        edge_src.append(leave_list[i+1])
        edge_tgt.append(leave_list[i])

    edge_index = [edge_src, edge_tgt]

    # set data flow edge type to 1
    for _ in range(dataflow_edge_num):
        edge_attr.append([1])

    # set data flow edge type to 3
    for _ in range(nexttoken_edge_num * 2):
        edge_attr.append([3])

    return x, edge_index, edge_attr, root_children_node_num


def get_subgraph_node_num(root_children_node_num, divide_node_num, max_subgraph_num):
    subgraph_node_num = []
    node_sum = 0
    real_graph_num = 0
    for num in root_children_node_num:
        node_sum += num
        if node_sum >= divide_node_num:
            subgraph_node_num.append(node_sum)
            node_sum = 0

    subgraph_node_num.append(node_sum)
    real_graph_num = len(subgraph_node_num)

    if real_graph_num >= max_subgraph_num:
        return subgraph_node_num[: max_subgraph_num], max_subgraph_num

    # print(len(subgraph_node_num))
    # if the last subgraph node num < divide_node_num/2, then put the last subgraph to the second to last subgraph
    if subgraph_node_num[-1] < divide_node_num/2 and len(subgraph_node_num) > 1:
        subgraph_node_num[-2] = subgraph_node_num[-2] + subgraph_node_num[-1]
        subgraph_node_num.pop() # remove the last element
        real_graph_num -= 1

    for _ in range(real_graph_num, max_subgraph_num):
            subgraph_node_num.append(0)

    return subgraph_node_num, real_graph_num
