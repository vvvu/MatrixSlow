from .node import Variable
from .graph import default_graph
"""
The node operations in Computational Graph
"""

def get_node_from_graph(node_name, name_scope = None, graph = None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None

def get_trainable_variables_from_graph(node_name = None, name_scope = None, graph = None):
    if graph is None:
        graph = default_graph
    if node_name is None:
        return [node for node in default_graph.nodes
                if isinstance(node, Variable) and node.trainable]

    if name_scope:
        node_name = name_scope + '/' + node_name
    return get_node_from_graph(node_name, graph = graph)

def update_node_value_in_graph(node_name, new_value, name_scope = None, graph = None):
    node = get_node_from_graph(node_name, name_scope, graph)
    assert node is not None
    assert node.value.shape == new_value.shape
    node.value = new_value

class name_scope:

    def __init__(self, name_scope):
        self.name_scope = name_scope

    '''
    with语句: 在Python中，有一些人物可能事先需要设置，事后进行清理工作，所以提供了with语句
    1. 使用with语句的对象必须有__enter__()和__exit__()方法
    2. 紧跟with后面的语句被求值后，返回对象的__enter__()方法被调用，返回值赋值给as后面的变量
    3. with语句执行完成后，调用__exit()__方法
    例如:
    with open("foo.txt") as file:
        data = file.read()
    执行顺序如下
    (1) open() -> __enter__() => file
    (2) file -> data = file.read() -> __exit()__
    '''
    def __enter__(self):
        default_graph.name_scope = self.name_scope
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        default_graph.name_scope = None