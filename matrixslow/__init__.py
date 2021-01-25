from matrixslow import core

# from . import ops
# from . import optimizer
# from . import layer

'''
1. 在Python中，如果当前目录存在有__init__.py文件，则表示该目录为一个Package
'''

default_graph = core.default_graph
get_node_from_graph = core.get_node_from_graph
name_scope = core.name_scope
Variable = core.Variable