class Graph:
    """
    Computational Graph
    """

    def __init__(self):
        self.nodes = [] # The nodes in Computational Graph
        self.name_scope = None

    def add_node(self, node):
        """
        Add node to Computational Graph
        :param node:
        :return:
        """
        self.nodes.append(node)

    def clear_jacobi(self):
        """
        Clear all nodes' jacobi matrix in the graph
        :return:
        """
        for node in self.nodes:
            node.clear_jacobi()

    def reset_value(self):
        """
        Reset all nodes' value
        :return:
        """
        for node in self.nodes:
            # 每个节点不递归清楚自己子节点的值(否则会多次Clear同一个Node)
            node.reset_value(False)

    # def draw(self, ax = None):

# 全局默认计算图
default_graph = Graph()