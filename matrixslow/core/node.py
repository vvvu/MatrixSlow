import abc
import numpy as np

from .graph import default_graph

class Node(object):
    """
    计算图节点基类
    The base class of Compuational Graph Node
    """

    def __init__(self, *parents, **kargs):
        # 计算图对象，默认为全局对象default_graph
        # The Computational Graph Object, the default is the global object default_graph
        self.graph = kargs.get('graph', default=default_graph)
        self.need_save = kargs.get('need_save', True)
        self.gen_node.name(**kargs)

        self.parents = list(parents)  # parent node
        self.children = []  # children node list
        self.value = None  # NumPy中的Matrix类
        self.jacobi = None
        # The Jacobi Matrix of result node to this node (结果节点对本节点的Jacobi Matrix)

        # 将本节点添加到父节点的子节点列表中去
        # Add this node the list of child nodes of the parent node
        for parent in self.parents:
            parent.children.append(self)

        # 将本节点添加到计算图中去
        # Add this node to the computational graph
        self.graph.add_node(self)

    def get_parent(self):
        return self.parents

    def get_children(self):
        return self.children

    def gen_node_name(self, **kargs):
        """
        生成node名称，如果用户不指定，则根据节点类型自动生成
        1. 不指定名称：节点类名 + 当前计算图中节点个数
        2. 指定name_scope: name_scope + 节点类名 + 当前计算图中节点个数
        :param kargs:
        :return:
        """
        self.name = kargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.node_count()
        ))

        if self.graph.name_scope:
            self.name = '{}/{}'.format(self.graph.name_scope, self.name)

    '''
    抽象方法，继承Node类的子类都需要覆盖该方法来实现自己特定的计算(根据node类型的不同)
    Abstract method: subclasses that inherit the Node class need to
    override this method to implement their own specific computations
    (depending on the node type)
    '''

    @abc.abstractmethod
    def compute(self):
        """
        抽象方法，根据父节点的值计算本节点的值
        Abstract method: Compute the value of this node based on the value
        of the parent node
        :return:
        """

    def compute(self):
        pass

    def forward(self):
        """
        前向传播计算本节点的值，如果父节点的值没有计算，则递归调用父节点的forward方法
        Forward propagation computes the value of this node, if the value
        of the parent node is not computed, then recursively call the forward()
        of the parent node
        :return:
        """
        for node in self.parents:  # 计算父节点的值
            if node.value is None:
                node.forward()

        self.compute()  # 计算本节点的值

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """
        抽象方法，计算本节点对某个父节点的Jacobi矩阵
        Abstract method: Compute the Jacobi Matrix of this node to parent node
        :param parent:
        :return:
        """

    def get_jacobi(self, parent):
        pass

    def backward(self, result):
        """
        反向传播，计算结果节点对本节点的Jacobi矩阵
        Back propagation(BP), compute the Jacobi matrix of the result node
        to this node
        :param result:
        :return:
        """

        '''
        1. np.eye() return a 2-D array with ones on the diagnoal
        and zeros elsewhere
        2. np.mat() interpret the input as a matrix
        '''
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
                '''
                如果节点本身是结果节点，节点对于节点本身的Jacobi Matrix为一个单位矩阵
                '''
            else:  # Compute the jacobi matrix
                self.jacobi = np.mat(np.zeros((result.dimension(), self.dimension())))

                for child in self.get_children():
                    if child.value is not None:
                        self.jacobi += child.backward(result) * child.get_jacobi(self)

        return self.jacobi

    def clear_jacobi(self):
        """
        清空结果节点对本节点的Jacobi Matrix
        Clear the Jacobi Matrix of the result node to this node
        :return:
        """
        self.jacobi = None

    def dimension(self):
        """
        返回本节点的值(Matrix in NumPy)展开成向量后的维数(rows * cols)
        Returns the dimension of the value of this node expanded into a vector
        :return:
        """
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        """
        返回本节点的值(Matrix in NumPy)展开成向量后的维数(行数，列数)
        Returns the dimension(number of rows, number of columns) of this node
        expanded into a vector
        :return:
        """
        return self.value.shape

    def reset_value(self, recursive=True):
        """
        重置本节点的值，且递归重置下游节点的值
        :param recursive:
        :return:
        """
        self.value = None

        if recursive:
            for child in self.get_children():
                child.reset_value()

class Variable(Node):
    """
    变量节点：变量节点不同于[计算节点]，变量节点的值并不是计算出来的，所以并不需要父节点
    """
    def __init__(self, dim, init = False, trainable = True, **kargs):
        """
        变量节点没有父节点，其构造函数接受变量的形状，是否初始化以及是否参与训练
        :param dim:
        :param init:
        :param trainable:
        :param kargs:
        """
        Node.__init__(self, **kargs)
        self.dim = dim

        # 如果需要初始化，则以Normal Distribution随机初始化变量的值
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))

        # Variable Node是否参与训练
        self.trainable = trainable

    def set_value(self, value):
        """
        为Variable Node赋值
        :param value:
        :return:
        """
        assert isinstance(value, np.matrix) and value.shape == self.dim
        '''
        The isinstance() function checks if the object (first argument) is
        an instance or subclass of classinfo class (second argument)
        '''
        # The value of this node is modified, reset the value of
        # all downstream nodes.
        self.reset_value()
        self.value = value
