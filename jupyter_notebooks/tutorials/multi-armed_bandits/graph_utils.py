import pydot

import random 

def add_recurse(g, parent, tree):
    if isinstance(tree, (list, set)):
        for node in tree:
            name = '%s_%03d' % (node, random.randint(0, 999))
            n = pydot.Node(name=name, label=node)
            g.add_node(n)
            g.add_edge(pydot.Edge(parent, n))
        return

    for node, children in tree.items():
        name = '%s_%03d' % (node, random.randint(0, 999))
        n = pydot.Node(name=name, label=node)
        g.add_node(n)
        g.add_edge(pydot.Edge(parent, n))
        add_recurse(g, n, children)

def tree_to_dot(tree):
    g = pydot.Dot(graph_type='digraph')

    for node, children in tree.items():
        name = '%s_%03d' % (node, random.randint(0, 999))
        n = pydot.Node(name=name, label=node)
        g.add_node(n)
        add_recurse(g, n, children)

    return g