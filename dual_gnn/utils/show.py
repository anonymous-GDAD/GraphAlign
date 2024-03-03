def show_stru(data):
    print(data)
    print('==============================================================')
    # 获取图的一些信息
    print(f'Number of nodes: {data.num_nodes}') # 节点数量
    print(f'Number of edges: {data.num_edges}') # 边数量
    print(f'Number of node features: {data.num_node_features}') # 节点属性的维度
    print(f'Number of node features: {data.num_features}') # 同样是节点属性的维度
    print(f'Number of edge features: {data.num_edge_features}') # 边属性的维度
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # 平均节点度
    print(f'if edge indices are ordered and do not contain duplicate entries.: {data.is_coalesced()}') # 是否边是有序的同时不含有重复的边
    print(f'Number of training nodes: {data.train_mask.sum()}') # 用作训练集的节点
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}') # 用作训练集的节点数占比
    print(f'Contains isolated nodes: {data.has_isolated_nodes()}') # 此图是否包含孤立的节点
    print(f'Contains self-loops: {data.has_self_loops()}')  # 此图是否包含自环的边
    print(f'Is undirected: {data.is_undirected()}')  # 此图是否是无向图