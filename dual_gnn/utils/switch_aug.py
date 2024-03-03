from dual_gnn.models.augmentation import subgraph, drop_nodes, mask_nodes, permute_edges,drop_edges


def switch_aug(data, aug, aug_ratio):
    if aug == 'dropN':
        data = drop_nodes(data, aug_ratio)
    elif aug == 'dropE':
        data = drop_edges(data, aug_ratio)
    elif aug == 'permE':
        data = permute_edges(data, aug_ratio)
    elif aug == 'maskN':
        data = mask_nodes(data, aug_ratio)
    elif aug == 'subgraph':
        data = subgraph(data, aug_ratio)
    elif aug == 'random':
        n = np.random.randint(2)
        if n == 0:
            data = drop_nodes(data, aug_ratio)
        elif n == 1:
            data = subgraph(data, aug_ratio)
            # data = subgraph(data, 0.5)
        else:
            print('augmentation error')
            assert False
    elif aug == 'none':
        None
    else:
        print('augmentation error')
        assert False

    return data
