def topo_sort(v, topo=[], visited=set()):
    """
    """
    if v not in visited:
        visited.add(v)
        for child in v.children:
            topo_sort(child, topo, visited)
        topo.append(v)
    return topo

