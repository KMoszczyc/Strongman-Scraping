def arg_sort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=True, )

# TODO: it doesnt work!
def remap_list(values_list, points_list):
    """Use index_list to reorder values list"""
    # index_list = arg_sort(points_list)
    # remaped_list = [values_list[index] for index in index_list]
    # print('-----------------------')
    # print(values_list, points_list)
    # print(index_list, remaped_list)

    values_points_dict = list(zip(values_list, points_list))
    # remapped_values = sorted(values_points_dict, key=values_points_dict.get)
    remapped_values = [value for value, points in sorted(values_points_dict, key=lambda x: x[1], reverse=True)]

    print('-----------------------')
    print(values_list, points_list)
    print(values_points_dict, remapped_values)

    return remapped_values



