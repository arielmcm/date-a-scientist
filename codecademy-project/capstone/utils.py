def get_column_mapping(column):
    mapping = column.value_counts().to_dict()
    counter = 0
    for k, v in mapping.items():
        counter += 1
        mapping[k] = counter
    return column.map(mapping)
