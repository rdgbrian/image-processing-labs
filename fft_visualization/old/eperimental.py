def see_partition(partition):
    if partition.dependants is None:
        return
    
    for dependant in partition.dependants:
        see_partition(dependant)
        if dependant.dependants is None:
            print("f" + str(partition.image_cord) + " | F" + partition.partition_string + str(partition.relative_cord))
        else:
            print("F" + partition.partition_string + str(partition.relative_cord))

def see_partition(partition):
    if partition.dependants is None:
        print("F" + partition.partition_string + str(partition.relative_cord) + "\tf" + str(partition.image_cord))
        return
    
    for dependant in partition.dependants:
        see_partition(dependant)
    print("F" + partition.partition_string + str(partition.relative_cord) +  f"\tW top={partition.weight_params_top} bottom={partition.weight_params_bottom}")