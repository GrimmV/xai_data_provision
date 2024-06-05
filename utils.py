
def combine_lists_uniquely(list1: list[str], list2: list[str]) -> list[str]:
    unique_l1 = set(list1)
    unique_l2 = set(list2)

    remainder = unique_l2 - unique_l1

    return sorted(list(unique_l1) + list(remainder))