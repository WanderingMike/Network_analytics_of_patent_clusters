def categorise_output(citations):
    '''This functions categorises the ML-readable output column forward citations'''

    if citations >= 20:
        return 3
    elif 10 <= citations <= 19:
        return 2
    elif 2 <= citations <= 9:
        return 1
    else:
        return 0

