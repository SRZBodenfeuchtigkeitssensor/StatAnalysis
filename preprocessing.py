'''Functions for preprocessing of data chunks

# TODO: get falling part of chunk (w/o constant one)'''


def get_moving_average(chunk, k_neighbors: int = 3):
    # TODO: adjust k to curve
    return chunk.rolling('8000s').mean()


def get_smoothness(chunk):
    '''returns std of the chunk.diffs() as a measure of the data's smoothness'''
    return chunk.diff().std()
