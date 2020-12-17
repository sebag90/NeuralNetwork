import numpy as np


def print_progress_bar(iteration, total, prefix = "", suffix = "", decimals = 1, length = 100, fill = "#", printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '.' * (length - filledLength)
    print(f"\r{prefix} [{bar}] {percent}% {suffix}", end = printEnd)
    if iteration == total:
        print()


def shuffle(x, y):
    """
    takes 2 vectors as input and shuffles them 
    keeping the correlation between the elements
    of the vectors

    example:
        input:
            [1, 2, 3, 4, 5] 
            [6, 7, 8, 9, 0]
        output:
            [3, 5, 2, 4, 1]
            [8, 0, 7, 9, 6]
    """
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]


def split_dataset(x, y, test_size=0.2):
    """
    shuffles a vector of instances x and corresponding
    labels y and divide them in test and train data sets 
    based on test_size (standard 80% train, 20%test)
    """
    x, y = shuffle(x, y)

    limit = x.shape[0] - int(x.shape[0]*test_size)

    x_train = x[:limit]
    y_train = y[:limit]
    x_test = x[limit:]
    y_test = y[limit:]

    return x_train, x_test, y_train, y_test