import numpy as np


def print_progress_bar(iteration, total, prefix = "", suffix = "", decimals = 1, length = 100, fill = "#", printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '.' * (length - filledLength)
    print(f"\r{prefix} [{bar}] {percent}% {suffix}", end = printEnd)
    if iteration == total:
        print()


def shuffle(a, b):
    np.random.seed()
    rnd_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rnd_state)
    np.random.shuffle(b)
    return a, b


def dataset(x, y, test_size=0.8):
    np.random.seed()
    rnd_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rnd_state)
    np.random.shuffle(y)

    limit = int(len(x)*test_size)

    x_train = x[:limit]
    y_train = y[:limit]
    x_test = x[limit:]
    y_test = y[limit:]

    return x_train, x_test, y_train, y_test