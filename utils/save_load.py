import pickle


def save(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    return None


def load(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj
