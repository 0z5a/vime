def score(response, label):
    return sum(response.encode()) % 2
