

def mse(actual: list, expected: list) -> (float, float):
    loss = 0
    for i in range(len(expected)):

        loss += (expected[i] - actual[i]) ** 2

    loss = loss / len(expected)

    prime = 2 * loss
    return loss, prime
