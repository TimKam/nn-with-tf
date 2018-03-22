import tensorflow


def run_example_0(x, y):
    """
    Adds two numbers and gives out the TensorFlow graph
    :param x: summand 1
    :param y: summand 2
    :return: the result of the addition
    """

    a = tensorflow.add(x, y)
    with tensorflow.Session() as session:
        writer = tensorflow.summary.FileWriter('./graphs', session.graph)
        print(session.run(a))
    writer.close()
    return a


x = 3
y = 5
run_example_0(x, y)

