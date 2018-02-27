import tensorflow as tensorflow

x = tensorflow.constant(2, name='x')
y = tensorflow.constant(3, name='y')

operation_1 = tensorflow.add(x, y)
operation_2 = tensorflow.multiply(x, y)
operation_3 = tensorflow.pow(operation_1, operation_2)

with tensorflow.Session() as session:
    writer = tensorflow.summary.FileWriter('../graphs', session.graph)
    print(session.run(operation_3))
writer.close()
# returns: (2 + 3) to the power of (2 * 3) = 5 to the power of 6 = 15625
"""
Exercise 1:
graph:
x-|
  |- add      - |
y-|             |
                |- power
x-|             |
  |- multiply - |
y-|
Exercise 2:

When the learning rate is too high...
 1) ...the training accuracy does not improve continuously:
    (graph)
 2) ...the loss function does not converge.
    (graph)
When the learning rate is too low...
    1) ...the training accuracy increases too slowly.
    2) ...the loss function diverges too slowly.
    
When the batch size is too low (10), the resulting model underfits:
(graphs)

When the batch size is too high (2000), the resulting model overfits:
(graphs)

When training for XX iterations with a batch size of XX and a learning rate of XX training accuracy and loss function saturate:


"""


