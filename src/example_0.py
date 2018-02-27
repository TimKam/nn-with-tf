import tensorflow

x = 3
y = 5
a = tensorflow.add(x, y)
print(a)
with tensorflow.Session() as session:
    writer = tensorflow.summary.FileWriter('./graphs', session.graph)
    print(session.run(a))
writer.close()

