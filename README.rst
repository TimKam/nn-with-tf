Sensing and perception example analyses
=======================================
This repository contains some example logs and analyses to play around with sensing and perception algorithms and tools.

Getting started
===============
To run the examples, proceed as follows:
* Install the dependencies (preferably in a `virtual Python environment <https://docs.python.org/3/tutorial/venv.html>`__) by running ``pip install -r requirements`` in the repository's root directory.
* Play with the examples in `./src/run_examples.py <./src/run_examples.py>`__.

You can run the convolutional network on FloydHub by executing:

.. code-block::

    floyd run --cpu --env tensorflow-1.3 --data /timkam/datasets/cifar/2:my_data "python src/floydhub.py"

If you want to run any of the other networks, adjust the code in `./src/floydhub.py <./src/floydhub.py>`__ accordingly.

To access the graph data used for the presentation run ``tensorboard --logdir=graphs/``.
