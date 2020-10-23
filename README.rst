pl-mricnn_predict
================================

.. image:: https://badge.fury.io/py/mricnn_predict.svg
    :target: https://badge.fury.io/py/mricnn_predict

.. image:: https://travis-ci.org/FNNDSC/mricnn_predict.svg?branch=master
    :target: https://travis-ci.org/FNNDSC/mricnn_predict

.. image:: https://img.shields.io/badge/python-3.5%2B-blue.svg
    :target: https://badge.fury.io/py/pl-mricnn_predict

.. contents:: Table of Contents


Abstract
--------

An app to predict segmented brain MRI images from a given unsegmented brain MRI


Synopsis
--------

.. code::

    python mricnn_predict.py                                           \
        [-v <level>] [--verbosity <level>]                          \
        [--version]                                                 \
        [--man]                                                     \
        [--meta]                                                    \
        <inputDir>
        <outputDir> 

Description
-----------

``mricnn_predict.py`` is a ChRIS-based application that...

Arguments
---------

.. code::

    [-v <level>] [--verbosity <level>]
    Verbosity level for app. Not used currently.

    [--version]
    If specified, print version number. 
    
    [--man]
    If specified, print (this) man page.

    [--meta]
    If specified, print plugin meta data.


Run
----

This ``plugin`` can be run in two modes: natively as a python package or as a containerized docker image.

Using PyPI
~~~~~~~~~~

To run from PyPI, simply do a 

.. code:: bash

    pip install mricnn_predict

and run with

.. code:: bash

    mricnn_predict.py --man /tmp /tmp

to get inline help. The app should also understand being called with only two positional arguments

.. code:: bash

    mricnn_predict.py /some/input/directory /destination/directory


Using ``docker run``
~~~~~~~~~~~~~~~~~~~~

To run using ``docker``, be sure to assign an "input" directory to ``/incoming`` and an output directory to ``/outgoing``. *Make sure that the* ``$(pwd)/out`` *directory is world writable!*

Now, prefix all calls with 

.. code:: bash

    docker run --rm -v $(pwd)/out:/outgoing                             \
            fnndsc/pl-mricnn_predict mricnn_predict.py                        \

Thus, getting inline help is:

.. code:: bash

    mkdir in out && chmod 777 out
    docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            fnndsc/pl-mricnn_predict mricnn_predict.py                        \
            --man                                                       \
            /incoming /outgoing

Examples
--------


.. code:: bash

    mkdir in out && chmod 777 out
    docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            fnndsc/pl-mricnn_predict mricnn_predict.py                  \
            --testDir test                                              \
            --model model                                               \
            /incoming /outgoing



