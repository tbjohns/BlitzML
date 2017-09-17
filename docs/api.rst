.. container:: note4github

   If you are reading this in the GitHub navigation tree, access the rendered
   documentation at https://tbjohns.github.io/BlitzML/ . 


BlitzML API reference
============================

This is the complete API reference for the BlitzML python package.

Training L1-regularized models
------------------------------

L1 regularization is a popular approach to training sparse models.  BlitzML
efficiently solves L1-regularized problems with a variety of loss functions.

Problem classes
~~~~~~~~~~~~~~~

.. module:: blitzml

.. autoclass:: LassoProblem
  :members:
  :inherited-members: 

.. autoclass:: SparseLogisticRegressionProblem

  Calls to ``solve`` and ``compute_max_l1_penalty`` use interfaces identical to
  the same methods in ``blitzml.LassoProblem``.

.. autoclass:: SparseHuberProblem

  Calls to ``solve`` and ``compute_max_l1_penalty`` use interfaces identical to
  those in ``blitzml.LassoProblem``.

.. autoclass:: SparseSquaredHingeProblem

  Calls to ``solve`` and ``compute_max_l1_penalty`` use interfaces identical to
  those in ``blitzml.LassoProblem``.

.. autoclass:: SparseSmoothedHingeProblem  

Solution classes
~~~~~~~~~~~~~~~~

.. module:: blitzml._sparse_linear

.. autoclass:: LassoSolution()
  :inherited-members: 
  :members:

.. autoclass:: SparseLogisticRegressionSolution()
  :members:

  Except for the following additional method, interface is identical to 
  ``LassoSolution``'s interface.

.. autoclass:: SparseHuberSolution()

  Interface is identical to ``LassoSolution``'s interface.

.. autoclass:: SparseSquaredHingeSolution()

  Interface is identical to ``LassoSolution``'s interface.

.. autoclass:: SparseSmoothedHingeSolution()

  Interface is identical to ``LassoSolution``'s interface.


Utility functions
-----------------

.. module:: blitzml

.. autofunction:: load_solution  

.. autofunction:: parse_log_directory

.. autofunction:: suppress_warnings  

.. autofunction:: unsuppress_warnings  

