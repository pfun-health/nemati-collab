#include "cmaes.h"

double c_generateCost(PyObject *self // pointer to class instance
		      )
{
  
  /* extension wrapper for _generateCost method in
     the pycardio.fit_model.optimize.Optimizer class */

  int N = a_p->N; // len of xmean
  PyObject *xList, *item;
  Py_ssize_t i, n;
  n = (Py_ssize_t) N;
  
  xList = PyList_New(n); // new list
  
  // convert xmean_p to PyListObject xList
  for(i=0; i<n; i++){
    PyObject *index = PyLong_FromSsize_t(i);
    item = PyFloat_FromDouble(xmean_p[(int) i]);
    PyObject_SetItem(xList, index, item);
    Py_DECREF(index);
    Py_DECREF(item);
  }

  PyObject *genCostName = PyUnicode_FromString("_generateCost");
  // Call the Optimizer._generateCost method (current instance)
  PyObject *costResult = PyObject_CallMethodObjArgs(self, genCostName, xList, NULL);
  double cost = PyFloat_AsDouble(costResult);
  if (cost != -1){
    return cost;
  }
  else {
    fprintf(stderr, "Error: _generateCost returned -1.\n");
    return -1;
  }

}
