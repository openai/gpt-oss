#include <Python.h>

#include <gpt-oss.h>

#include "module.h"


static int PyGPTOSSModel_init(PyGPTOSSModel* self, PyObject* args, PyObject* kwargs) {
    enum gptoss_status status;
    const char* filepath;

    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        return -1;
    }
    status = gptoss_model_create_from_file(filepath, &self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return -1;
    }
    return 0;
}

static void PyGPTOSSModel_dealloc(PyGPTOSSModel* self) {
    (void) gptoss_model_release(self->handle);
    self->handle = NULL;
    PyObject_Del((PyObject*) self);
}

static PyObject* PyGPTOSSModel_copy(PyGPTOSSModel* self) {
    PyGPTOSSModel* copy = (PyGPTOSSModel*) PyObject_New(PyGPTOSSModel, Py_TYPE(self));
    if (copy == NULL) {
        return NULL;
    }

    (void) gptoss_model_retain(self->handle);
    copy->handle = self->handle;
    return (PyObject*) copy;
}

static PyMethodDef PyGPTOSSModel_methods[] = {
    {"__copy__", (PyCFunction) PyGPTOSSModel_copy, METH_NOARGS, "Create a copy of the Model"},
    {NULL},
};

static PyObject *PyGPTOSSModel_get_max_context_length(PyGPTOSSModel* self, void* closure) {
    size_t max_context_length = 0;
    const enum gptoss_status status = gptoss_model_get_max_context_length(self->handle, &max_context_length);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    return PyLong_FromSize_t(max_context_length);
}

static PyObject *PyGPTOSSModel_get_tokenizer(PyGPTOSSModel* self, void* closure) {
    PyObject* args = PyTuple_Pack(1, self);
    if (args == NULL) {
        return NULL;
    }

    PyObject* tokenizer = PyObject_CallObject((PyObject*) &PyGPTOSSTokenizer_Type, args);
    Py_DECREF(args);
    return tokenizer;
}

static PyGetSetDef PyGPTOSSModel_getseters[] = {
    (PyGetSetDef) {
        .name = "max_context_length",
        .get = (getter) PyGPTOSSModel_get_max_context_length,
        .doc = "Maximum context length supported by the model",
    },
    (PyGetSetDef) {
        .name = "tokenizer",
        .get = (getter) PyGPTOSSModel_get_tokenizer,
        .doc = "Tokenizer object associated with the model",
    },
    {NULL}  // Sentinel
};

PyTypeObject PyGPTOSSModel_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gptoss.Model",
    .tp_basicsize = sizeof(PyGPTOSSModel),
    .tp_flags = 0
        | Py_TPFLAGS_DEFAULT
        | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Model object",
    .tp_methods = PyGPTOSSModel_methods,
    .tp_getset = PyGPTOSSModel_getseters,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) PyGPTOSSModel_init,
    .tp_dealloc = (destructor) PyGPTOSSModel_dealloc,
};
