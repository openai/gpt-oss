#include <Python.h>

#include <gpt-oss.h>

#include "module.h"


static int PyGPTOSSContext_init(PyGPTOSSContext* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"model", "context_length", "max_batch_tokens", NULL};
    PyObject* model = NULL;
    Py_ssize_t context_length = 0; // Default to 0 if None
    Py_ssize_t max_batch_tokens = 0; // Default to 0 if None

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|$ii", kwlist,
                                     &PyGPTOSSModel_Type, &model,
                                     &context_length, &max_batch_tokens))
    {
        return -1;
    }
    if (context_length < 0) {
        PyErr_SetString(PyExc_ValueError, "context_length must be a positive integer");
        return -1;
    }
    if (max_batch_tokens < 0) {
        PyErr_SetString(PyExc_ValueError, "max_batch_tokens must be a positive integer");
        return -1;
    }

    enum gptoss_status status = gptoss_context_create(
        ((const PyGPTOSSModel*) model)->handle,
        (size_t) context_length,
        (size_t) max_batch_tokens,
        &self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception
        goto error;
    }

    return 0;

error:
    gptoss_context_release(self->handle);
    self->handle = NULL;
    return -1;
}

static void PyGPTOSSContext_dealloc(PyGPTOSSContext* self) {
    (void) gptoss_context_release(self->handle);
    self->handle = NULL;
    PyObject_Del((PyObject*) self);
}

static PyObject* PyGPTOSSContext_copy(PyGPTOSSContext *self) {
    PyGPTOSSContext* copy = (PyGPTOSSContext*) PyObject_New(PyGPTOSSContext, Py_TYPE(self));
    if (copy == NULL) {
        return NULL;
    }

    (void) gptoss_context_retain(self->handle);
    copy->handle = self->handle;
    return (PyObject*) copy;
}

static PyObject* PyGPTOSSContext_append(PyGPTOSSContext* self, PyObject* arg) {
    if (PyBytes_Check(arg)) {
        char* string_ptr = NULL;
        Py_ssize_t string_size = 0;
        if (PyBytes_AsStringAndSize(arg, &string_ptr, &string_size) < 0) {
            return NULL;
        }

        const enum gptoss_status status = gptoss_context_append_chars(
            self->handle, string_ptr, string_size, /*num_tokens_out=*/NULL);
        if (status != gptoss_status_success) {
            // TODO: set exception
            return NULL;
        }

        Py_RETURN_NONE;
    } else if (PyUnicode_Check(arg)) {
        Py_ssize_t string_size = 0;
        const char* string_ptr = PyUnicode_AsUTF8AndSize(arg, &string_size);
        if (string_ptr == NULL) {
            return NULL;
        }

        const enum gptoss_status status = gptoss_context_append_chars(
            self->handle, string_ptr, string_size, /*num_tokens_out=*/NULL);
        if (status != gptoss_status_success) {
            // TODO: set exception
            return NULL;
        }

        Py_RETURN_NONE;
    } else if (PyLong_Check(arg)) {
        const unsigned long token_as_ulong = PyLong_AsUnsignedLong(arg);
        if (token_as_ulong == (unsigned long) -1 && PyErr_Occurred()) {
            return NULL;
        }

        const uint32_t token = (uint32_t) token_as_ulong;
        const enum gptoss_status status = gptoss_context_append_tokens(
            self->handle, /*num_tokens=*/1, &token);
        if (status != gptoss_status_success) {
            // TODO: set exception
            return NULL;
        }

        Py_RETURN_NONE;
    } else {
        PyErr_SetString(PyExc_TypeError, "expected a bytes or integer argument");
        return NULL;
    }
}

static PyObject* PyGPTOSSContext_process(PyGPTOSSContext* self) {
    const enum gptoss_status status = gptoss_context_process(self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* PyGPTOSSContext_sample(PyGPTOSSContext* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"max_output_tokens", "temperature", "seed", NULL};
    PyObject* token_list_obj = NULL;
    uint32_t* token_ptr = NULL;

    unsigned int max_output_tokens = 0;
    unsigned long long seed = 0;
    float temperature = 1.0f;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "I|$fK", kwlist,
            &max_output_tokens, &temperature, &seed))
    {
        return NULL;
    }

    token_ptr = (uint32_t*) PyMem_Malloc(max_output_tokens * sizeof(uint32_t));
    if (token_ptr == NULL) {
        goto error;
    }

    size_t num_tokens = 0;
    const enum gptoss_status status = gptoss_context_sample(
        self->handle, temperature, (uint64_t) seed,
        (size_t) max_output_tokens, token_ptr, &num_tokens);
    if (status != gptoss_status_success) {
        // TODO: set exception
        goto error;
    }

    token_list_obj = PyList_New((Py_ssize_t) num_tokens);
    if (token_list_obj == NULL) {
        goto error;
    }

    for (size_t t = 0; t < num_tokens; t++) {
        PyObject* token_obj = PyLong_FromUnsignedLong((unsigned long) token_ptr[t]);
        if (token_obj == NULL) {
            goto error;
        }

        PyList_SET_ITEM(token_list_obj, (Py_ssize_t) t, token_obj);
    }
    
    PyMem_Free(token_ptr);
    return token_list_obj;
    
error:
    PyMem_Free(token_ptr);
    Py_XDECREF(token_list_obj);
    return NULL;
}

static PyObject* PyGPTOSSContext_reset(PyGPTOSSContext* self) {
    const enum gptoss_status status = gptoss_context_reset(self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef PyGPTOSSContext_methods[] = {
    {"__copy__", (PyCFunction) PyGPTOSSContext_copy, METH_NOARGS, "Create a copy of the Context"},
    {"append", (PyCFunction) PyGPTOSSContext_append, METH_O, "Append bytes to the Context"},
    {"process", (PyCFunction) PyGPTOSSContext_process, METH_NOARGS, "Process tokens in the Context"},
    {"sample", (PyCFunction) PyGPTOSSContext_sample, METH_VARARGS | METH_KEYWORDS, "Sample token predictions from the Context"},
    {"reset", (PyCFunction) PyGPTOSSContext_reset, METH_NOARGS, "Discard the content of the Context"},
    {NULL},
};

static PyObject* PyGPTOSSContext_get_num_tokens(PyGPTOSSContext* self, void* closure) {
    size_t num_tokens = 0;
    const enum gptoss_status status = gptoss_context_get_num_tokens(self->handle, &num_tokens);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    return PyLong_FromSize_t(num_tokens);
}

static PyObject* PyGPTOSSContext_get_max_tokens(PyGPTOSSContext* self, void* closure) {
    size_t max_tokens = 0;
    const enum gptoss_status status = gptoss_context_get_max_tokens(self->handle, &max_tokens);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    return PyLong_FromSize_t(max_tokens);
}

static PyObject* PyGPTOSSContext_get_tokens(PyGPTOSSContext* self, void* closure) {
    PyObject* token_list_obj = NULL;
    uint32_t* token_ptr = NULL;

    size_t num_tokens = 0;
    gptoss_context_get_tokens(self->handle, /*tokens_out=*/NULL, /*max_tokens=*/0, &num_tokens);

    if (num_tokens != 0) {
        token_ptr = (uint32_t*) PyMem_Malloc(num_tokens * sizeof(uint32_t));
        if (token_ptr == NULL) {
            // TODO: set exception
            goto error;
        }

        enum gptoss_status status = gptoss_context_get_tokens(self->handle, token_ptr, /*max_tokens=*/num_tokens, &num_tokens);
        if (status != gptoss_status_success) {
            // TODO: set exception
            goto error;
        }
    }

    token_list_obj = PyList_New((Py_ssize_t) num_tokens);
    if (token_list_obj == NULL) {
        goto error;
    }

    for (size_t t = 0; t < num_tokens; t++) {
        PyObject* token_obj = PyLong_FromUnsignedLong((unsigned long) token_ptr[t]);
        if (token_obj == NULL) {
            goto error;
        }

        PyList_SET_ITEM(token_list_obj, (Py_ssize_t) t, token_obj);
    }

    PyMem_Free(token_ptr);
    return token_list_obj;

error:
    PyMem_Free(token_ptr);
    Py_XDECREF(token_list_obj);
    return NULL;
}

static PyGetSetDef PyGPTOSSContext_getseters[] = {
    (PyGetSetDef) {
        .name = "num_tokens",
        .get = (getter) PyGPTOSSContext_get_num_tokens,
        .doc = "Current number of tokens in the context",
    },
    (PyGetSetDef) {
        .name = "max_tokens",
        .get = (getter) PyGPTOSSContext_get_max_tokens,
        .doc = "Maximum number of tokens in the context",
    },
    (PyGetSetDef) {
        .name = "tokens",
        .get = (getter) PyGPTOSSContext_get_tokens,
        .doc = "List of token IDs in the context",
    },
    {NULL}  /* Sentinel */
};

PyTypeObject PyGPTOSSContext_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gptoss.Context",
    .tp_basicsize = sizeof(PyGPTOSSContext),
    .tp_flags = 0
        | Py_TPFLAGS_DEFAULT
        | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Context object",
    .tp_methods = PyGPTOSSContext_methods,
    .tp_getset = PyGPTOSSContext_getseters,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) PyGPTOSSContext_init,
    .tp_dealloc = (destructor) PyGPTOSSContext_dealloc,
};
