#include <Python.h>

#include <gpt-oss.h>

typedef struct {
    PyObject_HEAD
    gptoss_model_t handle;
} PyGPTOSSModel;

typedef struct {
    PyObject_HEAD
    gptoss_tokenizer_t handle;
} PyGPTOSSTokenizer;

typedef struct {
    PyObject_HEAD
    gptoss_context_t handle;
} PyGPTOSSContext;

extern PyTypeObject PyGPTOSSModel_Type;
extern PyTypeObject PyGPTOSSTokenizer_Type;
extern PyTypeObject PyGPTOSSContext_Type;
