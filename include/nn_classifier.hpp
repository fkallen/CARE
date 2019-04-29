#ifndef CARE_NN_CLASSIFIER_HPP
#define CARE_NN_CLASSIFIER_HPP


#include <featureextractor.hpp>
#include <config.hpp>


#include <python2.7/Python.h>

#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <string>

namespace care{

    struct NN_Correction_Classifier_Base{
    public:
        NN_Correction_Classifier_Base() : hasMoved(false){}

        NN_Correction_Classifier_Base(std::string psp, std::string mp):
            py_source_path(psp), call(Py_BuildValue("s", "infer"))
        {
            Py_Initialize();
            std::string path_cmd = "import sys; sys.path.insert(0, '"+py_source_path+"')";
            PyRun_SimpleString(path_cmd.c_str());
            PyObject* module = PyImport_ImportModule("deep_errorcorrector_conv");
            if(module == nullptr){
                PyErr_Print();
                throw std::runtime_error("PyImport_ImportModule");
            }
            PyObject* constructor = PyObject_GetAttrString(module, "Classifier");

            char format[] = "s";
            classifier = PyObject_CallFunction(constructor, format, mp.c_str());

            if(classifier == nullptr){
                PyErr_Print();
                throw std::runtime_error("PyObject_CallFunction(constructor )");
            }

            // Decrease Reference Counts?
        }

        ~NN_Correction_Classifier_Base() {
            if(!hasMoved)
                Py_Finalize();
        }

        NN_Correction_Classifier_Base(const NN_Correction_Classifier_Base&) = delete;
        NN_Correction_Classifier_Base& operator=(const NN_Correction_Classifier_Base&) = delete;

        NN_Correction_Classifier_Base(NN_Correction_Classifier_Base&& rhs){
            *this = std::move(rhs);
        }

        NN_Correction_Classifier_Base& operator=(NN_Correction_Classifier_Base&& rhs){
            hasMoved = rhs.hasMoved;
            py_source_path = std::move(rhs.py_source_path);
            classifier = rhs.classifier;
            call = rhs.call;

            rhs.hasMoved = true;

            return *this;
        }

        bool hasMoved;

        std::string py_source_path;
        PyObject* classifier;
        PyObject* call;
    };

    struct NN_Correction_Classifier {

        public:
            using ClassifierBase = NN_Correction_Classifier_Base;

        ClassifierBase* base;

        NN_Correction_Classifier() : NN_Correction_Classifier(nullptr){}

        NN_Correction_Classifier(ClassifierBase* ptr)
            : base(ptr){}

        std::vector<float> infer(const std::vector<MSAFeature3>& features) const{
            if(features.size() == 0)
                return {};

            PyObject* feature_list = PyList_New(features.size());
            for (size_t i = 0; i < features.size(); i++) {
                PyObject* sample = PyList_New(136);
                for (size_t j = 0; j < 4; j++) {
                    for (size_t k = 0; k < 17; k++) {
                        PyList_SetItem(sample, 34*j+2*k, Py_BuildValue("f", features[i].weights[k][j]));
                        PyList_SetItem(sample, 34*j+2*k+1, Py_BuildValue("f", features[i].counts[k][j]));
                    }
                }
                PyList_SetItem(feature_list, i, sample);
            }
            PyObject* result = PyObject_CallMethodObjArgs(base->classifier, base->call, feature_list, NULL);
            Py_DECREF(feature_list);
            std::vector<float> predictions(features.size());
            for (size_t i = 0; i < features.size(); i++) {
                predictions[i] = PyFloat_AsDouble(PyList_GetItem(result, i));
            }
            Py_DECREF(result);
            return predictions;
        }

        std::vector<float> infer(const std::vector<std::vector<float>>& features) const{
            if(features.size() == 0)
                return {};

            PyObject* feature_list = PyList_New(features.size());
            for (size_t i = 0; i < features.size(); i++) {
                PyObject* sample = PyList_New(features[i].size());
                for (size_t j = 0; j < features[0].size(); j++) {
                    PyList_SetItem(sample, j, Py_BuildValue("f", features[i][j]));
                }
                PyList_SetItem(feature_list, i, sample);
            }
            PyObject* result = PyObject_CallMethodObjArgs(base->classifier, base->call, feature_list, NULL);
            Py_DECREF(feature_list);
            std::vector<float> predictions(features.size());
            for (size_t i = 0; i < features.size(); i++) {
                predictions[i] = PyFloat_AsDouble(PyList_GetItem(result, i));
            }
            Py_DECREF(result);
            return predictions;
        }
    };







}




#endif
