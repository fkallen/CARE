#ifndef CARE_FOREST_CLASSIFIER
#define CARE_FOREST_CLASSIFIER

#include <config.hpp>

#include <dlfcn.h> //linux load shared object

#include <functional>
#include <string>
#include <limits>
#include <iostream>
#include <utility>
#include <stdexcept>

namespace care{

    struct ForestClassifier{

        std::string soFilename;
        void* soHandle = nullptr;
        std::function<std::pair<int, int>(double, double, double,
                                        double, double, double,
                                        double, double, double,
                                        double, double, double,
                                        double)> shouldCorrect_forest;

        ForestClassifier(){}

        ForestClassifier(std::string soFilename_) : soFilename(std::move(soFilename_)){
            //std::cerr << soFilename << '\n';
            bool failure = false;

            soHandle = dlopen(soFilename.c_str(), RTLD_NOW);
        	if (!soHandle) {
        		std::cerr << dlerror() << '\n';
        		failure = true;
        	}

            auto fptr = dlsym(soHandle, "shouldCorrect_forest");

            const char* error = dlerror();
        	if (error != NULL) {
        		std::cerr << error << '\n';
        		failure = true;
        	}

            if(failure){
                throw std::runtime_error("Cannot load forest object:" + soFilename);
            }

            shouldCorrect_forest = (std::pair<int, int> (*) (double, double, double,
                                            double, double, double,
                                            double, double, double,
                                            double, double, double,
                                            double)) fptr;
        }

        ~ForestClassifier(){
            if(soHandle != nullptr){
                int dlcloseretval = dlclose(soHandle);
            	if (dlcloseretval != 0) {
            		std::cerr << dlerror() << '\n';
            	}
            }
        }


        ForestClassifier(const ForestClassifier&) = delete;

        ForestClassifier(ForestClassifier&& rhs){
            *this = std::move(rhs);
        }

        ForestClassifier& operator=(const ForestClassifier&) = delete;

        ForestClassifier& operator=(ForestClassifier&& rhs){
            if(soHandle != nullptr){
                int dlcloseretval = dlclose(soHandle);
            	if (dlcloseretval != 0) {
            		std::cerr << dlerror() << '\n';
            	}
            }
            soFilename = std::move(rhs.soFilename);
            soHandle = std::move(rhs.soHandle);
            shouldCorrect_forest = std::move(rhs.shouldCorrect_forest);

            rhs.soFilename = "";
            rhs.soHandle = nullptr;
            rhs.shouldCorrect_forest = nullptr;

            return *this;
        }

        bool shouldCorrect(double position_support,
                            double position_coverage,
                            double alignment_coverage,
                            double dataset_coverage,
                            double min_support,
                            double min_coverage,
                            double max_support,
                            double max_coverage,
                            double mean_support,
                            double mean_coverage,
                            double median_support,
                            double median_coverage,
                            double maxgini,
                            double correction_fraction) const noexcept{


        auto forestresult = shouldCorrect_forest(position_support,
                                                position_coverage,
                                                alignment_coverage,
                                                dataset_coverage,
                                                min_support,
                                                min_coverage,
                                                max_support,
                                                max_coverage,
                                                mean_support,
                                                mean_coverage,
                                                median_support,
                                                median_coverage,
                                                maxgini);

            const int count_correct = forestresult.second;
            const int count_dontcorrect = forestresult.first;

            bool result = count_correct / (count_dontcorrect + count_correct) > correction_fraction;
            return result;
        }

    };
}


#endif
