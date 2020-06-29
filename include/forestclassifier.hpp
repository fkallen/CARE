#ifndef CARE_FOREST_CLASSIFIER
#define CARE_FOREST_CLASSIFIER

#include <config.hpp>

#include <soloader.hpp>

#include <string>
#include <iostream>
#include <utility>

namespace care{

    struct ForestClassifier{

        SoFunction<std::pair<int, int>(double, double, double,
                                        double, double, double,
                                        double, double, double,
                                        double, double, double,
                                        double)> function;

        ForestClassifier(){}

        ForestClassifier(std::string soFilename) 
            : function(soFilename, "shouldCorrect_forest")
        {
           
        }


        ForestClassifier(const ForestClassifier&) = delete;

        ForestClassifier(ForestClassifier&& rhs){
            *this = std::move(rhs);
        }

        ForestClassifier& operator=(const ForestClassifier&) = delete;

        ForestClassifier& operator=(ForestClassifier&& rhs){
            function = std::move(rhs.function);

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


            auto forestresult = function(
                position_support,
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
                maxgini
            );

            const int count_correct = forestresult.second;
            const int count_dontcorrect = forestresult.first;

            bool result = count_correct / (count_dontcorrect + count_correct) > correction_fraction;
            return result;
        }

    };
}


#endif
