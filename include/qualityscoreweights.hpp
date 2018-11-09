#ifndef CARE_CPU_QUALITY_SCORE_WEIGHTS_HPP
#define CARE_CPU_QUALITY_SCORE_WEIGHTS_HPP

#include <array>
#include <cmath>

namespace care{
namespace cpu{

    struct QualityScoreConversion{
        static constexpr int ASCII_BASE = 33;
        static constexpr double MIN_WEIGHT = 0.001;

        using Array_t = std::array<double, 256>;

        Array_t qscore_to_error_prob;
        Array_t qscore_to_weight;

        QualityScoreConversion(){
            for(int i = 0; i < 256; i++){
                if(i < ASCII_BASE)
                    qscore_to_error_prob[i] = 1.0;
                else
                    qscore_to_error_prob[i] = std::pow(10.0, -(i-ASCII_BASE)/10.0);
            }

            for(int i = 0; i < 256; i++){
                qscore_to_weight[i] = std::max(MIN_WEIGHT, 1.0 - qscore_to_error_prob[i]);
            }
        }

        double getErrorProbability(char c) const{
            return qscore_to_error_prob[static_cast<unsigned char>(c)];
        }

        double getWeight(char c) const{
            return qscore_to_weight[static_cast<unsigned char>(c)];
        }

        const Array_t& getWeights() const{
            return qscore_to_weight;
        }

        const Array_t& getErrorProbabilities() const{
            return qscore_to_error_prob;
        }
    };

}
}

#endif
