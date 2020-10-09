#ifndef CARE_STRING_GLUEING_HPP
#define CARE_STRING_GLUEING_HPP

#include <cpu_alignment.hpp>
#include <hostdevicefunctions.cuh>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>

namespace care{




struct Gluer{

    Gluer(int overlapLowerBound_, float mismatchRatioUpperBound_)
        : overlapLowerBound(overlapLowerBound_), mismatchRatioUpperBound(mismatchRatioUpperBound_)
    {

    }

    // Given strings s1 and s2, overlap them and glue them together to produce string of length resultlength
    // The resulting string begins with a prefix of s1 and ends with a suffix of s2.
    // If the overlap is bad, the result is empty

    std::optional<std::string> operator()(std::string_view s1, std::string_view s2, int resultlength) const{
        const auto s1begin = s1.begin();
        const auto s1end = s1.end();
        const auto s2begin = s2.begin();
        const auto s2end = s2.end();

        const int s1length = std::distance(s1begin, s1end);
        const int s2length = std::distance(s2begin, s2end);

        //the last position of rlString should be at position x in the combined string
        const int x = resultlength - 1;

        const int s2BeginInCombined = x - s2length + 1; //theoretical position of first character of s2 in the combined string
        const int s1EndInCombined = std::min(x, s1length - 1);
        const int overlapSize = std::min(
            std::max(0, s1EndInCombined - s2BeginInCombined + 1), // overlap can be at most the positions between begin of s1 and end of s1
            std::min(
                std::min(s1length, s2length), // overlap cannot be longer than any of both strings
                x+1 //overlap cannot be longer than specified segment length
            )
        );

        if(overlapSize >= overlapLowerBound){
            const int s2Start = std::max(0, -s2BeginInCombined);
            const int ham = cpu::hammingDistanceOverlap(
                s1begin + (s1EndInCombined+1) - overlapSize, s1end, 
                s2begin + s2Start, s2end
            );
            const float mismatchRatio = float(ham) / float(overlapSize);

            if(fleq(mismatchRatio, mismatchRatioUpperBound)){
                const int newLength = x+1;
                const int s11remaining = newLength - (s2length - s2Start);
                std::string sequence(newLength, 'F');

                const auto it = std::copy_n(s1begin, s11remaining, sequence.begin());
                std::copy(s2begin + s2Start, s2end, it);

                return std::optional<std::string>{std::move(sequence)};
            }else{
                return {}; //no result
            }
        }else{
            return {}; //no result
        }
    }


    int overlapLowerBound;
    float mismatchRatioUpperBound;

};




} //namespace care

#endif