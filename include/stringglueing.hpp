#ifndef CARE_STRING_GLUEING_HPP
#define CARE_STRING_GLUEING_HPP

#include <cpu_alignment.hpp>
#include <hostdevicefunctions.cuh>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <cassert>

namespace care{


struct GlueDecision{
    /*
        strings s1 and s2 will be combined to produce a string of length resultlength
        in the resulting string, s1[0] will be at index s1FirstResultIndex, 
        and s2[0] will be at index s2FirstResultIndex
    */
    int s1FirstResultIndex{};
    int s2FirstResultIndex{};
    int resultlength{};
    std::string_view s1{};
    std::string_view s2{};
};


struct MismatchRatioGlueDecider{
public:
    MismatchRatioGlueDecider(int overlapLowerBound_, float mismatchRatioUpperBound_)
        : overlapLowerBound(overlapLowerBound_), mismatchRatioUpperBound(mismatchRatioUpperBound_)
    {

    }

    // Given strings s1 and s2, check if it is possible to overlap them and glue them together to produce string of length resultlength
    // If the overlap is bad, the result is empty
    std::optional<GlueDecision> operator()(std::string_view s1, std::string_view s2, int resultlength) const{
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
                GlueDecision decision;

                decision.s1FirstResultIndex = 0;
                decision.s2FirstResultIndex = s2BeginInCombined >= 0 ? s2BeginInCombined : 0;
                decision.resultlength = resultlength;
                decision.s1 = s1;
                decision.s2 = s2;
                decision.s1.remove_suffix(s1length - (s1EndInCombined + 1));
                assert(s2Start <= int(decision.s2.size()));
                decision.s2.remove_prefix(s2Start);

                return decision;
            }else{
                return {}; //no result
            }
        }else{
            return {}; //no result
        }
    }
private:
    int overlapLowerBound;
    float mismatchRatioUpperBound;
};



struct MatchLengthGlueDecider{
public:
    MatchLengthGlueDecider(int overlapLowerBound_, float matchLengthLowerBound_)
        : overlapLowerBound(overlapLowerBound_), matchLengthLowerBound(matchLengthLowerBound_)
    {

    }

    // Given strings s1 and s2, check if it is possible to overlap them and glue them together to produce string of length resultlength
    // If the longest match is not long enough, the result is empty
    std::optional<GlueDecision> operator()(std::string_view s1, std::string_view s2, int resultlength) const{
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
            const int l = cpu::longestMatch(
                s1begin + (s1EndInCombined+1) - overlapSize, s1end, 
                s2begin + s2Start, s2end
            );

            if(l >= matchLengthLowerBound){
                GlueDecision decision;

                decision.s1FirstResultIndex = 0;
                decision.s2FirstResultIndex = s2BeginInCombined >= 0 ? s2BeginInCombined : 0;
                decision.resultlength = resultlength;
                decision.s1 = s1;
                decision.s2 = s2;
                decision.s1.remove_suffix(s1length - (s1EndInCombined + 1));
                assert(s2Start <= int(decision.s2.size()));
                decision.s2.remove_prefix(s2Start);

                return decision;
            }else{
                return {}; //no result
            }
        }else{
            return {}; //no result
        }
    }

private:
    int overlapLowerBound{};
    int matchLengthLowerBound{};
};


/*
    Very naive gluer. Take as much from s2 as possible, fill remaining positions at the left end with s1
*/
struct NaiveGluer{
public:

    std::string operator()(const GlueDecision& g) const{
        assert(g.s1FirstResultIndex == 0);
        assert(int(g.s1.size()) <= g.resultlength);
        assert(int(g.s2.size()) <= g.resultlength);

        std::string result(g.resultlength, '#');

        const int s11remaining = g.resultlength - g.s2.size();
        const auto it = std::copy_n(g.s1.begin(), s11remaining, result.begin());

        auto end = std::copy(g.s2.begin(), g.s2.end(), it);
        assert(end == result.end());

        return result;
    }

};


/*
    result string will begin with prefix s1[0 : min(s1length, originalReadLength)]
    result string will end with suffix s2[max(0, s2length - originalReadLength) : min(s2length, originalReadLength)]
    Gap is filled with either s1 or s2, depending on the weight of the position
*/
struct WeightedGapGluer{
public:
    WeightedGapGluer(int origLength) : originalReadLength(origLength){}

    std::string operator()(const GlueDecision& g) const{
        assert(g.s1FirstResultIndex == 0);
        assert(int(g.s1.size()) <= g.resultlength);
        assert(int(g.s2.size()) <= g.resultlength);

        std::string result(g.resultlength, '#');

        const int numToCopys1 = std::min(
            g.resultlength,
            std::min(int(g.s1.size()), originalReadLength)
        );
        const int numToCopys2 = std::min(
            g.resultlength,
            std::min(int(g.s2.size()), originalReadLength)
        );

        const auto gapbegin = std::copy_n(g.s1.begin(), numToCopys1, result.begin());      
        const auto gapend = result.begin() + result.size() - numToCopys2;
        std::copy_n(g.s2.begin() + g.s2.size() - numToCopys2, numToCopys2, gapend);

        //fill the gap
        if(std::distance(result.begin(), gapbegin) < std::distance(result.begin(), gapend)){

            auto getweight = [&](const auto& s, int pos){
                if(pos < 0 || pos >= int(s.size())){
                    return 0.0f;
                }else{
                    // original positions have weight 1, weight of other positions decreases with distance to original
                    if(pos < originalReadLength){
                        return 1.0f;
                    }else{
                        //linear interpolation
                        const int unoriginalpositions = s.size() - originalReadLength;
                        const float f = 1.0f / unoriginalpositions;

                        return 1.0f - f * (pos - originalReadLength);
                    }
                }
            };

            const int gapsize = std::distance(gapbegin, gapend);
            const int firstGapPos = std::distance(result.begin(), gapbegin);

            auto iter = gapbegin;
            for(int i = 0; i < gapsize; i++){
                const int positionInS1 = firstGapPos + i;
                const int positionInS2 = firstGapPos + i - g.s2FirstResultIndex;

                const float w1 = getweight(g.s1, positionInS1);
                const float w2 = getweight(g.s2, positionInS2);
                assert((w1 != 0.0f) || (w2 != 0.0f));

                if(fgeq(w1, w2)){
                    assert(positionInS1 < int(g.s1.size()));
                    *iter = g.s1[positionInS1];
                }else{
                    assert(positionInS2 < int(g.s2.size()));
                    *iter = g.s2[positionInS2];
                }
                //*iter = fgeq(getweight(g.s1, positionInS1), getweight(g.s2, positionInS2)) ? g.s1[positionInS1] : g.s2[positionInS2];

                ++iter;
            }
        }


        return result;
    }
private:
    int originalReadLength{};
};






struct MismatchRatioGluer{

    MismatchRatioGluer(int overlapLowerBound_, float mismatchRatioUpperBound_)
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



struct MatchLengthGluer{

    MatchLengthGluer(int overlapLowerBound_, float matchLengthLowerBound_)
        : overlapLowerBound(overlapLowerBound_), matchLengthLowerBound(matchLengthLowerBound_)
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
            const int l = cpu::longestMatch(
                s1begin + (s1EndInCombined+1) - overlapSize, s1end, 
                s2begin + s2Start, s2end
            );

            if(l >= matchLengthLowerBound){
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
    int matchLengthLowerBound;

};



} //namespace care

#endif