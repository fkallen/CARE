#include "../inc/multiple_sequence_alignment.hpp"
#include <algorithm>
#include <cstring>

namespace care{

namespace pileup{

    PileupImage::PileupImage(double m_coverage,
                int kmerlength){

        correctionSettings.m = m_coverage;
        correctionSettings.k = kmerlength;
    }

    PileupImage::PileupImage(const PileupImage& other){
        *this = other;
    }

    PileupImage::PileupImage(PileupImage&& other){
        *this = std::move(other);
    }

    PileupImage& PileupImage::operator=(const PileupImage& other){
        resize(other.max_n_columns);
        std::memcpy(h_As.data(), other.h_As.data(), sizeof(int) * other.max_n_columns);
        std::memcpy(h_Cs.data(), other.h_Cs.data(), sizeof(int) * other.max_n_columns);
        std::memcpy(h_Gs.data(), other.h_Gs.data(), sizeof(int) * other.max_n_columns);
        std::memcpy(h_Ts.data(), other.h_Ts.data(), sizeof(int) * other.max_n_columns);
        std::memcpy(h_Aweights.data(), other.h_Aweights.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_Cweights.data(), other.h_Cweights.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_Gweights.data(), other.h_Gweights.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_Tweights.data(), other.h_Tweights.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_consensus.data(), other.h_consensus.data(), sizeof(char) * other.max_n_columns);
        std::memcpy(h_support.data(), other.h_support.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_coverage.data(), other.h_coverage.data(), sizeof(int) * other.max_n_columns);
        std::memcpy(h_origWeights.data(), other.h_origWeights.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_origCoverage.data(), other.h_origCoverage.data(), sizeof(int) * other.max_n_columns);

        n_columns = other.n_columns;
        columnProperties = other.columnProperties;
        correctionSettings = other.correctionSettings;

        return *this;
    }



    PileupImage& PileupImage::operator=(PileupImage&& other){
        h_As = std::move(other.h_As);
        h_Cs = std::move(other.h_Cs);
        h_Gs = std::move(other.h_Gs);
        h_Ts = std::move(other.h_Ts);
        h_Aweights = std::move(other.h_Aweights);
        h_Cweights = std::move(other.h_Cweights);
        h_Gweights = std::move(other.h_Gweights);
        h_Tweights = std::move(other.h_Tweights);
        h_consensus = std::move(other.h_consensus);
        h_support = std::move(other.h_support);
        h_coverage = std::move(other.h_coverage);
        h_origWeights = std::move(other.h_origWeights);
        h_origCoverage = std::move(other.h_origCoverage);

        n_columns = other.n_columns;
        columnProperties = other.columnProperties;
        correctionSettings = other.correctionSettings;

        return *this;
    }


    void PileupImage::resize(int cols){

        if(cols > max_n_columns){
            const int newmaxcols = 1.5 * cols;

            h_consensus.resize(newmaxcols);
            h_support.resize(newmaxcols);
            h_coverage.resize(newmaxcols);
            h_origWeights.resize(newmaxcols);
            h_origCoverage.resize(newmaxcols);
            h_As.resize(newmaxcols);
            h_Cs.resize(newmaxcols);
            h_Gs.resize(newmaxcols);
            h_Ts.resize(newmaxcols);
            h_Aweights.resize(newmaxcols);
            h_Cweights.resize(newmaxcols);
            h_Gweights.resize(newmaxcols);
            h_Tweights.resize(newmaxcols);

            max_n_columns = newmaxcols;
        }

        n_columns = cols;
    }

    void PileupImage::clear(){
			auto zero = [](auto& vec){
				std::fill(vec.begin(), vec.end(), 0);
			};

			zero(h_support);
			zero(h_coverage);
			zero(h_origWeights);
			zero(h_origCoverage);
			zero(h_As);
			zero(h_Cs);
			zero(h_Gs);
			zero(h_Ts);
			zero(h_Aweights);
			zero(h_Cweights);
			zero(h_Gweights);
			zero(h_Tweights);

			std::fill(h_consensus.begin(), h_consensus.end(), '\0');
    }

    void PileupImage::destroy(){
            auto destroyvec = [](auto& vec){
                vec.clear();
                vec.shrink_to_fit();
            };

            destroyvec(h_support);
            destroyvec(h_coverage);
            destroyvec(h_origWeights);
            destroyvec(h_origCoverage);
            destroyvec(h_As);
            destroyvec(h_Cs);
            destroyvec(h_Gs);
            destroyvec(h_Ts);
            destroyvec(h_Aweights);
            destroyvec(h_Cweights);
            destroyvec(h_Gweights);
            destroyvec(h_Tweights);

            destroyvec(h_consensus);
    }


void PileupImage::cpu_find_consensus_internal(){
    for(int i = 0; i < columnProperties.columnsToCheck; i++){
        char cons = 'A';
        double consWeight = h_Aweights[i];
        if(h_Cweights[i] > consWeight){
            cons = 'C';
            consWeight = h_Cweights[i];
        }
        if(h_Gweights[i] > consWeight){
            cons = 'G';
            consWeight = h_Gweights[i];
        }
        if(h_Tweights[i] > consWeight){
            cons = 'T';
            consWeight = h_Tweights[i];
        }
        h_consensus[i] = cons;

        const double columnWeight = h_Aweights[i] + h_Cweights[i] + h_Gweights[i] + h_Tweights[i];
        h_support[i] = consWeight / columnWeight;
    }
}

PileupImage::CorrectionResult PileupImage::cpu_correct_sequence_internal(const std::string& sequence_to_correct,
                                    double estimatedErrorrate,
                                    double avg_support_threshold,
                                    double min_support_threshold,
                                    double min_coverage_threshold){

    cpu_find_consensus_internal();

    const int subjectlength = sequence_to_correct.length();

    CorrectionResult result;
    result.isCorrected = false;
    result.correctedSequence.resize(subjectlength);
    result.stats.avg_support = 0;
    result.stats.min_support = 1.0;
    result.stats.max_coverage = 0;
    result.stats.min_coverage = std::numeric_limits<int>::max();
    //get stats for subject columns
    for(int i = columnProperties.subjectColumnsBegin_incl; i < columnProperties.subjectColumnsEnd_excl; i++){
        assert(i < columnProperties.columnsToCheck);

        result.stats.avg_support += h_support[i];
        result.stats.min_support = std::min(h_support[i], result.stats.min_support);
        result.stats.max_coverage = std::max(h_coverage[i], result.stats.max_coverage);
        result.stats.min_coverage = std::min(h_coverage[i], result.stats.min_coverage);
    }

    result.stats.avg_support /= subjectlength;

    auto isGoodAvgSupport = [&](double avgsupport){
        return avgsupport >= avg_support_threshold;
    };
    auto isGoodMinSupport = [&](double minsupport){
        return minsupport >= min_support_threshold;
    };
    auto isGoodMinCoverage = [&](double mincoverage){
        return mincoverage >= min_coverage_threshold;
    };

    //TODO vary parameters
    result.stats.isHQ = isGoodAvgSupport(result.stats.avg_support)
                        && isGoodMinSupport(result.stats.min_support)
                        && isGoodMinCoverage(result.stats.min_coverage);

    result.stats.failedAvgSupport = !isGoodAvgSupport(result.stats.avg_support);
    result.stats.failedMinSupport = !isGoodMinSupport(result.stats.min_support);
    result.stats.failedMinCoverage = !isGoodMinCoverage(result.stats.min_coverage);

    if(result.stats.isHQ){
#if 1
        //correct anchor
        for(int i = 0; i < subjectlength; i++){
            const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;
            result.correctedSequence[i] = h_consensus[globalIndex];
        }
        result.isCorrected = true;
#endif
    }else{
#if 1
        //correct anchor


#if 1
        result.correctedSequence = sequence_to_correct;
        bool foundAColumn = false;
        for(int i = 0; i < subjectlength; i++){
            const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;

            if(h_support[globalIndex] > 0.5 && h_origCoverage[globalIndex] < min_coverage_threshold){
                double avgsupportkregion = 0;
                int c = 0;
                bool kregioncoverageisgood = true;

                for(int j = i - correctionSettings.k/2; j <= i + correctionSettings.k/2 && kregioncoverageisgood; j++){
                    if(j != i && j >= 0 && j < subjectlength){
                        avgsupportkregion += h_support[columnProperties.subjectColumnsBegin_incl + j];
                        kregioncoverageisgood &= (h_coverage[columnProperties.subjectColumnsBegin_incl + j] >= min_coverage_threshold);
                        c++;
                    }
                }

                avgsupportkregion /= c;
                if(kregioncoverageisgood && avgsupportkregion >= 1.0-estimatedErrorrate){
                    result.correctedSequence[i] = h_consensus[globalIndex];
                    foundAColumn = true;
                }
            }
        }

        result.isCorrected = foundAColumn;
#else
        result.correctedSequence = sequence_to_correct;
        const int regionsize = correctionSettings.k;
        bool foundAColumn = false;

        for(int columnindex = columnProperties.subjectColumnsBegin_incl - regionsize/2;
            columnindex < columnProperties.subjectColumnsEnd_excl;
            columnindex += regionsize){

            double supportsum = 0;
            double minsupport = std::numeric_limits<double>::max();
            double maxsupport = std::numeric_limits<double>::min();

            int origcoveragesum = 0;
            int minorigcoverage = std::numeric_limits<int>::max();
            int maxorigcoverage = std::numeric_limits<int>::min();

            int coveragesum = 0;
            int mincoverage = std::numeric_limits<int>::max();
            int maxcoverage = std::numeric_limits<int>::min();

            int c = 0;
            for(int i = 0; i < regionsize; i++){
                const int index = columnindex + i;
                if(0 <= index && index < columnProperties.columnsToCheck){
                    supportsum += h_support[index];
                    minsupport = std::min(minsupport, h_support[index]);
                    maxsupport = std::max(maxsupport, h_support[index]);

                    origcoveragesum += h_origCoverage[index];
                    minorigcoverage = std::min(minorigcoverage, h_origCoverage[index]);
                    maxorigcoverage = std::max(maxorigcoverage, h_origCoverage[index]);

                    coveragesum += h_coverage[index];
                    mincoverage = std::min(mincoverage, h_coverage[index]);
                    maxcoverage = std::max(maxcoverage, h_coverage[index]);

                    c++;
                }
            }
            const double avgsupport = supportsum / c;

            bool isHQregion = isGoodAvgSupport(avgsupport)
                               && isGoodMinSupport(minsupport)
                               && isGoodMinCoverage(mincoverage);

           if(isHQregion){
               //correct anchor
               for(int i = 0; i < regionsize; i++){
                   const int index = columnindex + i;
                   if(columnProperties.subjectColumnsBegin_incl <= index && index < columnProperties.subjectColumnsEnd_excl){
                       const int localindex = index - columnProperties.subjectColumnsBegin_incl;
                       result.correctedSequence[localindex] = h_consensus[index];
                   }
               }
               result.isCorrected = true;
           }else{
               for(int i = 0; i < regionsize; i++){
                   const int index = columnindex + i;
                   if(columnProperties.subjectColumnsBegin_incl <= index
                       && index < columnProperties.subjectColumnsEnd_excl){

                       if(h_support[index] > 0.5
                           && h_origCoverage[index] < min_coverage_threshold
                           && isGoodAvgSupport(avgsupport)
                           && mincoverage >= min_coverage_threshold){

                               const int localindex = index - columnProperties.subjectColumnsBegin_incl;
                               result.correctedSequence[localindex] = h_consensus[index];
                       }
                   }
               }
               result.isCorrected = true;
           }
        }

        result.isCorrected = foundAColumn;

#endif


#endif
    }

    return result;
}

#if 0

bool PileupImage::shouldCorrect(double min_support,
            double min_coverage,
            double max_support,
            double max_coverage,
            double mean_support,
            double mean_coverage,
            double median_support,
            double median_coverage) const{
  if ( max_coverage <= 103.5 ) {
    if ( median_coverage <= 66.5 ) {
      if ( min_support <= 0.733824968338 ) {
        if ( mean_coverage <= 31.7670440674 ) {
          if ( min_coverage <= 8.5 ) {
            return true;
          }
          else {  // if min_coverage > 8.5
            return false;
          }
        }
        else {  // if mean_coverage > 31.7670440674
          return true;
        }
      }
      else {  // if min_support > 0.733824968338
        if ( mean_coverage <= 60.6869735718 ) {
          return true;
        }
        else {  // if mean_coverage > 60.6869735718
          if ( max_coverage <= 77.5 ) {
            return true;
          }
          else {  // if max_coverage > 77.5
            return true;
          }
        }
      }
    }
    else {  // if median_coverage > 66.5
      if ( mean_coverage <= 66.9183044434 ) {
        if ( min_coverage <= 46.0 ) {
          return true;
        }
        else {  // if min_coverage > 46.0
          if ( mean_coverage <= 65.7979812622 ) {
            return false;
          }
          else {  // if mean_coverage > 65.7979812622
            return true;
          }
        }
      }
      else {  // if mean_coverage > 66.9183044434
        if ( median_support <= 0.951799988747 ) {
          if ( median_coverage <= 70.5 ) {
            return false;
          }
          else {  // if median_coverage > 70.5
            return true;
          }
        }
        else {  // if median_support > 0.951799988747
          if ( max_coverage <= 102.5 ) {
            return false;
          }
          else {  // if max_coverage > 102.5
            return true;
          }
        }
      }
    }
  }
  else {  // if max_coverage > 103.5
    if ( mean_coverage <= 89.6746063232 ) {
      if ( median_support <= 0.927744984627 ) {
        if ( max_support <= 0.906044960022 ) {
          if ( min_support <= 0.805135011673 ) {
            return true;
          }
          else {  // if min_support > 0.805135011673
            return true;
          }
        }
        else {  // if max_support > 0.906044960022
          if ( min_support <= 0.91034001112 ) {
            return false;
          }
          else {  // if min_support > 0.91034001112
            return true;
          }
        }
      }
      else {  // if median_support > 0.927744984627
        if ( median_coverage <= 75.0 ) {
          return true;
        }
        else {  // if median_coverage > 75.0
          if ( median_coverage <= 91.5 ) {
            return false;
          }
          else {  // if median_coverage > 91.5
            return true;
          }
        }
      }
    }
    else {  // if mean_coverage > 89.6746063232
      if ( mean_support <= 0.915690004826 ) {
        if ( mean_support <= 0.855924963951 ) {
          if ( min_support <= 0.793125033379 ) {
            return true;
          }
          else {  // if min_support > 0.793125033379
            return true;
          }
        }
        else {  // if mean_support > 0.855924963951
          if ( max_coverage <= 233.5 ) {
            return false;
          }
          else {  // if max_coverage > 233.5
            return false;
          }
        }
      }
      else {  // if mean_support > 0.915690004826
        if ( min_support <= 0.951964974403 ) {
          if ( mean_support <= 0.951475024223 ) {
            return false;
          }
          else {  // if mean_support > 0.951475024223
            return true;
          }
        }
        else {  // if min_support > 0.951964974403
          if ( min_support <= 0.96180999279 ) {
            return false;
          }
          else {  // if min_support > 0.96180999279
            return false;
          }
        }
      }
    }
  }
}

#else


bool PileupImage::shouldCorrect(double min_support, double min_coverage, double max_support, double max_coverage,
    double mean_support, double mean_coverage, double median_support, double median_coverage) const{
  if ( min_support <= 0.933934986591 ) {
    if ( median_support <= 0.870015025139 ) {
      if ( median_support <= 0.818364977837 ) {
        if ( min_coverage <= 46.5 ) {
          if ( max_coverage <= 81.5 ) {
            if ( mean_coverage <= 14.2330322266 ) {
              if ( max_coverage <= 41.5 ) {
                if ( min_coverage <= 9.5 ) {
                  if ( mean_coverage <= 12.9128789902 ) {
                    if ( mean_coverage <= 12.9045448303 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 12.9045448303
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 12.9128789902
                    if ( min_coverage <= 8.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 8.5
                      return true;
                    }
                  }
                }
                else {  // if min_coverage > 9.5
                  if ( min_support <= 0.51266002655 ) {
                    return false;
                  }
                  else {  // if min_support > 0.51266002655
                    if ( max_coverage <= 23.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 23.5
                      return true;
                    }
                  }
                }
              }
              else {  // if max_coverage > 41.5
                if ( min_support <= 0.60119497776 ) {
                  if ( max_coverage <= 42.5 ) {
                    if ( min_support <= 0.547885000706 ) {
                      return true;
                    }
                    else {  // if min_support > 0.547885000706
                      return true;
                    }
                  }
                  else {  // if max_coverage > 42.5
                    return true;
                  }
                }
                else {  // if min_support > 0.60119497776
                  if ( mean_support <= 0.601590037346 ) {
                    return false;
                  }
                  else {  // if mean_support > 0.601590037346
                    if ( mean_coverage <= 12.7279415131 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 12.7279415131
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if mean_coverage > 14.2330322266
              if ( min_coverage <= 2.5 ) {
                if ( max_coverage <= 54.5 ) {
                  if ( max_coverage <= 30.5 ) {
                    if ( min_support <= 0.544324994087 ) {
                      return false;
                    }
                    else {  // if min_support > 0.544324994087
                      return true;
                    }
                  }
                  else {  // if max_coverage > 30.5
                    if ( mean_coverage <= 23.514705658 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 23.514705658
                      return true;
                    }
                  }
                }
                else {  // if max_coverage > 54.5
                  if ( median_coverage <= 13.25 ) {
                    if ( mean_coverage <= 14.4411764145 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 14.4411764145
                      return true;
                    }
                  }
                  else {  // if median_coverage > 13.25
                    if ( median_coverage <= 24.25 ) {
                      return true;
                    }
                    else {  // if median_coverage > 24.25
                      return true;
                    }
                  }
                }
              }
              else {  // if min_coverage > 2.5
                if ( max_support <= 0.50013500452 ) {
                  if ( min_support <= 0.500110030174 ) {
                    return true;
                  }
                  else {  // if min_support > 0.500110030174
                    if ( max_coverage <= 53.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 53.5
                      return true;
                    }
                  }
                }
                else {  // if max_support > 0.50013500452
                  if ( median_coverage <= 17.75 ) {
                    if ( mean_coverage <= 14.242647171 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 14.242647171
                      return true;
                    }
                  }
                  else {  // if median_coverage > 17.75
                    if ( mean_coverage <= 15.397436142 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 15.397436142
                      return true;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_coverage > 81.5
            if ( mean_support <= 0.758054971695 ) {
              if ( max_support <= 0.758035004139 ) {
                if ( mean_support <= 0.753515005112 ) {
                  if ( mean_coverage <= 37.1819839478 ) {
                    if ( median_coverage <= 37.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 37.5
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 37.1819839478
                    if ( median_coverage <= 54.25 ) {
                      return true;
                    }
                    else {  // if median_coverage > 54.25
                      return true;
                    }
                  }
                }
                else {  // if mean_support > 0.753515005112
                  if ( max_support <= 0.753525018692 ) {
                    return false;
                  }
                  else {  // if max_support > 0.753525018692
                    if ( max_coverage <= 102.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 102.5
                      return true;
                    }
                  }
                }
              }
              else {  // if max_support > 0.758035004139
                return false;
              }
            }
            else {  // if mean_support > 0.758054971695
              if ( mean_coverage <= 27.7941169739 ) {
                return false;
              }
              else {  // if mean_coverage > 27.7941169739
                if ( max_coverage <= 111.5 ) {
                  if ( min_coverage <= 21.5 ) {
                    if ( mean_coverage <= 38.6568641663 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 38.6568641663
                      return true;
                    }
                  }
                  else {  // if min_coverage > 21.5
                    if ( min_coverage <= 25.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 25.5
                      return true;
                    }
                  }
                }
                else {  // if max_coverage > 111.5
                  if ( mean_coverage <= 75.5588226318 ) {
                    return true;
                  }
                  else {  // if mean_coverage > 75.5588226318
                    if ( min_coverage <= 19.0 ) {
                      return false;
                    }
                    else {  // if min_coverage > 19.0
                      return true;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if min_coverage > 46.5
          if ( max_support <= 0.765484988689 ) {
            if ( max_coverage <= 509.0 ) {
              if ( mean_support <= 0.728345036507 ) {
                if ( min_coverage <= 298.5 ) {
                  if ( max_coverage <= 153.5 ) {
                    if ( max_support <= 0.668524980545 ) {
                      return true;
                    }
                    else {  // if max_support > 0.668524980545
                      return true;
                    }
                  }
                  else {  // if max_coverage > 153.5
                    if ( max_support <= 0.567404985428 ) {
                      return true;
                    }
                    else {  // if max_support > 0.567404985428
                      return true;
                    }
                  }
                }
                else {  // if min_coverage > 298.5
                  if ( max_coverage <= 331.5 ) {
                    if ( max_support <= 0.626259982586 ) {
                      return false;
                    }
                    else {  // if max_support > 0.626259982586
                      return false;
                    }
                  }
                  else {  // if max_coverage > 331.5
                    if ( mean_support <= 0.563279986382 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.563279986382
                      return true;
                    }
                  }
                }
              }
              else {  // if mean_support > 0.728345036507
                if ( max_coverage <= 175.5 ) {
                  if ( min_coverage <= 91.5 ) {
                    if ( max_coverage <= 149.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 149.5
                      return true;
                    }
                  }
                  else {  // if min_coverage > 91.5
                    if ( mean_support <= 0.74998497963 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.74998497963
                      return true;
                    }
                  }
                }
                else {  // if max_coverage > 175.5
                  if ( max_coverage <= 215.5 ) {
                    if ( median_support <= 0.753280043602 ) {
                      return true;
                    }
                    else {  // if median_support > 0.753280043602
                      return true;
                    }
                  }
                  else {  // if max_coverage > 215.5
                    if ( min_coverage <= 157.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 157.5
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if max_coverage > 509.0
              if ( mean_support <= 0.719455003738 ) {
                return false;
              }
              else {  // if mean_support > 0.719455003738
                if ( min_coverage <= 416.0 ) {
                  return true;
                }
                else {  // if min_coverage > 416.0
                  return false;
                }
              }
            }
          }
          else {  // if max_support > 0.765484988689
            if ( max_coverage <= 217.5 ) {
              if ( max_support <= 0.791215002537 ) {
                if ( mean_coverage <= 58.8388900757 ) {
                  if ( median_support <= 0.779829978943 ) {
                    if ( mean_coverage <= 57.9954528809 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 57.9954528809
                      return false;
                    }
                  }
                  else {  // if median_support > 0.779829978943
                    if ( median_coverage <= 55.5 ) {
                      return false;
                    }
                    else {  // if median_coverage > 55.5
                      return false;
                    }
                  }
                }
                else {  // if mean_coverage > 58.8388900757
                  if ( median_support <= 0.782774984837 ) {
                    if ( max_coverage <= 199.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 199.5
                      return true;
                    }
                  }
                  else {  // if median_support > 0.782774984837
                    if ( min_coverage <= 75.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 75.5
                      return true;
                    }
                  }
                }
              }
              else {  // if max_support > 0.791215002537
                if ( median_coverage <= 111.25 ) {
                  if ( min_coverage <= 81.5 ) {
                    if ( max_coverage <= 139.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 139.5
                      return true;
                    }
                  }
                  else {  // if min_coverage > 81.5
                    if ( mean_coverage <= 110.405883789 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 110.405883789
                      return true;
                    }
                  }
                }
                else {  // if median_coverage > 111.25
                  if ( min_coverage <= 160.5 ) {
                    if ( mean_coverage <= 175.147064209 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 175.147064209
                      return true;
                    }
                  }
                  else {  // if min_coverage > 160.5
                    if ( mean_coverage <= 180.146469116 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 180.146469116
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if max_coverage > 217.5
              if ( min_coverage <= 78.5 ) {
                return false;
              }
              else {  // if min_coverage > 78.5
                if ( mean_support <= 0.805374979973 ) {
                  if ( median_coverage <= 159.0 ) {
                    if ( min_coverage <= 105.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 105.5
                      return false;
                    }
                  }
                  else {  // if median_coverage > 159.0
                    if ( max_support <= 0.79540002346 ) {
                      return true;
                    }
                    else {  // if max_support > 0.79540002346
                      return true;
                    }
                  }
                }
                else {  // if mean_support > 0.805374979973
                  if ( max_coverage <= 268.5 ) {
                    if ( mean_support <= 0.805384993553 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.805384993553
                      return true;
                    }
                  }
                  else {  // if max_coverage > 268.5
                    if ( max_support <= 0.805410027504 ) {
                      return false;
                    }
                    else {  // if max_support > 0.805410027504
                      return true;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if median_support > 0.818364977837
        if ( min_coverage <= 46.5 ) {
          if ( min_coverage <= 2.5 ) {
            if ( max_support <= 0.818414986134 ) {
              if ( max_coverage <= 43.0 ) {
                return true;
              }
              else {  // if max_coverage > 43.0
                if ( max_coverage <= 47.5 ) {
                  return false;
                }
                else {  // if max_coverage > 47.5
                  if ( max_coverage <= 63.0 ) {
                    return true;
                  }
                  else {  // if max_coverage > 63.0
                    return false;
                  }
                }
              }
            }
            else {  // if max_support > 0.818414986134
              if ( max_coverage <= 69.5 ) {
                if ( mean_support <= 0.818684995174 ) {
                  if ( min_support <= 0.818674981594 ) {
                    if ( median_coverage <= 20.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 20.5
                      return true;
                    }
                  }
                  else {  // if min_support > 0.818674981594
                    if ( median_coverage <= 18.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 18.5
                      return false;
                    }
                  }
                }
                else {  // if mean_support > 0.818684995174
                  if ( max_coverage <= 45.5 ) {
                    if ( min_support <= 0.818755030632 ) {
                      return true;
                    }
                    else {  // if min_support > 0.818755030632
                      return true;
                    }
                  }
                  else {  // if max_coverage > 45.5
                    if ( median_support <= 0.869984984398 ) {
                      return true;
                    }
                    else {  // if median_support > 0.869984984398
                      return true;
                    }
                  }
                }
              }
              else {  // if max_coverage > 69.5
                if ( median_coverage <= 21.25 ) {
                  if ( mean_support <= 0.855629980564 ) {
                    if ( mean_coverage <= 23.5 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 23.5
                      return true;
                    }
                  }
                  else {  // if mean_support > 0.855629980564
                    if ( mean_support <= 0.856965005398 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.856965005398
                      return true;
                    }
                  }
                }
                else {  // if median_coverage > 21.25
                  if ( median_coverage <= 41.75 ) {
                    if ( max_support <= 0.834004998207 ) {
                      return true;
                    }
                    else {  // if max_support > 0.834004998207
                      return true;
                    }
                  }
                  else {  // if median_coverage > 41.75
                    if ( mean_coverage <= 39.0588226318 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 39.0588226318
                      return true;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_coverage > 2.5
            if ( min_coverage <= 36.5 ) {
              if ( max_coverage <= 69.5 ) {
                if ( median_coverage <= 48.25 ) {
                  if ( median_coverage <= 39.25 ) {
                    if ( mean_coverage <= 38.7752075195 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 38.7752075195
                      return true;
                    }
                  }
                  else {  // if median_coverage > 39.25
                    if ( max_coverage <= 62.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 62.5
                      return true;
                    }
                  }
                }
                else {  // if median_coverage > 48.25
                  if ( median_coverage <= 50.25 ) {
                    if ( min_support <= 0.821689963341 ) {
                      return false;
                    }
                    else {  // if min_support > 0.821689963341
                      return true;
                    }
                  }
                  else {  // if median_coverage > 50.25
                    if ( median_coverage <= 51.75 ) {
                      return true;
                    }
                    else {  // if median_coverage > 51.75
                      return true;
                    }
                  }
                }
              }
              else {  // if max_coverage > 69.5
                if ( max_coverage <= 114.5 ) {
                  if ( mean_coverage <= 73.2426452637 ) {
                    if ( mean_coverage <= 64.4411773682 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 64.4411773682
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 73.2426452637
                    if ( mean_coverage <= 73.272064209 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 73.272064209
                      return true;
                    }
                  }
                }
                else {  // if max_coverage > 114.5
                  if ( max_coverage <= 136.5 ) {
                    return true;
                  }
                  else {  // if max_coverage > 136.5
                    if ( median_support <= 0.827455043793 ) {
                      return false;
                    }
                    else {  // if median_support > 0.827455043793
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if min_coverage > 36.5
              if ( mean_coverage <= 66.7196044922 ) {
                if ( median_coverage <= 67.75 ) {
                  if ( max_coverage <= 67.5 ) {
                    if ( mean_coverage <= 53.8090896606 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 53.8090896606
                      return true;
                    }
                  }
                  else {  // if max_coverage > 67.5
                    if ( mean_support <= 0.861755013466 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.861755013466
                      return true;
                    }
                  }
                }
                else {  // if median_coverage > 67.75
                  if ( min_support <= 0.86928498745 ) {
                    if ( max_coverage <= 79.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 79.5
                      return true;
                    }
                  }
                  else {  // if min_support > 0.86928498745
                    return false;
                  }
                }
              }
              else {  // if mean_coverage > 66.7196044922
                if ( min_coverage <= 42.5 ) {
                  if ( max_support <= 0.867145001888 ) {
                    if ( median_coverage <= 74.25 ) {
                      return true;
                    }
                    else {  // if median_coverage > 74.25
                      return true;
                    }
                  }
                  else {  // if max_support > 0.867145001888
                    if ( mean_support <= 0.867200016975 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.867200016975
                      return true;
                    }
                  }
                }
                else {  // if min_coverage > 42.5
                  if ( max_support <= 0.831514954567 ) {
                    if ( min_support <= 0.820874989033 ) {
                      return true;
                    }
                    else {  // if min_support > 0.820874989033
                      return true;
                    }
                  }
                  else {  // if max_support > 0.831514954567
                    if ( min_support <= 0.831659972668 ) {
                      return false;
                    }
                    else {  // if min_support > 0.831659972668
                      return true;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if min_coverage > 46.5
          if ( median_support <= 0.845565021038 ) {
            if ( max_coverage <= 273.5 ) {
              if ( max_coverage <= 155.5 ) {
                if ( median_support <= 0.838455021381 ) {
                  if ( median_coverage <= 96.75 ) {
                    if ( mean_coverage <= 70.1213226318 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 70.1213226318
                      return true;
                    }
                  }
                  else {  // if median_coverage > 96.75
                    if ( min_coverage <= 136.0 ) {
                      return true;
                    }
                    else {  // if min_coverage > 136.0
                      return false;
                    }
                  }
                }
                else {  // if median_support > 0.838455021381
                  if ( max_coverage <= 75.5 ) {
                    if ( mean_coverage <= 65.6161651611 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 65.6161651611
                      return false;
                    }
                  }
                  else {  // if max_coverage > 75.5
                    if ( min_support <= 0.83965498209 ) {
                      return true;
                    }
                    else {  // if min_support > 0.83965498209
                      return true;
                    }
                  }
                }
              }
              else {  // if max_coverage > 155.5
                if ( mean_support <= 0.830354988575 ) {
                  if ( max_coverage <= 227.5 ) {
                    if ( min_coverage <= 170.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 170.5
                      return true;
                    }
                  }
                  else {  // if max_coverage > 227.5
                    if ( mean_support <= 0.818395018578 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.818395018578
                      return true;
                    }
                  }
                }
                else {  // if mean_support > 0.830354988575
                  if ( mean_coverage <= 202.323516846 ) {
                    if ( min_coverage <= 149.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 149.5
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 202.323516846
                    if ( mean_coverage <= 251.083328247 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 251.083328247
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if max_coverage > 273.5
              if ( min_coverage <= 523.0 ) {
                if ( median_support <= 0.837305009365 ) {
                  if ( max_coverage <= 278.5 ) {
                    if ( min_coverage <= 249.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 249.5
                      return false;
                    }
                  }
                  else {  // if max_coverage > 278.5
                    if ( min_support <= 0.819669961929 ) {
                      return true;
                    }
                    else {  // if min_support > 0.819669961929
                      return true;
                    }
                  }
                }
                else {  // if median_support > 0.837305009365
                  if ( median_coverage <= 260.75 ) {
                    if ( min_coverage <= 223.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 223.5
                      return true;
                    }
                  }
                  else {  // if median_coverage > 260.75
                    if ( mean_support <= 0.837975025177 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.837975025177
                      return true;
                    }
                  }
                }
              }
              else {  // if min_coverage > 523.0
                return false;
              }
            }
          }
          else {  // if median_support > 0.845565021038
            if ( mean_coverage <= 123.46862793 ) {
              if ( mean_coverage <= 112.320510864 ) {
                if ( max_support <= 0.855944991112 ) {
                  if ( max_support <= 0.845664978027 ) {
                    if ( mean_coverage <= 79.7003631592 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 79.7003631592
                      return true;
                    }
                  }
                  else {  // if max_support > 0.845664978027
                    if ( min_coverage <= 92.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 92.5
                      return true;
                    }
                  }
                }
                else {  // if max_support > 0.855944991112
                  if ( max_coverage <= 99.5 ) {
                    if ( max_coverage <= 74.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 74.5
                      return true;
                    }
                  }
                  else {  // if max_coverage > 99.5
                    if ( max_coverage <= 159.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 159.5
                      return false;
                    }
                  }
                }
              }
              else {  // if mean_coverage > 112.320510864
                if ( mean_support <= 0.851119995117 ) {
                  if ( min_coverage <= 71.5 ) {
                    if ( mean_support <= 0.849269986153 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.849269986153
                      return false;
                    }
                  }
                  else {  // if min_coverage > 71.5
                    if ( median_coverage <= 113.25 ) {
                      return true;
                    }
                    else {  // if median_coverage > 113.25
                      return true;
                    }
                  }
                }
                else {  // if mean_support > 0.851119995117
                  if ( min_coverage <= 70.5 ) {
                    if ( max_support <= 0.868939995766 ) {
                      return false;
                    }
                    else {  // if max_support > 0.868939995766
                      return true;
                    }
                  }
                  else {  // if min_coverage > 70.5
                    if ( min_coverage <= 107.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 107.5
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if mean_coverage > 123.46862793
              if ( max_coverage <= 277.5 ) {
                if ( max_coverage <= 179.5 ) {
                  if ( min_coverage <= 144.5 ) {
                    if ( median_support <= 0.863255023956 ) {
                      return true;
                    }
                    else {  // if median_support > 0.863255023956
                      return true;
                    }
                  }
                  else {  // if min_coverage > 144.5
                    if ( max_coverage <= 174.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 174.5
                      return true;
                    }
                  }
                }
                else {  // if max_coverage > 179.5
                  if ( max_support <= 0.863494992256 ) {
                    if ( max_coverage <= 251.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 251.5
                      return true;
                    }
                  }
                  else {  // if max_support > 0.863494992256
                    if ( mean_coverage <= 163.147064209 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 163.147064209
                      return true;
                    }
                  }
                }
              }
              else {  // if max_coverage > 277.5
                if ( max_coverage <= 309.5 ) {
                  if ( max_support <= 0.84932500124 ) {
                    if ( min_coverage <= 170.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 170.5
                      return true;
                    }
                  }
                  else {  // if max_support > 0.84932500124
                    if ( max_coverage <= 286.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 286.5
                      return true;
                    }
                  }
                }
                else {  // if max_coverage > 309.5
                  if ( max_coverage <= 365.5 ) {
                    if ( mean_support <= 0.863904953003 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.863904953003
                      return true;
                    }
                  }
                  else {  // if max_coverage > 365.5
                    if ( max_coverage <= 410.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 410.5
                      return true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    else {  // if median_support > 0.870015025139
      if ( min_coverage <= 46.5 ) {
        if ( min_coverage <= 44.5 ) {
          if ( min_coverage <= 1.5 ) {
            if ( max_coverage <= 96.5 ) {
              if ( max_coverage <= 59.5 ) {
                if ( mean_coverage <= 31.5588226318 ) {
                  if ( median_coverage <= 6.5 ) {
                    if ( mean_coverage <= 13.2941169739 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 13.2941169739
                      return true;
                    }
                  }
                  else {  // if median_coverage > 6.5
                    if ( median_support <= 0.929980039597 ) {
                      return true;
                    }
                    else {  // if median_support > 0.929980039597
                      return true;
                    }
                  }
                }
                else {  // if mean_coverage > 31.5588226318
                  if ( mean_support <= 0.890534996986 ) {
                    return false;
                  }
                  else {  // if mean_support > 0.890534996986
                    return true;
                  }
                }
              }
              else {  // if max_coverage > 59.5
                if ( median_coverage <= 7.5 ) {
                  if ( max_coverage <= 70.5 ) {
                    if ( median_support <= 0.871729969978 ) {
                      return false;
                    }
                    else {  // if median_support > 0.871729969978
                      return true;
                    }
                  }
                  else {  // if max_coverage > 70.5
                    if ( mean_coverage <= 21.3823547363 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 21.3823547363
                      return false;
                    }
                  }
                }
                else {  // if median_coverage > 7.5
                  if ( max_coverage <= 74.5 ) {
                    if ( median_coverage <= 11.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 11.5
                      return true;
                    }
                  }
                  else {  // if max_coverage > 74.5
                    if ( max_support <= 0.922140002251 ) {
                      return true;
                    }
                    else {  // if max_support > 0.922140002251
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if max_coverage > 96.5
              if ( min_support <= 0.932209968567 ) {
                if ( median_coverage <= 27.5 ) {
                  if ( mean_support <= 0.917744994164 ) {
                    if ( max_support <= 0.881559967995 ) {
                      return false;
                    }
                    else {  // if max_support > 0.881559967995
                      return true;
                    }
                  }
                  else {  // if mean_support > 0.917744994164
                    return false;
                  }
                }
                else {  // if median_coverage > 27.5
                  if ( median_support <= 0.873860001564 ) {
                    return false;
                  }
                  else {  // if median_support > 0.873860001564
                    if ( median_support <= 0.919295012951 ) {
                      return true;
                    }
                    else {  // if median_support > 0.919295012951
                      return true;
                    }
                  }
                }
              }
              else {  // if min_support > 0.932209968567
                if ( median_support <= 0.93291002512 ) {
                  return false;
                }
                else {  // if median_support > 0.93291002512
                  return true;
                }
              }
            }
          }
          else {  // if min_coverage > 1.5
            if ( median_coverage <= 19.75 ) {
              if ( max_support <= 0.878684997559 ) {
                if ( median_support <= 0.878650009632 ) {
                  if ( median_support <= 0.876870036125 ) {
                    return true;
                  }
                  else {  // if median_support > 0.876870036125
                    if ( median_support <= 0.876894950867 ) {
                      return false;
                    }
                    else {  // if median_support > 0.876894950867
                      return true;
                    }
                  }
                }
                else {  // if median_support > 0.878650009632
                  return false;
                }
              }
              else {  // if max_support > 0.878684997559
                if ( mean_support <= 0.920244991779 ) {
                  if ( mean_coverage <= 25.454044342 ) {
                    if ( mean_coverage <= 20.9705886841 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 20.9705886841
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 25.454044342
                    if ( mean_coverage <= 25.5 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 25.5
                      return true;
                    }
                  }
                }
                else {  // if mean_support > 0.920244991779
                  if ( min_support <= 0.920269966125 ) {
                    return false;
                  }
                  else {  // if min_support > 0.920269966125
                    if ( max_coverage <= 34.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 34.5
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if median_coverage > 19.75
              if ( mean_coverage <= 17.2339744568 ) {
                return false;
              }
              else {  // if mean_coverage > 17.2339744568
                if ( max_coverage <= 93.5 ) {
                  if ( min_coverage <= 35.5 ) {
                    if ( median_coverage <= 69.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 69.5
                      return true;
                    }
                  }
                  else {  // if min_coverage > 35.5
                    if ( min_support <= 0.870034992695 ) {
                      return false;
                    }
                    else {  // if min_support > 0.870034992695
                      return true;
                    }
                  }
                }
                else {  // if max_coverage > 93.5
                  if ( mean_coverage <= 44.8529434204 ) {
                    if ( mean_coverage <= 44.7784309387 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 44.7784309387
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 44.8529434204
                    if ( median_coverage <= 85.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 85.5
                      return true;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if min_coverage > 44.5
          if ( max_coverage <= 75.5 ) {
            if ( mean_coverage <= 61.055557251 ) {
              if ( median_support <= 0.889034986496 ) {
                if ( mean_coverage <= 53.7222213745 ) {
                  if ( max_support <= 0.872400045395 ) {
                    return false;
                  }
                  else {  // if max_support > 0.872400045395
                    if ( mean_coverage <= 51.944442749 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 51.944442749
                      return true;
                    }
                  }
                }
                else {  // if mean_coverage > 53.7222213745
                  if ( mean_coverage <= 54.7944450378 ) {
                    if ( mean_coverage <= 54.6833343506 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 54.6833343506
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 54.7944450378
                    if ( min_support <= 0.872964978218 ) {
                      return true;
                    }
                    else {  // if min_support > 0.872964978218
                      return true;
                    }
                  }
                }
              }
              else {  // if median_support > 0.889034986496
                if ( min_support <= 0.889710009098 ) {
                  return false;
                }
                else {  // if min_support > 0.889710009098
                  if ( min_support <= 0.922384977341 ) {
                    if ( mean_support <= 0.922024965286 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.922024965286
                      return false;
                    }
                  }
                  else {  // if min_support > 0.922384977341
                    if ( min_support <= 0.926975011826 ) {
                      return true;
                    }
                    else {  // if min_support > 0.926975011826
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if mean_coverage > 61.055557251
              if ( min_support <= 0.925289988518 ) {
                if ( median_coverage <= 64.5 ) {
                  if ( median_coverage <= 63.5 ) {
                    if ( mean_support <= 0.913390040398 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.913390040398
                      return true;
                    }
                  }
                  else {  // if median_coverage > 63.5
                    if ( mean_coverage <= 62.6746063232 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 62.6746063232
                      return false;
                    }
                  }
                }
                else {  // if median_coverage > 64.5
                  return false;
                }
              }
              else {  // if min_support > 0.925289988518
                return true;
              }
            }
          }
          else {  // if max_coverage > 75.5
            if ( median_coverage <= 109.0 ) {
              if ( mean_coverage <= 70.5669631958 ) {
                if ( mean_coverage <= 70.53125 ) {
                  if ( median_coverage <= 68.25 ) {
                    if ( max_coverage <= 83.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 83.5
                      return true;
                    }
                  }
                  else {  // if median_coverage > 68.25
                    if ( mean_coverage <= 66.4494934082 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 66.4494934082
                      return true;
                    }
                  }
                }
                else {  // if mean_coverage > 70.53125
                  return false;
                }
              }
              else {  // if mean_coverage > 70.5669631958
                if ( median_support <= 0.93008005619 ) {
                  if ( mean_coverage <= 84.0882339478 ) {
                    if ( mean_support <= 0.878464996815 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.878464996815
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 84.0882339478
                    if ( mean_coverage <= 84.2647094727 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 84.2647094727
                      return true;
                    }
                  }
                }
                else {  // if median_support > 0.93008005619
                  if ( max_support <= 0.930104970932 ) {
                    return false;
                  }
                  else {  // if max_support > 0.930104970932
                    if ( median_coverage <= 70.25 ) {
                      return true;
                    }
                    else {  // if median_coverage > 70.25
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if median_coverage > 109.0
              return false;
            }
          }
        }
      }
      else {  // if min_coverage > 46.5
        if ( mean_support <= 0.906395018101 ) {
          if ( mean_coverage <= 146.179138184 ) {
            if ( min_support <= 0.89632499218 ) {
              if ( median_coverage <= 117.25 ) {
                if ( mean_support <= 0.882035017014 ) {
                  if ( mean_coverage <= 61.0357131958 ) {
                    if ( median_coverage <= 59.75 ) {
                      return true;
                    }
                    else {  // if median_coverage > 59.75
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 61.0357131958
                    if ( min_coverage <= 83.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 83.5
                      return true;
                    }
                  }
                }
                else {  // if mean_support > 0.882035017014
                  if ( min_coverage <= 100.5 ) {
                    if ( max_coverage <= 64.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 64.5
                      return true;
                    }
                  }
                  else {  // if min_coverage > 100.5
                    if ( min_coverage <= 107.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 107.5
                      return true;
                    }
                  }
                }
              }
              else {  // if median_coverage > 117.25
                if ( mean_support <= 0.878885030746 ) {
                  if ( max_coverage <= 153.5 ) {
                    if ( min_coverage <= 131.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 131.5
                      return false;
                    }
                  }
                  else {  // if max_coverage > 153.5
                    if ( median_support <= 0.874675035477 ) {
                      return true;
                    }
                    else {  // if median_support > 0.874675035477
                      return true;
                    }
                  }
                }
                else {  // if mean_support > 0.878885030746
                  if ( max_coverage <= 167.5 ) {
                    if ( min_coverage <= 115.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 115.5
                      return true;
                    }
                  }
                  else {  // if max_coverage > 167.5
                    if ( mean_coverage <= 146.067871094 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 146.067871094
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if min_support > 0.89632499218
              if ( mean_coverage <= 112.618179321 ) {
                if ( mean_support <= 0.902604997158 ) {
                  if ( mean_coverage <= 112.466064453 ) {
                    if ( median_coverage <= 74.75 ) {
                      return true;
                    }
                    else {  // if median_coverage > 74.75
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 112.466064453
                    if ( mean_coverage <= 112.54196167 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 112.54196167
                      return true;
                    }
                  }
                }
                else {  // if mean_support > 0.902604997158
                  if ( max_coverage <= 147.5 ) {
                    if ( min_coverage <= 95.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 95.5
                      return false;
                    }
                  }
                  else {  // if max_coverage > 147.5
                    if ( mean_support <= 0.903584957123 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.903584957123
                      return false;
                    }
                  }
                }
              }
              else {  // if mean_coverage > 112.618179321
                if ( mean_support <= 0.902864992619 ) {
                  if ( mean_coverage <= 112.771240234 ) {
                    if ( max_support <= 0.900059998035 ) {
                      return false;
                    }
                    else {  // if max_support > 0.900059998035
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 112.771240234
                    if ( mean_coverage <= 145.937255859 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 145.937255859
                      return false;
                    }
                  }
                }
                else {  // if mean_support > 0.902864992619
                  if ( mean_coverage <= 128.125488281 ) {
                    if ( max_coverage <= 137.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 137.5
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 128.125488281
                    if ( mean_coverage <= 132.970581055 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 132.970581055
                      return true;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_coverage > 146.179138184
            if ( mean_coverage <= 299.411743164 ) {
              if ( max_support <= 0.882664978504 ) {
                if ( max_coverage <= 282.5 ) {
                  if ( max_coverage <= 188.5 ) {
                    if ( min_coverage <= 112.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 112.5
                      return true;
                    }
                  }
                  else {  // if max_coverage > 188.5
                    if ( mean_coverage <= 187.727935791 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 187.727935791
                      return false;
                    }
                  }
                }
                else {  // if max_coverage > 282.5
                  if ( median_coverage <= 275.75 ) {
                    if ( mean_coverage <= 252.656860352 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 252.656860352
                      return true;
                    }
                  }
                  else {  // if median_coverage > 275.75
                    if ( min_coverage <= 286.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 286.5
                      return false;
                    }
                  }
                }
              }
              else {  // if max_support > 0.882664978504
                if ( mean_coverage <= 185.894439697 ) {
                  if ( mean_coverage <= 158.45803833 ) {
                    if ( median_support <= 0.899614989758 ) {
                      return true;
                    }
                    else {  // if median_support > 0.899614989758
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 158.45803833
                    if ( min_coverage <= 124.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 124.5
                      return false;
                    }
                  }
                }
                else {  // if mean_coverage > 185.894439697
                  if ( mean_support <= 0.899134993553 ) {
                    if ( max_coverage <= 267.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 267.5
                      return false;
                    }
                  }
                  else {  // if mean_support > 0.899134993553
                    if ( min_coverage <= 166.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 166.5
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if mean_coverage > 299.411743164
              if ( mean_support <= 0.890455007553 ) {
                if ( mean_coverage <= 351.617645264 ) {
                  if ( min_coverage <= 260.5 ) {
                    if ( median_coverage <= 343.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 343.5
                      return false;
                    }
                  }
                  else {  // if min_coverage > 260.5
                    if ( mean_support <= 0.870399951935 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.870399951935
                      return true;
                    }
                  }
                }
                else {  // if mean_coverage > 351.617645264
                  if ( min_support <= 0.870180010796 ) {
                    if ( min_support <= 0.870044946671 ) {
                      return true;
                    }
                    else {  // if min_support > 0.870044946671
                      return false;
                    }
                  }
                  else {  // if min_support > 0.870180010796
                    if ( mean_support <= 0.876410007477 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.876410007477
                      return true;
                    }
                  }
                }
              }
              else {  // if mean_support > 0.890455007553
                if ( median_coverage <= 345.75 ) {
                  if ( mean_coverage <= 339.79119873 ) {
                    if ( max_coverage <= 348.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 348.5
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 339.79119873
                    if ( mean_support <= 0.896394968033 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.896394968033
                      return false;
                    }
                  }
                }
                else {  // if median_coverage > 345.75
                  if ( mean_support <= 0.901719987392 ) {
                    if ( min_support <= 0.890740036964 ) {
                      return false;
                    }
                    else {  // if min_support > 0.890740036964
                      return true;
                    }
                  }
                  else {  // if mean_support > 0.901719987392
                    if ( median_support <= 0.902425050735 ) {
                      return false;
                    }
                    else {  // if median_support > 0.902425050735
                      return true;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_support > 0.906395018101
          if ( mean_coverage <= 168.922622681 ) {
            if ( max_support <= 0.922264993191 ) {
              if ( min_coverage <= 88.5 ) {
                if ( max_support <= 0.912265002728 ) {
                  if ( min_coverage <= 68.5 ) {
                    if ( min_support <= 0.90649497509 ) {
                      return true;
                    }
                    else {  // if min_support > 0.90649497509
                      return true;
                    }
                  }
                  else {  // if min_coverage > 68.5
                    if ( mean_coverage <= 90.3030319214 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 90.3030319214
                      return true;
                    }
                  }
                }
                else {  // if max_support > 0.912265002728
                  if ( min_coverage <= 60.5 ) {
                    if ( max_coverage <= 98.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 98.5
                      return true;
                    }
                  }
                  else {  // if min_coverage > 60.5
                    if ( median_support <= 0.915825009346 ) {
                      return true;
                    }
                    else {  // if median_support > 0.915825009346
                      return false;
                    }
                  }
                }
              }
              else {  // if min_coverage > 88.5
                if ( min_support <= 0.915894985199 ) {
                  if ( median_coverage <= 133.75 ) {
                    if ( max_coverage <= 110.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 110.5
                      return false;
                    }
                  }
                  else {  // if median_coverage > 133.75
                    if ( mean_coverage <= 129.558837891 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 129.558837891
                      return false;
                    }
                  }
                }
                else {  // if min_support > 0.915894985199
                  if ( min_coverage <= 117.5 ) {
                    if ( mean_coverage <= 104.132476807 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 104.132476807
                      return false;
                    }
                  }
                  else {  // if min_coverage > 117.5
                    if ( mean_support <= 0.915955007076 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.915955007076
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if max_support > 0.922264993191
              if ( min_coverage <= 109.5 ) {
                if ( min_coverage <= 83.5 ) {
                  if ( mean_support <= 0.925565004349 ) {
                    if ( mean_coverage <= 61.1055526733 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 61.1055526733
                      return false;
                    }
                  }
                  else {  // if mean_support > 0.925565004349
                    if ( min_coverage <= 66.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 66.5
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 83.5
                  if ( max_support <= 0.929475009441 ) {
                    if ( mean_coverage <= 103.654762268 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 103.654762268
                      return false;
                    }
                  }
                  else {  // if max_support > 0.929475009441
                    if ( median_support <= 0.93303501606 ) {
                      return false;
                    }
                    else {  // if median_support > 0.93303501606
                      return false;
                    }
                  }
                }
              }
              else {  // if min_coverage > 109.5
                if ( mean_support <= 0.925554990768 ) {
                  if ( median_coverage <= 157.25 ) {
                    if ( mean_coverage <= 160.147064209 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 160.147064209
                      return false;
                    }
                  }
                  else {  // if median_coverage > 157.25
                    if ( max_support <= 0.92461502552 ) {
                      return false;
                    }
                    else {  // if max_support > 0.92461502552
                      return false;
                    }
                  }
                }
                else {  // if mean_support > 0.925554990768
                  if ( mean_coverage <= 148.594116211 ) {
                    if ( mean_coverage <= 117.920455933 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 117.920455933
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 148.594116211
                    if ( max_coverage <= 219.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 219.5
                      return true;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_coverage > 168.922622681
            if ( min_coverage <= 173.5 ) {
              if ( min_support <= 0.922214984894 ) {
                if ( min_coverage <= 145.5 ) {
                  if ( mean_coverage <= 188.558822632 ) {
                    if ( median_support <= 0.916960000992 ) {
                      return false;
                    }
                    else {  // if median_support > 0.916960000992
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 188.558822632
                    if ( max_coverage <= 234.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 234.5
                      return true;
                    }
                  }
                }
                else {  // if min_coverage > 145.5
                  if ( max_support <= 0.909824967384 ) {
                    if ( max_coverage <= 239.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 239.5
                      return false;
                    }
                  }
                  else {  // if max_support > 0.909824967384
                    if ( mean_support <= 0.910364985466 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.910364985466
                      return false;
                    }
                  }
                }
              }
              else {  // if min_support > 0.922214984894
                if ( mean_coverage <= 192.735290527 ) {
                  if ( mean_support <= 0.931535005569 ) {
                    if ( min_coverage <= 119.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 119.5
                      return false;
                    }
                  }
                  else {  // if mean_support > 0.931535005569
                    if ( median_support <= 0.932474970818 ) {
                      return false;
                    }
                    else {  // if median_support > 0.932474970818
                      return false;
                    }
                  }
                }
                else {  // if mean_coverage > 192.735290527
                  if ( median_support <= 0.922975003719 ) {
                    if ( mean_coverage <= 196.852935791 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 196.852935791
                      return false;
                    }
                  }
                  else {  // if median_support > 0.922975003719
                    if ( mean_coverage <= 198.264709473 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 198.264709473
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if min_coverage > 173.5
              if ( max_support <= 0.918954968452 ) {
                if ( mean_support <= 0.910004973412 ) {
                  if ( max_coverage <= 350.5 ) {
                    if ( mean_coverage <= 199.126693726 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 199.126693726
                      return false;
                    }
                  }
                  else {  // if max_coverage > 350.5
                    if ( median_coverage <= 364.5 ) {
                      return false;
                    }
                    else {  // if median_coverage > 364.5
                      return true;
                    }
                  }
                }
                else {  // if mean_support > 0.910004973412
                  if ( min_coverage <= 190.5 ) {
                    if ( median_coverage <= 183.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 183.5
                      return false;
                    }
                  }
                  else {  // if min_coverage > 190.5
                    if ( max_support <= 0.91891503334 ) {
                      return false;
                    }
                    else {  // if max_support > 0.91891503334
                      return true;
                    }
                  }
                }
              }
              else {  // if max_support > 0.918954968452
                if ( min_coverage <= 207.5 ) {
                  if ( max_support <= 0.929165005684 ) {
                    if ( mean_coverage <= 213.676483154 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 213.676483154
                      return false;
                    }
                  }
                  else {  // if max_support > 0.929165005684
                    if ( max_coverage <= 298.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 298.5
                      return true;
                    }
                  }
                }
                else {  // if min_coverage > 207.5
                  if ( min_support <= 0.92782497406 ) {
                    if ( mean_support <= 0.925355017185 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.925355017185
                      return false;
                    }
                  }
                  else {  // if min_support > 0.92782497406
                    if ( max_coverage <= 295.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 295.5
                      return false;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if min_support > 0.933934986591
    if ( min_coverage <= 46.5 ) {
      if ( min_coverage <= 43.5 ) {
        if ( min_coverage <= 1.5 ) {
          if ( max_coverage <= 91.5 ) {
            if ( max_coverage <= 60.5 ) {
              if ( max_coverage <= 40.5 ) {
                if ( mean_coverage <= 21.5980377197 ) {
                  if ( max_support <= 0.953610002995 ) {
                    if ( mean_coverage <= 20.0294113159 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 20.0294113159
                      return true;
                    }
                  }
                  else {  // if max_support > 0.953610002995
                    if ( max_support <= 0.953740000725 ) {
                      return false;
                    }
                    else {  // if max_support > 0.953740000725
                      return true;
                    }
                  }
                }
                else {  // if mean_coverage > 21.5980377197
                  if ( min_support <= 0.95039498806 ) {
                    return false;
                  }
                  else {  // if min_support > 0.95039498806
                    return true;
                  }
                }
              }
              else {  // if max_coverage > 40.5
                if ( mean_support <= 0.945814967155 ) {
                  if ( median_support <= 0.94580501318 ) {
                    if ( mean_coverage <= 20.757352829 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 20.757352829
                      return true;
                    }
                  }
                  else {  // if median_support > 0.94580501318
                    if ( median_coverage <= 21.0 ) {
                      return true;
                    }
                    else {  // if median_coverage > 21.0
                      return false;
                    }
                  }
                }
                else {  // if mean_support > 0.945814967155
                  if ( median_coverage <= 36.5 ) {
                    if ( mean_coverage <= 25.0678730011 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 25.0678730011
                      return true;
                    }
                  }
                  else {  // if median_coverage > 36.5
                    if ( min_support <= 0.957149982452 ) {
                      return false;
                    }
                    else {  // if min_support > 0.957149982452
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if max_coverage > 60.5
              if ( min_support <= 0.951415002346 ) {
                if ( max_support <= 0.951404988766 ) {
                  if ( max_coverage <= 75.5 ) {
                    if ( mean_support <= 0.950659990311 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.950659990311
                      return true;
                    }
                  }
                  else {  // if max_coverage > 75.5
                    if ( min_support <= 0.936810016632 ) {
                      return true;
                    }
                    else {  // if min_support > 0.936810016632
                      return true;
                    }
                  }
                }
                else {  // if max_support > 0.951404988766
                  if ( mean_coverage <= 33.4705886841 ) {
                    return false;
                  }
                  else {  // if mean_coverage > 33.4705886841
                    return true;
                  }
                }
              }
              else {  // if min_support > 0.951415002346
                if ( mean_coverage <= 25.0294113159 ) {
                  if ( min_support <= 0.958109974861 ) {
                    if ( median_coverage <= 17.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 17.5
                      return true;
                    }
                  }
                  else {  // if min_support > 0.958109974861
                    if ( max_coverage <= 65.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 65.5
                      return true;
                    }
                  }
                }
                else {  // if mean_coverage > 25.0294113159
                  if ( max_coverage <= 74.5 ) {
                    if ( mean_support <= 0.965539991856 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.965539991856
                      return true;
                    }
                  }
                  else {  // if max_coverage > 74.5
                    if ( mean_coverage <= 32.4392166138 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 32.4392166138
                      return true;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_coverage > 91.5
            if ( median_support <= 0.935000002384 ) {
              if ( mean_coverage <= 47.9411773682 ) {
                return false;
              }
              else {  // if mean_coverage > 47.9411773682
                return true;
              }
            }
            else {  // if median_support > 0.935000002384
              if ( min_support <= 0.954400002956 ) {
                if ( mean_support <= 0.953620016575 ) {
                  if ( mean_coverage <= 47.3823547363 ) {
                    if ( mean_coverage <= 47.3235282898 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 47.3235282898
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 47.3823547363
                    if ( max_coverage <= 121.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 121.5
                      return true;
                    }
                  }
                }
                else {  // if mean_support > 0.953620016575
                  if ( max_support <= 0.954159975052 ) {
                    return false;
                  }
                  else {  // if max_support > 0.954159975052
                    if ( min_support <= 0.954290032387 ) {
                      return true;
                    }
                    else {  // if min_support > 0.954290032387
                      return false;
                    }
                  }
                }
              }
              else {  // if min_support > 0.954400002956
                if ( median_coverage <= 66.5 ) {
                  if ( max_coverage <= 102.5 ) {
                    if ( max_support <= 0.987689971924 ) {
                      return true;
                    }
                    else {  // if max_support > 0.987689971924
                      return true;
                    }
                  }
                  else {  // if max_coverage > 102.5
                    if ( mean_coverage <= 38.0588226318 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 38.0588226318
                      return true;
                    }
                  }
                }
                else {  // if median_coverage > 66.5
                  if ( min_support <= 0.970390021801 ) {
                    return false;
                  }
                  else {  // if min_support > 0.970390021801
                    if ( mean_coverage <= 63.0294113159 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 63.0294113159
                      return true;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if min_coverage > 1.5
          if ( min_support <= 0.978585004807 ) {
            if ( min_coverage <= 41.5 ) {
              if ( max_coverage <= 118.5 ) {
                if ( median_coverage <= 42.25 ) {
                  if ( max_coverage <= 94.5 ) {
                    if ( min_coverage <= 23.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 23.5
                      return true;
                    }
                  }
                  else {  // if max_coverage > 94.5
                    if ( mean_coverage <= 41.7058830261 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 41.7058830261
                      return true;
                    }
                  }
                }
                else {  // if median_coverage > 42.25
                  if ( min_coverage <= 7.5 ) {
                    if ( max_coverage <= 110.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 110.5
                      return true;
                    }
                  }
                  else {  // if min_coverage > 7.5
                    if ( max_coverage <= 69.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 69.5
                      return true;
                    }
                  }
                }
              }
              else {  // if max_coverage > 118.5
                if ( min_coverage <= 20.5 ) {
                  if ( min_coverage <= 6.5 ) {
                    return true;
                  }
                  else {  // if min_coverage > 6.5
                    if ( median_support <= 0.978314995766 ) {
                      return true;
                    }
                    else {  // if median_support > 0.978314995766
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 20.5
                  if ( min_coverage <= 34.5 ) {
                    if ( max_coverage <= 136.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 136.5
                      return true;
                    }
                  }
                  else {  // if min_coverage > 34.5
                    if ( max_support <= 0.952494978905 ) {
                      return true;
                    }
                    else {  // if max_support > 0.952494978905
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if min_coverage > 41.5
              if ( max_coverage <= 85.5 ) {
                if ( median_coverage <= 67.75 ) {
                  if ( max_coverage <= 72.5 ) {
                    if ( mean_coverage <= 57.819442749 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 57.819442749
                      return true;
                    }
                  }
                  else {  // if max_coverage > 72.5
                    if ( median_support <= 0.93501996994 ) {
                      return true;
                    }
                    else {  // if median_support > 0.93501996994
                      return true;
                    }
                  }
                }
                else {  // if median_coverage > 67.75
                  if ( max_support <= 0.96727502346 ) {
                    if ( median_support <= 0.960340023041 ) {
                      return true;
                    }
                    else {  // if median_support > 0.960340023041
                      return false;
                    }
                  }
                  else {  // if max_support > 0.96727502346
                    return true;
                  }
                }
              }
              else {  // if max_coverage > 85.5
                if ( max_coverage <= 156.5 ) {
                  if ( mean_support <= 0.934464991093 ) {
                    if ( median_coverage <= 59.5 ) {
                      return false;
                    }
                    else {  // if median_coverage > 59.5
                      return true;
                    }
                  }
                  else {  // if mean_support > 0.934464991093
                    if ( min_support <= 0.940034985542 ) {
                      return true;
                    }
                    else {  // if min_support > 0.940034985542
                      return true;
                    }
                  }
                }
                else {  // if max_coverage > 156.5
                  if ( mean_support <= 0.948819994926 ) {
                    return true;
                  }
                  else {  // if mean_support > 0.948819994926
                    return false;
                  }
                }
              }
            }
          }
          else {  // if min_support > 0.978585004807
            if ( median_coverage <= 111.5 ) {
              if ( min_support <= 0.986384987831 ) {
                if ( median_support <= 0.986374974251 ) {
                  if ( max_coverage <= 68.5 ) {
                    if ( mean_coverage <= 56.8888893127 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 56.8888893127
                      return false;
                    }
                  }
                  else {  // if max_coverage > 68.5
                    if ( min_coverage <= 11.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 11.5
                      return true;
                    }
                  }
                }
                else {  // if median_support > 0.986374974251
                  if ( max_coverage <= 140.5 ) {
                    if ( min_coverage <= 16.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 16.5
                      return true;
                    }
                  }
                  else {  // if max_coverage > 140.5
                    return false;
                  }
                }
              }
              else {  // if min_support > 0.986384987831
                if ( mean_coverage <= 45.8529434204 ) {
                  if ( mean_support <= 0.987134993076 ) {
                    return true;
                  }
                  else {  // if mean_support > 0.987134993076
                    return false;
                  }
                }
                else {  // if mean_coverage > 45.8529434204
                  if ( max_coverage <= 138.5 ) {
                    if ( min_coverage <= 7.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 7.5
                      return true;
                    }
                  }
                  else {  // if max_coverage > 138.5
                    if ( min_support <= 0.987800002098 ) {
                      return true;
                    }
                    else {  // if min_support > 0.987800002098
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if median_coverage > 111.5
              if ( max_coverage <= 178.0 ) {
                return false;
              }
              else {  // if max_coverage > 178.0
                return true;
              }
            }
          }
        }
      }
      else {  // if min_coverage > 43.5
        if ( max_coverage <= 79.5 ) {
          if ( mean_support <= 0.982669949532 ) {
            if ( min_support <= 0.978374958038 ) {
              if ( median_coverage <= 45.25 ) {
                return false;
              }
              else {  // if median_coverage > 45.25
                if ( median_coverage <= 51.75 ) {
                  return true;
                }
                else {  // if median_coverage > 51.75
                  if ( max_coverage <= 67.5 ) {
                    if ( min_coverage <= 44.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 44.5
                      return true;
                    }
                  }
                  else {  // if max_coverage > 67.5
                    if ( mean_coverage <= 61.6410255432 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 61.6410255432
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if min_support > 0.978374958038
              if ( median_coverage <= 65.25 ) {
                if ( max_coverage <= 76.5 ) {
                  if ( mean_coverage <= 61.0261459351 ) {
                    if ( mean_coverage <= 54.4277801514 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 54.4277801514
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 61.0261459351
                    if ( mean_coverage <= 61.7333335876 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 61.7333335876
                      return true;
                    }
                  }
                }
                else {  // if max_coverage > 76.5
                  if ( max_support <= 0.980504989624 ) {
                    if ( mean_support <= 0.980324983597 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.980324983597
                      return false;
                    }
                  }
                  else {  // if max_support > 0.980504989624
                    return true;
                  }
                }
              }
              else {  // if median_coverage > 65.25
                return false;
              }
            }
          }
          else {  // if mean_support > 0.982669949532
            return true;
          }
        }
        else {  // if max_coverage > 79.5
          if ( max_coverage <= 88.5 ) {
            if ( min_coverage <= 45.5 ) {
              if ( max_coverage <= 80.5 ) {
                if ( mean_coverage <= 62.1143798828 ) {
                  if ( mean_coverage <= 62.1010093689 ) {
                    if ( mean_coverage <= 60.055557251 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 60.055557251
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 62.1010093689
                    return false;
                  }
                }
                else {  // if mean_coverage > 62.1143798828
                  return true;
                }
              }
              else {  // if max_coverage > 80.5
                if ( median_coverage <= 58.75 ) {
                  return true;
                }
                else {  // if median_coverage > 58.75
                  if ( mean_coverage <= 61.2287597656 ) {
                    if ( mean_coverage <= 61.2111129761 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 61.2111129761
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 61.2287597656
                    if ( mean_coverage <= 63.7638893127 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 63.7638893127
                      return true;
                    }
                  }
                }
              }
            }
            else {  // if min_coverage > 45.5
              if ( mean_coverage <= 58.0063018799 ) {
                return false;
              }
              else {  // if mean_coverage > 58.0063018799
                if ( median_coverage <= 60.75 ) {
                  if ( max_coverage <= 80.5 ) {
                    if ( mean_coverage <= 62.1825408936 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 62.1825408936
                      return false;
                    }
                  }
                  else {  // if max_coverage > 80.5
                    return true;
                  }
                }
                else {  // if median_coverage > 60.75
                  if ( mean_coverage <= 68.7207794189 ) {
                    if ( mean_coverage <= 63.9027786255 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 63.9027786255
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 68.7207794189
                    if ( mean_coverage <= 68.944442749 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 68.944442749
                      return true;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_coverage > 88.5
            if ( mean_support <= 0.970094978809 ) {
              if ( max_support <= 0.970025002956 ) {
                if ( mean_coverage <= 78.6176452637 ) {
                  if ( mean_coverage <= 78.5753631592 ) {
                    if ( median_support <= 0.93588000536 ) {
                      return true;
                    }
                    else {  // if median_support > 0.93588000536
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 78.5753631592
                    if ( max_coverage <= 117.0 ) {
                      return true;
                    }
                    else {  // if max_coverage > 117.0
                      return false;
                    }
                  }
                }
                else {  // if mean_coverage > 78.6176452637
                  if ( max_support <= 0.968824982643 ) {
                    return true;
                  }
                  else {  // if max_support > 0.968824982643
                    if ( mean_support <= 0.968860030174 ) {
                      return true;
                    }
                    else {  // if mean_support > 0.968860030174
                      return true;
                    }
                  }
                }
              }
              else {  // if max_support > 0.970025002956
                if ( mean_coverage <= 66.6555557251 ) {
                  return false;
                }
                else {  // if mean_coverage > 66.6555557251
                  if ( mean_support <= 0.970035016537 ) {
                    return false;
                  }
                  else {  // if mean_support > 0.970035016537
                    return true;
                  }
                }
              }
            }
            else {  // if mean_support > 0.970094978809
              if ( max_coverage <= 91.5 ) {
                if ( min_support <= 0.978459954262 ) {
                  return true;
                }
                else {  // if min_support > 0.978459954262
                  if ( mean_support <= 0.978544950485 ) {
                    if ( min_coverage <= 45.0 ) {
                      return false;
                    }
                    else {  // if min_coverage > 45.0
                      return true;
                    }
                  }
                  else {  // if mean_support > 0.978544950485
                    if ( median_coverage <= 72.25 ) {
                      return true;
                    }
                    else {  // if median_coverage > 72.25
                      return true;
                    }
                  }
                }
              }
              else {  // if max_coverage > 91.5
                if ( median_coverage <= 97.5 ) {
                  if ( mean_support <= 0.976330041885 ) {
                    return true;
                  }
                  else {  // if mean_support > 0.976330041885
                    if ( max_support <= 0.976345002651 ) {
                      return true;
                    }
                    else {  // if max_support > 0.976345002651
                      return true;
                    }
                  }
                }
                else {  // if median_coverage > 97.5
                  if ( max_coverage <= 129.5 ) {
                    if ( mean_coverage <= 92.9117584229 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 92.9117584229
                      return false;
                    }
                  }
                  else {  // if max_coverage > 129.5
                    if ( median_coverage <= 98.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 98.5
                      return true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    else {  // if min_coverage > 46.5
      if ( max_support <= 0.962215006351 ) {
        if ( min_coverage <= 136.5 ) {
          if ( median_support <= 0.94840502739 ) {
            if ( min_coverage <= 104.5 ) {
              if ( max_support <= 0.940105021 ) {
                if ( min_coverage <= 83.5 ) {
                  if ( mean_coverage <= 129.352935791 ) {
                    if ( min_coverage <= 47.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 47.5
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 129.352935791
                    if ( mean_coverage <= 131.235290527 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 131.235290527
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 83.5
                  if ( max_coverage <= 126.5 ) {
                    if ( mean_coverage <= 94.0 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 94.0
                      return false;
                    }
                  }
                  else {  // if max_coverage > 126.5
                    if ( max_coverage <= 179.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 179.5
                      return false;
                    }
                  }
                }
              }
              else {  // if max_support > 0.940105021
                if ( min_coverage <= 75.5 ) {
                  if ( mean_support <= 0.942894995213 ) {
                    if ( median_coverage <= 107.5 ) {
                      return false;
                    }
                    else {  // if median_coverage > 107.5
                      return false;
                    }
                  }
                  else {  // if mean_support > 0.942894995213
                    if ( mean_coverage <= 67.9212112427 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 67.9212112427
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 75.5
                  if ( max_coverage <= 122.5 ) {
                    if ( median_support <= 0.945549964905 ) {
                      return false;
                    }
                    else {  // if median_support > 0.945549964905
                      return false;
                    }
                  }
                  else {  // if max_coverage > 122.5
                    if ( min_support <= 0.940874993801 ) {
                      return false;
                    }
                    else {  // if min_support > 0.940874993801
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if min_coverage > 104.5
              if ( min_support <= 0.943395018578 ) {
                if ( max_coverage <= 186.5 ) {
                  if ( min_support <= 0.939634978771 ) {
                    if ( max_coverage <= 183.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 183.5
                      return false;
                    }
                  }
                  else {  // if min_support > 0.939634978771
                    if ( min_coverage <= 107.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 107.5
                      return false;
                    }
                  }
                }
                else {  // if max_coverage > 186.5
                  if ( min_support <= 0.937654972076 ) {
                    if ( min_coverage <= 117.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 117.5
                      return false;
                    }
                  }
                  else {  // if min_support > 0.937654972076
                    if ( max_support <= 0.93774497509 ) {
                      return false;
                    }
                    else {  // if max_support > 0.93774497509
                      return false;
                    }
                  }
                }
              }
              else {  // if min_support > 0.943395018578
                if ( max_coverage <= 154.5 ) {
                  if ( mean_coverage <= 144.444442749 ) {
                    if ( mean_support <= 0.945254981518 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.945254981518
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 144.444442749
                    return true;
                  }
                }
                else {  // if max_coverage > 154.5
                  if ( median_coverage <= 168.5 ) {
                    if ( max_coverage <= 220.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 220.5
                      return true;
                    }
                  }
                  else {  // if median_coverage > 168.5
                    if ( min_coverage <= 133.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 133.5
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_support > 0.94840502739
            if ( min_support <= 0.95572501421 ) {
              if ( min_coverage <= 94.5 ) {
                if ( mean_support <= 0.950904965401 ) {
                  if ( mean_coverage <= 62.6417121887 ) {
                    if ( median_support <= 0.950625002384 ) {
                      return false;
                    }
                    else {  // if median_support > 0.950625002384
                      return true;
                    }
                  }
                  else {  // if mean_coverage > 62.6417121887
                    if ( max_coverage <= 113.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 113.5
                      return false;
                    }
                  }
                }
                else {  // if mean_support > 0.950904965401
                  if ( min_coverage <= 48.5 ) {
                    if ( max_coverage <= 79.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 79.5
                      return false;
                    }
                  }
                  else {  // if min_coverage > 48.5
                    if ( median_coverage <= 53.75 ) {
                      return true;
                    }
                    else {  // if median_coverage > 53.75
                      return false;
                    }
                  }
                }
              }
              else {  // if min_coverage > 94.5
                if ( mean_coverage <= 152.392303467 ) {
                  if ( max_support <= 0.949384987354 ) {
                    if ( mean_support <= 0.948485016823 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.948485016823
                      return false;
                    }
                  }
                  else {  // if max_support > 0.949384987354
                    if ( max_coverage <= 141.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 141.5
                      return false;
                    }
                  }
                }
                else {  // if mean_coverage > 152.392303467
                  if ( median_coverage <= 165.25 ) {
                    if ( median_coverage <= 145.25 ) {
                      return true;
                    }
                    else {  // if median_coverage > 145.25
                      return false;
                    }
                  }
                  else {  // if median_coverage > 165.25
                    if ( max_coverage <= 184.5 ) {
                      return true;
                    }
                    else {  // if max_coverage > 184.5
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if min_support > 0.95572501421
              if ( min_coverage <= 96.5 ) {
                if ( median_support <= 0.958145022392 ) {
                  if ( min_coverage <= 73.5 ) {
                    if ( max_coverage <= 121.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 121.5
                      return false;
                    }
                  }
                  else {  // if min_coverage > 73.5
                    if ( max_coverage <= 145.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 145.5
                      return false;
                    }
                  }
                }
                else {  // if median_support > 0.958145022392
                  if ( max_coverage <= 150.5 ) {
                    if ( mean_coverage <= 55.2575759888 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 55.2575759888
                      return false;
                    }
                  }
                  else {  // if max_coverage > 150.5
                    if ( mean_coverage <= 122.5 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 122.5
                      return false;
                    }
                  }
                }
              }
              else {  // if min_coverage > 96.5
                if ( mean_coverage <= 141.138092041 ) {
                  if ( mean_support <= 0.958554983139 ) {
                    if ( mean_coverage <= 129.466064453 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 129.466064453
                      return false;
                    }
                  }
                  else {  // if mean_support > 0.958554983139
                    if ( median_coverage <= 143.75 ) {
                      return false;
                    }
                    else {  // if median_coverage > 143.75
                      return false;
                    }
                  }
                }
                else {  // if mean_coverage > 141.138092041
                  if ( max_coverage <= 233.5 ) {
                    if ( min_coverage <= 104.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 104.5
                      return false;
                    }
                  }
                  else {  // if max_coverage > 233.5
                    if ( median_coverage <= 189.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 189.5
                      return false;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if min_coverage > 136.5
          if ( mean_coverage <= 198.449493408 ) {
            if ( median_support <= 0.948454976082 ) {
              if ( mean_support <= 0.939554989338 ) {
                if ( mean_coverage <= 179.242645264 ) {
                  if ( min_coverage <= 167.5 ) {
                    if ( min_coverage <= 152.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 152.5
                      return false;
                    }
                  }
                  else {  // if min_coverage > 167.5
                    if ( min_support <= 0.93737500906 ) {
                      return false;
                    }
                    else {  // if min_support > 0.93737500906
                      return true;
                    }
                  }
                }
                else {  // if mean_coverage > 179.242645264
                  if ( max_coverage <= 236.5 ) {
                    if ( mean_coverage <= 181.029418945 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 181.029418945
                      return false;
                    }
                  }
                  else {  // if max_coverage > 236.5
                    if ( mean_coverage <= 190.941177368 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 190.941177368
                      return false;
                    }
                  }
                }
              }
              else {  // if mean_support > 0.939554989338
                if ( mean_coverage <= 179.679489136 ) {
                  if ( max_coverage <= 227.5 ) {
                    if ( max_support <= 0.939755022526 ) {
                      return false;
                    }
                    else {  // if max_support > 0.939755022526
                      return false;
                    }
                  }
                  else {  // if max_coverage > 227.5
                    return true;
                  }
                }
                else {  // if mean_coverage > 179.679489136
                  if ( min_support <= 0.941635012627 ) {
                    if ( mean_support <= 0.941269993782 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.941269993782
                      return false;
                    }
                  }
                  else {  // if min_support > 0.941635012627
                    if ( min_coverage <= 155.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 155.5
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if median_support > 0.948454976082
              if ( mean_support <= 0.9544749856 ) {
                if ( median_coverage <= 175.75 ) {
                  if ( mean_coverage <= 174.878677368 ) {
                    if ( max_coverage <= 169.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 169.5
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 174.878677368
                    if ( mean_coverage <= 174.970581055 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 174.970581055
                      return false;
                    }
                  }
                }
                else {  // if median_coverage > 175.75
                  if ( max_coverage <= 181.5 ) {
                    return true;
                  }
                  else {  // if max_coverage > 181.5
                    if ( max_coverage <= 261.0 ) {
                      return false;
                    }
                    else {  // if max_coverage > 261.0
                      return true;
                    }
                  }
                }
              }
              else {  // if mean_support > 0.9544749856
                if ( min_coverage <= 156.5 ) {
                  if ( median_support <= 0.961335003376 ) {
                    if ( max_coverage <= 254.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 254.5
                      return true;
                    }
                  }
                  else {  // if median_support > 0.961335003376
                    if ( median_support <= 0.962105035782 ) {
                      return false;
                    }
                    else {  // if median_support > 0.962105035782
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 156.5
                  if ( max_support <= 0.957275032997 ) {
                    if ( max_support <= 0.957185029984 ) {
                      return false;
                    }
                    else {  // if max_support > 0.957185029984
                      return false;
                    }
                  }
                  else {  // if max_support > 0.957275032997
                    if ( mean_support <= 0.962175011635 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.962175011635
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_coverage > 198.449493408
            if ( mean_coverage <= 225.878677368 ) {
              if ( max_support <= 0.946645021439 ) {
                if ( min_support <= 0.934045016766 ) {
                  if ( median_coverage <= 216.0 ) {
                    if ( min_coverage <= 191.5 ) {
                      return true;
                    }
                    else {  // if min_coverage > 191.5
                      return false;
                    }
                  }
                  else {  // if median_coverage > 216.0
                    return false;
                  }
                }
                else {  // if min_support > 0.934045016766
                  if ( max_support <= 0.939785003662 ) {
                    if ( mean_coverage <= 223.970581055 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 223.970581055
                      return false;
                    }
                  }
                  else {  // if max_support > 0.939785003662
                    if ( mean_coverage <= 225.870834351 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 225.870834351
                      return true;
                    }
                  }
                }
              }
              else {  // if max_support > 0.946645021439
                if ( min_support <= 0.956155002117 ) {
                  if ( mean_coverage <= 225.739501953 ) {
                    if ( max_coverage <= 263.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 263.5
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 225.739501953
                    if ( median_coverage <= 225.5 ) {
                      return true;
                    }
                    else {  // if median_coverage > 225.5
                      return false;
                    }
                  }
                }
                else {  // if min_support > 0.956155002117
                  if ( median_coverage <= 190.5 ) {
                    return true;
                  }
                  else {  // if median_coverage > 190.5
                    if ( max_coverage <= 279.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 279.5
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if mean_coverage > 225.878677368
              if ( median_support <= 0.948194980621 ) {
                if ( min_coverage <= 250.5 ) {
                  if ( max_coverage <= 253.5 ) {
                    if ( median_support <= 0.933955013752 ) {
                      return true;
                    }
                    else {  // if median_support > 0.933955013752
                      return false;
                    }
                  }
                  else {  // if max_coverage > 253.5
                    if ( max_coverage <= 320.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 320.5
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 250.5
                  if ( median_coverage <= 442.5 ) {
                    if ( median_support <= 0.937714993954 ) {
                      return false;
                    }
                    else {  // if median_support > 0.937714993954
                      return false;
                    }
                  }
                  else {  // if median_coverage > 442.5
                    if ( min_coverage <= 412.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 412.5
                      return false;
                    }
                  }
                }
              }
              else {  // if median_support > 0.948194980621
                if ( min_coverage <= 188.5 ) {
                  if ( mean_coverage <= 229.264709473 ) {
                    if ( max_coverage <= 257.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 257.5
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 229.264709473
                    if ( mean_coverage <= 229.382354736 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 229.382354736
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 188.5
                  if ( min_coverage <= 245.5 ) {
                    if ( max_support <= 0.962204992771 ) {
                      return false;
                    }
                    else {  // if max_support > 0.962204992771
                      return false;
                    }
                  }
                  else {  // if min_coverage > 245.5
                    if ( median_coverage <= 333.25 ) {
                      return false;
                    }
                    else {  // if median_coverage > 333.25
                      return false;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if max_support > 0.962215006351
        if ( mean_support <= 0.976544976234 ) {
          if ( mean_coverage <= 149.939331055 ) {
            if ( max_support <= 0.968724966049 ) {
              if ( min_coverage <= 77.5 ) {
                if ( median_support <= 0.965624988079 ) {
                  if ( max_coverage <= 193.0 ) {
                    if ( max_support <= 0.962554991245 ) {
                      return false;
                    }
                    else {  // if max_support > 0.962554991245
                      return false;
                    }
                  }
                  else {  // if max_coverage > 193.0
                    if ( max_support <= 0.963490009308 ) {
                      return true;
                    }
                    else {  // if max_support > 0.963490009308
                      return false;
                    }
                  }
                }
                else {  // if median_support > 0.965624988079
                  if ( max_support <= 0.968715012074 ) {
                    if ( median_coverage <= 139.0 ) {
                      return false;
                    }
                    else {  // if median_coverage > 139.0
                      return true;
                    }
                  }
                  else {  // if max_support > 0.968715012074
                    if ( min_coverage <= 51.0 ) {
                      return true;
                    }
                    else {  // if min_coverage > 51.0
                      return false;
                    }
                  }
                }
              }
              else {  // if min_coverage > 77.5
                if ( min_support <= 0.966764986515 ) {
                  if ( mean_coverage <= 127.297058105 ) {
                    if ( min_support <= 0.962374985218 ) {
                      return false;
                    }
                    else {  // if min_support > 0.962374985218
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 127.297058105
                    if ( mean_coverage <= 149.918746948 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 149.918746948
                      return true;
                    }
                  }
                }
                else {  // if min_support > 0.966764986515
                  if ( mean_coverage <= 140.727935791 ) {
                    if ( mean_coverage <= 83.1999969482 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 83.1999969482
                      return false;
                    }
                  }
                  else {  // if mean_coverage > 140.727935791
                    if ( min_support <= 0.968715012074 ) {
                      return false;
                    }
                    else {  // if min_support > 0.968715012074
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if max_support > 0.968724966049
              if ( median_coverage <= 122.75 ) {
                if ( min_support <= 0.972734987736 ) {
                  if ( min_coverage <= 68.5 ) {
                    if ( mean_support <= 0.97272503376 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.97272503376
                      return false;
                    }
                  }
                  else {  // if min_coverage > 68.5
                    if ( median_support <= 0.971355021 ) {
                      return false;
                    }
                    else {  // if median_support > 0.971355021
                      return false;
                    }
                  }
                }
                else {  // if min_support > 0.972734987736
                  if ( median_coverage <= 109.25 ) {
                    if ( max_coverage <= 177.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 177.5
                      return true;
                    }
                  }
                  else {  // if median_coverage > 109.25
                    if ( min_coverage <= 55.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 55.5
                      return false;
                    }
                  }
                }
              }
              else {  // if median_coverage > 122.75
                if ( max_coverage <= 204.5 ) {
                  if ( max_support <= 0.972464978695 ) {
                    if ( median_coverage <= 131.75 ) {
                      return false;
                    }
                    else {  // if median_coverage > 131.75
                      return false;
                    }
                  }
                  else {  // if max_support > 0.972464978695
                    if ( mean_coverage <= 116.529411316 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 116.529411316
                      return false;
                    }
                  }
                }
                else {  // if max_coverage > 204.5
                  if ( median_coverage <= 143.5 ) {
                    if ( min_support <= 0.975285053253 ) {
                      return false;
                    }
                    else {  // if min_support > 0.975285053253
                      return false;
                    }
                  }
                  else {  // if median_coverage > 143.5
                    if ( median_coverage <= 148.0 ) {
                      return true;
                    }
                    else {  // if median_coverage > 148.0
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_coverage > 149.939331055
            if ( min_coverage <= 170.5 ) {
              if ( median_support <= 0.968195021152 ) {
                if ( mean_coverage <= 163.904541016 ) {
                  if ( median_coverage <= 167.5 ) {
                    if ( mean_coverage <= 163.885620117 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 163.885620117
                      return true;
                    }
                  }
                  else {  // if median_coverage > 167.5
                    if ( min_coverage <= 120.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 120.5
                      return false;
                    }
                  }
                }
                else {  // if mean_coverage > 163.904541016
                  if ( max_support <= 0.963575005531 ) {
                    if ( median_coverage <= 219.0 ) {
                      return false;
                    }
                    else {  // if median_coverage > 219.0
                      return true;
                    }
                  }
                  else {  // if max_support > 0.963575005531
                    if ( min_coverage <= 140.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 140.5
                      return false;
                    }
                  }
                }
              }
              else {  // if median_support > 0.968195021152
                if ( min_coverage <= 144.5 ) {
                  if ( median_coverage <= 192.5 ) {
                    if ( mean_support <= 0.972874999046 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.972874999046
                      return false;
                    }
                  }
                  else {  // if median_coverage > 192.5
                    if ( min_coverage <= 126.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 126.5
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 144.5
                  if ( median_coverage <= 190.75 ) {
                    if ( min_support <= 0.976535022259 ) {
                      return false;
                    }
                    else {  // if min_support > 0.976535022259
                      return false;
                    }
                  }
                  else {  // if median_coverage > 190.75
                    if ( min_coverage <= 149.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 149.5
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if min_coverage > 170.5
              if ( min_coverage <= 200.5 ) {
                if ( max_support <= 0.970565021038 ) {
                  if ( min_coverage <= 188.5 ) {
                    if ( max_coverage <= 275.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 275.5
                      return false;
                    }
                  }
                  else {  // if min_coverage > 188.5
                    if ( median_support <= 0.962234973907 ) {
                      return false;
                    }
                    else {  // if median_support > 0.962234973907
                      return false;
                    }
                  }
                }
                else {  // if max_support > 0.970565021038
                  if ( max_coverage <= 315.0 ) {
                    if ( max_coverage <= 275.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 275.5
                      return false;
                    }
                  }
                  else {  // if max_coverage > 315.0
                    if ( mean_coverage <= 252.352935791 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 252.352935791
                      return true;
                    }
                  }
                }
              }
              else {  // if min_coverage > 200.5
                if ( min_coverage <= 217.5 ) {
                  if ( max_support <= 0.967864990234 ) {
                    if ( mean_support <= 0.967854976654 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.967854976654
                      return false;
                    }
                  }
                  else {  // if max_support > 0.967864990234
                    if ( mean_coverage <= 268.058837891 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 268.058837891
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 217.5
                  if ( max_support <= 0.964325010777 ) {
                    if ( mean_support <= 0.96429502964 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.96429502964
                      return false;
                    }
                  }
                  else {  // if max_support > 0.964325010777
                    if ( mean_coverage <= 307.811767578 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 307.811767578
                      return false;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_support > 0.976544976234
          if ( mean_coverage <= 163.029418945 ) {
            if ( min_support <= 0.984054982662 ) {
              if ( median_coverage <= 133.75 ) {
                if ( min_support <= 0.981374979019 ) {
                  if ( median_support <= 0.979255020618 ) {
                    if ( min_coverage <= 74.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 74.5
                      return false;
                    }
                  }
                  else {  // if median_support > 0.979255020618
                    if ( max_coverage <= 64.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 64.5
                      return false;
                    }
                  }
                }
                else {  // if min_support > 0.981374979019
                  if ( median_coverage <= 69.25 ) {
                    if ( min_coverage <= 63.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 63.5
                      return false;
                    }
                  }
                  else {  // if median_coverage > 69.25
                    if ( mean_coverage <= 67.5277786255 ) {
                      return true;
                    }
                    else {  // if mean_coverage > 67.5277786255
                      return false;
                    }
                  }
                }
              }
              else {  // if median_coverage > 133.75
                if ( max_coverage <= 229.5 ) {
                  if ( median_support <= 0.979075014591 ) {
                    if ( max_coverage <= 214.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 214.5
                      return false;
                    }
                  }
                  else {  // if median_support > 0.979075014591
                    if ( min_coverage <= 123.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 123.5
                      return false;
                    }
                  }
                }
                else {  // if max_coverage > 229.5
                  return true;
                }
              }
            }
            else {  // if min_support > 0.984054982662
              if ( mean_support <= 0.988695025444 ) {
                if ( min_support <= 0.986925005913 ) {
                  if ( min_coverage <= 118.5 ) {
                    if ( mean_coverage <= 75.9258270264 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 75.9258270264
                      return false;
                    }
                  }
                  else {  // if min_coverage > 118.5
                    if ( max_support <= 0.986874997616 ) {
                      return false;
                    }
                    else {  // if max_support > 0.986874997616
                      return false;
                    }
                  }
                }
                else {  // if min_support > 0.986925005913
                  if ( max_coverage <= 161.5 ) {
                    if ( max_coverage <= 118.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 118.5
                      return false;
                    }
                  }
                  else {  // if max_coverage > 161.5
                    if ( max_coverage <= 191.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 191.5
                      return false;
                    }
                  }
                }
              }
              else {  // if mean_support > 0.988695025444
                if ( max_coverage <= 73.0 ) {
                  return true;
                }
                else {  // if max_coverage > 73.0
                  if ( median_support <= 0.992715001106 ) {
                    if ( mean_support <= 0.991365015507 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.991365015507
                      return false;
                    }
                  }
                  else {  // if median_support > 0.992715001106
                    if ( mean_support <= 0.994075000286 ) {
                      return false;
                    }
                    else {  // if mean_support > 0.994075000286
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_coverage > 163.029418945
            if ( min_coverage <= 170.5 ) {
              if ( mean_support <= 0.984674990177 ) {
                if ( mean_coverage <= 247.176483154 ) {
                  if ( max_support <= 0.979835033417 ) {
                    if ( median_coverage <= 169.75 ) {
                      return false;
                    }
                    else {  // if median_coverage > 169.75
                      return false;
                    }
                  }
                  else {  // if max_support > 0.979835033417
                    if ( mean_coverage <= 165.125488281 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 165.125488281
                      return false;
                    }
                  }
                }
                else {  // if mean_coverage > 247.176483154
                  return true;
                }
              }
              else {  // if mean_support > 0.984674990177
                if ( min_support <= 0.989634990692 ) {
                  if ( median_coverage <= 242.5 ) {
                    if ( min_support <= 0.989614963531 ) {
                      return false;
                    }
                    else {  // if min_support > 0.989614963531
                      return false;
                    }
                  }
                  else {  // if median_coverage > 242.5
                    if ( median_coverage <= 256.0 ) {
                      return true;
                    }
                    else {  // if median_coverage > 256.0
                      return false;
                    }
                  }
                }
                else {  // if min_support > 0.989634990692
                  if ( min_support <= 0.994785010815 ) {
                    if ( max_coverage <= 170.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 170.5
                      return false;
                    }
                  }
                  else {  // if min_support > 0.994785010815
                    if ( mean_coverage <= 163.297058105 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 163.297058105
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if min_coverage > 170.5
              if ( median_support <= 0.986564993858 ) {
                if ( min_coverage <= 206.5 ) {
                  if ( mean_support <= 0.980875015259 ) {
                    if ( max_coverage <= 290.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 290.5
                      return false;
                    }
                  }
                  else {  // if mean_support > 0.980875015259
                    if ( min_support <= 0.986544966698 ) {
                      return false;
                    }
                    else {  // if min_support > 0.986544966698
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 206.5
                  if ( median_coverage <= 251.25 ) {
                    if ( min_coverage <= 216.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 216.5
                      return false;
                    }
                  }
                  else {  // if median_coverage > 251.25
                    if ( median_coverage <= 293.25 ) {
                      return false;
                    }
                    else {  // if median_coverage > 293.25
                      return false;
                    }
                  }
                }
              }
              else {  // if median_support > 0.986564993858
                if ( min_coverage <= 218.5 ) {
                  if ( max_support <= 0.991724967957 ) {
                    if ( max_coverage <= 302.5 ) {
                      return false;
                    }
                    else {  // if max_coverage > 302.5
                      return false;
                    }
                  }
                  else {  // if max_support > 0.991724967957
                    if ( median_coverage <= 196.25 ) {
                      return false;
                    }
                    else {  // if median_coverage > 196.25
                      return false;
                    }
                  }
                }
                else {  // if min_coverage > 218.5
                  if ( min_support <= 0.991864979267 ) {
                    if ( mean_coverage <= 252.029418945 ) {
                      return false;
                    }
                    else {  // if mean_coverage > 252.029418945
                      return false;
                    }
                  }
                  else {  // if min_support > 0.991864979267
                    if ( min_coverage <= 239.5 ) {
                      return false;
                    }
                    else {  // if min_coverage > 239.5
                      return false;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

#endif

std::vector<PileupImage::Feature> PileupImage::getFeaturesOfNonConsensusPositions(
                                const std::string& sequence,
                                int k,
                                double support_threshold) const{
    auto isValidIndex = [&](int i){
        return columnProperties.subjectColumnsBegin_incl <= i
            && i < columnProperties.subjectColumnsEnd_excl;
    };

    std::vector<Feature> result;

    for(int i = columnProperties.subjectColumnsBegin_incl;
        i < columnProperties.subjectColumnsEnd_excl;
        i++){

        const int localindex = i - columnProperties.subjectColumnsBegin_incl;

        if(h_support[i] >= support_threshold && h_consensus[i] != sequence[localindex]){
            Feature f;
            f.position = localindex;

            int begin = i-k/2;
            int end = i+k/2;

            for(int j = 0; j < k; j++){
                if(!isValidIndex(begin))
                    begin++;
                if(!isValidIndex(end))
                    end--;
            }
            end++;

            f.min_support = *std::min_element(h_support.begin() + begin, h_support.begin() + end);
            f.min_coverage = *std::min_element(h_coverage.begin() + begin, h_coverage.begin() + end);
            f.max_support = *std::max_element(h_support.begin() + begin, h_support.begin() + end);
            f.max_coverage = *std::max_element(h_coverage.begin() + begin, h_coverage.begin() + end);
            f.mean_support = std::accumulate(h_support.begin() + begin, h_support.begin() + end, 0.0) / (end - begin);
            f.mean_coverage = std::accumulate(h_coverage.begin() + begin, h_coverage.begin() + end, 0.0) / (end - begin);

            std::array<double, 32> arr;

            std::copy(h_support.begin() + begin, h_support.begin() + end, arr.begin());
            std::nth_element(arr.begin(), arr.begin() + arr.size()/2, arr.end());
            f.median_support = arr[arr.size()/2];

            std::copy(h_coverage.begin() + begin, h_coverage.begin() + end, arr.begin());
            std::nth_element(arr.begin(), arr.begin() + arr.size()/2, arr.end());
            f.median_coverage = arr[arr.size()/2];

            result.emplace_back(f);
        }
    }

    return result;
}

PileupImage::CorrectionResult PileupImage::cpu_correct_sequence_internal_RF(const std::string& sequence_to_correct){

    cpu_find_consensus_internal();

    const int subjectlength = sequence_to_correct.length();

    CorrectionResult result;
    result.isCorrected = false;
    result.correctedSequence.resize(subjectlength);
    result.stats.avg_support = 0;
    result.stats.min_support = 1.0;
    result.stats.max_coverage = 0;
    result.stats.min_coverage = std::numeric_limits<int>::max();
    result.correctedSequence = sequence_to_correct;

    auto features = getFeaturesOfNonConsensusPositions(sequence_to_correct,
                                                        correctionSettings.k,
                                                        0.5);

    for(const auto& feature : features){
        if(!shouldCorrect(feature.min_support,
                    feature.min_coverage,
                    feature.max_support,
                    feature.max_coverage,
                    feature.mean_support,
                    feature.mean_coverage,
                    feature.median_support,
                    feature.median_coverage)){

            const int globalIndex = columnProperties.subjectColumnsBegin_incl + feature.position;
            result.correctedSequence[feature.position] = h_consensus[globalIndex];
        }
    }

    result.isCorrected = true;

    return result;
}


}
}
