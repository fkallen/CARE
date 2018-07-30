#include "../inc/ecolisrr490124.hpp"

namespace ecoli_srr490124{

    //coverage is normalized to number of reads in msa
    bool shouldCorrect(double min_col_support, double min_col_coverage,
        double max_col_support, double max_col_coverage,
        double mean_col_support, double mean_col_coverage,
        double median_col_support, double median_col_coverage,
        double maxgini) {
      if ( median_col_support <= 0.978500008583 ) {
        if ( median_col_support <= 0.943500041962 ) {
          if ( min_col_coverage <= 0.96369600296 ) {
            if ( median_col_coverage <= 0.00778211420402 ) {
              if ( max_col_coverage <= 0.24736829102 ) {
                if ( max_col_coverage <= 0.167229786515 ) {
                  if ( max_col_coverage <= 0.119449183345 ) {
                    if ( mean_col_support <= 0.665258228779 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.665258228779
                      return 0.0287118538723 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.119449183345
                    if ( min_col_coverage <= 0.00776197947562 ) {
                      return 0.0520382867872 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00776197947562
                      return 0.489795918367 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.167229786515
                  if ( max_col_coverage <= 0.210697084665 ) {
                    if ( median_col_coverage <= 0.00652530277148 ) {
                      return 0.0839297724155 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00652530277148
                      return 0.0515323182404 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.210697084665
                    if ( min_col_coverage <= 0.00670018605888 ) {
                      return 0.102076839805 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00670018605888
                      return 0.206255475048 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.24736829102
                if ( min_col_coverage <= 0.00543578621 ) {
                  if ( mean_col_support <= 0.965088248253 ) {
                    if ( mean_col_coverage <= 0.141360074282 ) {
                      return 0.158828036055 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.141360074282
                      return 0.444444444444 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.965088248253
                    if ( min_col_coverage <= 0.00353983417153 ) {
                      return 0.249131944444 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00353983417153
                      return false;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.00543578621
                  if ( mean_col_coverage <= 0.127378940582 ) {
                    if ( min_col_support <= 0.537500023842 ) {
                      return 0.461950059453 < maxgini;
                    }
                    else {  // if min_col_support > 0.537500023842
                      return 0.246574878491 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.127378940582
                    if ( mean_col_support <= 0.897676467896 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.897676467896
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.00778211420402
              if ( median_col_support <= 0.858500003815 ) {
                if ( max_col_support <= 0.992499947548 ) {
                  if ( max_col_support <= 0.979499995708 ) {
                    if ( median_col_coverage <= 0.0102874133736 ) {
                      return 0.375 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0102874133736
                      return 0.0152600101067 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.979499995708
                    if ( min_col_support <= 0.577499985695 ) {
                      return 0.034141751806 < maxgini;
                    }
                    else {  // if min_col_support > 0.577499985695
                      return 0.0243780349765 < maxgini;
                    }
                  }
                }
                else {  // if max_col_support > 0.992499947548
                  if ( min_col_coverage <= 0.88042396307 ) {
                    if ( min_col_support <= 0.584499955177 ) {
                      return 0.048396458445 < maxgini;
                    }
                    else {  // if min_col_support > 0.584499955177
                      return 0.040817520016 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.88042396307
                    if ( max_col_coverage <= 0.991508126259 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.991508126259
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.858500003815
                if ( max_col_coverage <= 0.598166465759 ) {
                  if ( max_col_support <= 0.996500015259 ) {
                    if ( max_col_support <= 0.994500041008 ) {
                      return 0.0216776130606 < maxgini;
                    }
                    else {  // if max_col_support > 0.994500041008
                      return 0.0300066758235 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.996500015259
                    if ( min_col_coverage <= 0.00594354979694 ) {
                      return 0.0517552275621 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00594354979694
                      return 0.0363085890249 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.598166465759
                  if ( min_col_support <= 0.567499995232 ) {
                    if ( min_col_coverage <= 0.891560673714 ) {
                      return 0.058505283671 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.891560673714
                      return 0.444444444444 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.567499995232
                    if ( max_col_coverage <= 0.697074353695 ) {
                      return 0.0283205379063 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.697074353695
                      return 0.020844786233 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.96369600296
            if ( min_col_support <= 0.705000042915 ) {
              if ( max_col_coverage <= 0.982238054276 ) {
                return 0.0 < maxgini;
              }
              else {  // if max_col_coverage > 0.982238054276
                if ( min_col_coverage <= 0.996677696705 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.996677696705
                  if ( min_col_coverage <= 0.99671036005 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.99671036005
                    return false;
                  }
                }
              }
            }
            else {  // if min_col_support > 0.705000042915
              return 0.0 < maxgini;
            }
          }
        }
        else {  // if median_col_support > 0.943500041962
          if ( max_col_coverage <= 0.608728468418 ) {
            if ( median_col_coverage <= 0.00578872393817 ) {
              if ( max_col_coverage <= 0.231923952699 ) {
                if ( max_col_coverage <= 0.171507805586 ) {
                  if ( min_col_coverage <= 0.00298954336904 ) {
                    if ( min_col_support <= 0.368000000715 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.368000000715
                      return 0.0575844293488 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00298954336904
                    if ( max_col_coverage <= 0.137766867876 ) {
                      return 0.0163762173915 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.137766867876
                      return 0.0303080652299 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.171507805586
                  if ( median_col_coverage <= 0.00292826397344 ) {
                    if ( median_col_support <= 0.96850001812 ) {
                      return 0.125530649387 < maxgini;
                    }
                    else {  // if median_col_support > 0.96850001812
                      return 0.0677131425054 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.00292826397344
                    if ( max_col_coverage <= 0.187707781792 ) {
                      return 0.0426902729066 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.187707781792
                      return 0.0656903841366 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.231923952699
                if ( max_col_coverage <= 0.273953348398 ) {
                  if ( min_col_coverage <= 0.00577201787382 ) {
                    if ( min_col_support <= 0.779500007629 ) {
                      return 0.0927086325768 < maxgini;
                    }
                    else {  // if min_col_support > 0.779500007629
                      return 0.156763687104 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00577201787382
                    if ( mean_col_coverage <= 0.0964807868004 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0964807868004
                      return false;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.273953348398
                  if ( min_col_support <= 0.728999972343 ) {
                    if ( max_col_coverage <= 0.27461206913 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.27461206913
                      return 0.0837551281368 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.728999972343
                    if ( mean_col_coverage <= 0.0898835733533 ) {
                      return 0.489795918367 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0898835733533
                      return 0.268677461245 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.00578872393817
              if ( median_col_support <= 0.964499950409 ) {
                if ( min_col_support <= 0.934499979019 ) {
                  if ( min_col_coverage <= 0.00298954336904 ) {
                    if ( median_col_coverage <= 0.025470521301 ) {
                      return 0.0319608734734 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.025470521301
                      return 0.0989627316363 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00298954336904
                    if ( min_col_support <= 0.890499949455 ) {
                      return 0.0257643981683 < maxgini;
                    }
                    else {  // if min_col_support > 0.890499949455
                      return 0.0302864330029 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.934499979019
                  if ( mean_col_support <= 0.9627353549 ) {
                    if ( max_col_support <= 0.996500015259 ) {
                      return 0.1171875 < maxgini;
                    }
                    else {  // if max_col_support > 0.996500015259
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.9627353549
                    if ( mean_col_coverage <= 0.41524168849 ) {
                      return 0.0380918444812 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.41524168849
                      return 0.0305113737398 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.964499950409
                if ( min_col_support <= 0.933500051498 ) {
                  if ( min_col_coverage <= 0.00309119746089 ) {
                    if ( median_col_coverage <= 0.0728426128626 ) {
                      return 0.0428349310369 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0728426128626
                      return 0.367309458219 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00309119746089
                    if ( min_col_support <= 0.904500007629 ) {
                      return 0.019151603918 < maxgini;
                    }
                    else {  // if min_col_support > 0.904500007629
                      return 0.0222763737711 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.933500051498
                  if ( median_col_support <= 0.972499966621 ) {
                    if ( mean_col_support <= 0.979970574379 ) {
                      return 0.0357773040206 < maxgini;
                    }
                    else {  // if mean_col_support > 0.979970574379
                      return 0.0291012253543 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.972499966621
                    if ( min_col_support <= 0.949499964714 ) {
                      return 0.0216253832269 < maxgini;
                    }
                    else {  // if min_col_support > 0.949499964714
                      return 0.0262649965052 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.608728468418
            if ( min_col_support <= 0.608500003815 ) {
              if ( min_col_coverage <= 0.945092499256 ) {
                if ( mean_col_coverage <= 0.656672298908 ) {
                  if ( max_col_coverage <= 0.860595583916 ) {
                    if ( median_col_coverage <= 0.617447435856 ) {
                      return 0.0590135128605 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.617447435856
                      return 0.444444444444 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.860595583916
                    if ( min_col_coverage <= 0.250996470451 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.250996470451
                      return false;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.656672298908
                  if ( min_col_coverage <= 0.584698200226 ) {
                    if ( mean_col_coverage <= 0.668732881546 ) {
                      return 0.1638 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.668732881546
                      return 0.382244809689 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.584698200226
                    if ( max_col_coverage <= 0.856929659843 ) {
                      return 0.119986850756 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.856929659843
                      return 0.253159822863 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.945092499256
                if ( min_col_coverage <= 0.963345468044 ) {
                  if ( min_col_coverage <= 0.954941868782 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.954941868782
                    return 0.0 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.963345468044
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.608500003815
              if ( max_col_coverage <= 0.73086130619 ) {
                if ( median_col_support <= 0.96850001812 ) {
                  if ( mean_col_support <= 0.921882390976 ) {
                    if ( mean_col_support <= 0.920029401779 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.920029401779
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.921882390976
                    if ( min_col_support <= 0.880499958992 ) {
                      return 0.016705395186 < maxgini;
                    }
                    else {  // if min_col_support > 0.880499958992
                      return 0.0229907379673 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.96850001812
                  if ( mean_col_support <= 0.935500025749 ) {
                    if ( mean_col_support <= 0.935176491737 ) {
                      return 0.277777777778 < maxgini;
                    }
                    else {  // if mean_col_support > 0.935176491737
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.935500025749
                    if ( min_col_support <= 0.914499998093 ) {
                      return 0.0125045868936 < maxgini;
                    }
                    else {  // if min_col_support > 0.914499998093
                      return 0.0173807156634 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.73086130619
                if ( min_col_coverage <= 0.989845275879 ) {
                  if ( max_col_coverage <= 0.830890059471 ) {
                    if ( mean_col_support <= 0.938117623329 ) {
                      return 0.197530864198 < maxgini;
                    }
                    else {  // if mean_col_support > 0.938117623329
                      return 0.0129404119696 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.830890059471
                    if ( min_col_support <= 0.638499975204 ) {
                      return 0.127066115702 < maxgini;
                    }
                    else {  // if min_col_support > 0.638499975204
                      return 0.0063126608555 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.989845275879
                  if ( min_col_support <= 0.685500025749 ) {
                    if ( mean_col_coverage <= 0.99936491251 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.99936491251
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.685500025749
                    if ( median_col_coverage <= 0.990322470665 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.990322470665
                      return 0.0997229916898 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.978500008583
        if ( mean_col_coverage <= 0.538221478462 ) {
          if ( median_col_support <= 0.990499973297 ) {
            if ( max_col_coverage <= 0.445287823677 ) {
              if ( mean_col_support <= 0.953369319439 ) {
                if ( min_col_support <= 0.671499967575 ) {
                  if ( mean_col_support <= 0.92191183567 ) {
                    if ( min_col_coverage <= 0.267056524754 ) {
                      return 0.154238227147 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.267056524754
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.92191183567
                    if ( min_col_support <= 0.631500005722 ) {
                      return 0.0440201391425 < maxgini;
                    }
                    else {  // if min_col_support > 0.631500005722
                      return 0.0886181783005 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.671499967575
                  if ( median_col_coverage <= 0.003144685179 ) {
                    if ( median_col_support <= 0.983000040054 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.983000040054
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.003144685179
                    if ( mean_col_support <= 0.95335829258 ) {
                      return 0.100977777778 < maxgini;
                    }
                    else {  // if mean_col_support > 0.95335829258
                      return false;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.953369319439
                if ( min_col_coverage <= 0.203351795673 ) {
                  if ( max_col_coverage <= 0.343956291676 ) {
                    if ( min_col_coverage <= 0.0726199001074 ) {
                      return 0.0164874000654 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0726199001074
                      return 0.0236129927385 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.343956291676
                    if ( min_col_support <= 0.956499993801 ) {
                      return 0.0120324538076 < maxgini;
                    }
                    else {  // if min_col_support > 0.956499993801
                      return 0.0163837582833 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.203351795673
                  if ( max_col_coverage <= 0.411954909563 ) {
                    if ( min_col_coverage <= 0.203356251121 ) {
                      return 0.290657439446 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.203356251121
                      return 0.0254028730922 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.411954909563
                    if ( min_col_coverage <= 0.228830888867 ) {
                      return 0.0139911971326 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.228830888867
                      return 0.0221028096934 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.445287823677
              if ( mean_col_support <= 0.924117684364 ) {
                if ( min_col_support <= 0.511999964714 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_support > 0.511999964714
                  if ( min_col_coverage <= 0.312133789062 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.312133789062
                    if ( max_col_coverage <= 0.557925105095 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.557925105095
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.924117684364
                if ( min_col_support <= 0.636500000954 ) {
                  if ( max_col_coverage <= 0.624402761459 ) {
                    if ( mean_col_support <= 0.974852919579 ) {
                      return 0.0636305306582 < maxgini;
                    }
                    else {  // if mean_col_support > 0.974852919579
                      return false;
                    }
                  }
                  else {  // if max_col_coverage > 0.624402761459
                    if ( max_col_support <= 0.99950003624 ) {
                      return false;
                    }
                    else {  // if max_col_support > 0.99950003624
                      return 0.225509409649 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.636500000954
                  if ( median_col_support <= 0.986500024796 ) {
                    if ( min_col_support <= 0.956499993801 ) {
                      return 0.0142360475577 < maxgini;
                    }
                    else {  // if min_col_support > 0.956499993801
                      return 0.019813489418 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.986500024796
                    if ( min_col_support <= 0.96850001812 ) {
                      return 0.0108320313895 < maxgini;
                    }
                    else {  // if min_col_support > 0.96850001812
                      return 0.0152590494045 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.990499973297
            if ( mean_col_support <= 0.98297059536 ) {
              if ( mean_col_coverage <= 0.5382193923 ) {
                if ( min_col_support <= 0.841500043869 ) {
                  if ( min_col_coverage <= 0.00325204106048 ) {
                    if ( median_col_coverage <= 0.00955034047365 ) {
                      return 0.0270471527813 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00955034047365
                      return 0.0741132187002 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00325204106048
                    if ( max_col_coverage <= 0.635318398476 ) {
                      return 0.0230804473682 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.635318398476
                      return 0.118048236096 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.841500043869
                  if ( min_col_coverage <= 0.00251572718844 ) {
                    if ( max_col_coverage <= 0.119496569037 ) {
                      return 0.423440453686 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.119496569037
                      return 0.127066115702 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00251572718844
                    if ( max_col_coverage <= 0.179760441184 ) {
                      return 0.0783672851288 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.179760441184
                      return 0.0202681452317 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.5382193923
                return false;
              }
            }
            else {  // if mean_col_support > 0.98297059536
              if ( median_col_support <= 0.993499994278 ) {
                if ( max_col_coverage <= 0.4420543015 ) {
                  if ( min_col_coverage <= 0.201204225421 ) {
                    if ( min_col_support <= 0.982499957085 ) {
                      return 0.00950344179017 < maxgini;
                    }
                    else {  // if min_col_support > 0.982499957085
                      return 0.0181053576868 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.201204225421
                    if ( min_col_coverage <= 0.201213389635 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.201213389635
                      return 0.0184968524448 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.4420543015
                  if ( max_col_coverage <= 0.594215154648 ) {
                    if ( min_col_support <= 0.976500034332 ) {
                      return 0.00950094898309 < maxgini;
                    }
                    else {  // if min_col_support > 0.976500034332
                      return 0.0129545748827 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.594215154648
                    if ( min_col_support <= 0.986500024796 ) {
                      return 0.00786998363651 < maxgini;
                    }
                    else {  // if min_col_support > 0.986500024796
                      return 0.0107650337413 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.993499994278
                if ( max_col_coverage <= 0.571727752686 ) {
                  if ( mean_col_support <= 0.990207195282 ) {
                    if ( mean_col_support <= 0.990193724632 ) {
                      return 0.0198951887172 < maxgini;
                    }
                    else {  // if mean_col_support > 0.990193724632
                      return 0.408163265306 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.990207195282
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.00979041135283 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.00653944323423 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.571727752686
                  if ( min_col_support <= 0.778999984264 ) {
                    if ( min_col_support <= 0.776499986649 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_support > 0.776499986649
                      return 0.46875 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.778999984264
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.00745199497967 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.00489229865658 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.538221478462
          if ( min_col_support <= 0.601500034332 ) {
            if ( min_col_coverage <= 0.795537471771 ) {
              if ( median_col_support <= 0.993499994278 ) {
                if ( mean_col_coverage <= 0.744422078133 ) {
                  if ( min_col_coverage <= 0.455196231604 ) {
                    if ( max_col_coverage <= 0.678542554379 ) {
                      return 0.183626033058 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.678542554379
                      return 0.408163265306 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.455196231604
                    if ( mean_col_support <= 0.967323541641 ) {
                      return 0.112654531794 < maxgini;
                    }
                    else {  // if mean_col_support > 0.967323541641
                      return 0.2405373195 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.744422078133
                  if ( min_col_coverage <= 0.562738060951 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.562738060951
                    if ( mean_col_support <= 0.961499929428 ) {
                      return 0.190405804891 < maxgini;
                    }
                    else {  // if mean_col_support > 0.961499929428
                      return 0.394965277778 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.993499994278
                if ( min_col_support <= 0.543500006199 ) {
                  if ( min_col_coverage <= 0.477325409651 ) {
                    if ( max_col_coverage <= 0.669313549995 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.669313549995
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.477325409651
                    if ( mean_col_coverage <= 0.679551839828 ) {
                      return 0.277777777778 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.679551839828
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.543500006199
                  if ( median_col_coverage <= 0.764746189117 ) {
                    if ( max_col_coverage <= 0.987341761589 ) {
                      return 0.244897959184 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.987341761589
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.764746189117
                    if ( mean_col_coverage <= 0.871473670006 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.871473670006
                      return 0.336734693878 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.795537471771
              if ( min_col_coverage <= 0.922877311707 ) {
                if ( max_col_coverage <= 0.971952795982 ) {
                  if ( mean_col_coverage <= 0.899097681046 ) {
                    if ( median_col_coverage <= 0.865466296673 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.865466296673
                      return 0.21875 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.899097681046
                    if ( median_col_coverage <= 0.933527946472 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.933527946472
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.971952795982
                  if ( median_col_support <= 0.996500015259 ) {
                    if ( median_col_support <= 0.983500003815 ) {
                      return 0.152777777778 < maxgini;
                    }
                    else {  // if median_col_support > 0.983500003815
                      return 0.499117293976 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.996500015259
                    if ( mean_col_coverage <= 0.950986266136 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.950986266136
                      return false;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.922877311707
                if ( min_col_support <= 0.5 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_support > 0.5
                  if ( min_col_support <= 0.557500004768 ) {
                    if ( median_col_support <= 0.980000019073 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.980000019073
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.557500004768
                    if ( median_col_support <= 0.990499973297 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.601500034332
            if ( median_col_support <= 0.990499973297 ) {
              if ( median_col_coverage <= 0.624509811401 ) {
                if ( median_col_support <= 0.986500024796 ) {
                  if ( min_col_support <= 0.958500027657 ) {
                    if ( mean_col_support <= 0.970205903053 ) {
                      return 0.0379261927292 < maxgini;
                    }
                    else {  // if mean_col_support > 0.970205903053
                      return 0.00865605593947 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.958500027657
                    if ( mean_col_support <= 0.995558857918 ) {
                      return 0.0138683249794 < maxgini;
                    }
                    else {  // if mean_col_support > 0.995558857918
                      return 0.224765868887 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.986500024796
                  if ( min_col_support <= 0.956499993801 ) {
                    if ( mean_col_coverage <= 0.759864330292 ) {
                      return 0.00553935879333 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.759864330292
                      return 0.444444444444 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.956499993801
                    if ( mean_col_support <= 0.996676445007 ) {
                      return 0.00882683662458 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996676445007
                      return 0.076070973434 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.624509811401
                if ( min_col_support <= 0.630499958992 ) {
                  if ( median_col_coverage <= 0.775032937527 ) {
                    if ( median_col_coverage <= 0.713273644447 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.713273644447
                      return 0.249131944444 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.775032937527
                    if ( max_col_coverage <= 0.864510834217 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.864510834217
                      return 0.433075550268 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.630499958992
                  if ( mean_col_support <= 0.9627353549 ) {
                    if ( max_col_coverage <= 0.993423938751 ) {
                      return 0.0883058984911 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.993423938751
                      return 0.478298611111 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.9627353549
                    if ( max_col_coverage <= 0.864293158054 ) {
                      return 0.00589184630303 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.864293158054
                      return 0.00264573563422 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.990499973297
              if ( min_col_support <= 0.667500019073 ) {
                if ( min_col_coverage <= 0.865798354149 ) {
                  if ( mean_col_coverage <= 0.7611156106 ) {
                    if ( max_col_coverage <= 0.949276387691 ) {
                      return 0.0532515992853 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.949276387691
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.7611156106
                    if ( mean_col_support <= 0.958735227585 ) {
                      return 0.499540863177 < maxgini;
                    }
                    else {  // if mean_col_support > 0.958735227585
                      return 0.271968291449 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.865798354149
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( mean_col_coverage <= 0.937221050262 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.937221050262
                      return 0.336734693878 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( min_col_support <= 0.634000003338 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.634000003338
                      return 0.498269896194 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.667500019073
                if ( mean_col_support <= 0.965647101402 ) {
                  if ( max_col_coverage <= 0.955079495907 ) {
                    if ( mean_col_coverage <= 0.930160403252 ) {
                      return 0.0496571869041 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.930160403252
                      return false;
                    }
                  }
                  else {  // if max_col_coverage > 0.955079495907
                    if ( min_col_support <= 0.688499987125 ) {
                      return 0.244897959184 < maxgini;
                    }
                    else {  // if min_col_support > 0.688499987125
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.965647101402
                  if ( mean_col_coverage <= 0.700577497482 ) {
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.0052367926648 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.00278915779617 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.700577497482
                    if ( min_col_support <= 0.738499999046 ) {
                      return 0.0758310249307 < maxgini;
                    }
                    else {  // if min_col_support > 0.738499999046
                      return 0.000967734917716 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }






    bool shouldCorrect0(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( mean_col_support <= 0.985891222954 ) {
        if ( mean_col_support <= 0.962355136871 ) {
          if ( median_col_coverage <= 0.943439304829 ) {
            if ( max_col_coverage <= 0.373840391636 ) {
              if ( max_col_support <= 0.990499973297 ) {
                if ( median_col_support <= 0.966500043869 ) {
                  if ( median_col_support <= 0.547500014305 ) {
                    if ( median_col_coverage <= 0.257952094078 ) {
                      return 0.00916305641939 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.257952094078
                      return 0.213039485767 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.547500014305
                    if ( max_col_support <= 0.980499982834 ) {
                      return 0.0143026124562 < maxgini;
                    }
                    else {  // if max_col_support > 0.980499982834
                      return 0.0276313307912 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.966500043869
                  return false;
                }
              }
              else {  // if max_col_support > 0.990499973297
                if ( min_col_coverage <= 0.00605145096779 ) {
                  if ( median_col_support <= 0.9375 ) {
                    if ( median_col_coverage <= 0.00625652400777 ) {
                      return 0.0639543128913 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00625652400777
                      return 0.0465718724351 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.9375
                    if ( max_col_coverage <= 0.273996978998 ) {
                      return 0.0374479000993 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.273996978998
                      return 0.126233137912 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.00605145096779
                  if ( median_col_coverage <= 0.172000914812 ) {
                    if ( median_col_support <= 0.87549996376 ) {
                      return 0.0456747495957 < maxgini;
                    }
                    else {  // if median_col_support > 0.87549996376
                      return 0.0331556504022 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.172000914812
                    if ( max_col_support <= 0.999000012875 ) {
                      return 0.0380412616778 < maxgini;
                    }
                    else {  // if max_col_support > 0.999000012875
                      return 0.049177889687 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.373840391636
              if ( min_col_support <= 0.575500011444 ) {
                if ( max_col_coverage <= 0.809999465942 ) {
                  if ( min_col_coverage <= 0.449886053801 ) {
                    if ( median_col_coverage <= 0.293622910976 ) {
                      return 0.0384722113846 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.293622910976
                      return 0.0466024671217 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.449886053801
                    if ( max_col_support <= 0.997500002384 ) {
                      return 0.0345101002992 < maxgini;
                    }
                    else {  // if max_col_support > 0.997500002384
                      return 0.0722112199369 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.809999465942
                  if ( mean_col_coverage <= 0.905526280403 ) {
                    if ( min_col_coverage <= 0.81365609169 ) {
                      return 0.130479874693 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.81365609169
                      return 0.498866213152 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.905526280403
                    if ( max_col_coverage <= 0.970375239849 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.970375239849
                      return 0.375 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.575500011444
                if ( min_col_support <= 0.935500025749 ) {
                  if ( max_col_support <= 0.993499994278 ) {
                    if ( min_col_support <= 0.630499958992 ) {
                      return 0.0290318095025 < maxgini;
                    }
                    else {  // if min_col_support > 0.630499958992
                      return 0.0195984326975 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.993499994278
                    if ( min_col_coverage <= 0.0639987289906 ) {
                      return 0.13359940915 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0639987289906
                      return 0.0339023282592 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.935500025749
                  if ( mean_col_support <= 0.961529493332 ) {
                    if ( min_col_coverage <= 0.228805303574 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.228805303574
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.961529493332
                    if ( median_col_coverage <= 0.313920348883 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.313920348883
                      return 0.444444444444 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.943439304829
            if ( mean_col_support <= 0.945852994919 ) {
              if ( max_col_coverage <= 0.995546460152 ) {
                if ( mean_col_coverage <= 0.964455127716 ) {
                  if ( median_col_support <= 0.871999979019 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.871999979019
                    return 0.0 < maxgini;
                  }
                }
                else {  // if mean_col_coverage > 0.964455127716
                  if ( mean_col_support <= 0.921294033527 ) {
                    if ( median_col_support <= 0.754500031471 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.754500031471
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.921294033527
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if max_col_coverage > 0.995546460152
                if ( mean_col_coverage <= 0.985585808754 ) {
                  if ( mean_col_coverage <= 0.983260571957 ) {
                    if ( mean_col_support <= 0.941264629364 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.941264629364
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.983260571957
                    return 0.0 < maxgini;
                  }
                }
                else {  // if mean_col_coverage > 0.985585808754
                  if ( mean_col_support <= 0.900088191032 ) {
                    if ( min_col_support <= 0.457000017166 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.457000017166
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.900088191032
                    return false;
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.945852994919
              if ( min_col_coverage <= 0.980537176132 ) {
                if ( mean_col_support <= 0.956499993801 ) {
                  if ( median_col_support <= 0.981500029564 ) {
                    if ( median_col_support <= 0.931999981403 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.931999981403
                      return 0.0997229916898 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.981500029564
                    if ( median_col_coverage <= 0.958151400089 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.958151400089
                      return 0.493827160494 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.956499993801
                  if ( mean_col_coverage <= 0.976203680038 ) {
                    if ( median_col_support <= 0.992500007153 ) {
                      return 0.0798611111111 < maxgini;
                    }
                    else {  // if median_col_support > 0.992500007153
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.976203680038
                    if ( median_col_coverage <= 0.977345347404 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.977345347404
                      return 0.375 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.980537176132
                return false;
              }
            }
          }
        }
        else {  // if mean_col_support > 0.962355136871
          if ( mean_col_coverage <= 0.997792363167 ) {
            if ( mean_col_coverage <= 0.534271121025 ) {
              if ( median_col_coverage <= 0.00556329311803 ) {
                if ( max_col_coverage <= 0.232224017382 ) {
                  if ( min_col_coverage <= 0.00294551439583 ) {
                    if ( median_col_support <= 0.975499987602 ) {
                      return 0.100827262182 < maxgini;
                    }
                    else {  // if median_col_support > 0.975499987602
                      return 0.0372151485508 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00294551439583
                    if ( min_col_support <= 0.945999979973 ) {
                      return 0.035223460562 < maxgini;
                    }
                    else {  // if min_col_support > 0.945999979973
                      return false;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.232224017382
                  if ( median_col_support <= 0.964499950409 ) {
                    if ( median_col_coverage <= 0.00353983417153 ) {
                      return 0.20371616376 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00353983417153
                      return 0.329063274508 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.964499950409
                    if ( max_col_coverage <= 0.23227635026 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.23227635026
                      return 0.0957790442597 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.00556329311803
                if ( median_col_coverage <= 0.320484369993 ) {
                  if ( median_col_coverage <= 0.0812681019306 ) {
                    if ( median_col_coverage <= 0.0795933827758 ) {
                      return 0.0255521135611 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0795933827758
                      return 0.0165037115622 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0812681019306
                    if ( min_col_support <= 0.93850004673 ) {
                      return 0.0275643827061 < maxgini;
                    }
                    else {  // if min_col_support > 0.93850004673
                      return 0.0331080293136 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.320484369993
                  if ( mean_col_support <= 0.974147081375 ) {
                    if ( mean_col_coverage <= 0.53414696455 ) {
                      return 0.0287790782875 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.53414696455
                      return 0.1128 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.974147081375
                    if ( min_col_support <= 0.903499960899 ) {
                      return 0.0172987120338 < maxgini;
                    }
                    else {  // if min_col_support > 0.903499960899
                      return 0.025195694145 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.534271121025
              if ( mean_col_coverage <= 0.973666727543 ) {
                if ( mean_col_coverage <= 0.635700345039 ) {
                  if ( mean_col_coverage <= 0.635698497295 ) {
                    if ( median_col_support <= 0.965499997139 ) {
                      return 0.0226414140132 < maxgini;
                    }
                    else {  // if median_col_support > 0.965499997139
                      return 0.0152761336945 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.635698497295
                    if ( max_col_coverage <= 0.705172419548 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.705172419548
                      return 0.48 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.635700345039
                  if ( median_col_support <= 0.993499994278 ) {
                    if ( median_col_support <= 0.988499999046 ) {
                      return 0.0112816657009 < maxgini;
                    }
                    else {  // if median_col_support > 0.988499999046
                      return 0.0338108338785 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.993499994278
                    if ( max_col_coverage <= 0.89499437809 ) {
                      return 0.0586833775442 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.89499437809
                      return 0.255974638517 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.973666727543
                if ( min_col_support <= 0.653499960899 ) {
                  if ( median_col_support <= 0.992499947548 ) {
                    if ( min_col_coverage <= 0.951753258705 ) {
                      return 0.197530864198 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.951753258705
                      return false;
                    }
                  }
                  else {  // if median_col_support > 0.992499947548
                    if ( mean_col_support <= 0.975147128105 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.975147128105
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.653499960899
                  if ( max_col_coverage <= 0.989346027374 ) {
                    if ( mean_col_coverage <= 0.98044449091 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.98044449091
                      return false;
                    }
                  }
                  else {  // if max_col_coverage > 0.989346027374
                    if ( mean_col_support <= 0.970852971077 ) {
                      return 0.0867768595041 < maxgini;
                    }
                    else {  // if mean_col_support > 0.970852971077
                      return 0.0170441950394 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.997792363167
            if ( min_col_coverage <= 0.990902304649 ) {
              if ( mean_col_support <= 0.969852924347 ) {
                return false;
              }
              else {  // if mean_col_support > 0.969852924347
                return 0.0 < maxgini;
              }
            }
            else {  // if min_col_coverage > 0.990902304649
              if ( mean_col_coverage <= 0.998040258884 ) {
                return false;
              }
              else {  // if mean_col_coverage > 0.998040258884
                if ( min_col_coverage <= 0.996844947338 ) {
                  if ( median_col_support <= 0.997500002384 ) {
                    if ( median_col_coverage <= 0.996441245079 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.996441245079
                      return 0.444444444444 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.997500002384
                    return false;
                  }
                }
                else {  // if min_col_coverage > 0.996844947338
                  if ( median_col_support <= 0.991500020027 ) {
                    if ( mean_col_support <= 0.972823500633 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.972823500633
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.991500020027
                    if ( min_col_coverage <= 0.998626351357 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.998626351357
                      return false;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.985891222954
        if ( mean_col_coverage <= 0.522721767426 ) {
          if ( mean_col_coverage <= 0.357107460499 ) {
            if ( mean_col_support <= 0.992279171944 ) {
              if ( min_col_support <= 0.946500003338 ) {
                if ( min_col_coverage <= 0.00278164655901 ) {
                  if ( min_col_support <= 0.895500004292 ) {
                    if ( max_col_coverage <= 0.141133204103 ) {
                      return 0.0241655678067 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.141133204103
                      return 0.00187441259605 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.895500004292
                    if ( median_col_coverage <= 0.0697964429855 ) {
                      return 0.0397676418367 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0697964429855
                      return 0.287334593573 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.00278164655901
                  if ( median_col_support <= 0.949499964714 ) {
                    if ( mean_col_coverage <= 0.12042209506 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.12042209506
                      return false;
                    }
                  }
                  else {  // if median_col_support > 0.949499964714
                    if ( min_col_coverage <= 0.0649501308799 ) {
                      return 0.0134945706791 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0649501308799
                      return 0.0178908353286 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.946500003338
                if ( max_col_coverage <= 0.538863778114 ) {
                  if ( min_col_coverage <= 0.332378387451 ) {
                    if ( min_col_support <= 0.96749997139 ) {
                      return 0.022958539461 < maxgini;
                    }
                    else {  // if min_col_support > 0.96749997139
                      return 0.0283490007117 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.332378387451
                    if ( median_col_coverage <= 0.33668076992 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.33668076992
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.538863778114
                  if ( mean_col_coverage <= 0.349092483521 ) {
                    if ( mean_col_support <= 0.991911709309 ) {
                      return 0.222148760331 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991911709309
                      return 0.497041420118 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.349092483521
                    if ( mean_col_support <= 0.987264633179 ) {
                      return 0.277777777778 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987264633179
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.992279171944
              if ( median_col_support <= 0.993499994278 ) {
                if ( max_col_coverage <= 0.448730349541 ) {
                  if ( mean_col_support <= 0.995782375336 ) {
                    if ( min_col_support <= 0.972499966621 ) {
                      return 0.0146323797865 < maxgini;
                    }
                    else {  // if min_col_support > 0.972499966621
                      return 0.0215257185929 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.995782375336
                    if ( min_col_coverage <= 0.225319176912 ) {
                      return 0.0132514559561 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.225319176912
                      return 0.0184329678604 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.448730349541
                  if ( median_col_coverage <= 0.283485621214 ) {
                    if ( min_col_coverage <= 0.267747044563 ) {
                      return 0.00822944933752 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.267747044563
                      return 0.0600925425155 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.283485621214
                    if ( median_col_coverage <= 0.28349712491 ) {
                      return 0.18 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.28349712491
                      return 0.0157765646774 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.993499994278
                if ( min_col_support <= 0.983500003815 ) {
                  if ( mean_col_support <= 0.996834874153 ) {
                    if ( min_col_coverage <= 0.22781419754 ) {
                      return 0.00579228522862 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.22781419754
                      return 0.0112592413092 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.996834874153
                    if ( median_col_coverage <= 0.00283688516356 ) {
                      return 0.0475907198096 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00283688516356
                      return 0.00455626392203 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.983500003815
                  if ( mean_col_support <= 0.998029470444 ) {
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.0162390690703 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.00848847234462 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.998029470444
                    if ( mean_col_coverage <= 0.326476484537 ) {
                      return 0.0032764913898 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.326476484537
                      return 0.00966160806862 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.357107460499
            if ( min_col_support <= 0.985499978065 ) {
              if ( min_col_support <= 0.976500034332 ) {
                if ( max_col_coverage <= 0.578215003014 ) {
                  if ( median_col_support <= 0.985499978065 ) {
                    if ( min_col_coverage <= 0.178269743919 ) {
                      return 0.249131944444 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.178269743919
                      return 0.0201314337932 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.985499978065
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.014240339923 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.0100474503729 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.578215003014
                  if ( mean_col_support <= 0.992088198662 ) {
                    if ( median_col_coverage <= 0.501552820206 ) {
                      return 0.0161384720782 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.501552820206
                      return 0.244897959184 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992088198662
                    if ( mean_col_coverage <= 0.421377122402 ) {
                      return 0.0343875 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.421377122402
                      return 0.00932346080998 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.976500034332
                if ( median_col_support <= 0.991500020027 ) {
                  if ( max_col_coverage <= 0.465457558632 ) {
                    if ( max_col_coverage <= 0.46440154314 ) {
                      return 0.0192528872002 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.46440154314
                      return 0.0325725181549 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.465457558632
                    if ( median_col_support <= 0.986500024796 ) {
                      return 0.020511253985 < maxgini;
                    }
                    else {  // if median_col_support > 0.986500024796
                      return 0.0151526981418 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.991500020027
                  if ( mean_col_coverage <= 0.522720873356 ) {
                    if ( min_col_coverage <= 0.253825962543 ) {
                      return 0.00142602422943 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.253825962543
                      return 0.00969917944457 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.522720873356
                    if ( max_col_coverage <= 0.606223165989 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.606223165989
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.985499978065
              if ( median_col_support <= 0.993499994278 ) {
                if ( min_col_coverage <= 0.20892521739 ) {
                  if ( min_col_support <= 0.986500024796 ) {
                    if ( median_col_coverage <= 0.290619492531 ) {
                      return 0.244897959184 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.290619492531
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.986500024796
                    return 0.0 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.20892521739
                  if ( min_col_coverage <= 0.258800983429 ) {
                    if ( max_col_coverage <= 0.44519045949 ) {
                      return 0.0327777777778 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.44519045949
                      return 0.00264900196301 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.258800983429
                    if ( min_col_support <= 0.992499947548 ) {
                      return 0.0137322352351 < maxgini;
                    }
                    else {  // if min_col_support > 0.992499947548
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.993499994278
                if ( max_col_coverage <= 0.562134444714 ) {
                  if ( median_col_support <= 0.996500015259 ) {
                    if ( min_col_support <= 0.995499968529 ) {
                      return 0.00958249333204 < maxgini;
                    }
                    else {  // if min_col_support > 0.995499968529
                      return 0.0607766243752 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.996500015259
                    if ( mean_col_support <= 0.999088168144 ) {
                      return 0.00688264834357 < maxgini;
                    }
                    else {  // if mean_col_support > 0.999088168144
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.562134444714
                  if ( mean_col_coverage <= 0.479104965925 ) {
                    if ( min_col_coverage <= 0.41160517931 ) {
                      return 0.0037957851624 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.41160517931
                      return 0.0280317153703 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.479104965925
                    if ( mean_col_coverage <= 0.47910618782 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.47910618782
                      return 0.00731927454666 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.522721767426
          if ( mean_col_coverage <= 0.678763866425 ) {
            if ( min_col_coverage <= 0.494957983494 ) {
              if ( max_col_coverage <= 0.617237925529 ) {
                if ( min_col_support <= 0.988499999046 ) {
                  if ( min_col_support <= 0.943500041962 ) {
                    if ( min_col_coverage <= 0.449948847294 ) {
                      return 0.00126023894448 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.449948847294
                      return 0.00750659586277 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.943500041962
                    if ( mean_col_coverage <= 0.523802042007 ) {
                      return 0.00715506875283 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.523802042007
                      return 0.0113988596809 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.988499999046
                  if ( mean_col_coverage <= 0.529920220375 ) {
                    if ( mean_col_support <= 0.995029389858 ) {
                      return 0.045846282704 < maxgini;
                    }
                    else {  // if mean_col_support > 0.995029389858
                      return 0.00736590307008 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.529920220375
                    if ( max_col_coverage <= 0.617229700089 ) {
                      return 0.00417571029865 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.617229700089
                      return 0.1171875 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.617237925529
                if ( median_col_support <= 0.990499973297 ) {
                  if ( min_col_coverage <= 0.494335353374 ) {
                    if ( median_col_support <= 0.983500003815 ) {
                      return 0.0158327969715 < maxgini;
                    }
                    else {  // if median_col_support > 0.983500003815
                      return 0.0105556528494 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.494335353374
                    if ( median_col_coverage <= 0.525440990925 ) {
                      return 0.0286735834196 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.525440990925
                      return 0.00725191120808 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.990499973297
                  if ( min_col_coverage <= 0.198479533195 ) {
                    if ( max_col_coverage <= 0.949955582619 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.949955582619
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.198479533195
                    if ( min_col_coverage <= 0.458260834217 ) {
                      return 0.0057162357047 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.458260834217
                      return 0.00466989750818 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.494957983494
              if ( mean_col_support <= 0.994029402733 ) {
                if ( median_col_support <= 0.984500050545 ) {
                  if ( mean_col_support <= 0.991441130638 ) {
                    if ( min_col_support <= 0.959499955177 ) {
                      return 0.00928564683434 < maxgini;
                    }
                    else {  // if min_col_support > 0.959499955177
                      return 0.0150752710023 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.991441130638
                    if ( mean_col_support <= 0.991500020027 ) {
                      return 0.029733001068 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991500020027
                      return 0.0152161325167 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.984500050545
                  if ( min_col_support <= 0.964499950409 ) {
                    if ( max_col_support <= 0.996500015259 ) {
                      return 0.0768 < maxgini;
                    }
                    else {  // if max_col_support > 0.996500015259
                      return 0.00503533126913 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.964499950409
                    if ( max_col_support <= 0.995499968529 ) {
                      return 0.205516495403 < maxgini;
                    }
                    else {  // if max_col_support > 0.995499968529
                      return 0.00896861980011 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.994029402733
                if ( median_col_support <= 0.993499994278 ) {
                  if ( max_col_coverage <= 0.565434455872 ) {
                    if ( median_col_coverage <= 0.521633327007 ) {
                      return 0.32 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.521633327007
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.565434455872
                    if ( max_col_coverage <= 0.785648226738 ) {
                      return 0.00537814746085 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.785648226738
                      return 0.0108899539875 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( min_col_support <= 0.991500020027 ) {
                    if ( median_col_coverage <= 0.49852091074 ) {
                      return 0.0997229916898 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.49852091074
                      return 0.0029216195459 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.991500020027
                    if ( max_col_coverage <= 0.85873901844 ) {
                      return 0.00208512192157 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.85873901844
                      return 0.46875 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.678763866425
            if ( mean_col_support <= 0.995088219643 ) {
              if ( mean_col_coverage <= 0.760866045952 ) {
                if ( median_col_support <= 0.989500045776 ) {
                  if ( mean_col_support <= 0.987205862999 ) {
                    if ( mean_col_support <= 0.986088275909 ) {
                      return 0.00265815584487 < maxgini;
                    }
                    else {  // if mean_col_support > 0.986088275909
                      return 0.0102211953462 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.987205862999
                    if ( min_col_support <= 0.960500001907 ) {
                      return 0.00416824920274 < maxgini;
                    }
                    else {  // if min_col_support > 0.960500001907
                      return 0.00771055092482 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.989500045776
                  if ( median_col_coverage <= 0.651528894901 ) {
                    if ( min_col_coverage <= 0.646700382233 ) {
                      return 0.00445439952584 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.646700382233
                      return 0.277777777778 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.651528894901
                    if ( min_col_coverage <= 0.628501713276 ) {
                      return 0.00173802996974 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.628501713276
                      return 0.00301832503244 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.760866045952
                if ( min_col_coverage <= 0.983816564083 ) {
                  if ( min_col_support <= 0.979499995708 ) {
                    if ( mean_col_support <= 0.991558790207 ) {
                      return 0.00326046626718 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991558790207
                      return 0.00116208834853 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.979499995708
                    if ( min_col_support <= 0.986500024796 ) {
                      return 0.00441978947285 < maxgini;
                    }
                    else {  // if min_col_support > 0.986500024796
                      return 0.00119248690538 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.983816564083
                  if ( min_col_coverage <= 0.983849227428 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.983849227428
                    if ( mean_col_coverage <= 0.998193502426 ) {
                      return 0.00904958677686 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.998193502426
                      return 0.0870910378927 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.995088219643
              if ( max_col_coverage <= 0.809556424618 ) {
                if ( mean_col_coverage <= 0.696396350861 ) {
                  if ( mean_col_coverage <= 0.696388363838 ) {
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.095 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.00236695167593 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.696388363838
                    if ( mean_col_support <= 0.995499968529 ) {
                      return 0.336734693878 < maxgini;
                    }
                    else {  // if mean_col_support > 0.995499968529
                      return 0.0149245293685 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.696396350861
                  if ( mean_col_support <= 0.996264696121 ) {
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.0719713512097 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.00273048359387 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.996264696121
                    if ( min_col_coverage <= 0.668521106243 ) {
                      return 0.00123793107415 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.668521106243
                      return 0.00332202001027 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.809556424618
                if ( min_col_coverage <= 0.627465963364 ) {
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( max_col_coverage <= 0.874782204628 ) {
                      return 0.0113632653061 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.874782204628
                      return 0.0798611111111 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( mean_col_coverage <= 0.790160298347 ) {
                      return 0.00198121319726 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.790160298347
                      return 0.0349539646154 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.627465963364
                  if ( min_col_support <= 0.991500020027 ) {
                    if ( mean_col_coverage <= 0.906330406666 ) {
                      return 0.000927239991156 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.906330406666
                      return 0.000272438075377 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.991500020027
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.000650438977516 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return 0.000131097966763 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    bool shouldCorrect1(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( max_col_coverage <= 0.576480031013 ) {
        if ( max_col_coverage <= 0.416897565126 ) {
          if ( mean_col_support <= 0.97357738018 ) {
            if ( min_col_coverage <= 0.006033196114 ) {
              if ( min_col_coverage <= 0.003016598057 ) {
                if ( mean_col_coverage <= 0.0451844781637 ) {
                  if ( min_col_coverage <= 0.00167978345416 ) {
                    if ( min_col_support <= 0.698500037193 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_support > 0.698500037193
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.00167978345416
                    if ( max_col_coverage <= 0.121096074581 ) {
                      return 0.0285654274312 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.121096074581
                      return 0.0628861554491 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.0451844781637
                  if ( min_col_support <= 0.716500043869 ) {
                    if ( median_col_support <= 0.944499969482 ) {
                      return 0.0715683225631 < maxgini;
                    }
                    else {  // if median_col_support > 0.944499969482
                      return 0.0388730317201 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.716500043869
                    if ( median_col_coverage <= 0.00298954336904 ) {
                      return 0.113283546628 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00298954336904
                      return 0.0692659437717 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.003016598057
                if ( median_col_support <= 0.949499964714 ) {
                  if ( median_col_coverage <= 0.0062402645126 ) {
                    if ( median_col_coverage <= 0.00535476207733 ) {
                      return 0.0674986442813 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00535476207733
                      return 0.0894140738818 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0062402645126
                    if ( median_col_coverage <= 0.0480624884367 ) {
                      return 0.0432905085212 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0480624884367
                      return 0.111575311244 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.949499964714
                  if ( median_col_coverage <= 0.0458865985274 ) {
                    if ( max_col_coverage <= 0.254153013229 ) {
                      return 0.0278802646479 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.254153013229
                      return 0.0819569369702 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0458865985274
                    if ( min_col_support <= 0.724500000477 ) {
                      return 0.0208465322682 < maxgini;
                    }
                    else {  // if min_col_support > 0.724500000477
                      return 0.11065807636 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.006033196114
              if ( min_col_support <= 0.654500007629 ) {
                if ( max_col_support <= 0.991500020027 ) {
                  if ( median_col_support <= 0.547500014305 ) {
                    if ( max_col_support <= 0.983500003815 ) {
                      return 0.00617858405037 < maxgini;
                    }
                    else {  // if max_col_support > 0.983500003815
                      return 0.0265401992812 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.547500014305
                    if ( max_col_support <= 0.979499995708 ) {
                      return 0.0160941517362 < maxgini;
                    }
                    else {  // if max_col_support > 0.979499995708
                      return 0.0309091857943 < maxgini;
                    }
                  }
                }
                else {  // if max_col_support > 0.991500020027
                  if ( mean_col_coverage <= 0.383326530457 ) {
                    if ( min_col_support <= 0.439500004053 ) {
                      return 0.0321014567248 < maxgini;
                    }
                    else {  // if min_col_support > 0.439500004053
                      return 0.0473820325047 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.383326530457
                    if ( median_col_support <= 0.942000031471 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.942000031471
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.654500007629
                if ( min_col_coverage <= 0.0877628847957 ) {
                  if ( max_col_coverage <= 0.412890344858 ) {
                    if ( max_col_coverage <= 0.371982455254 ) {
                      return 0.0307433467014 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.371982455254
                      return 0.0639214117019 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.412890344858
                    if ( min_col_coverage <= 0.032849252224 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.032849252224
                      return 0.133914337337 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.0877628847957
                  if ( min_col_support <= 0.960999965668 ) {
                    if ( median_col_support <= 0.935500025749 ) {
                      return 0.0411534531111 < maxgini;
                    }
                    else {  // if median_col_support > 0.935500025749
                      return 0.0319448773351 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.960999965668
                    return false;
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.97357738018
            if ( median_col_coverage <= 0.00578872393817 ) {
              if ( mean_col_support <= 0.987710118294 ) {
                if ( median_col_support <= 0.978500008583 ) {
                  if ( min_col_coverage <= 0.00577201787382 ) {
                    if ( max_col_coverage <= 0.232525572181 ) {
                      return 0.0510667499875 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.232525572181
                      return 0.204764542936 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00577201787382
                    if ( median_col_support <= 0.965499997139 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.965499997139
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.978500008583
                  if ( min_col_coverage <= 0.00246609514579 ) {
                    if ( min_col_coverage <= 0.00245399144478 ) {
                      return 0.087768 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00245399144478
                      return 0.356949702542 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00246609514579
                    if ( max_col_coverage <= 0.35060852766 ) {
                      return 0.0237236140388 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.35060852766
                      return 0.48 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.987710118294
                if ( min_col_support <= 0.802500009537 ) {
                  if ( mean_col_coverage <= 0.0581699386239 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if mean_col_coverage > 0.0581699386239
                    if ( mean_col_support <= 0.988205909729 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.988205909729
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.802500009537
                  if ( mean_col_support <= 0.990300893784 ) {
                    if ( median_col_support <= 0.964499950409 ) {
                      return 0.263671875 < maxgini;
                    }
                    else {  // if median_col_support > 0.964499950409
                      return 0.0242695510204 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.990300893784
                    if ( min_col_coverage <= 0.0023781247437 ) {
                      return 0.0361323346757 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0023781247437
                      return 0.00896360461955 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.00578872393817
              if ( min_col_support <= 0.961500048637 ) {
                if ( mean_col_support <= 0.98602938652 ) {
                  if ( mean_col_support <= 0.980639576912 ) {
                    if ( min_col_support <= 0.905499994755 ) {
                      return 0.0245038496953 < maxgini;
                    }
                    else {  // if min_col_support > 0.905499994755
                      return 0.0337255060518 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.980639576912
                    if ( min_col_coverage <= 0.201581090689 ) {
                      return 0.0249425222407 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.201581090689
                      return 0.0314789644132 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.98602938652
                  if ( min_col_coverage <= 0.0604312606156 ) {
                    if ( max_col_coverage <= 0.316647678614 ) {
                      return 0.0105025093494 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.316647678614
                      return 0.0237082527522 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0604312606156
                    if ( max_col_coverage <= 0.287116676569 ) {
                      return 0.0257171442701 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.287116676569
                      return 0.01905567686 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.961500048637
                if ( mean_col_support <= 0.994767069817 ) {
                  if ( mean_col_support <= 0.976000070572 ) {
                    if ( mean_col_coverage <= 0.296551913023 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.296551913023
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.976000070572
                    if ( mean_col_support <= 0.988355100155 ) {
                      return 0.033534311659 < maxgini;
                    }
                    else {  // if mean_col_support > 0.988355100155
                      return 0.0229851593901 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.994767069817
                  if ( min_col_support <= 0.978500008583 ) {
                    if ( mean_col_support <= 0.995899558067 ) {
                      return 0.0125720307126 < maxgini;
                    }
                    else {  // if mean_col_support > 0.995899558067
                      return 0.00734057438845 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.978500008583
                    if ( median_col_coverage <= 0.253508806229 ) {
                      return 0.0135973364169 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.253508806229
                      return 0.0197545979383 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.416897565126
          if ( mean_col_support <= 0.983264803886 ) {
            if ( min_col_coverage <= 0.0785348638892 ) {
              if ( mean_col_support <= 0.975499987602 ) {
                if ( min_col_support <= 0.909999966621 ) {
                  if ( median_col_support <= 0.858999967575 ) {
                    if ( mean_col_support <= 0.856970608234 ) {
                      return 0.091292925502 < maxgini;
                    }
                    else {  // if mean_col_support > 0.856970608234
                      return 0.00526312125368 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.858999967575
                    if ( min_col_coverage <= 0.0783025324345 ) {
                      return 0.116321245327 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0783025324345
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.909999966621
                  if ( min_col_coverage <= 0.0740761458874 ) {
                    if ( max_col_coverage <= 0.468230009079 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.468230009079
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.0740761458874
                    return false;
                  }
                }
              }
              else {  // if mean_col_support > 0.975499987602
                if ( mean_col_support <= 0.979705870152 ) {
                  if ( mean_col_coverage <= 0.236416816711 ) {
                    if ( mean_col_support <= 0.979398190975 ) {
                      return 0.0886965927528 < maxgini;
                    }
                    else {  // if mean_col_support > 0.979398190975
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.236416816711
                    if ( mean_col_coverage <= 0.242276906967 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.242276906967
                      return 0.470868014269 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.979705870152
                  if ( min_col_support <= 0.823500037193 ) {
                    if ( max_col_coverage <= 0.460302352905 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.460302352905
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.823500037193
                    if ( min_col_support <= 0.916000008583 ) {
                      return 0.0614764424288 < maxgini;
                    }
                    else {  // if min_col_support > 0.916000008583
                      return 0.209750566893 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.0785348638892
              if ( mean_col_support <= 0.947970569134 ) {
                if ( median_col_support <= 0.978500008583 ) {
                  if ( median_col_coverage <= 0.300287425518 ) {
                    if ( max_col_coverage <= 0.462049901485 ) {
                      return 0.0334205833198 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.462049901485
                      return 0.0268121850938 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.300287425518
                    if ( max_col_support <= 0.99950003624 ) {
                      return 0.0315714945944 < maxgini;
                    }
                    else {  // if max_col_support > 0.99950003624
                      return 0.0449477499705 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.978500008583
                  if ( min_col_support <= 0.705000042915 ) {
                    if ( median_col_coverage <= 0.423927843571 ) {
                      return 0.129712693598 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.423927843571
                      return 0.358533272974 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.705000042915
                    return false;
                  }
                }
              }
              else {  // if mean_col_support > 0.947970569134
                if ( mean_col_support <= 0.973911762238 ) {
                  if ( min_col_support <= 0.87450003624 ) {
                    if ( median_col_support <= 0.946500003338 ) {
                      return 0.0314254140145 < maxgini;
                    }
                    else {  // if median_col_support > 0.946500003338
                      return 0.0222931965173 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.87450003624
                    if ( max_col_coverage <= 0.536273479462 ) {
                      return 0.0349539015441 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.536273479462
                      return 0.0303157526976 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.973911762238
                  if ( min_col_support <= 0.932500004768 ) {
                    if ( median_col_support <= 0.999000012875 ) {
                      return 0.022203557672 < maxgini;
                    }
                    else {  // if median_col_support > 0.999000012875
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.932500004768
                    if ( mean_col_support <= 0.980441153049 ) {
                      return 0.0362473913892 < maxgini;
                    }
                    else {  // if mean_col_support > 0.980441153049
                      return 0.0287630316333 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.983264803886
            if ( min_col_coverage <= 0.12369312346 ) {
              if ( median_col_support <= 0.979499995708 ) {
                if ( max_col_coverage <= 0.470012247562 ) {
                  if ( max_col_coverage <= 0.425043553114 ) {
                    if ( median_col_coverage <= 0.199841946363 ) {
                      return 0.0113632653061 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.199841946363
                      return false;
                    }
                  }
                  else {  // if max_col_coverage > 0.425043553114
                    if ( mean_col_coverage <= 0.269526958466 ) {
                      return 0.061050401885 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.269526958466
                      return 0.134262465374 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.470012247562
                  if ( median_col_coverage <= 0.177922934294 ) {
                    if ( mean_col_support <= 0.983676433563 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983676433563
                      return 0.14201183432 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.177922934294
                    if ( max_col_coverage <= 0.470084279776 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.470084279776
                      return 0.327538422777 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.979499995708
                if ( median_col_coverage <= 0.17524215579 ) {
                  if ( min_col_coverage <= 0.113569036126 ) {
                    if ( mean_col_coverage <= 0.248534411192 ) {
                      return 0.0198787116337 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.248534411192
                      return 0.00122774662087 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.113569036126
                    if ( min_col_support <= 0.942499995232 ) {
                      return 0.1128 < maxgini;
                    }
                    else {  // if min_col_support > 0.942499995232
                      return 0.0152954585111 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.17524215579
                  if ( median_col_support <= 0.982499957085 ) {
                    if ( mean_col_coverage <= 0.274159729481 ) {
                      return 0.231111111111 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.274159729481
                      return 0.0491490713933 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.982499957085
                    if ( mean_col_support <= 0.995205938816 ) {
                      return 0.032108233761 < maxgini;
                    }
                    else {  // if mean_col_support > 0.995205938816
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.12369312346
              if ( min_col_support <= 0.976500034332 ) {
                if ( median_col_support <= 0.982499957085 ) {
                  if ( min_col_support <= 0.946500003338 ) {
                    if ( median_col_support <= 0.971500039101 ) {
                      return 0.0266490155209 < maxgini;
                    }
                    else {  // if median_col_support > 0.971500039101
                      return 0.017422175097 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.946500003338
                    if ( min_col_support <= 0.959499955177 ) {
                      return 0.0226848656666 < maxgini;
                    }
                    else {  // if min_col_support > 0.959499955177
                      return 0.0255313259303 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.982499957085
                  if ( mean_col_support <= 0.99426472187 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.01581412692 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.0109720549375 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.99426472187
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.00941994138192 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.00445223325651 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.976500034332
                if ( median_col_support <= 0.992499947548 ) {
                  if ( max_col_coverage <= 0.454222202301 ) {
                    if ( min_col_coverage <= 0.219936311245 ) {
                      return 0.0106650673361 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.219936311245
                      return 0.0214335930424 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.454222202301
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.0188815941363 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.0151638823415 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.992499947548
                  if ( median_col_support <= 0.996500015259 ) {
                    if ( min_col_coverage <= 0.21542981267 ) {
                      return 0.00390242410865 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.21542981267
                      return 0.010511242819 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.996500015259
                    if ( mean_col_coverage <= 0.507448792458 ) {
                      return 0.00495176366396 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.507448792458
                      return 0.0174468491954 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if max_col_coverage > 0.576480031013
        if ( mean_col_support <= 0.981970608234 ) {
          if ( median_col_coverage <= 0.990882158279 ) {
            if ( median_col_support <= 0.993499994278 ) {
              if ( max_col_coverage <= 0.681666493416 ) {
                if ( median_col_coverage <= 0.429871141911 ) {
                  if ( min_col_support <= 0.96749997139 ) {
                    if ( median_col_support <= 0.940500020981 ) {
                      return 0.0281181327359 < maxgini;
                    }
                    else {  // if median_col_support > 0.940500020981
                      return 0.0200235160544 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.96749997139
                    if ( max_col_coverage <= 0.589520394802 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.589520394802
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.429871141911
                  if ( mean_col_coverage <= 0.462300896645 ) {
                    return false;
                  }
                  else {  // if mean_col_coverage > 0.462300896645
                    if ( median_col_coverage <= 0.540447115898 ) {
                      return 0.0277035710212 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.540447115898
                      return 0.0222409551812 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.681666493416
                if ( min_col_coverage <= 0.965689182281 ) {
                  if ( min_col_coverage <= 0.813330769539 ) {
                    if ( min_col_support <= 0.590499997139 ) {
                      return 0.0729996735236 < maxgini;
                    }
                    else {  // if min_col_support > 0.590499997139
                      return 0.0171284591557 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.813330769539
                    if ( max_col_coverage <= 0.864568352699 ) {
                      return 0.426035502959 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.864568352699
                      return 0.0440269067581 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.965689182281
                  if ( mean_col_coverage <= 0.978129506111 ) {
                    if ( min_col_support <= 0.771000027657 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.771000027657
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.978129506111
                    if ( median_col_coverage <= 0.972993373871 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.972993373871
                      return 0.270992755841 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.993499994278
              if ( median_col_coverage <= 0.762602686882 ) {
                if ( min_col_support <= 0.543500006199 ) {
                  if ( mean_col_coverage <= 0.558673620224 ) {
                    if ( max_col_coverage <= 0.592210412025 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.592210412025
                      return 0.124444444444 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.558673620224
                    if ( mean_col_support <= 0.970646977425 ) {
                      return 0.499929138322 < maxgini;
                    }
                    else {  // if mean_col_support > 0.970646977425
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.543500006199
                  if ( mean_col_support <= 0.974088311195 ) {
                    if ( min_col_support <= 0.569499969482 ) {
                      return 0.314545762903 < maxgini;
                    }
                    else {  // if min_col_support > 0.569499969482
                      return 0.0961811263372 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.974088311195
                    if ( min_col_coverage <= 0.693159103394 ) {
                      return 0.020323060771 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.693159103394
                      return 0.174190738859 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.762602686882
                if ( median_col_support <= 0.996500015259 ) {
                  if ( max_col_support <= 0.99950003624 ) {
                    return false;
                  }
                  else {  // if max_col_support > 0.99950003624
                    if ( min_col_support <= 0.606999993324 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.606999993324
                      return 0.18 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.996500015259
                  if ( mean_col_coverage <= 0.974360585213 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.494243769456 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.974360585213
                    if ( min_col_coverage <= 0.910184204578 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.910184204578
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.990882158279
            if ( min_col_support <= 0.698500037193 ) {
              if ( min_col_coverage <= 0.987663626671 ) {
                return false;
              }
              else {  // if min_col_coverage > 0.987663626671
                if ( mean_col_coverage <= 0.994536697865 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_coverage > 0.994536697865
                  if ( mean_col_coverage <= 0.997343838215 ) {
                    if ( min_col_support <= 0.650499999523 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.650499999523
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.997343838215
                    if ( mean_col_support <= 0.977264761925 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.977264761925
                      return 0.375 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.698500037193
              if ( median_col_support <= 0.995499968529 ) {
                if ( min_col_coverage <= 0.997002005577 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_coverage > 0.997002005577
                  if ( median_col_coverage <= 0.998599410057 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.998599410057
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if median_col_support > 0.995499968529
                if ( mean_col_coverage <= 0.999374270439 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_coverage > 0.999374270439
                  return false;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.981970608234
          if ( median_col_coverage <= 0.540131926537 ) {
            if ( mean_col_coverage <= 0.535330176353 ) {
              if ( min_col_support <= 0.980499982834 ) {
                if ( mean_col_support <= 0.992852926254 ) {
                  if ( min_col_support <= 0.942499995232 ) {
                    if ( max_col_coverage <= 0.793682932854 ) {
                      return 0.0124839969477 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.793682932854
                      return 0.197530864198 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.942499995232
                    if ( min_col_coverage <= 0.226735398173 ) {
                      return 0.0933912627551 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.226735398173
                      return 0.0173340825467 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.992852926254
                  if ( min_col_support <= 0.970499992371 ) {
                    if ( min_col_support <= 0.940500020981 ) {
                      return 0.0122862199614 < maxgini;
                    }
                    else {  // if min_col_support > 0.940500020981
                      return 0.00616218419422 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.970499992371
                    if ( median_col_coverage <= 0.50577545166 ) {
                      return 0.00941093923723 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.50577545166
                      return 0.0575034293553 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.980499982834
                if ( mean_col_support <= 0.99638235569 ) {
                  if ( mean_col_coverage <= 0.535329818726 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.0158353049268 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.0108873301423 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.535329818726
                    if ( min_col_support <= 0.986000001431 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_support > 0.986000001431
                      return 0.444444444444 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.99638235569
                  if ( mean_col_coverage <= 0.535315394402 ) {
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.00814066548476 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.00512639568052 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.535315394402
                    if ( mean_col_coverage <= 0.535316228867 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.535316228867
                      return 0.021736505253 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.535330176353
              if ( median_col_support <= 0.990499973297 ) {
                if ( max_col_support <= 0.997500002384 ) {
                  if ( min_col_support <= 0.977499961853 ) {
                    if ( min_col_support <= 0.966500043869 ) {
                      return 0.0240593652134 < maxgini;
                    }
                    else {  // if min_col_support > 0.966500043869
                      return 0.00761686364543 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.977499961853
                    if ( min_col_coverage <= 0.430271327496 ) {
                      return 0.46875 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.430271327496
                      return 0.0347381706879 < maxgini;
                    }
                  }
                }
                else {  // if max_col_support > 0.997500002384
                  if ( min_col_support <= 0.918500006199 ) {
                    if ( median_col_coverage <= 0.492915630341 ) {
                      return 0.0115987686345 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.492915630341
                      return 0.00660839105372 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.918500006199
                    if ( median_col_support <= 0.986500024796 ) {
                      return 0.0142281945192 < maxgini;
                    }
                    else {  // if median_col_support > 0.986500024796
                      return 0.00950569082636 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.990499973297
                if ( max_col_coverage <= 0.654592752457 ) {
                  if ( median_col_coverage <= 0.444903850555 ) {
                    if ( min_col_coverage <= 0.409520149231 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.409520149231
                      return 0.2688 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.444903850555
                    if ( mean_col_support <= 0.99644112587 ) {
                      return 0.00601609849502 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99644112587
                      return 0.00418723143437 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.654592752457
                  if ( mean_col_support <= 0.997441112995 ) {
                    if ( median_col_coverage <= 0.342265427113 ) {
                      return 0.277777777778 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.342265427113
                      return 0.00463201846109 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.997441112995
                    if ( median_col_coverage <= 0.538955926895 ) {
                      return 0.00203765376125 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.538955926895
                      return 0.00446526007136 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.540131926537
            if ( max_col_coverage <= 0.791143655777 ) {
              if ( min_col_support <= 0.982499957085 ) {
                if ( median_col_support <= 0.988499999046 ) {
                  if ( min_col_support <= 0.964499950409 ) {
                    if ( mean_col_support <= 0.987911820412 ) {
                      return 0.0111195958255 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987911820412
                      return 0.00703912301549 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.964499950409
                    if ( median_col_support <= 0.984500050545 ) {
                      return 0.0155992028715 < maxgini;
                    }
                    else {  // if median_col_support > 0.984500050545
                      return 0.00979980417288 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.988499999046
                  if ( min_col_coverage <= 0.597371339798 ) {
                    if ( mean_col_coverage <= 0.591314375401 ) {
                      return 0.00222568163548 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.591314375401
                      return 0.00442813689275 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.597371339798
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.00376676162725 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.00149346525186 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.982499957085
                if ( median_col_coverage <= 0.603073596954 ) {
                  if ( median_col_support <= 0.992499947548 ) {
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.00937497234139 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.00569651172229 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.992499947548
                    if ( min_col_coverage <= 0.60163807869 ) {
                      return 0.00301020808695 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.60163807869
                      return 0.42 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.603073596954
                  if ( min_col_coverage <= 0.670384883881 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.0103703886897 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.00255794424684 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.670384883881
                    if ( min_col_coverage <= 0.670397341251 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.670397341251
                      return 0.00539079620161 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.791143655777
              if ( min_col_support <= 0.715499997139 ) {
                return false;
              }
              else {  // if min_col_support > 0.715499997139
                if ( median_col_support <= 0.991500020027 ) {
                  if ( median_col_coverage <= 0.679870605469 ) {
                    if ( min_col_coverage <= 0.579770922661 ) {
                      return 0.0105608625234 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.579770922661
                      return 0.00626108914245 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.679870605469
                    if ( median_col_coverage <= 0.813759386539 ) {
                      return 0.00399518629927 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.813759386539
                      return 0.00132815397363 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.991500020027
                  if ( mean_col_support <= 0.982205867767 ) {
                    if ( mean_col_support <= 0.982088267803 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.982088267803
                      return 0.18836565097 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.982205867767
                    if ( mean_col_support <= 0.996323466301 ) {
                      return 0.00153130677037 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996323466301
                      return 0.000531033797408 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    bool shouldCorrect2(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( min_col_support <= 0.923500001431 ) {
        if ( median_col_support <= 0.947499990463 ) {
          if ( max_col_coverage <= 0.515408575535 ) {
            if ( median_col_coverage <= 0.00685715209693 ) {
              if ( max_col_coverage <= 0.174212098122 ) {
                if ( median_col_coverage <= 0.00227531581186 ) {
                  if ( min_col_support <= 0.465999990702 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.465999990702
                    if ( min_col_coverage <= 0.00227015046403 ) {
                      return 0.0896885813149 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00227015046403
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.00227531581186
                  if ( max_col_coverage <= 0.139045342803 ) {
                    if ( mean_col_support <= 0.923533916473 ) {
                      return 0.0414845313873 < maxgini;
                    }
                    else {  // if mean_col_support > 0.923533916473
                      return 0.0294712401696 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.139045342803
                    if ( min_col_coverage <= 0.00268456852064 ) {
                      return 0.0973431108401 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00268456852064
                      return 0.0590320200961 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.174212098122
                if ( mean_col_support <= 0.891499996185 ) {
                  if ( mean_col_support <= 0.885735273361 ) {
                    if ( mean_col_support <= 0.885676503181 ) {
                      return 0.0541276550068 < maxgini;
                    }
                    else {  // if mean_col_support > 0.885676503181
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.885735273361
                    if ( min_col_support <= 0.617499947548 ) {
                      return 0.00660058769352 < maxgini;
                    }
                    else {  // if min_col_support > 0.617499947548
                      return 0.0508128544423 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.891499996185
                  if ( mean_col_coverage <= 0.11244559288 ) {
                    if ( median_col_support <= 0.864500045776 ) {
                      return 0.0799585225205 < maxgini;
                    }
                    else {  // if median_col_support > 0.864500045776
                      return 0.100886798476 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.11244559288
                    if ( mean_col_support <= 0.971382379532 ) {
                      return 0.224676390951 < maxgini;
                    }
                    else {  // if mean_col_support > 0.971382379532
                      return 0.428641975309 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.00685715209693
              if ( mean_col_support <= 0.942297160625 ) {
                if ( min_col_support <= 0.439500004053 ) {
                  if ( mean_col_coverage <= 0.29671895504 ) {
                    if ( median_col_coverage <= 0.0197879951447 ) {
                      return 0.0406429642409 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0197879951447
                      return 0.0246436748259 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.29671895504
                    if ( median_col_coverage <= 0.376572161913 ) {
                      return 0.052729977577 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.376572161913
                      return 0.160899653979 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.439500004053
                  if ( min_col_support <= 0.627499997616 ) {
                    if ( max_col_support <= 0.996500015259 ) {
                      return 0.0311602111639 < maxgini;
                    }
                    else {  // if max_col_support > 0.996500015259
                      return 0.0492599837993 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.627499997616
                    if ( max_col_coverage <= 0.377685010433 ) {
                      return 0.0411758591928 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.377685010433
                      return 0.037264713405 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.942297160625
                if ( min_col_coverage <= 0.00297177489847 ) {
                  if ( median_col_support <= 0.945500016212 ) {
                    if ( median_col_support <= 0.930500030518 ) {
                      return 0.0520953581852 < maxgini;
                    }
                    else {  // if median_col_support > 0.930500030518
                      return 0.105463444998 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.945500016212
                    return 0.0 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.00297177489847
                  if ( max_col_coverage <= 0.222556442022 ) {
                    if ( median_col_coverage <= 0.0573157146573 ) {
                      return 0.0224300851515 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0573157146573
                      return 0.0335602615876 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.222556442022
                    if ( median_col_coverage <= 0.0111890081316 ) {
                      return 0.111439149736 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0111890081316
                      return 0.0363921066456 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.515408575535
            if ( median_col_support <= 0.848500013351 ) {
              if ( median_col_coverage <= 0.933841824532 ) {
                if ( mean_col_coverage <= 0.449950098991 ) {
                  if ( max_col_support <= 0.997500002384 ) {
                    if ( max_col_coverage <= 0.518784165382 ) {
                      return 0.0371256314672 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.518784165382
                      return 0.019209345562 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.997500002384
                    if ( min_col_support <= 0.790500044823 ) {
                      return 0.0344379620271 < maxgini;
                    }
                    else {  // if min_col_support > 0.790500044823
                      return 0.0788853679392 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.449950098991
                  if ( min_col_support <= 0.554499983788 ) {
                    if ( min_col_coverage <= 0.412008583546 ) {
                      return 0.0434093903042 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.412008583546
                      return 0.0722810101987 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.554499983788
                    if ( mean_col_coverage <= 0.449953436852 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.449953436852
                      return 0.0337221731675 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.933841824532
                if ( min_col_coverage <= 0.931366801262 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_coverage > 0.931366801262
                  if ( min_col_support <= 0.660499989986 ) {
                    if ( mean_col_support <= 0.900088191032 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.900088191032
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.660499989986
                    return 0.0 < maxgini;
                  }
                }
              }
            }
            else {  // if median_col_support > 0.848500013351
              if ( min_col_support <= 0.547500014305 ) {
                if ( median_col_support <= 0.924499988556 ) {
                  if ( min_col_coverage <= 0.879424571991 ) {
                    if ( max_col_coverage <= 0.882636845112 ) {
                      return 0.0458178167575 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.882636845112
                      return 0.165289256198 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.879424571991
                    if ( mean_col_support <= 0.932941198349 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.932941198349
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.924499988556
                  if ( mean_col_coverage <= 0.681881070137 ) {
                    if ( median_col_support <= 0.93850004673 ) {
                      return 0.108503938712 < maxgini;
                    }
                    else {  // if median_col_support > 0.93850004673
                      return 0.0266617969321 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.681881070137
                    if ( median_col_support <= 0.933500051498 ) {
                      return 0.0798611111111 < maxgini;
                    }
                    else {  // if median_col_support > 0.933500051498
                      return 0.310650887574 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.547500014305
                if ( median_col_support <= 0.918500006199 ) {
                  if ( min_col_support <= 0.795500040054 ) {
                    if ( max_col_coverage <= 0.566419899464 ) {
                      return 0.0352913618368 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.566419899464
                      return 0.0248674693409 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.795500040054
                    if ( max_col_coverage <= 0.729261815548 ) {
                      return 0.0363852173235 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.729261815548
                      return 0.0224546405366 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.918500006199
                  if ( median_col_coverage <= 0.503095984459 ) {
                    if ( median_col_coverage <= 0.503083229065 ) {
                      return 0.028196852684 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.503083229065
                      return 0.260355029586 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.503095984459
                    if ( mean_col_coverage <= 0.530588567257 ) {
                      return 0.3046875 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.530588567257
                      return 0.0208295185033 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.947499990463
          if ( mean_col_coverage <= 0.995585799217 ) {
            if ( max_col_coverage <= 0.587176918983 ) {
              if ( mean_col_coverage <= 0.0967811048031 ) {
                if ( max_col_coverage <= 0.232525572181 ) {
                  if ( max_col_coverage <= 0.208652585745 ) {
                    if ( max_col_coverage <= 0.112375743687 ) {
                      return 0.0230324197364 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.112375743687
                      return 0.0299201168097 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.208652585745
                    if ( mean_col_coverage <= 0.0853203982115 ) {
                      return 0.0599912547685 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0853203982115
                      return 0.0297898987377 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.232525572181
                  if ( mean_col_coverage <= 0.0967582166195 ) {
                    if ( median_col_coverage <= 0.00412763468921 ) {
                      return 0.117296976656 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00412763468921
                      return 0.0440592417524 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.0967582166195
                    if ( min_col_coverage <= 0.00339567381889 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.00339567381889
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.0967811048031
                if ( median_col_coverage <= 0.00834493990988 ) {
                  if ( median_col_support <= 0.965499997139 ) {
                    if ( mean_col_coverage <= 0.123465642333 ) {
                      return 0.135214568919 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.123465642333
                      return 0.362462712113 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.965499997139
                    if ( mean_col_support <= 0.952353000641 ) {
                      return 0.42 < maxgini;
                    }
                    else {  // if mean_col_support > 0.952353000641
                      return 0.0673781143404 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.00834493990988
                  if ( min_col_coverage <= 0.0035650737118 ) {
                    if ( max_col_coverage <= 0.336422145367 ) {
                      return 0.0355743214183 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.336422145367
                      return 0.202697042124 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0035650737118
                    if ( mean_col_support <= 0.980594217777 ) {
                      return 0.0251162949826 < maxgini;
                    }
                    else {  // if mean_col_support > 0.980594217777
                      return 0.0173769699597 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.587176918983
              if ( median_col_support <= 0.996500015259 ) {
                if ( mean_col_support <= 0.972088217735 ) {
                  if ( median_col_coverage <= 0.831297874451 ) {
                    if ( mean_col_support <= 0.948205888271 ) {
                      return 0.0958159151794 < maxgini;
                    }
                    else {  // if mean_col_support > 0.948205888271
                      return 0.0260920515666 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.831297874451
                    if ( min_col_coverage <= 0.957086086273 ) {
                      return 0.221154862841 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.957086086273
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.972088217735
                  if ( min_col_support <= 0.650499999523 ) {
                    if ( max_col_coverage <= 0.847122073174 ) {
                      return 0.0899206524353 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.847122073174
                      return 0.460223537147 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.650499999523
                    if ( median_col_coverage <= 0.586097717285 ) {
                      return 0.0135675767259 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.586097717285
                      return 0.00591842149463 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.996500015259
                if ( median_col_support <= 0.997500002384 ) {
                  if ( mean_col_coverage <= 0.980501830578 ) {
                    if ( max_col_coverage <= 0.988776683807 ) {
                      return 0.0349770870599 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.988776683807
                      return 0.130447440609 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.980501830578
                    if ( mean_col_coverage <= 0.980583667755 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.980583667755
                      return 0.27173119065 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.997500002384
                  if ( mean_col_support <= 0.976823568344 ) {
                    if ( max_col_coverage <= 0.725136160851 ) {
                      return 0.32 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.725136160851
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.976823568344
                    if ( mean_col_support <= 0.982794046402 ) {
                      return 0.284888426408 < maxgini;
                    }
                    else {  // if mean_col_support > 0.982794046402
                      return 0.0100097567295 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.995585799217
            if ( min_col_coverage <= 0.992970108986 ) {
              if ( min_col_support <= 0.608999967575 ) {
                if ( median_col_coverage <= 0.996684551239 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.996684551239
                  if ( mean_col_support <= 0.972735285759 ) {
                    if ( median_col_support <= 0.993000030518 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.993000030518
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.972735285759
                    if ( median_col_support <= 0.996999979019 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.996999979019
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.608999967575
                if ( mean_col_coverage <= 0.995615363121 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.995615363121
                  if ( median_col_support <= 0.99849998951 ) {
                    if ( max_col_support <= 0.99950003624 ) {
                      return false;
                    }
                    else {  // if max_col_support > 0.99950003624
                      return 0.0348727285459 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99849998951
                    if ( min_col_support <= 0.899999976158 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.899999976158
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.992970108986
              if ( median_col_support <= 0.979499995708 ) {
                if ( mean_col_support <= 0.974294126034 ) {
                  if ( median_col_support <= 0.975499987602 ) {
                    if ( min_col_support <= 0.677999973297 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.677999973297
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.975499987602
                    return false;
                  }
                }
                else {  // if mean_col_support > 0.974294126034
                  return 0.0 < maxgini;
                }
              }
              else {  // if median_col_support > 0.979499995708
                if ( min_col_support <= 0.733999967575 ) {
                  if ( min_col_support <= 0.633000016212 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.633000016212
                    if ( min_col_coverage <= 0.997287094593 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.997287094593
                      return 0.277777777778 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.733999967575
                  if ( median_col_support <= 0.997500002384 ) {
                    if ( min_col_support <= 0.769999980927 ) {
                      return 0.132653061224 < maxgini;
                    }
                    else {  // if min_col_support > 0.769999980927
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.997500002384
                    if ( median_col_coverage <= 0.996409416199 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.996409416199
                      return 0.137174211248 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.923500001431
        if ( mean_col_coverage <= 0.52930188179 ) {
          if ( median_col_support <= 0.983500003815 ) {
            if ( min_col_coverage <= 0.274250626564 ) {
              if ( mean_col_coverage <= 0.334368258715 ) {
                if ( max_col_coverage <= 0.189678534865 ) {
                  if ( mean_col_support <= 0.967882275581 ) {
                    if ( mean_col_support <= 0.967617630959 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.967617630959
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.967882275581
                    if ( median_col_coverage <= 0.00271370913833 ) {
                      return 0.139934286757 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00271370913833
                      return 0.0149264332763 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.189678534865
                  if ( median_col_coverage <= 0.00406361883506 ) {
                    if ( mean_col_coverage <= 0.140945404768 ) {
                      return 0.0602640714732 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.140945404768
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.00406361883506
                    if ( median_col_support <= 0.972499966621 ) {
                      return 0.0328641236036 < maxgini;
                    }
                    else {  // if median_col_support > 0.972499966621
                      return 0.0246593505766 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.334368258715
                if ( median_col_coverage <= 0.214040964842 ) {
                  if ( max_col_coverage <= 0.552034378052 ) {
                    if ( median_col_coverage <= 0.195993259549 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.195993259549
                      return 0.265927977839 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.552034378052
                    return 0.0 < maxgini;
                  }
                }
                else {  // if median_col_coverage > 0.214040964842
                  if ( mean_col_support <= 0.980441153049 ) {
                    if ( mean_col_coverage <= 0.33552518487 ) {
                      return 0.044581107283 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.33552518487
                      return 0.0308163952502 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.980441153049
                    if ( mean_col_support <= 0.989500045776 ) {
                      return 0.0236592475007 < maxgini;
                    }
                    else {  // if mean_col_support > 0.989500045776
                      return 0.0191730718613 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.274250626564
              if ( mean_col_coverage <= 0.405919283628 ) {
                if ( median_col_support <= 0.975499987602 ) {
                  if ( median_col_support <= 0.963500022888 ) {
                    if ( min_col_support <= 0.945500016212 ) {
                      return 0.0338061677815 < maxgini;
                    }
                    else {  // if min_col_support > 0.945500016212
                      return 0.0495298817558 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.963500022888
                    if ( max_col_support <= 0.99849998951 ) {
                      return 0.0218951267003 < maxgini;
                    }
                    else {  // if max_col_support > 0.99849998951
                      return 0.0291258315873 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.975499987602
                  if ( max_col_coverage <= 0.459241807461 ) {
                    if ( median_col_support <= 0.979499995708 ) {
                      return 0.028475068856 < maxgini;
                    }
                    else {  // if median_col_support > 0.979499995708
                      return 0.0218771750317 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.459241807461
                    if ( min_col_coverage <= 0.325449168682 ) {
                      return 0.0192003783385 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.325449168682
                      return 0.0276235607638 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.405919283628
                if ( max_col_coverage <= 0.545746445656 ) {
                  if ( median_col_coverage <= 0.304951667786 ) {
                    if ( median_col_coverage <= 0.304134696722 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.304134696722
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.304951667786
                    if ( min_col_coverage <= 0.338418513536 ) {
                      return 0.0208581779298 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.338418513536
                      return 0.024991516221 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.545746445656
                  if ( median_col_support <= 0.969500005245 ) {
                    if ( median_col_coverage <= 0.48053753376 ) {
                      return 0.0288201180279 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.48053753376
                      return 0.0131755647536 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.969500005245
                    if ( mean_col_coverage <= 0.52926915884 ) {
                      return 0.0200575354957 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.52926915884
                      return 0.167573964497 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.983500003815
            if ( min_col_coverage <= 0.286178946495 ) {
              if ( min_col_coverage <= 0.0606718063354 ) {
                if ( min_col_coverage <= 0.00277393031865 ) {
                  if ( mean_col_support <= 0.992193758488 ) {
                    if ( min_col_support <= 0.952499985695 ) {
                      return 0.0304041986436 < maxgini;
                    }
                    else {  // if min_col_support > 0.952499985695
                      return 0.128792373719 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992193758488
                    if ( min_col_support <= 0.981500029564 ) {
                      return 0.00875895460135 < maxgini;
                    }
                    else {  // if min_col_support > 0.981500029564
                      return 0.14954029205 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.00277393031865
                  if ( max_col_coverage <= 0.329052865505 ) {
                    if ( min_col_support <= 0.971500039101 ) {
                      return 0.00605754350681 < maxgini;
                    }
                    else {  // if min_col_support > 0.971500039101
                      return 0.0100185312729 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.329052865505
                    if ( mean_col_coverage <= 0.206279873848 ) {
                      return 0.0159281767824 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.206279873848
                      return 0.0376824238669 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.0606718063354
                if ( median_col_support <= 0.992499947548 ) {
                  if ( median_col_support <= 0.989500045776 ) {
                    if ( min_col_coverage <= 0.0606768839061 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.0606768839061
                      return 0.0196534765777 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.989500045776
                    if ( min_col_coverage <= 0.201465249062 ) {
                      return 0.0138769158639 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.201465249062
                      return 0.0165570790512 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.992499947548
                  if ( median_col_coverage <= 0.233754813671 ) {
                    if ( max_col_coverage <= 0.337997674942 ) {
                      return 0.011319451937 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.337997674942
                      return 0.00516532873607 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.233754813671
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.0139841301505 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.00722123706425 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.286178946495
              if ( median_col_support <= 0.992499947548 ) {
                if ( median_col_support <= 0.986500024796 ) {
                  if ( min_col_coverage <= 0.335157334805 ) {
                    if ( min_col_support <= 0.972499966621 ) {
                      return 0.0116735013858 < maxgini;
                    }
                    else {  // if min_col_support > 0.972499966621
                      return 0.0178668996028 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.335157334805
                    if ( min_col_support <= 0.954499959946 ) {
                      return 0.0129372000682 < maxgini;
                    }
                    else {  // if min_col_support > 0.954499959946
                      return 0.0199625909589 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.986500024796
                  if ( min_col_support <= 0.96850001812 ) {
                    if ( median_col_coverage <= 0.342664897442 ) {
                      return 0.0144898676111 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.342664897442
                      return 0.0103047492822 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.96850001812
                    if ( median_col_coverage <= 0.444720506668 ) {
                      return 0.0148241454826 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.444720506668
                      return 0.0126708159709 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.992499947548
                if ( median_col_support <= 0.994500041008 ) {
                  if ( max_col_coverage <= 0.576829195023 ) {
                    if ( max_col_support <= 0.996500015259 ) {
                      return false;
                    }
                    else {  // if max_col_support > 0.996500015259
                      return 0.0108131255855 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.576829195023
                    if ( min_col_support <= 0.983500003815 ) {
                      return 0.00600278904927 < maxgini;
                    }
                    else {  // if min_col_support > 0.983500003815
                      return 0.00868775859224 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.994500041008
                  if ( max_col_coverage <= 0.5337215662 ) {
                    if ( mean_col_coverage <= 0.455210030079 ) {
                      return 0.0080499763374 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.455210030079
                      return 0.0135097941184 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.5337215662
                    if ( median_col_coverage <= 0.405164480209 ) {
                      return 0.00360990485017 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.405164480209
                      return 0.00613260108264 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.52930188179
          if ( min_col_coverage <= 0.593681812286 ) {
            if ( min_col_coverage <= 0.500880300999 ) {
              if ( min_col_support <= 0.977499961853 ) {
                if ( min_col_coverage <= 0.500877201557 ) {
                  if ( median_col_support <= 0.985499978065 ) {
                    if ( mean_col_support <= 0.984676480293 ) {
                      return 0.0230796565561 < maxgini;
                    }
                    else {  // if mean_col_support > 0.984676480293
                      return 0.0149684606205 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.985499978065
                    if ( median_col_coverage <= 0.419358879328 ) {
                      return 0.0932333717801 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.419358879328
                      return 0.00746446323177 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.500877201557
                  if ( median_col_support <= 0.983500003815 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if median_col_support > 0.983500003815
                    return false;
                  }
                }
              }
              else {  // if min_col_support > 0.977499961853
                if ( mean_col_coverage <= 0.543449878693 ) {
                  if ( min_col_support <= 0.990499973297 ) {
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0124155569278 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.00586454044535 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.990499973297
                    if ( max_col_coverage <= 0.607325494289 ) {
                      return 0.00115740701934 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.607325494289
                      return 0.00448734973283 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.543449878693
                  if ( mean_col_support <= 0.994911789894 ) {
                    if ( min_col_coverage <= 0.460211336613 ) {
                      return 0.0131543913772 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.460211336613
                      return 0.00977750834912 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994911789894
                    if ( mean_col_support <= 0.997088193893 ) {
                      return 0.00544874196656 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997088193893
                      return 0.00306201493783 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.500880300999
              if ( mean_col_support <= 0.991676449776 ) {
                if ( min_col_coverage <= 0.593676686287 ) {
                  if ( min_col_support <= 0.964499950409 ) {
                    if ( mean_col_support <= 0.983676433563 ) {
                      return 0.0197974548726 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983676433563
                      return 0.00965993262774 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.964499950409
                    if ( mean_col_support <= 0.990911722183 ) {
                      return 0.0169289000512 < maxgini;
                    }
                    else {  // if mean_col_support > 0.990911722183
                      return 0.0106998867691 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.593676686287
                  if ( median_col_coverage <= 0.620767474174 ) {
                    if ( median_col_support <= 0.985000014305 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.985000014305
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.620767474174
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.991676449776
                if ( mean_col_support <= 0.995029389858 ) {
                  if ( median_col_coverage <= 0.616450309753 ) {
                    if ( mean_col_support <= 0.994323611259 ) {
                      return 0.00744041066482 < maxgini;
                    }
                    else {  // if mean_col_support > 0.994323611259
                      return 0.0059488972374 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.616450309753
                    if ( median_col_coverage <= 0.620335459709 ) {
                      return 0.00294475395464 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.620335459709
                      return 0.0059339369339 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.995029389858
                  if ( median_col_coverage <= 0.554463505745 ) {
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.00574587895484 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.00318791760579 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.554463505745
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.00429553564652 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.00230216787512 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.593681812286
            if ( mean_col_support <= 0.994794130325 ) {
              if ( median_col_support <= 0.984500050545 ) {
                if ( median_col_coverage <= 0.75928580761 ) {
                  if ( median_col_support <= 0.974500000477 ) {
                    if ( min_col_support <= 0.96850001812 ) {
                      return 0.0134705304088 < maxgini;
                    }
                    else {  // if min_col_support > 0.96850001812
                      return 0.06365646531 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.974500000477
                    if ( min_col_coverage <= 0.637408971786 ) {
                      return 0.00978981587171 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.637408971786
                      return 0.0071541813329 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.75928580761
                  if ( max_col_coverage <= 0.902799546719 ) {
                    if ( mean_col_support <= 0.990852952003 ) {
                      return 0.00693816549574 < maxgini;
                    }
                    else {  // if mean_col_support > 0.990852952003
                      return 0.000857632775265 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.902799546719
                    if ( mean_col_support <= 0.987676441669 ) {
                      return 0.00370010181594 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987676441669
                      return 0.0015254793991 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.984500050545
                if ( median_col_support <= 0.989500045776 ) {
                  if ( max_col_coverage <= 0.860845267773 ) {
                    if ( median_col_coverage <= 0.598232030869 ) {
                      return 0.18 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.598232030869
                      return 0.00623851373071 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.860845267773
                    if ( min_col_coverage <= 0.983796834946 ) {
                      return 0.00206427813239 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.983796834946
                      return 0.13624567474 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.989500045776
                  if ( max_col_coverage <= 0.826800704002 ) {
                    if ( mean_col_support <= 0.992205858231 ) {
                      return 0.00118898953232 < maxgini;
                    }
                    else {  // if mean_col_support > 0.992205858231
                      return 0.00404517944929 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.826800704002
                    if ( mean_col_coverage <= 0.874471127987 ) {
                      return 0.0019321248365 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.874471127987
                      return 0.000462912932123 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.994794130325
              if ( min_col_coverage <= 0.635060071945 ) {
                if ( median_col_support <= 0.995499968529 ) {
                  if ( mean_col_coverage <= 0.634099304676 ) {
                    if ( mean_col_coverage <= 0.63390994072 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.63390994072
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.634099304676
                    if ( mean_col_coverage <= 0.786046981812 ) {
                      return 0.00277889013153 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.786046981812
                      return 0.0353242585787 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.995499968529
                  if ( median_col_support <= 0.997500002384 ) {
                    if ( mean_col_coverage <= 0.691892147064 ) {
                      return 0.00224664878364 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.691892147064
                      return 0.000974424457455 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.997500002384
                    if ( mean_col_support <= 0.99849998951 ) {
                      return 0.00136695992374 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99849998951
                      return 0.000419412803597 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.635060071945
                if ( mean_col_support <= 0.996264696121 ) {
                  if ( max_col_coverage <= 0.770613908768 ) {
                    if ( min_col_coverage <= 0.664331197739 ) {
                      return 0.0072915690345 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.664331197739
                      return 0.03076171875 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.770613908768
                    if ( median_col_coverage <= 0.870350003242 ) {
                      return 0.00191421594804 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.870350003242
                      return 0.00032932651832 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.996264696121
                  if ( mean_col_coverage <= 0.728463172913 ) {
                    if ( mean_col_coverage <= 0.72846275568 ) {
                      return 0.00118650588289 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.72846275568
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.728463172913
                    if ( min_col_support <= 0.993499994278 ) {
                      return 0.000524092331302 < maxgini;
                    }
                    else {  // if min_col_support > 0.993499994278
                      return 0.000155908948226 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    bool shouldCorrect3(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( median_col_support <= 0.977499961853 ) {
        if ( median_col_support <= 0.943500041962 ) {
          if ( min_col_support <= 0.634500026703 ) {
            if ( max_col_coverage <= 0.979089975357 ) {
              if ( median_col_support <= 0.560500025749 ) {
                if ( median_col_coverage <= 0.549834430218 ) {
                  if ( median_col_coverage <= 0.00292046112008 ) {
                    if ( median_col_coverage <= 0.00285321217962 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00285321217962
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.00292046112008
                    if ( max_col_support <= 0.983500003815 ) {
                      return 0.0125184714273 < maxgini;
                    }
                    else {  // if max_col_support > 0.983500003815
                      return 0.0350341173599 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.549834430218
                  if ( median_col_support <= 0.439999997616 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.439999997616
                    if ( min_col_coverage <= 0.515729486942 ) {
                      return 0.310650887574 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.515729486942
                      return 0.0665873959572 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.560500025749
                if ( mean_col_coverage <= 0.0400786623359 ) {
                  if ( min_col_coverage <= 0.00268456852064 ) {
                    if ( max_col_coverage <= 0.145891487598 ) {
                      return 0.0813977348368 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.145891487598
                      return 0.48 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00268456852064
                    if ( median_col_coverage <= 0.0296592470258 ) {
                      return 0.0280201548492 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0296592470258
                      return 0.176085663296 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.0400786623359
                  if ( min_col_support <= 0.439500004053 ) {
                    if ( median_col_coverage <= 0.195860981941 ) {
                      return 0.0284136167132 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.195860981941
                      return 0.0698555141966 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.439500004053
                    if ( mean_col_coverage <= 0.26081943512 ) {
                      return 0.0500160814156 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.26081943512
                      return 0.0425412881453 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.979089975357
              if ( max_col_coverage <= 0.979630947113 ) {
                return false;
              }
              else {  // if max_col_coverage > 0.979630947113
                if ( median_col_coverage <= 0.950498819351 ) {
                  if ( min_col_coverage <= 0.891560673714 ) {
                    if ( min_col_coverage <= 0.363731026649 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.363731026649
                      return 0.0410076478167 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.891560673714
                    if ( max_col_coverage <= 0.99213796854 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.99213796854
                      return 0.244897959184 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.950498819351
                  if ( min_col_support <= 0.546000003815 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.546000003815
                    if ( mean_col_coverage <= 0.976847529411 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.976847529411
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.634500026703
            if ( median_col_coverage <= 0.00570614542812 ) {
              if ( mean_col_support <= 0.952289938927 ) {
                if ( min_col_support <= 0.866500020027 ) {
                  if ( min_col_coverage <= 0.00561010930687 ) {
                    if ( min_col_support <= 0.671499967575 ) {
                      return 0.0793973591061 < maxgini;
                    }
                    else {  // if min_col_support > 0.671499967575
                      return 0.0550185216797 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00561010930687
                    if ( min_col_support <= 0.680999994278 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_support > 0.680999994278
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.866500020027
                  if ( min_col_coverage <= 0.00332357967272 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.00332357967272
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.952289938927
                if ( min_col_coverage <= 0.00291971443221 ) {
                  if ( min_col_coverage <= 0.00285307271406 ) {
                    if ( min_col_coverage <= 0.00277393031865 ) {
                      return 0.080318405106 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00277393031865
                      return 0.136768960835 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00285307271406
                    if ( max_col_coverage <= 0.138929367065 ) {
                      return 0.0525931336742 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.138929367065
                      return 0.244327580288 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.00291971443221
                  if ( median_col_support <= 0.861500024796 ) {
                    if ( median_col_coverage <= 0.00355249876156 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00355249876156
                      return false;
                    }
                  }
                  else {  // if median_col_support > 0.861500024796
                    if ( mean_col_coverage <= 0.0776474326849 ) {
                      return 0.0647396156798 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0776474326849
                      return 0.125684112498 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.00570614542812
              if ( max_col_support <= 0.99950003624 ) {
                if ( median_col_coverage <= 0.982293605804 ) {
                  if ( median_col_support <= 0.885499954224 ) {
                    if ( max_col_support <= 0.992499947548 ) {
                      return 0.0224678140307 < maxgini;
                    }
                    else {  // if max_col_support > 0.992499947548
                      return 0.0354697103389 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.885499954224
                    if ( min_col_coverage <= 0.890383005142 ) {
                      return 0.02549792562 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.890383005142
                      return 0.255 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.982293605804
                  return false;
                }
              }
              else {  // if max_col_support > 0.99950003624
                if ( median_col_coverage <= 0.491948455572 ) {
                  if ( median_col_support <= 0.898499965668 ) {
                    if ( median_col_coverage <= 0.00679119117558 ) {
                      return 0.0633592602121 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00679119117558
                      return 0.0407804794201 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.898499965668
                    if ( mean_col_coverage <= 0.0888746827841 ) {
                      return 0.0266175414607 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0888746827841
                      return 0.0356159567201 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.491948455572
                  if ( mean_col_support <= 0.934558808804 ) {
                    if ( median_col_coverage <= 0.897406339645 ) {
                      return 0.0342396122238 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.897406339645
                      return 0.46875 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.934558808804
                    if ( min_col_support <= 0.837499976158 ) {
                      return 0.0189970219914 < maxgini;
                    }
                    else {  // if min_col_support > 0.837499976158
                      return 0.0276807030267 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.943500041962
          if ( min_col_coverage <= 0.446857273579 ) {
            if ( mean_col_coverage <= 0.353509783745 ) {
              if ( min_col_support <= 0.93850004673 ) {
                if ( mean_col_support <= 0.979710161686 ) {
                  if ( min_col_support <= 0.904500007629 ) {
                    if ( mean_col_coverage <= 0.00785543676466 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.00785543676466
                      return 0.0281651930616 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.904500007629
                    if ( mean_col_support <= 0.970735311508 ) {
                      return 0.0416343063191 < maxgini;
                    }
                    else {  // if mean_col_support > 0.970735311508
                      return 0.0308112007094 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.979710161686
                  if ( mean_col_coverage <= 0.0941876843572 ) {
                    if ( mean_col_coverage <= 0.0941830277443 ) {
                      return 0.0356951153667 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0941830277443
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.0941876843572
                    if ( mean_col_coverage <= 0.353366672993 ) {
                      return 0.0239409439878 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.353366672993
                      return 0.134625390219 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.93850004673
                if ( max_col_coverage <= 0.103876203299 ) {
                  if ( mean_col_coverage <= 0.0520541146398 ) {
                    return false;
                  }
                  else {  // if mean_col_coverage > 0.0520541146398
                    return 0.0 < maxgini;
                  }
                }
                else {  // if max_col_coverage > 0.103876203299
                  if ( mean_col_coverage <= 0.353509068489 ) {
                    if ( mean_col_support <= 0.983902692795 ) {
                      return 0.0374300365829 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983902692795
                      return 0.0280674735144 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.353509068489
                    return false;
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.353509783745
              if ( median_col_support <= 0.96749997139 ) {
                if ( min_col_coverage <= 0.446852505207 ) {
                  if ( min_col_support <= 0.887500047684 ) {
                    if ( mean_col_support <= 0.958794116974 ) {
                      return 0.0369414897598 < maxgini;
                    }
                    else {  // if mean_col_support > 0.958794116974
                      return 0.0204407499682 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.887500047684
                    if ( mean_col_coverage <= 0.423020392656 ) {
                      return 0.0310790594756 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.423020392656
                      return 0.0275479298724 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.446852505207
                  if ( median_col_coverage <= 0.475054234266 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.475054234266
                    return false;
                  }
                }
              }
              else {  // if median_col_support > 0.96749997139
                if ( mean_col_coverage <= 0.68891364336 ) {
                  if ( max_col_support <= 0.99950003624 ) {
                    if ( mean_col_coverage <= 0.45633661747 ) {
                      return 0.0227310657991 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.45633661747
                      return 0.0159407360397 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.99950003624
                    if ( min_col_support <= 0.944499969482 ) {
                      return 0.0191835131886 < maxgini;
                    }
                    else {  // if min_col_support > 0.944499969482
                      return 0.0270070763378 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.68891364336
                  if ( median_col_coverage <= 0.49657279253 ) {
                    if ( mean_col_support <= 0.97370582819 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.97370582819
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.49657279253
                    return 0.0 < maxgini;
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.446857273579
            if ( mean_col_coverage <= 0.997319817543 ) {
              if ( mean_col_coverage <= 0.637451291084 ) {
                if ( mean_col_coverage <= 0.637450814247 ) {
                  if ( median_col_coverage <= 0.501879692078 ) {
                    if ( min_col_coverage <= 0.500778794289 ) {
                      return 0.0219732613985 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.500778794289
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.501879692078
                    if ( mean_col_coverage <= 0.637424826622 ) {
                      return 0.0175650037714 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.637424826622
                      return 0.14201183432 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.637450814247
                  if ( min_col_coverage <= 0.576666653156 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.576666653156
                    return false;
                  }
                }
              }
              else {  // if mean_col_coverage > 0.637451291084
                if ( mean_col_support <= 0.941205859184 ) {
                  if ( max_col_coverage <= 0.858358740807 ) {
                    if ( median_col_coverage <= 0.663625836372 ) {
                      return 0.0753556324491 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.663625836372
                      return 0.207612456747 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.858358740807
                    if ( mean_col_coverage <= 0.972133159637 ) {
                      return 0.399930747922 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.972133159637
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.941205859184
                  if ( max_col_coverage <= 0.838737368584 ) {
                    if ( max_col_support <= 0.992499947548 ) {
                      return 0.095 < maxgini;
                    }
                    else {  // if max_col_support > 0.992499947548
                      return 0.0130864159797 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.838737368584
                    if ( median_col_coverage <= 0.543833136559 ) {
                      return 0.149937526031 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.543833136559
                      return 0.00690011684874 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.997319817543
              if ( min_col_support <= 0.68700003624 ) {
                if ( min_col_support <= 0.577499985695 ) {
                  return false;
                }
                else {  // if min_col_support > 0.577499985695
                  if ( min_col_coverage <= 0.997149527073 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.997149527073
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if min_col_support > 0.68700003624
                if ( min_col_coverage <= 0.992126941681 ) {
                  if ( median_col_support <= 0.969500005245 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if median_col_support > 0.969500005245
                    if ( mean_col_coverage <= 0.998768568039 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.998768568039
                      return false;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.992126941681
                  return 0.0 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.977499961853
        if ( mean_col_coverage <= 0.538221478462 ) {
          if ( median_col_support <= 0.989500045776 ) {
            if ( mean_col_coverage <= 0.347836405039 ) {
              if ( max_col_coverage <= 0.316136240959 ) {
                if ( median_col_coverage <= 0.102644145489 ) {
                  if ( min_col_coverage <= 0.00252207205631 ) {
                    if ( mean_col_coverage <= 0.0738265663385 ) {
                      return 0.0672480470721 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0738265663385
                      return 0.0442068199128 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00252207205631
                    if ( min_col_coverage <= 0.101023137569 ) {
                      return 0.0181357440375 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.101023137569
                      return 0.332409972299 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.102644145489
                  if ( min_col_support <= 0.954499959946 ) {
                    if ( median_col_support <= 0.981500029564 ) {
                      return 0.0261539921413 < maxgini;
                    }
                    else {  // if median_col_support > 0.981500029564
                      return 0.0195603613082 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.954499959946
                    if ( mean_col_support <= 0.992264688015 ) {
                      return 0.0343714455277 < maxgini;
                    }
                    else {  // if mean_col_support > 0.992264688015
                      return 0.0260979272357 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.316136240959
                if ( min_col_coverage <= 0.212991699576 ) {
                  if ( max_col_coverage <= 0.379002094269 ) {
                    if ( median_col_coverage <= 0.213563576341 ) {
                      return 0.017364407642 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.213563576341
                      return 0.0226969578053 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.379002094269
                    if ( mean_col_support <= 0.96697062254 ) {
                      return 0.0630245413744 < maxgini;
                    }
                    else {  // if mean_col_support > 0.96697062254
                      return 0.0140486936998 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.212991699576
                  if ( max_col_coverage <= 0.422709107399 ) {
                    if ( median_col_support <= 0.988499999046 ) {
                      return 0.0261974343733 < maxgini;
                    }
                    else {  // if median_col_support > 0.988499999046
                      return 0.020528287512 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.422709107399
                    if ( median_col_coverage <= 0.32143804431 ) {
                      return 0.0194721671338 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.32143804431
                      return 0.375 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.347836405039
              if ( mean_col_support <= 0.924117684364 ) {
                if ( min_col_coverage <= 0.386448502541 ) {
                  if ( min_col_support <= 0.522499978542 ) {
                    if ( min_col_support <= 0.511999964714 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_support > 0.511999964714
                      return 0.277777777778 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.522499978542
                    if ( median_col_coverage <= 0.347382217646 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.347382217646
                      return false;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.386448502541
                  if ( max_col_coverage <= 0.575287938118 ) {
                    return false;
                  }
                  else {  // if max_col_coverage > 0.575287938118
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.924117684364
                if ( min_col_coverage <= 0.098180487752 ) {
                  if ( min_col_support <= 0.808499991894 ) {
                    if ( median_col_coverage <= 0.176049530506 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.176049530506
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.808499991894
                    if ( mean_col_support <= 0.977235198021 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.977235198021
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.098180487752
                  if ( median_col_support <= 0.986500024796 ) {
                    if ( min_col_support <= 0.636500000954 ) {
                      return 0.0832580740423 < maxgini;
                    }
                    else {  // if min_col_support > 0.636500000954
                      return 0.0184260596765 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.986500024796
                    if ( min_col_support <= 0.978500008583 ) {
                      return 0.0132021998438 < maxgini;
                    }
                    else {  // if min_col_support > 0.978500008583
                      return 0.0172488599203 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.989500045776
            if ( max_col_coverage <= 0.457276642323 ) {
              if ( mean_col_coverage <= 0.077147603035 ) {
                if ( median_col_coverage <= 0.0249264314771 ) {
                  if ( max_col_coverage <= 0.251642048359 ) {
                    if ( max_col_coverage <= 0.0478731319308 ) {
                      return 0.0692439180733 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.0478731319308
                      return 0.0206127686449 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.251642048359
                    if ( min_col_coverage <= 0.00344831682742 ) {
                      return 0.21875 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00344831682742
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.0249264314771
                  if ( min_col_coverage <= 0.00343054183759 ) {
                    if ( mean_col_support <= 0.979499995708 ) {
                      return 0.108997365387 < maxgini;
                    }
                    else {  // if mean_col_support > 0.979499995708
                      return 0.0540818385159 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00343054183759
                    if ( min_col_coverage <= 0.00580552779138 ) {
                      return 0.0447526367927 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00580552779138
                      return 0.00475056689342 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.077147603035
                if ( mean_col_coverage <= 0.276776999235 ) {
                  if ( min_col_coverage <= 0.078781247139 ) {
                    if ( min_col_support <= 0.891499996185 ) {
                      return 0.0160978330982 < maxgini;
                    }
                    else {  // if min_col_support > 0.891499996185
                      return 0.00584260120026 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.078781247139
                    if ( median_col_coverage <= 0.15109616518 ) {
                      return 0.0181704468443 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.15109616518
                      return 0.010940696986 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.276776999235
                  if ( median_col_support <= 0.995499968529 ) {
                    if ( max_col_coverage <= 0.403621286154 ) {
                      return 0.0188592477812 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.403621286154
                      return 0.0143366193843 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.995499968529
                    if ( mean_col_coverage <= 0.322530984879 ) {
                      return 0.00337712083278 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.322530984879
                      return 0.010216500933 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.457276642323
              if ( median_col_support <= 0.994500041008 ) {
                if ( mean_col_coverage <= 0.538220763206 ) {
                  if ( max_col_coverage <= 0.594216823578 ) {
                    if ( min_col_support <= 0.529500007629 ) {
                      return 0.146712875446 < maxgini;
                    }
                    else {  // if min_col_support > 0.529500007629
                      return 0.0117823439499 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.594216823578
                    if ( min_col_coverage <= 0.294893532991 ) {
                      return 0.0203084953685 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.294893532991
                      return 0.00849587772561 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.538220763206
                  return false;
                }
              }
              else {  // if median_col_support > 0.994500041008
                if ( min_col_support <= 0.503000020981 ) {
                  if ( mean_col_support <= 0.96585303545 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if mean_col_support > 0.96585303545
                    return false;
                  }
                }
                else {  // if min_col_support > 0.503000020981
                  if ( min_col_support <= 0.655499994755 ) {
                    if ( median_col_coverage <= 0.433997690678 ) {
                      return 0.149937526031 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.433997690678
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.655499994755
                    if ( mean_col_coverage <= 0.509359478951 ) {
                      return 0.00692437984379 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.509359478951
                      return 0.00469653321149 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.538221478462
          if ( median_col_support <= 0.991500020027 ) {
            if ( min_col_support <= 0.604499995708 ) {
              if ( mean_col_coverage <= 0.9820510149 ) {
                if ( max_col_coverage <= 0.889922142029 ) {
                  if ( max_col_support <= 0.99950003624 ) {
                    if ( max_col_coverage <= 0.829953253269 ) {
                      return 0.348440853428 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.829953253269
                      return false;
                    }
                  }
                  else {  // if max_col_support > 0.99950003624
                    if ( min_col_coverage <= 0.397190868855 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.397190868855
                      return 0.147049423817 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.889922142029
                  if ( min_col_coverage <= 0.797438621521 ) {
                    if ( min_col_coverage <= 0.575905740261 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.575905740261
                      return 0.316929692888 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.797438621521
                    if ( min_col_coverage <= 0.800077676773 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.800077676773
                      return 0.493296398892 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.9820510149
                return false;
              }
            }
            else {  // if min_col_support > 0.604499995708
              if ( mean_col_support <= 0.961911797523 ) {
                if ( min_col_support <= 0.678499996662 ) {
                  if ( median_col_coverage <= 0.977423071861 ) {
                    if ( max_col_coverage <= 0.989961385727 ) {
                      return 0.0280913581573 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.989961385727
                      return 0.451843043995 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.977423071861
                    return false;
                  }
                }
                else {  // if min_col_support > 0.678499996662
                  if ( min_col_coverage <= 0.656446516514 ) {
                    if ( max_col_coverage <= 0.934610128403 ) {
                      return 0.0579193584317 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.934610128403
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.656446516514
                    if ( min_col_coverage <= 0.661112248898 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.661112248898
                      return 0.343858131488 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.961911797523
                if ( mean_col_support <= 0.992323517799 ) {
                  if ( median_col_coverage <= 0.610425651073 ) {
                    if ( min_col_support <= 0.956499993801 ) {
                      return 0.00817818613762 < maxgini;
                    }
                    else {  // if min_col_support > 0.956499993801
                      return 0.0140752697122 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.610425651073
                    if ( min_col_support <= 0.628000020981 ) {
                      return 0.275937435967 < maxgini;
                    }
                    else {  // if min_col_support > 0.628000020981
                      return 0.00558621590894 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.992323517799
                  if ( min_col_coverage <= 0.600587368011 ) {
                    if ( min_col_support <= 0.973500013351 ) {
                      return 0.00597162120795 < maxgini;
                    }
                    else {  // if min_col_support > 0.973500013351
                      return 0.00867079184625 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.600587368011
                    if ( max_col_coverage <= 0.788930892944 ) {
                      return 0.00659107791407 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.788930892944
                      return 0.00352987944889 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.991500020027
            if ( min_col_coverage <= 0.602534294128 ) {
              if ( median_col_coverage <= 0.532668292522 ) {
                if ( min_col_support <= 0.544000029564 ) {
                  if ( mean_col_support <= 0.968235373497 ) {
                    if ( mean_col_support <= 0.944117665291 ) {
                      return 0.489795918367 < maxgini;
                    }
                    else {  // if mean_col_support > 0.944117665291
                      return 0.104938271605 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.968235373497
                    if ( mean_col_coverage <= 0.551754593849 ) {
                      return 0.408163265306 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.551754593849
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.544000029564
                  if ( mean_col_support <= 0.956705868244 ) {
                    if ( min_col_support <= 0.661000013351 ) {
                      return 0.207612456747 < maxgini;
                    }
                    else {  // if min_col_support > 0.661000013351
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.956705868244
                    if ( mean_col_coverage <= 0.653276920319 ) {
                      return 0.0046464894903 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.653276920319
                      return 0.0739644970414 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.532668292522
                if ( median_col_support <= 0.994500041008 ) {
                  if ( min_col_support <= 0.569499969482 ) {
                    if ( mean_col_coverage <= 0.62522560358 ) {
                      return 0.0454299621417 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.62522560358
                      return 0.425361570248 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.569499969482
                    if ( min_col_support <= 0.701499998569 ) {
                      return 0.0644444444444 < maxgini;
                    }
                    else {  // if min_col_support > 0.701499998569
                      return 0.00423885965015 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.994500041008
                  if ( median_col_support <= 0.997500002384 ) {
                    if ( min_col_coverage <= 0.578727781773 ) {
                      return 0.0025809926742 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.578727781773
                      return 0.00202567357034 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.997500002384
                    if ( mean_col_support <= 0.978941202164 ) {
                      return 0.4296875 < maxgini;
                    }
                    else {  // if mean_col_support > 0.978941202164
                      return 0.000902150317532 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.602534294128
              if ( min_col_coverage <= 0.984323263168 ) {
                if ( median_col_coverage <= 0.66272687912 ) {
                  if ( min_col_support <= 0.579499959946 ) {
                    if ( median_col_coverage <= 0.662650406361 ) {
                      return 0.157550535077 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.662650406361
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.579499959946
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.00248134036459 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.00130275046864 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.66272687912
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( max_col_coverage <= 0.77422362566 ) {
                      return 0.00495011449335 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.77422362566
                      return 0.00178536344269 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( min_col_support <= 0.653499960899 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.653499960899
                      return 0.000438374415848 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.984323263168
                if ( min_col_support <= 0.695500016212 ) {
                  if ( median_col_coverage <= 0.997474074364 ) {
                    if ( mean_col_support <= 0.975029349327 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.975029349327
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.997474074364
                    if ( mean_col_support <= 0.976352930069 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.976352930069
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.695500016212
                  if ( median_col_coverage <= 0.997621893883 ) {
                    if ( mean_col_support <= 0.988558769226 ) {
                      return 0.225651577503 < maxgini;
                    }
                    else {  // if mean_col_support > 0.988558769226
                      return 0.00226207497979 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.997621893883
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.000753011941368 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    bool shouldCorrect4(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( mean_col_coverage <= 0.505620241165 ) {
        if ( mean_col_support <= 0.980485320091 ) {
          if ( mean_col_support <= 0.953242659569 ) {
            if ( max_col_coverage <= 0.348108291626 ) {
              if ( max_col_support <= 0.992499947548 ) {
                if ( mean_col_support <= 0.706499934196 ) {
                  if ( min_col_coverage <= 0.0105639044195 ) {
                    if ( median_col_support <= 0.490499973297 ) {
                      return 0.46875 < maxgini;
                    }
                    else {  // if median_col_support > 0.490499973297
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0105639044195
                    if ( median_col_coverage <= 0.0629254877567 ) {
                      return 0.0122159434712 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0629254877567
                      return 0.00279050681073 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.706499934196
                  if ( min_col_coverage <= 0.133111342788 ) {
                    if ( mean_col_support <= 0.859441161156 ) {
                      return 0.0255765555249 < maxgini;
                    }
                    else {  // if mean_col_support > 0.859441161156
                      return 0.0137544985075 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.133111342788
                    if ( min_col_support <= 0.533499956131 ) {
                      return 0.0473500457182 < maxgini;
                    }
                    else {  // if min_col_support > 0.533499956131
                      return 0.0305503210612 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_support > 0.992499947548
                if ( median_col_coverage <= 0.00631913123652 ) {
                  if ( max_col_coverage <= 0.167142897844 ) {
                    if ( mean_col_support <= 0.889560818672 ) {
                      return 0.0276927225932 < maxgini;
                    }
                    else {  // if mean_col_support > 0.889560818672
                      return 0.0470195172901 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.167142897844
                    if ( median_col_support <= 0.864500045776 ) {
                      return 0.0611795746824 < maxgini;
                    }
                    else {  // if median_col_support > 0.864500045776
                      return 0.102998184309 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.00631913123652
                  if ( median_col_coverage <= 0.157569393516 ) {
                    if ( min_col_support <= 0.645500004292 ) {
                      return 0.0462587677303 < maxgini;
                    }
                    else {  // if min_col_support > 0.645500004292
                      return 0.0388310868463 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.157569393516
                    if ( median_col_support <= 0.846500039101 ) {
                      return 0.0594003213572 < maxgini;
                    }
                    else {  // if median_col_support > 0.846500039101
                      return 0.0454001737061 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.348108291626
              if ( median_col_coverage <= 0.00949376635253 ) {
                if ( median_col_coverage <= 0.00830329023302 ) {
                  if ( median_col_coverage <= 0.00394966593012 ) {
                    if ( median_col_support <= 0.910499989986 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.910499989986
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.00394966593012
                    return 0.0 < maxgini;
                  }
                }
                else {  // if median_col_coverage > 0.00830329023302
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.00949376635253
                if ( median_col_support <= 0.995499968529 ) {
                  if ( max_col_support <= 0.995499968529 ) {
                    if ( max_col_support <= 0.993499994278 ) {
                      return 0.0233412307607 < maxgini;
                    }
                    else {  // if max_col_support > 0.993499994278
                      return 0.0322361633917 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.995499968529
                    if ( median_col_support <= 0.891499996185 ) {
                      return 0.0414630788621 < maxgini;
                    }
                    else {  // if median_col_support > 0.891499996185
                      return 0.03582582803 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.995499968529
                  if ( min_col_support <= 0.539999961853 ) {
                    if ( max_col_coverage <= 0.526553988457 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.526553988457
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.539999961853
                    return 0.0 < maxgini;
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.953242659569
            if ( median_col_coverage <= 0.00520156929269 ) {
              if ( max_col_coverage <= 0.232054024935 ) {
                if ( mean_col_coverage <= 0.0472782440484 ) {
                  if ( max_col_coverage <= 0.117129981518 ) {
                    if ( mean_col_coverage <= 0.0129480361938 ) {
                      return 0.132653061224 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0129480361938
                      return 0.0138030142663 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.117129981518
                    if ( max_col_coverage <= 0.117176860571 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.117176860571
                      return 0.0356873163754 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.0472782440484
                  if ( median_col_coverage <= 0.00291121425107 ) {
                    if ( min_col_support <= 0.694499969482 ) {
                      return 0.0327777777778 < maxgini;
                    }
                    else {  // if min_col_support > 0.694499969482
                      return 0.105681637191 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.00291121425107
                    if ( min_col_support <= 0.619500041008 ) {
                      return 0.0296144961993 < maxgini;
                    }
                    else {  // if min_col_support > 0.619500041008
                      return 0.0492806954424 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.232054024935
                if ( mean_col_coverage <= 0.0764906853437 ) {
                  if ( min_col_coverage <= 0.00352117046714 ) {
                    if ( max_col_coverage <= 0.242613255978 ) {
                      return 0.454299621417 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.242613255978
                      return 0.137174211248 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00352117046714
                    if ( max_col_coverage <= 0.245551601052 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.245551601052
                      return false;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.0764906853437
                  if ( max_col_coverage <= 0.294221937656 ) {
                    if ( median_col_support <= 0.978500008583 ) {
                      return 0.144711118826 < maxgini;
                    }
                    else {  // if median_col_support > 0.978500008583
                      return 0.0151124739107 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.294221937656
                    if ( median_col_coverage <= 0.00353983417153 ) {
                      return 0.186368 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00353983417153
                      return 0.474802165764 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.00520156929269
              if ( median_col_support <= 0.948500037193 ) {
                if ( mean_col_coverage <= 0.36591565609 ) {
                  if ( min_col_support <= 0.87349998951 ) {
                    if ( min_col_coverage <= 0.00261438358575 ) {
                      return 0.137737386621 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00261438358575
                      return 0.033024166254 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.87349998951
                    if ( median_col_coverage <= 0.345806747675 ) {
                      return 0.0378264221332 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.345806747675
                      return false;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.36591565609
                  if ( median_col_coverage <= 0.312076717615 ) {
                    if ( mean_col_support <= 0.971441149712 ) {
                      return 0.0195086319303 < maxgini;
                    }
                    else {  // if mean_col_support > 0.971441149712
                      return 0.0314604898271 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.312076717615
                    if ( mean_col_coverage <= 0.505600094795 ) {
                      return 0.0322219347951 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.505600094795
                      return 0.237812128419 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.948500037193
                if ( mean_col_coverage <= 0.117452599108 ) {
                  if ( max_col_coverage <= 0.158891305327 ) {
                    if ( median_col_coverage <= 0.0683705210686 ) {
                      return 0.0286357422014 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0683705210686
                      return 0.12084201369 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.158891305327
                    if ( min_col_coverage <= 0.0920364111662 ) {
                      return 0.0177092256692 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0920364111662
                      return false;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.117452599108
                  if ( median_col_coverage <= 0.0355929471552 ) {
                    if ( mean_col_coverage <= 0.117453239858 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.117453239858
                      return 0.0700725126549 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0355929471552
                    if ( min_col_coverage <= 0.00353983417153 ) {
                      return 0.0637426104382 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00353983417153
                      return 0.0266249816348 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.980485320091
          if ( min_col_support <= 0.976500034332 ) {
            if ( max_col_coverage <= 0.445408582687 ) {
              if ( max_col_coverage <= 0.306409984827 ) {
                if ( mean_col_coverage <= 0.15450437367 ) {
                  if ( median_col_coverage <= 0.00243605719879 ) {
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.150761978109 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.0274389413989 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.00243605719879
                    if ( max_col_coverage <= 0.284709870815 ) {
                      return 0.0209153696538 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.284709870815
                      return 0.0376723402447 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.15450437367
                  if ( median_col_coverage <= 0.101395130157 ) {
                    if ( mean_col_coverage <= 0.168536305428 ) {
                      return 0.0158612727091 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.168536305428
                      return 0.00746472439768 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.101395130157
                    if ( max_col_coverage <= 0.306390166283 ) {
                      return 0.0271185567429 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.306390166283
                      return 0.108612467682 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.306409984827
                if ( median_col_support <= 0.982499957085 ) {
                  if ( median_col_coverage <= 0.00343054183759 ) {
                    if ( median_col_support <= 0.978500008583 ) {
                      return 0.488165680473 < maxgini;
                    }
                    else {  // if median_col_support > 0.978500008583
                      return 0.132653061224 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.00343054183759
                    if ( median_col_coverage <= 0.243981078267 ) {
                      return 0.0231252855748 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.243981078267
                      return 0.0281355582425 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.982499957085
                  if ( median_col_support <= 0.989500045776 ) {
                    if ( median_col_support <= 0.985499978065 ) {
                      return 0.0201586703639 < maxgini;
                    }
                    else {  // if median_col_support > 0.985499978065
                      return 0.0175497592978 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.989500045776
                    if ( median_col_coverage <= 0.00346621498466 ) {
                      return 0.18 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00346621498466
                      return 0.0111934326647 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.445408582687
              if ( mean_col_support <= 0.991188287735 ) {
                if ( min_col_coverage <= 0.111492373049 ) {
                  if ( mean_col_coverage <= 0.281502008438 ) {
                    if ( median_col_support <= 0.972499966621 ) {
                      return 0.110726643599 < maxgini;
                    }
                    else {  // if median_col_support > 0.972499966621
                      return 0.00647242368022 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.281502008438
                    if ( mean_col_support <= 0.984088301659 ) {
                      return 0.301783264746 < maxgini;
                    }
                    else {  // if mean_col_support > 0.984088301659
                      return 0.132108641975 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.111492373049
                  if ( median_col_support <= 0.979499995708 ) {
                    if ( mean_col_coverage <= 0.414485454559 ) {
                      return 0.0244250200194 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.414485454559
                      return 0.0225352195738 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.979499995708
                    if ( median_col_support <= 0.985499978065 ) {
                      return 0.0180467809363 < maxgini;
                    }
                    else {  // if median_col_support > 0.985499978065
                      return 0.013676426009 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.991188287735
                if ( mean_col_support <= 0.994558811188 ) {
                  if ( min_col_support <= 0.96850001812 ) {
                    if ( mean_col_support <= 0.991970658302 ) {
                      return 0.0125674939049 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991970658302
                      return 0.00990689382471 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.96850001812
                    if ( mean_col_coverage <= 0.505612194538 ) {
                      return 0.0157022894908 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.505612194538
                      return 0.32 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.994558811188
                  if ( min_col_support <= 0.941499948502 ) {
                    if ( min_col_coverage <= 0.358038365841 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.358038365841
                      return 0.184688581315 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.941499948502
                    if ( min_col_support <= 0.974500000477 ) {
                      return 0.00607712986979 < maxgini;
                    }
                    else {  // if min_col_support > 0.974500000477
                      return 0.00922718090865 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.976500034332
            if ( mean_col_support <= 0.995617628098 ) {
              if ( mean_col_support <= 0.993735313416 ) {
                if ( median_col_support <= 0.981500029564 ) {
                  if ( min_col_coverage <= 0.152079731226 ) {
                    if ( min_col_support <= 0.977499961853 ) {
                      return 0.139793668547 < maxgini;
                    }
                    else {  // if min_col_support > 0.977499961853
                      return 0.0360239037118 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.152079731226
                    if ( max_col_support <= 0.99849998951 ) {
                      return 0.0420226579849 < maxgini;
                    }
                    else {  // if max_col_support > 0.99849998951
                      return 0.0253709468403 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.981500029564
                  if ( mean_col_coverage <= 0.344774961472 ) {
                    if ( max_col_coverage <= 0.383374929428 ) {
                      return 0.0324746844672 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.383374929428
                      return 0.0258916206938 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.344774961472
                    if ( min_col_coverage <= 0.195078700781 ) {
                      return 0.135199652778 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.195078700781
                      return 0.0184922398881 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.993735313416
                if ( min_col_coverage <= 0.286137878895 ) {
                  if ( max_col_coverage <= 0.436146944761 ) {
                    if ( min_col_support <= 0.981500029564 ) {
                      return 0.0183924917784 < maxgini;
                    }
                    else {  // if min_col_support > 0.981500029564
                      return 0.0253136292055 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.436146944761
                    if ( min_col_coverage <= 0.286131441593 ) {
                      return 0.0141139535632 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.286131441593
                      return 0.135633551457 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.286137878895
                  if ( min_col_coverage <= 0.451258510351 ) {
                    if ( max_col_support <= 0.99950003624 ) {
                      return 0.00283687370868 < maxgini;
                    }
                    else {  // if max_col_support > 0.99950003624
                      return 0.014846030565 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.451258510351
                    if ( min_col_coverage <= 0.451267361641 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.451267361641
                      return 0.0321015425911 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.995617628098
              if ( median_col_support <= 0.994500041008 ) {
                if ( mean_col_coverage <= 0.346013724804 ) {
                  if ( mean_col_coverage <= 0.346012890339 ) {
                    if ( median_col_coverage <= 0.00285996776074 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00285996776074
                      return 0.0163336564305 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.346012890339
                    return false;
                  }
                }
                else {  // if mean_col_coverage > 0.346013724804
                  if ( min_col_support <= 0.988499999046 ) {
                    if ( min_col_support <= 0.984500050545 ) {
                      return 0.00860287098389 < maxgini;
                    }
                    else {  // if min_col_support > 0.984500050545
                      return 0.0108827344455 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.988499999046
                    if ( min_col_support <= 0.990499973297 ) {
                      return 0.0145253731497 < maxgini;
                    }
                    else {  // if min_col_support > 0.990499973297
                      return 0.0110310762108 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.994500041008
                if ( median_col_support <= 0.997500002384 ) {
                  if ( mean_col_coverage <= 0.380898028612 ) {
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.0133806232564 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.00730796625403 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.380898028612
                    if ( median_col_coverage <= 0.390612334013 ) {
                      return 0.00550556166192 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.390612334013
                      return 0.00717617334855 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.997500002384
                  if ( min_col_support <= 0.977499961853 ) {
                    if ( max_col_coverage <= 0.230336502194 ) {
                      return 0.0773933402706 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.230336502194
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.977499961853
                    if ( mean_col_support <= 0.998500049114 ) {
                      return 0.00550374798027 < maxgini;
                    }
                    else {  // if mean_col_support > 0.998500049114
                      return 0.00234266529067 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_coverage > 0.505620241165
        if ( mean_col_support <= 0.980558872223 ) {
          if ( median_col_coverage <= 0.983579218388 ) {
            if ( mean_col_support <= 0.970617651939 ) {
              if ( min_col_coverage <= 0.907855033875 ) {
                if ( min_col_support <= 0.567499995232 ) {
                  if ( min_col_coverage <= 0.784161806107 ) {
                    if ( min_col_coverage <= 0.71948236227 ) {
                      return 0.0700845817207 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.71948236227
                      return 0.330906508876 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.784161806107
                    if ( max_col_coverage <= 0.971250772476 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.971250772476
                      return 0.47181822887 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.567499995232
                  if ( min_col_support <= 0.672500014305 ) {
                    if ( max_col_coverage <= 0.934057235718 ) {
                      return 0.0354808997772 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.934057235718
                      return 0.153880444908 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.672500014305
                    if ( max_col_coverage <= 0.668813765049 ) {
                      return 0.0301379423955 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.668813765049
                      return 0.0205712470073 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.907855033875
                if ( min_col_support <= 0.619500041008 ) {
                  if ( min_col_coverage <= 0.975481271744 ) {
                    if ( max_col_coverage <= 0.998583555222 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.998583555222
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.975481271744
                    return 0.0 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.619500041008
                  if ( mean_col_support <= 0.965088248253 ) {
                    if ( min_col_support <= 0.65649998188 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.65649998188
                      return 0.233303875773 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.965088248253
                    if ( mean_col_support <= 0.966499984264 ) {
                      return 0.0997229916898 < maxgini;
                    }
                    else {  // if mean_col_support > 0.966499984264
                      return 0.0210502489814 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.970617651939
              if ( min_col_support <= 0.599500000477 ) {
                if ( max_col_coverage <= 0.902165234089 ) {
                  if ( min_col_support <= 0.573500037193 ) {
                    if ( mean_col_support <= 0.973500013351 ) {
                      return 0.293367346939 < maxgini;
                    }
                    else {  // if mean_col_support > 0.973500013351
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.573500037193
                    if ( min_col_coverage <= 0.7441188097 ) {
                      return 0.131784115487 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.7441188097
                      return false;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.902165234089
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( median_col_coverage <= 0.922790706158 ) {
                      return 0.310650887574 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.922790706158
                      return false;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( median_col_coverage <= 0.943562626839 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.943562626839
                      return false;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.599500000477
                if ( min_col_support <= 0.650499999523 ) {
                  if ( max_col_coverage <= 0.831267356873 ) {
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.0543791872463 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.48 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.831267356873
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.39143663825 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.650499999523
                  if ( mean_col_coverage <= 0.643661379814 ) {
                    if ( max_col_support <= 0.986000001431 ) {
                      return false;
                    }
                    else {  // if max_col_support > 0.986000001431
                      return 0.0206496089677 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.643661379814
                    if ( max_col_support <= 0.992499947548 ) {
                      return 0.158790170132 < maxgini;
                    }
                    else {  // if max_col_support > 0.992499947548
                      return 0.0106772849588 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.983579218388
            if ( min_col_support <= 0.694999992847 ) {
              if ( mean_col_support <= 0.975558757782 ) {
                if ( min_col_support <= 0.619500041008 ) {
                  if ( min_col_support <= 0.557000041008 ) {
                    if ( min_col_support <= 0.541000008583 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.541000008583
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.557000041008
                    if ( mean_col_coverage <= 0.998292088509 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.998292088509
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.619500041008
                  if ( max_col_coverage <= 0.998724460602 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if max_col_coverage > 0.998724460602
                    if ( min_col_coverage <= 0.977006018162 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.977006018162
                      return false;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.975558757782
                if ( median_col_coverage <= 0.995401859283 ) {
                  if ( mean_col_coverage <= 0.99470937252 ) {
                    return false;
                  }
                  else {  // if mean_col_coverage > 0.99470937252
                    return 0.0 < maxgini;
                  }
                }
                else {  // if median_col_coverage > 0.995401859283
                  if ( mean_col_coverage <= 0.999812066555 ) {
                    if ( min_col_support <= 0.638499975204 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.638499975204
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.999812066555
                    return false;
                  }
                }
              }
            }
            else {  // if min_col_support > 0.694999992847
              if ( max_col_coverage <= 0.991112709045 ) {
                return false;
              }
              else {  // if max_col_coverage > 0.991112709045
                if ( max_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if max_col_support > 0.99950003624
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( median_col_support <= 0.980499982834 ) {
                      return 0.120707596254 < maxgini;
                    }
                    else {  // if median_col_support > 0.980499982834
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( mean_col_coverage <= 0.998304605484 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.998304605484
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.980558872223
          if ( median_col_support <= 0.989500045776 ) {
            if ( max_col_coverage <= 0.75953322649 ) {
              if ( min_col_coverage <= 0.483899772167 ) {
                if ( median_col_support <= 0.986500024796 ) {
                  if ( median_col_support <= 0.979499995708 ) {
                    if ( median_col_coverage <= 0.501675069332 ) {
                      return 0.0199085321798 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.501675069332
                      return 0.0146786217338 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.979499995708
                    if ( min_col_support <= 0.956499993801 ) {
                      return 0.0111539622058 < maxgini;
                    }
                    else {  // if min_col_support > 0.956499993801
                      return 0.0167739138349 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.986500024796
                  if ( min_col_support <= 0.961500048637 ) {
                    if ( median_col_coverage <= 0.456640005112 ) {
                      return 0.011220223018 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.456640005112
                      return 0.00698986596286 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.961500048637
                    if ( mean_col_coverage <= 0.526530921459 ) {
                      return 0.0142599143858 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.526530921459
                      return 0.0103518623088 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.483899772167
                if ( median_col_support <= 0.984500050545 ) {
                  if ( min_col_coverage <= 0.488771051168 ) {
                    if ( median_col_coverage <= 0.555394053459 ) {
                      return 0.0102240796219 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.555394053459
                      return 0.0713305898491 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.488771051168
                    if ( median_col_coverage <= 0.53165769577 ) {
                      return 0.0159533691139 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.53165769577
                      return 0.0129902173878 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.984500050545
                  if ( mean_col_support <= 0.996676504612 ) {
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.010322408581 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.00772616125117 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.996676504612
                    if ( median_col_support <= 0.988499999046 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.988499999046
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.75953322649
              if ( median_col_coverage <= 0.708443403244 ) {
                if ( mean_col_coverage <= 0.678517103195 ) {
                  if ( median_col_support <= 0.977499961853 ) {
                    if ( mean_col_support <= 0.986617684364 ) {
                      return 0.0130206551369 < maxgini;
                    }
                    else {  // if mean_col_support > 0.986617684364
                      return 0.0290728298844 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.977499961853
                    if ( mean_col_coverage <= 0.678492426872 ) {
                      return 0.00881235418543 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.678492426872
                      return 0.0658574380165 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.678517103195
                  if ( max_col_support <= 0.993499994278 ) {
                    if ( mean_col_support <= 0.98402941227 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.98402941227
                      return false;
                    }
                  }
                  else {  // if max_col_support > 0.993499994278
                    if ( median_col_coverage <= 0.594072699547 ) {
                      return 0.0239485766758 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.594072699547
                      return 0.00700378309495 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.708443403244
                if ( min_col_coverage <= 0.621520280838 ) {
                  if ( mean_col_coverage <= 0.872872710228 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if mean_col_coverage > 0.872872710228
                    return false;
                  }
                }
                else {  // if min_col_coverage > 0.621520280838
                  if ( max_col_coverage <= 0.86870098114 ) {
                    if ( min_col_coverage <= 0.715582609177 ) {
                      return 0.00515333628898 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.715582609177
                      return 0.00872130235693 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.86870098114
                    if ( min_col_coverage <= 0.984933018684 ) {
                      return 0.00220974043921 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.984933018684
                      return 0.0431444636678 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.989500045776
            if ( mean_col_coverage <= 0.663151502609 ) {
              if ( mean_col_coverage <= 0.663150906563 ) {
                if ( median_col_coverage <= 0.509917855263 ) {
                  if ( min_col_support <= 0.991500020027 ) {
                    if ( mean_col_support <= 0.995499968529 ) {
                      return 0.00866934667189 < maxgini;
                    }
                    else {  // if mean_col_support > 0.995499968529
                      return 0.00580538906312 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.991500020027
                    if ( mean_col_coverage <= 0.514132261276 ) {
                      return 0.00914736089935 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.514132261276
                      return 0.00395119084668 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.509917855263
                  if ( max_col_support <= 0.996500015259 ) {
                    if ( min_col_support <= 0.962499976158 ) {
                      return 0.32 < maxgini;
                    }
                    else {  // if min_col_support > 0.962499976158
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.996500015259
                    if ( mean_col_coverage <= 0.539231181145 ) {
                      return 0.0392 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.539231181145
                      return 0.00396785070122 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.663150906563
                return false;
              }
            }
            else {  // if mean_col_coverage > 0.663151502609
              if ( min_col_coverage <= 0.655196070671 ) {
                if ( mean_col_support <= 0.996147036552 ) {
                  if ( median_col_support <= 0.991500020027 ) {
                    if ( min_col_coverage <= 0.652594327927 ) {
                      return 0.00445303759524 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.652594327927
                      return 0.00990074503106 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.991500020027
                    if ( min_col_support <= 0.985499978065 ) {
                      return 0.00242630482226 < maxgini;
                    }
                    else {  // if min_col_support > 0.985499978065
                      return 0.00440924172833 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.996147036552
                  if ( max_col_coverage <= 0.770920753479 ) {
                    if ( min_col_coverage <= 0.637245476246 ) {
                      return 0.00239953102802 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.637245476246
                      return 0.00570682745709 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.770920753479
                    if ( min_col_coverage <= 0.586707532406 ) {
                      return 0.00328406336565 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.586707532406
                      return 0.00112987502544 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.655196070671
                if ( min_col_support <= 0.733500003815 ) {
                  if ( median_col_support <= 0.997500002384 ) {
                    if ( min_col_support <= 0.721000015736 ) {
                      return 0.104938271605 < maxgini;
                    }
                    else {  // if min_col_support > 0.721000015736
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.997500002384
                    if ( max_col_coverage <= 0.987804889679 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.987804889679
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.733500003815
                  if ( min_col_support <= 0.809499979019 ) {
                    if ( median_col_coverage <= 0.998659491539 ) {
                      return 0.0207016683882 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.998659491539
                      return 0.408163265306 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.809499979019
                    if ( max_col_coverage <= 0.767288327217 ) {
                      return 0.0109910910355 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.767288327217
                      return 0.000810625011737 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    bool shouldCorrect5(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( max_col_coverage <= 0.581996798515 ) {
        if ( mean_col_support <= 0.980414271355 ) {
          if ( mean_col_support <= 0.956303775311 ) {
            if ( min_col_support <= 0.653499960899 ) {
              if ( min_col_support <= 0.432500004768 ) {
                if ( median_col_coverage <= 0.241146147251 ) {
                  if ( max_col_support <= 0.985499978065 ) {
                    if ( median_col_coverage <= 0.0104011446238 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.0104011446238
                      return 0.0103550972269 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.985499978065
                    if ( mean_col_support <= 0.763382315636 ) {
                      return 0.019789192109 < maxgini;
                    }
                    else {  // if mean_col_support > 0.763382315636
                      return 0.0311007409599 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.241146147251
                  if ( median_col_coverage <= 0.241181790829 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.241181790829
                    if ( max_col_coverage <= 0.356118828058 ) {
                      return 0.184089414859 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.356118828058
                      return 0.0530404931957 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.432500004768
                if ( max_col_support <= 0.99950003624 ) {
                  if ( max_col_support <= 0.979499995708 ) {
                    if ( mean_col_coverage <= 0.239044174552 ) {
                      return 0.0108281424996 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.239044174552
                      return 0.0223795305227 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.979499995708
                    if ( max_col_coverage <= 0.410361111164 ) {
                      return 0.0385151572428 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.410361111164
                      return 0.0293012288772 < maxgini;
                    }
                  }
                }
                else {  // if max_col_support > 0.99950003624
                  if ( mean_col_support <= 0.938121378422 ) {
                    if ( max_col_coverage <= 0.372738957405 ) {
                      return 0.0509225308518 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.372738957405
                      return 0.0438690848824 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.938121378422
                    if ( min_col_support <= 0.517500042915 ) {
                      return 0.0246963991916 < maxgini;
                    }
                    else {  // if min_col_support > 0.517500042915
                      return 0.0411374610431 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.653499960899
              if ( min_col_coverage <= 0.00601505115628 ) {
                if ( median_col_coverage <= 0.00349041214213 ) {
                  if ( mean_col_support <= 0.956258296967 ) {
                    if ( mean_col_support <= 0.937121331692 ) {
                      return 0.0474918988791 < maxgini;
                    }
                    else {  // if mean_col_support > 0.937121331692
                      return 0.0753622248981 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.956258296967
                    if ( min_col_support <= 0.655499994755 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.655499994755
                      return 0.281183431953 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.00349041214213
                  if ( min_col_support <= 0.897500038147 ) {
                    if ( min_col_coverage <= 0.00234466907568 ) {
                      return 0.173946534549 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00234466907568
                      return 0.048196628314 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.897500038147
                    return false;
                  }
                }
              }
              else {  // if min_col_coverage > 0.00601505115628
                if ( min_col_support <= 0.888499975204 ) {
                  if ( median_col_coverage <= 0.0747793018818 ) {
                    if ( median_col_coverage <= 0.0105671025813 ) {
                      return 0.0522736999556 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0105671025813
                      return 0.0319510830798 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0747793018818
                    if ( median_col_support <= 0.90750002861 ) {
                      return 0.0390537651945 < maxgini;
                    }
                    else {  // if median_col_support > 0.90750002861
                      return 0.0336948807476 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.888499975204
                  if ( mean_col_coverage <= 0.0737695395947 ) {
                    if ( median_col_coverage <= 0.0416865721345 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.0416865721345
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.0737695395947
                    if ( max_col_support <= 0.994500041008 ) {
                      return 0.0268306680164 < maxgini;
                    }
                    else {  // if max_col_support > 0.994500041008
                      return 0.0530045431329 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.956303775311
            if ( min_col_coverage <= 0.0051746526733 ) {
              if ( median_col_support <= 0.964499950409 ) {
                if ( median_col_coverage <= 0.00509554985911 ) {
                  if ( max_col_coverage <= 0.220566019416 ) {
                    if ( min_col_coverage <= 0.00291121425107 ) {
                      return 0.109207125199 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00291121425107
                      return 0.0524654418123 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.220566019416
                    if ( min_col_coverage <= 0.00353983417153 ) {
                      return 0.140011547683 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00353983417153
                      return 0.25369391613 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.00509554985911
                  if ( max_col_coverage <= 0.337677180767 ) {
                    if ( min_col_support <= 0.707499980927 ) {
                      return 0.0231001852037 < maxgini;
                    }
                    else {  // if min_col_support > 0.707499980927
                      return 0.0447917110754 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.337677180767
                    if ( mean_col_support <= 0.961499929428 ) {
                      return 0.0475907198096 < maxgini;
                    }
                    else {  // if mean_col_support > 0.961499929428
                      return 0.390539516876 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.964499950409
                if ( mean_col_support <= 0.980405926704 ) {
                  if ( median_col_support <= 0.99849998951 ) {
                    if ( mean_col_coverage <= 0.164823457599 ) {
                      return 0.03095392869 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.164823457599
                      return 0.132653061224 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99849998951
                    if ( mean_col_support <= 0.956400632858 ) {
                      return 0.408163265306 < maxgini;
                    }
                    else {  // if mean_col_support > 0.956400632858
                      return 0.0406276108002 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.980405926704
                  if ( mean_col_coverage <= 0.0758414268494 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if mean_col_coverage > 0.0758414268494
                    if ( mean_col_coverage <= 0.0781555995345 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.0781555995345
                      return 0.241305016928 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.0051746526733
              if ( max_col_coverage <= 0.491577744484 ) {
                if ( min_col_coverage <= 0.0425293892622 ) {
                  if ( max_col_coverage <= 0.329842835665 ) {
                    if ( max_col_coverage <= 0.266318023205 ) {
                      return 0.0158898332244 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.266318023205
                      return 0.0275396023958 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.329842835665
                    if ( median_col_coverage <= 0.00843481533229 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.00843481533229
                      return 0.0938881268099 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.0425293892622
                  if ( median_col_support <= 0.962499976158 ) {
                    if ( min_col_support <= 0.934499979019 ) {
                      return 0.0330481147259 < maxgini;
                    }
                    else {  // if min_col_support > 0.934499979019
                      return 0.0439839441129 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.962499976158
                    if ( min_col_coverage <= 0.408992886543 ) {
                      return 0.0245754195258 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.408992886543
                      return 0.415224913495 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.491577744484
                if ( min_col_support <= 0.575500011444 ) {
                  if ( median_col_support <= 0.996999979019 ) {
                    if ( min_col_coverage <= 0.244321405888 ) {
                      return 0.255 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.244321405888
                      return 0.0784593012268 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.996999979019
                    return false;
                  }
                }
                else {  // if min_col_support > 0.575500011444
                  if ( min_col_coverage <= 0.122044071555 ) {
                    if ( median_col_coverage <= 0.196052074432 ) {
                      return 0.198738324612 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.196052074432
                      return 0.497777777778 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.122044071555
                    if ( max_col_support <= 0.978999972343 ) {
                      return false;
                    }
                    else {  // if max_col_support > 0.978999972343
                      return 0.0263496200548 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.980414271355
          if ( median_col_coverage <= 0.311632484198 ) {
            if ( mean_col_coverage <= 0.353900611401 ) {
              if ( min_col_support <= 0.979499995708 ) {
                if ( median_col_coverage <= 0.00271370913833 ) {
                  if ( min_col_support <= 0.941499948502 ) {
                    if ( median_col_support <= 0.971500039101 ) {
                      return 0.207026069242 < maxgini;
                    }
                    else {  // if median_col_support > 0.971500039101
                      return 0.0430771594332 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.941499948502
                    if ( median_col_coverage <= 0.00224467180669 ) {
                      return 0.0768 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00224467180669
                      return 0.00679493801653 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.00271370913833
                  if ( median_col_coverage <= 0.269401460886 ) {
                    if ( min_col_support <= 0.920500040054 ) {
                      return 0.018651990387 < maxgini;
                    }
                    else {  // if min_col_support > 0.920500040054
                      return 0.0214593512932 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.269401460886
                    if ( min_col_coverage <= 0.244783893228 ) {
                      return 0.0193810766386 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.244783893228
                      return 0.0255816960036 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.979499995708
                if ( mean_col_support <= 0.996029496193 ) {
                  if ( min_col_coverage <= 0.222885623574 ) {
                    if ( mean_col_coverage <= 0.301234722137 ) {
                      return 0.0224616389279 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.301234722137
                      return 0.0132546504077 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.222885623574
                    if ( min_col_support <= 0.980499982834 ) {
                      return 0.0199839874553 < maxgini;
                    }
                    else {  // if min_col_support > 0.980499982834
                      return 0.0265488480396 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.996029496193
                  if ( mean_col_support <= 0.998029470444 ) {
                    if ( median_col_coverage <= 0.00300841825083 ) {
                      return 0.408163265306 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00300841825083
                      return 0.0134505993955 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.998029470444
                    if ( mean_col_coverage <= 0.353312313557 ) {
                      return 0.00511763390349 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.353312313557
                      return 0.0775431078461 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.353900611401
              if ( median_col_support <= 0.989500045776 ) {
                if ( max_col_coverage <= 0.419518619776 ) {
                  if ( min_col_support <= 0.866500020027 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.866500020027
                    if ( median_col_support <= 0.979499995708 ) {
                      return 0.119808 < maxgini;
                    }
                    else {  // if median_col_support > 0.979499995708
                      return 0.0436463647959 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.419518619776
                  if ( median_col_coverage <= 0.21416041255 ) {
                    if ( min_col_coverage <= 0.157289132476 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.157289132476
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.21416041255
                    if ( median_col_coverage <= 0.304863989353 ) {
                      return 0.0171043190036 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.304863989353
                      return 0.0215355904054 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.989500045776
                if ( median_col_coverage <= 0.311588287354 ) {
                  if ( min_col_support <= 0.969500005245 ) {
                    if ( median_col_coverage <= 0.311491668224 ) {
                      return 0.00470286551642 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.311491668224
                      return 0.0454299621417 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.969500005245
                    if ( median_col_coverage <= 0.284648835659 ) {
                      return 0.00572818017202 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.284648835659
                      return 0.0115520163596 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.311588287354
                  if ( min_col_coverage <= 0.293151140213 ) {
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.128418549346 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.293151140213
                    if ( mean_col_coverage <= 0.368286430836 ) {
                      return 0.48 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.368286430836
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.311632484198
            if ( median_col_support <= 0.985499978065 ) {
              if ( mean_col_coverage <= 0.322858631611 ) {
                if ( median_col_support <= 0.978000044823 ) {
                  return false;
                }
                else {  // if median_col_support > 0.978000044823
                  return 0.0 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.322858631611
                if ( min_col_support <= 0.946500003338 ) {
                  if ( median_col_support <= 0.976500034332 ) {
                    if ( max_col_coverage <= 0.5190346241 ) {
                      return 0.0251498769755 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.5190346241
                      return 0.0204152830128 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.976500034332
                    if ( min_col_coverage <= 0.418679893017 ) {
                      return 0.0155737223133 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.418679893017
                      return 0.023873302703 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.946500003338
                  if ( median_col_support <= 0.977499961853 ) {
                    if ( median_col_support <= 0.949499964714 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.949499964714
                      return 0.0273136820635 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.977499961853
                    if ( mean_col_support <= 0.980852901936 ) {
                      return 0.396694214876 < maxgini;
                    }
                    else {  // if mean_col_support > 0.980852901936
                      return 0.0199258188456 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.985499978065
              if ( mean_col_support <= 0.995676517487 ) {
                if ( min_col_support <= 0.96850001812 ) {
                  if ( mean_col_coverage <= 0.382504582405 ) {
                    if ( max_col_support <= 0.99849998951 ) {
                      return 0.051466164684 < maxgini;
                    }
                    else {  // if max_col_support > 0.99849998951
                      return 0.0146605799584 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.382504582405
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.0120717316874 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.00891532758825 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.96850001812
                  if ( min_col_support <= 0.982499957085 ) {
                    if ( mean_col_support <= 0.993911802769 ) {
                      return 0.0168228661652 < maxgini;
                    }
                    else {  // if mean_col_support > 0.993911802769
                      return 0.0116163251596 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.982499957085
                    if ( min_col_support <= 0.988499999046 ) {
                      return 0.0175132435695 < maxgini;
                    }
                    else {  // if min_col_support > 0.988499999046
                      return 0.0107289679281 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.995676517487
                if ( min_col_coverage <= 0.380271464586 ) {
                  if ( median_col_support <= 0.996500015259 ) {
                    if ( max_col_coverage <= 0.475529819727 ) {
                      return 0.0113451368383 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.475529819727
                      return 0.00774728188355 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.996500015259
                    if ( median_col_coverage <= 0.443472743034 ) {
                      return 0.00356703799411 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.443472743034
                      return 0.297520661157 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.380271464586
                  if ( median_col_coverage <= 0.393421828747 ) {
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.0586465393784 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.393421828747
                    if ( mean_col_coverage <= 0.464622497559 ) {
                      return 0.0137522412098 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.464622497559
                      return 0.00956283321133 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if max_col_coverage > 0.581996798515
        if ( median_col_coverage <= 0.540131926537 ) {
          if ( mean_col_support <= 0.987676501274 ) {
            if ( median_col_coverage <= 0.30014577508 ) {
              if ( mean_col_support <= 0.985617637634 ) {
                if ( median_col_coverage <= 0.299559056759 ) {
                  if ( median_col_support <= 0.953500032425 ) {
                    if ( max_col_coverage <= 0.58456659317 ) {
                      return 0.21875 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.58456659317
                      return 0.0375449280007 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.953500032425
                    if ( mean_col_support <= 0.974735319614 ) {
                      return 0.0205106177347 < maxgini;
                    }
                    else {  // if mean_col_support > 0.974735319614
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.299559056759
                  if ( min_col_coverage <= 0.26308208704 ) {
                    if ( min_col_coverage <= 0.222911119461 ) {
                      return 0.46875 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.222911119461
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.26308208704
                    return false;
                  }
                }
              }
              else {  // if mean_col_support > 0.985617637634
                if ( median_col_support <= 0.972499966621 ) {
                  if ( median_col_coverage <= 0.275609314442 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.275609314442
                    if ( mean_col_support <= 0.985823512077 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.985823512077
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.972499966621
                  return 0.0 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.30014577508
              if ( min_col_support <= 0.581499993801 ) {
                if ( max_col_coverage <= 0.984435796738 ) {
                  if ( mean_col_support <= 0.968264698982 ) {
                    if ( min_col_support <= 0.331000000238 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.331000000238
                      return 0.0470377804818 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.968264698982
                    if ( median_col_coverage <= 0.487422972918 ) {
                      return 0.210680108116 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.487422972918
                      return 0.497488994978 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.984435796738
                  return false;
                }
              }
              else {  // if min_col_support > 0.581499993801
                if ( min_col_support <= 0.74950003624 ) {
                  if ( max_col_coverage <= 0.955851972103 ) {
                    if ( median_col_coverage <= 0.43175047636 ) {
                      return 0.0220868213077 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.43175047636
                      return 0.029679785502 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.955851972103
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.355029585799 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.74950003624
                  if ( mean_col_coverage <= 0.554444253445 ) {
                    if ( min_col_coverage <= 0.374617695808 ) {
                      return 0.0191943038679 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.374617695808
                      return 0.0223829039853 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.554444253445
                    if ( mean_col_coverage <= 0.555645942688 ) {
                      return 0.0128718775627 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.555645942688
                      return 0.0185175776512 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.987676501274
            if ( min_col_support <= 0.986500024796 ) {
              if ( mean_col_coverage <= 0.543144285679 ) {
                if ( median_col_support <= 0.989500045776 ) {
                  if ( min_col_support <= 0.956499993801 ) {
                    if ( min_col_support <= 0.939499974251 ) {
                      return 0.00703381068807 < maxgini;
                    }
                    else {  // if min_col_support > 0.939499974251
                      return 0.0116808262976 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.956499993801
                    if ( median_col_support <= 0.986500024796 ) {
                      return 0.0180239241332 < maxgini;
                    }
                    else {  // if median_col_support > 0.986500024796
                      return 0.0130761338934 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.989500045776
                  if ( median_col_coverage <= 0.4693364501 ) {
                    if ( max_col_coverage <= 0.709185123444 ) {
                      return 0.00818282022596 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.709185123444
                      return 0.0480189349112 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.4693364501
                    if ( min_col_coverage <= 0.440420031548 ) {
                      return 0.00861674740744 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.440420031548
                      return 0.00638343565726 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.543144285679
                if ( mean_col_support <= 0.994617700577 ) {
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( mean_col_coverage <= 0.543347716331 ) {
                      return 0.0027548156956 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.543347716331
                      return 0.011120381059 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( min_col_support <= 0.966500043869 ) {
                      return 0.00317480224695 < maxgini;
                    }
                    else {  // if min_col_support > 0.966500043869
                      return 0.00784071489425 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.994617700577
                  if ( min_col_support <= 0.975499987602 ) {
                    if ( median_col_coverage <= 0.349580347538 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.349580347538
                      return 0.0023814295116 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.975499987602
                    if ( min_col_coverage <= 0.504795432091 ) {
                      return 0.00501790585041 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.504795432091
                      return 0.00680534513446 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.986500024796
              if ( mean_col_support <= 0.997088193893 ) {
                if ( median_col_support <= 0.987499952316 ) {
                  if ( min_col_coverage <= 0.453482329845 ) {
                    if ( median_col_coverage <= 0.484731793404 ) {
                      return 0.158790170132 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.484731793404
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.453482329845
                    return 0.0 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.987499952316
                  if ( mean_col_coverage <= 0.538509607315 ) {
                    if ( mean_col_coverage <= 0.538509368896 ) {
                      return 0.0104042477298 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.538509368896
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.538509607315
                    if ( median_col_coverage <= 0.540064513683 ) {
                      return 0.00608849376071 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.540064513683
                      return 0.0422338388282 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.997088193893
                if ( mean_col_coverage <= 0.543309926987 ) {
                  if ( min_col_coverage <= 0.411479353905 ) {
                    if ( min_col_coverage <= 0.276662647724 ) {
                      return 0.0220358315035 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.276662647724
                      return 0.0029887988699 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.411479353905
                    if ( min_col_coverage <= 0.411486983299 ) {
                      return 0.297520661157 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.411486983299
                      return 0.00517509048089 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.543309926987
                  if ( min_col_coverage <= 0.490210831165 ) {
                    if ( min_col_coverage <= 0.490182310343 ) {
                      return 0.0032078863061 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.490182310343
                      return 0.0291198524594 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.490210831165
                    if ( min_col_support <= 0.991500020027 ) {
                      return 0.00288355312439 < maxgini;
                    }
                    else {  // if min_col_support > 0.991500020027
                      return 0.00158162056682 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.540131926537
          if ( min_col_support <= 0.595499992371 ) {
            if ( median_col_coverage <= 0.836125731468 ) {
              if ( median_col_support <= 0.993499994278 ) {
                if ( median_col_support <= 0.952499985695 ) {
                  if ( median_col_support <= 0.926499962807 ) {
                    if ( max_col_coverage <= 0.63577491045 ) {
                      return 0.2112 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.63577491045
                      return 0.0561820570193 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.926499962807
                    if ( min_col_support <= 0.579499959946 ) {
                      return 0.134390334452 < maxgini;
                    }
                    else {  // if min_col_support > 0.579499959946
                      return 0.0365170362358 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.952499985695
                  if ( median_col_coverage <= 0.682921171188 ) {
                    if ( median_col_coverage <= 0.540436148643 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.540436148643
                      return 0.141206461873 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.682921171188
                    if ( median_col_support <= 0.985499978065 ) {
                      return 0.222455841873 < maxgini;
                    }
                    else {  // if median_col_support > 0.985499978065
                      return 0.395890281836 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.993499994278
                if ( mean_col_coverage <= 0.677643477917 ) {
                  if ( median_col_coverage <= 0.614965736866 ) {
                    if ( mean_col_support <= 0.967411756516 ) {
                      return 0.4921875 < maxgini;
                    }
                    else {  // if mean_col_support > 0.967411756516
                      return 0.18836565097 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.614965736866
                    return 0.0 < maxgini;
                  }
                }
                else {  // if mean_col_coverage > 0.677643477917
                  if ( max_col_coverage <= 0.768919229507 ) {
                    if ( min_col_support <= 0.527500033379 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.527500033379
                      return false;
                    }
                  }
                  else {  // if max_col_coverage > 0.768919229507
                    if ( max_col_coverage <= 0.803654551506 ) {
                      return 0.3046875 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.803654551506
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.836125731468
              if ( median_col_support <= 0.984500050545 ) {
                if ( mean_col_support <= 0.922558784485 ) {
                  if ( mean_col_coverage <= 0.961834251881 ) {
                    if ( max_col_coverage <= 0.945052862167 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.945052862167
                      return 0.42 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.961834251881
                    if ( mean_col_support <= 0.90038228035 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.90038228035
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.922558784485
                  if ( min_col_support <= 0.537000000477 ) {
                    if ( min_col_support <= 0.514999985695 ) {
                      return 0.1171875 < maxgini;
                    }
                    else {  // if min_col_support > 0.514999985695
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.537000000477
                    if ( mean_col_support <= 0.941382408142 ) {
                      return 0.385633270321 < maxgini;
                    }
                    else {  // if mean_col_support > 0.941382408142
                      return 0.0739644970414 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.984500050545
                if ( median_col_coverage <= 0.967445731163 ) {
                  if ( max_col_coverage <= 0.995718359947 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return false;
                    }
                  }
                  else {  // if max_col_coverage > 0.995718359947
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.495580393016 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.967445731163
                  if ( median_col_support <= 0.989500045776 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.989500045776
                    if ( min_col_support <= 0.5 ) {
                      return 0.375 < maxgini;
                    }
                    else {  // if min_col_support > 0.5
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.595499992371
            if ( median_col_support <= 0.984500050545 ) {
              if ( mean_col_support <= 0.964088201523 ) {
                if ( mean_col_coverage <= 0.995368242264 ) {
                  if ( median_col_support <= 0.62349998951 ) {
                    if ( median_col_coverage <= 0.576875507832 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.576875507832
                      return false;
                    }
                  }
                  else {  // if median_col_support > 0.62349998951
                    if ( median_col_coverage <= 0.95477604866 ) {
                      return 0.023519824627 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.95477604866
                      return 0.334993209597 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.995368242264
                  if ( mean_col_support <= 0.961205840111 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.961205840111
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.964088201523
                if ( max_col_coverage <= 0.791545033455 ) {
                  if ( min_col_support <= 0.597000002861 ) {
                    if ( median_col_coverage <= 0.590485930443 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.590485930443
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.597000002861
                    if ( mean_col_coverage <= 0.64150583744 ) {
                      return 0.0143303544435 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.64150583744
                      return 0.0123873403723 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.791545033455
                  if ( min_col_coverage <= 0.983202934265 ) {
                    if ( min_col_support <= 0.659500002861 ) {
                      return 0.140266262755 < maxgini;
                    }
                    else {  // if min_col_support > 0.659500002861
                      return 0.00697027397367 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.983202934265
                    if ( mean_col_support <= 0.977441132069 ) {
                      return 0.498866213152 < maxgini;
                    }
                    else {  // if mean_col_support > 0.977441132069
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.984500050545
              if ( mean_col_support <= 0.965264678001 ) {
                if ( median_col_support <= 0.992499947548 ) {
                  if ( mean_col_coverage <= 0.808123171329 ) {
                    if ( mean_col_coverage <= 0.760670423508 ) {
                      return 0.0229414603669 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.760670423508
                      return 0.158790170132 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.808123171329
                    if ( min_col_support <= 0.712000012398 ) {
                      return 0.358533272974 < maxgini;
                    }
                    else {  // if min_col_support > 0.712000012398
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.992499947548
                  if ( min_col_coverage <= 0.900532364845 ) {
                    if ( min_col_support <= 0.690500020981 ) {
                      return 0.150701715304 < maxgini;
                    }
                    else {  // if min_col_support > 0.690500020981
                      return 0.478596908442 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.900532364845
                    if ( min_col_coverage <= 0.933422267437 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.933422267437
                      return false;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.965264678001
                if ( median_col_coverage <= 0.653231620789 ) {
                  if ( max_col_coverage <= 0.921747922897 ) {
                    if ( median_col_coverage <= 0.617588877678 ) {
                      return 0.00472171788248 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.617588877678
                      return 0.00343483326445 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.921747922897
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.0219151172873 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.094592657808 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.653231620789
                  if ( min_col_coverage <= 0.747444629669 ) {
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.00375085339737 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.000962243647861 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.747444629669
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.00168712911494 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.000481644555251 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    bool shouldCorrect6(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( min_col_support <= 0.925500035286 ) {
        if ( median_col_support <= 0.948500037193 ) {
          if ( min_col_coverage <= 0.967793822289 ) {
            if ( median_col_support <= 0.909500002861 ) {
              if ( median_col_coverage <= 0.00904979370534 ) {
                if ( min_col_coverage <= 0.00222967937589 ) {
                  if ( mean_col_support <= 0.944411754608 ) {
                    if ( min_col_coverage <= 0.00220281421207 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00220281421207
                      return 0.46875 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.944411754608
                    if ( min_col_coverage <= 0.00215357355773 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00215357355773
                      return false;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.00222967937589
                  if ( min_col_coverage <= 0.00705469585955 ) {
                    if ( mean_col_coverage <= 0.0771322771907 ) {
                      return 0.0534587470016 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0771322771907
                      return 0.108368106875 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00705469585955
                    if ( median_col_coverage <= 0.00778211420402 ) {
                      return 0.126674429885 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00778211420402
                      return 0.0477269158742 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.00904979370534
                if ( max_col_support <= 0.99950003624 ) {
                  if ( median_col_support <= 0.567499995232 ) {
                    if ( mean_col_coverage <= 0.613809168339 ) {
                      return 0.0195943806774 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.613809168339
                      return 0.0831758034026 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.567499995232
                    if ( mean_col_coverage <= 0.879601359367 ) {
                      return 0.0309990838436 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.879601359367
                      return 0.375 < maxgini;
                    }
                  }
                }
                else {  // if max_col_support > 0.99950003624
                  if ( median_col_support <= 0.826499998569 ) {
                    if ( mean_col_coverage <= 0.924358129501 ) {
                      return 0.0467274660937 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.924358129501
                      return 0.488521579431 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.826499998569
                    if ( max_col_coverage <= 0.584670960903 ) {
                      return 0.0407287052487 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.584670960903
                      return 0.0296452071319 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.909500002861
              if ( median_col_coverage <= 0.00697269290686 ) {
                if ( max_col_coverage <= 0.232054024935 ) {
                  if ( mean_col_coverage <= 0.0445690527558 ) {
                    if ( min_col_support <= 0.358500003815 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if min_col_support > 0.358500003815
                      return 0.0266263083225 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.0445690527558
                    if ( mean_col_coverage <= 0.0445724874735 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.0445724874735
                      return 0.0631675885272 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.232054024935
                  if ( max_col_coverage <= 0.232593268156 ) {
                    if ( min_col_coverage <= 0.00259477132931 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.00259477132931
                      return 0.339444444444 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.232593268156
                    if ( median_col_support <= 0.910500049591 ) {
                      return 0.367309458219 < maxgini;
                    }
                    else {  // if median_col_support > 0.910500049591
                      return 0.150475044613 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.00697269290686
                if ( median_col_coverage <= 0.484774529934 ) {
                  if ( max_col_coverage <= 0.481717437506 ) {
                    if ( max_col_support <= 0.993499994278 ) {
                      return 0.0173054227217 < maxgini;
                    }
                    else {  // if max_col_support > 0.993499994278
                      return 0.0350844403863 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.481717437506
                    if ( median_col_coverage <= 0.325958132744 ) {
                      return 0.0222789609712 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.325958132744
                      return 0.0309083185371 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.484774529934
                  if ( min_col_support <= 0.576499998569 ) {
                    if ( mean_col_coverage <= 0.980197787285 ) {
                      return 0.0856282203347 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.980197787285
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.576499998569
                    if ( median_col_support <= 0.919499993324 ) {
                      return 0.0288744050987 < maxgini;
                    }
                    else {  // if median_col_support > 0.919499993324
                      return 0.0204505961375 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.967793822289
            if ( mean_col_coverage <= 0.991657733917 ) {
              if ( min_col_coverage <= 0.974890649319 ) {
                if ( min_col_support <= 0.736500024796 ) {
                  return false;
                }
                else {  // if min_col_support > 0.736500024796
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.974890649319
                return 0.0 < maxgini;
              }
            }
            else {  // if mean_col_coverage > 0.991657733917
              return false;
            }
          }
        }
        else {  // if median_col_support > 0.948500037193
          if ( max_col_coverage <= 0.586977303028 ) {
            if ( median_col_support <= 0.964499950409 ) {
              if ( min_col_support <= 0.889500021935 ) {
                if ( median_col_coverage <= 0.00512164738029 ) {
                  if ( min_col_coverage <= 0.00353983417153 ) {
                    if ( mean_col_support <= 0.972064554691 ) {
                      return 0.0434179869821 < maxgini;
                    }
                    else {  // if mean_col_support > 0.972064554691
                      return 0.0801653942001 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00353983417153
                    if ( mean_col_coverage <= 0.102023087442 ) {
                      return 0.0725968516079 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.102023087442
                      return 0.405838376646 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.00512164738029
                  if ( mean_col_coverage <= 0.386108607054 ) {
                    if ( median_col_coverage <= 0.353141069412 ) {
                      return 0.0256659256561 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.353141069412
                      return 0.174817898023 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.386108607054
                    if ( min_col_support <= 0.559499979019 ) {
                      return 0.0612290663647 < maxgini;
                    }
                    else {  // if min_col_support > 0.559499979019
                      return 0.0220288884728 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.889500021935
                if ( mean_col_support <= 0.96026456356 ) {
                  if ( median_col_coverage <= 0.185413986444 ) {
                    if ( median_col_coverage <= 0.166323035955 ) {
                      return 0.165289256198 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.166323035955
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.185413986444
                    if ( median_col_support <= 0.949499964714 ) {
                      return 0.172335600907 < maxgini;
                    }
                    else {  // if median_col_support > 0.949499964714
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.96026456356
                  if ( min_col_coverage <= 0.00221733842045 ) {
                    if ( mean_col_coverage <= 0.0830083712935 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.0830083712935
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00221733842045
                    if ( median_col_coverage <= 0.00349041214213 ) {
                      return 0.0766009231792 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00349041214213
                      return 0.0292247278824 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.964499950409
              if ( mean_col_coverage <= 0.0927200168371 ) {
                if ( mean_col_support <= 0.986179172993 ) {
                  if ( min_col_coverage <= 0.00298063270748 ) {
                    if ( max_col_coverage <= 0.257819533348 ) {
                      return 0.052128035208 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.257819533348
                      return 0.396694214876 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00298063270748
                    if ( mean_col_coverage <= 0.0927170142531 ) {
                      return 0.0257310635662 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0927170142531
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.986179172993
                  if ( min_col_support <= 0.923500001431 ) {
                    if ( min_col_coverage <= 0.00192497239914 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.00192497239914
                      return 0.0119143700072 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.923500001431
                    if ( median_col_coverage <= 0.00350263761356 ) {
                      return 0.0839500297442 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00350263761356
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.0927200168371
                if ( min_col_coverage <= 0.00336135411635 ) {
                  if ( max_col_coverage <= 0.273021101952 ) {
                    if ( min_col_coverage <= 0.00246002827771 ) {
                      return 0.0922681359045 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00246002827771
                      return 0.0283794896299 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.273021101952
                    if ( median_col_support <= 0.977499961853 ) {
                      return 0.128824063625 < maxgini;
                    }
                    else {  // if median_col_support > 0.977499961853
                      return 0.0521713253268 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.00336135411635
                  if ( median_col_coverage <= 0.00578872393817 ) {
                    if ( max_col_coverage <= 0.335641145706 ) {
                      return 0.0652848794741 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.335641145706
                      return 0.351239669421 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.00578872393817
                    if ( median_col_coverage <= 0.0689245611429 ) {
                      return 0.0107222181919 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0689245611429
                      return 0.0193904418409 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.586977303028
            if ( mean_col_support <= 0.971617639065 ) {
              if ( min_col_coverage <= 0.912722706795 ) {
                if ( mean_col_coverage <= 0.852694988251 ) {
                  if ( min_col_support <= 0.605499982834 ) {
                    if ( median_col_coverage <= 0.680687010288 ) {
                      return 0.139635704802 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.680687010288
                      return 0.336209303047 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.605499982834
                    if ( max_col_coverage <= 0.988286972046 ) {
                      return 0.0207669729551 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.988286972046
                      return 0.302448979592 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.852694988251
                  if ( mean_col_coverage <= 0.852802813053 ) {
                    return false;
                  }
                  else {  // if mean_col_coverage > 0.852802813053
                    if ( mean_col_support <= 0.951382339001 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if mean_col_support > 0.951382339001
                      return 0.165361221527 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.912722706795
                if ( mean_col_support <= 0.964470505714 ) {
                  if ( mean_col_coverage <= 0.994987487793 ) {
                    if ( max_col_coverage <= 0.998727738857 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.998727738857
                      return 0.472612041648 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.994987487793
                    if ( min_col_coverage <= 0.976089060307 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.976089060307
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.964470505714
                  if ( min_col_support <= 0.659999966621 ) {
                    if ( min_col_coverage <= 0.975600242615 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.975600242615
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.659999966621
                    if ( max_col_coverage <= 0.993560791016 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.993560791016
                      return 0.139206844722 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.971617639065
              if ( median_col_coverage <= 0.995215058327 ) {
                if ( median_col_coverage <= 0.553801655769 ) {
                  if ( mean_col_support <= 0.982205867767 ) {
                    if ( max_col_coverage <= 0.884852468967 ) {
                      return 0.0172652839205 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.884852468967
                      return 0.314098750744 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.982205867767
                    if ( min_col_support <= 0.891499996185 ) {
                      return 0.00512493107223 < maxgini;
                    }
                    else {  // if min_col_support > 0.891499996185
                      return 0.00883360184357 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.553801655769
                  if ( mean_col_coverage <= 0.97850382328 ) {
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.00738920128764 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.110945152355 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.97850382328
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.0162451694411 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return 0.233380205516 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.995215058327
                if ( min_col_coverage <= 0.990902304649 ) {
                  if ( median_col_coverage <= 0.995786428452 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.995786428452
                    if ( median_col_coverage <= 0.998666644096 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.998666644096
                      return 0.0786079011019 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.990902304649
                  if ( mean_col_support <= 0.97723531723 ) {
                    if ( min_col_support <= 0.702000021935 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.702000021935
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.97723531723
                    if ( min_col_coverage <= 0.991264879704 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.991264879704
                      return 0.149447128042 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.925500035286
        if ( mean_col_coverage <= 0.530745506287 ) {
          if ( min_col_support <= 0.975499987602 ) {
            if ( median_col_coverage <= 0.311350464821 ) {
              if ( mean_col_coverage <= 0.346961677074 ) {
                if ( mean_col_support <= 0.988656938076 ) {
                  if ( mean_col_support <= 0.983485341072 ) {
                    if ( median_col_coverage <= 0.00268422579393 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00268422579393
                      return 0.0336772790915 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.983485341072
                    if ( min_col_coverage <= 0.247021272779 ) {
                      return 0.0263728962865 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.247021272779
                      return 0.0343021267039 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.988656938076
                  if ( mean_col_support <= 0.992121398449 ) {
                    if ( min_col_support <= 0.956499993801 ) {
                      return 0.0161851862668 < maxgini;
                    }
                    else {  // if min_col_support > 0.956499993801
                      return 0.0239143609746 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992121398449
                    if ( median_col_coverage <= 0.0736724585295 ) {
                      return 0.00755656636529 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0736724585295
                      return 0.0149459613939 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.346961677074
                if ( median_col_support <= 0.983500003815 ) {
                  if ( mean_col_support <= 0.973970651627 ) {
                    if ( min_col_support <= 0.947499990463 ) {
                      return 0.0443377223059 < maxgini;
                    }
                    else {  // if min_col_support > 0.947499990463
                      return 0.23256850078 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.973970651627
                    if ( mean_col_coverage <= 0.379982024431 ) {
                      return 0.0250537719612 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.379982024431
                      return 0.0154512142166 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.983500003815
                  if ( min_col_coverage <= 0.259141266346 ) {
                    if ( min_col_coverage <= 0.185498744249 ) {
                      return 0.0480990805029 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.185498744249
                      return 0.00940741935999 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.259141266346
                    if ( mean_col_coverage <= 0.373079866171 ) {
                      return 0.0161204724799 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.373079866171
                      return 0.00986390926276 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.311350464821
              if ( median_col_support <= 0.981500029564 ) {
                if ( max_col_support <= 0.980499982834 ) {
                  if ( mean_col_coverage <= 0.452527880669 ) {
                    return false;
                  }
                  else {  // if mean_col_coverage > 0.452527880669
                    if ( median_col_support <= 0.939999997616 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.939999997616
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if max_col_support > 0.980499982834
                  if ( median_col_support <= 0.975499987602 ) {
                    if ( median_col_support <= 0.96850001812 ) {
                      return 0.03037157267 < maxgini;
                    }
                    else {  // if median_col_support > 0.96850001812
                      return 0.0255819029016 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.975499987602
                    if ( min_col_coverage <= 0.219195768237 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.219195768237
                      return 0.0210085758701 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.981500029564
                if ( max_col_coverage <= 0.498236328363 ) {
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( max_col_coverage <= 0.497035562992 ) {
                      return 0.0171907374349 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.497035562992
                      return 0.0296822317976 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( median_col_coverage <= 0.326305687428 ) {
                      return 0.0122465378018 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.326305687428
                      return 0.00823904200212 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.498236328363
                  if ( median_col_support <= 0.986500024796 ) {
                    if ( mean_col_coverage <= 0.458857297897 ) {
                      return 0.0153279023805 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.458857297897
                      return 0.0175505205944 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.986500024796
                    if ( mean_col_coverage <= 0.521433234215 ) {
                      return 0.0107351581309 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.521433234215
                      return 0.00734093458858 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.975499987602
            if ( mean_col_support <= 0.995485305786 ) {
              if ( max_col_coverage <= 0.446016073227 ) {
                if ( mean_col_coverage <= 0.199247449636 ) {
                  if ( mean_col_coverage <= 0.199241131544 ) {
                    if ( mean_col_support <= 0.98814702034 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.98814702034
                      return 0.0298263580301 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.199241131544
                    return false;
                  }
                }
                else {  // if mean_col_coverage > 0.199247449636
                  if ( max_col_coverage <= 0.445996522903 ) {
                    if ( max_col_coverage <= 0.385077625513 ) {
                      return 0.0241970967831 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.385077625513
                      return 0.0215160526884 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.445996522903
                    if ( min_col_coverage <= 0.273474186659 ) {
                      return 0.375 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.273474186659
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.446016073227
                if ( max_col_coverage <= 0.554884135723 ) {
                  if ( median_col_support <= 0.988499999046 ) {
                    if ( mean_col_coverage <= 0.379034101963 ) {
                      return 0.0168941274385 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.379034101963
                      return 0.020114172421 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.988499999046
                    if ( median_col_coverage <= 0.260993033648 ) {
                      return 0.00355599583034 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.260993033648
                      return 0.0155619331027 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.554884135723
                  if ( median_col_coverage <= 0.49896043539 ) {
                    if ( mean_col_coverage <= 0.381219744682 ) {
                      return 0.104938271605 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.381219744682
                      return 0.0140097181497 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.49896043539
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.0795090148442 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.0264269770408 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.995485305786
              if ( max_col_coverage <= 0.487658798695 ) {
                if ( median_col_support <= 0.993499994278 ) {
                  if ( mean_col_coverage <= 0.395151555538 ) {
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0160556318301 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.0130818871433 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.395151555538
                    if ( mean_col_coverage <= 0.395152568817 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.395152568817
                      return 0.0199375974036 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( max_col_coverage <= 0.487649261951 ) {
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.0104667448164 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.00347427131981 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.487649261951
                    if ( mean_col_support <= 0.996617674828 ) {
                      return 0.297520661157 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996617674828
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.487658798695
                if ( mean_col_support <= 0.997323513031 ) {
                  if ( max_col_coverage <= 0.601325273514 ) {
                    if ( min_col_coverage <= 0.344746053219 ) {
                      return 0.00817765703143 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.344746053219
                      return 0.0106542891065 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.601325273514
                    if ( mean_col_coverage <= 0.530739068985 ) {
                      return 0.00679778999105 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.530739068985
                      return 0.14201183432 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.997323513031
                  if ( min_col_coverage <= 0.385277688503 ) {
                    if ( median_col_coverage <= 0.430989325047 ) {
                      return 0.00531081256083 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.430989325047
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.385277688503
                    if ( median_col_coverage <= 0.408412128687 ) {
                      return 0.0189508837796 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.408412128687
                      return 0.00644805254835 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.530745506287
          if ( mean_col_support <= 0.992676496506 ) {
            if ( mean_col_support <= 0.98402941227 ) {
              if ( median_col_support <= 0.966500043869 ) {
                if ( max_col_support <= 0.985499978065 ) {
                  if ( median_col_coverage <= 0.502688169479 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.502688169479
                    return false;
                  }
                }
                else {  // if max_col_support > 0.985499978065
                  if ( median_col_coverage <= 0.677698373795 ) {
                    if ( median_col_coverage <= 0.677675485611 ) {
                      return 0.0231990474063 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.677675485611
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.677698373795
                    if ( min_col_coverage <= 0.64334499836 ) {
                      return 0.0305529131986 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.64334499836
                      return 0.00694034383064 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.966500043869
                if ( mean_col_coverage <= 0.624272346497 ) {
                  if ( mean_col_coverage <= 0.624270439148 ) {
                    if ( median_col_coverage <= 0.568761587143 ) {
                      return 0.0213412057999 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.568761587143
                      return 0.0320731361951 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.624270439148
                    return false;
                  }
                }
                else {  // if mean_col_coverage > 0.624272346497
                  if ( mean_col_coverage <= 0.953184783459 ) {
                    if ( mean_col_coverage <= 0.671478152275 ) {
                      return 0.015550991297 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.671478152275
                      return 0.0107723879289 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.953184783459
                    if ( median_col_support <= 0.979499995708 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.979499995708
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.98402941227
              if ( median_col_support <= 0.984500050545 ) {
                if ( min_col_coverage <= 0.60528498888 ) {
                  if ( mean_col_coverage <= 0.655170202255 ) {
                    if ( max_col_coverage <= 0.802769005299 ) {
                      return 0.0143553178533 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.802769005299
                      return 0.0559535301191 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.655170202255
                    if ( max_col_coverage <= 0.702170848846 ) {
                      return 0.0997229916898 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.702170848846
                      return 0.011572204433 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.60528498888
                  if ( median_col_coverage <= 0.74006664753 ) {
                    if ( max_col_coverage <= 0.784772574902 ) {
                      return 0.0114935853437 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.784772574902
                      return 0.00806387816119 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.74006664753
                    if ( max_col_support <= 0.99950003624 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_support > 0.99950003624
                      return 0.0037924586948 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.984500050545
                if ( mean_col_coverage <= 0.691915035248 ) {
                  if ( min_col_support <= 0.960500001907 ) {
                    if ( min_col_coverage <= 0.64396750927 ) {
                      return 0.00649212022967 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.64396750927
                      return 0.0928019036288 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.960500001907
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.0120327146079 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.00883448209283 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.691915035248
                  if ( min_col_support <= 0.972499966621 ) {
                    if ( min_col_support <= 0.942499995232 ) {
                      return 0.00107436082847 < maxgini;
                    }
                    else {  // if min_col_support > 0.942499995232
                      return 0.00350360662622 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.972499966621
                    if ( min_col_coverage <= 0.750266551971 ) {
                      return 0.00751757226878 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.750266551971
                      return 0.000559597046261 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.992676496506
            if ( mean_col_coverage <= 0.689119458199 ) {
              if ( min_col_coverage <= 0.498863637447 ) {
                if ( min_col_support <= 0.990499973297 ) {
                  if ( median_col_support <= 0.991500020027 ) {
                    if ( median_col_support <= 0.979499995708 ) {
                      return 0.156734693878 < maxgini;
                    }
                    else {  // if median_col_support > 0.979499995708
                      return 0.0097783859311 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.991500020027
                    if ( mean_col_support <= 0.997205853462 ) {
                      return 0.00502478607828 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997205853462
                      return 0.00307377399456 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.990499973297
                  if ( mean_col_coverage <= 0.668861269951 ) {
                    if ( median_col_coverage <= 0.533204972744 ) {
                      return 0.00410240494109 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.533204972744
                      return 0.00172741277029 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.668861269951
                    if ( min_col_coverage <= 0.497021079063 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.497021079063
                      return 0.48 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.498863637447
                if ( mean_col_support <= 0.99638235569 ) {
                  if ( mean_col_support <= 0.994911789894 ) {
                    if ( min_col_coverage <= 0.592863798141 ) {
                      return 0.00661824738175 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.592863798141
                      return 0.00465884197915 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994911789894
                    if ( max_col_coverage <= 0.877995491028 ) {
                      return 0.004397516025 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.877995491028
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.99638235569
                  if ( median_col_support <= 0.995499968529 ) {
                    if ( median_col_coverage <= 0.650101423264 ) {
                      return 0.00323564167411 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.650101423264
                      return 0.0087871750868 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.995499968529
                    if ( min_col_support <= 0.993499994278 ) {
                      return 0.00215327187533 < maxgini;
                    }
                    else {  // if min_col_support > 0.993499994278
                      return 0.0014637391885 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.689119458199
              if ( median_col_support <= 0.993499994278 ) {
                if ( min_col_coverage <= 0.657042741776 ) {
                  if ( median_col_support <= 0.989500045776 ) {
                    if ( min_col_coverage <= 0.651746153831 ) {
                      return 0.00604350599293 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.651746153831
                      return 0.0115698561853 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.989500045776
                    if ( max_col_coverage <= 0.748853206635 ) {
                      return 0.0121699291962 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.748853206635
                      return 0.00331640584392 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.657042741776
                  if ( min_col_coverage <= 0.798199534416 ) {
                    if ( median_col_support <= 0.991500020027 ) {
                      return 0.00356250404307 < maxgini;
                    }
                    else {  // if median_col_support > 0.991500020027
                      return 0.00171361998873 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.798199534416
                    if ( max_col_coverage <= 0.920341849327 ) {
                      return 0.00291120196517 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.920341849327
                      return 0.000930604327757 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.993499994278
                if ( median_col_coverage <= 0.691514015198 ) {
                  if ( max_col_coverage <= 0.975861787796 ) {
                    if ( max_col_coverage <= 0.774016320705 ) {
                      return 0.00257050772005 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.774016320705
                      return 0.00117731367939 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.975861787796
                    if ( mean_col_coverage <= 0.793212771416 ) {
                      return 0.15572657311 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.793212771416
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.691514015198
                  if ( median_col_coverage <= 0.902491927147 ) {
                    if ( median_col_coverage <= 0.902487635612 ) {
                      return 0.000453069634354 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.902487635612
                      return 0.366230677765 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.902491927147
                    if ( median_col_coverage <= 0.983710050583 ) {
                      return 8.31720853472e-05 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.983710050583
                      return 0.000688912727677 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    bool shouldCorrect7(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( min_col_support <= 0.925500035286 ) {
        if ( median_col_support <= 0.943500041962 ) {
          if ( max_col_support <= 0.99950003624 ) {
            if ( max_col_support <= 0.984500050545 ) {
              if ( mean_col_support <= 0.724970579147 ) {
                if ( median_col_coverage <= 0.490797549486 ) {
                  if ( min_col_support <= 0.471499979496 ) {
                    if ( max_col_coverage <= 0.228280812502 ) {
                      return 0.00950712968736 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.228280812502
                      return 0.00324832663144 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.471499979496
                    if ( max_col_support <= 0.947499990463 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_support > 0.947499990463
                      return 0.0274128292661 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.490797549486
                  if ( min_col_support <= 0.338999986649 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.338999986649
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.724970579147
                if ( median_col_support <= 0.485500007868 ) {
                  if ( min_col_support <= 0.385500013828 ) {
                    if ( median_col_coverage <= 0.0803822129965 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0803822129965
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.385500013828
                    return 0.0 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.485500007868
                  if ( min_col_support <= 0.536499977112 ) {
                    if ( mean_col_coverage <= 0.175450742245 ) {
                      return 0.0125821702648 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.175450742245
                      return 0.0301955966615 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.536499977112
                    if ( median_col_coverage <= 0.249174892902 ) {
                      return 0.0142491055923 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.249174892902
                      return 0.0214550591755 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_support > 0.984500050545
              if ( median_col_support <= 0.877499997616 ) {
                if ( min_col_support <= 0.869500041008 ) {
                  if ( mean_col_coverage <= 0.876220226288 ) {
                    if ( min_col_coverage <= 0.639658153057 ) {
                      return 0.0340851485438 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.639658153057
                      return 0.0975534725821 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.876220226288
                    if ( max_col_coverage <= 0.959468066692 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.959468066692
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.869500041008
                  if ( min_col_support <= 0.870999991894 ) {
                    if ( mean_col_coverage <= 0.464248478413 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.464248478413
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.870999991894
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if median_col_support > 0.877499997616
                if ( min_col_coverage <= 0.920768976212 ) {
                  if ( min_col_support <= 0.850499987602 ) {
                    if ( median_col_coverage <= 0.0395104065537 ) {
                      return 0.297520661157 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0395104065537
                      return 0.0240429919961 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.850499987602
                    if ( min_col_support <= 0.851500034332 ) {
                      return 0.0629057377812 < maxgini;
                    }
                    else {  // if min_col_support > 0.851500034332
                      return 0.0276031561169 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.920768976212
                  if ( median_col_coverage <= 0.964275479317 ) {
                    if ( mean_col_support <= 0.955705881119 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.955705881119
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.964275479317
                    if ( median_col_coverage <= 0.980401158333 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.980401158333
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_support > 0.99950003624
            if ( median_col_support <= 0.836500048637 ) {
              if ( mean_col_support <= 0.756323575974 ) {
                if ( median_col_coverage <= 0.61129117012 ) {
                  if ( min_col_support <= 0.432500004768 ) {
                    if ( min_col_support <= 0.374500006437 ) {
                      return 0.0110290726081 < maxgini;
                    }
                    else {  // if min_col_support > 0.374500006437
                      return 0.0227693875731 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.432500004768
                    if ( min_col_support <= 0.447499990463 ) {
                      return 0.0491994148613 < maxgini;
                    }
                    else {  // if min_col_support > 0.447499990463
                      return 0.0373115398043 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.61129117012
                  if ( mean_col_support <= 0.726352930069 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.726352930069
                    if ( mean_col_support <= 0.748911738396 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.748911738396
                      return false;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.756323575974
                if ( median_col_support <= 0.655499994755 ) {
                  if ( min_col_support <= 0.643499970436 ) {
                    if ( median_col_support <= 0.583500027657 ) {
                      return 0.0451765898315 < maxgini;
                    }
                    else {  // if median_col_support > 0.583500027657
                      return 0.0531117875861 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.643499970436
                    if ( median_col_support <= 0.653499960899 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.653499960899
                      return false;
                    }
                  }
                }
                else {  // if median_col_support > 0.655499994755
                  if ( mean_col_coverage <= 0.973783493042 ) {
                    if ( mean_col_support <= 0.930060744286 ) {
                      return 0.045663613545 < maxgini;
                    }
                    else {  // if mean_col_support > 0.930060744286
                      return 0.0346503328823 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.973783493042
                    if ( min_col_coverage <= 0.965257287025 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.965257287025
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.836500048637
              if ( max_col_coverage <= 0.541222453117 ) {
                if ( mean_col_support <= 0.955895781517 ) {
                  if ( median_col_support <= 0.905499994755 ) {
                    if ( min_col_coverage <= 0.00760457478464 ) {
                      return 0.0578550453165 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00760457478464
                      return 0.040377097789 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.905499994755
                    if ( min_col_support <= 0.606500029564 ) {
                      return 0.0449357505898 < maxgini;
                    }
                    else {  // if min_col_support > 0.606500029564
                      return 0.0377914722805 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.955895781517
                  if ( median_col_support <= 0.910500049591 ) {
                    if ( min_col_support <= 0.890499949455 ) {
                      return 0.0410990513859 < maxgini;
                    }
                    else {  // if min_col_support > 0.890499949455
                      return 0.0769112549698 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.910500049591
                    if ( max_col_coverage <= 0.541029572487 ) {
                      return 0.0367105531733 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.541029572487
                      return 0.119511090991 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.541222453117
                if ( median_col_coverage <= 0.966131567955 ) {
                  if ( min_col_coverage <= 0.56213581562 ) {
                    if ( median_col_coverage <= 0.497802197933 ) {
                      return 0.0319949299147 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.497802197933
                      return 0.027477522154 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.56213581562
                    if ( min_col_support <= 0.568500041962 ) {
                      return 0.0622962082177 < maxgini;
                    }
                    else {  // if min_col_support > 0.568500041962
                      return 0.0201987398663 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.966131567955
                  if ( median_col_support <= 0.868000030518 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.868000030518
                    if ( median_col_coverage <= 0.968118667603 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.968118667603
                      return 0.255 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.943500041962
          if ( min_col_coverage <= 0.983690023422 ) {
            if ( mean_col_support <= 0.981279194355 ) {
              if ( median_col_support <= 0.990499973297 ) {
                if ( mean_col_coverage <= 0.510622143745 ) {
                  if ( median_col_coverage <= 0.00513479672372 ) {
                    if ( median_col_support <= 0.964499950409 ) {
                      return 0.057991920353 < maxgini;
                    }
                    else {  // if median_col_support > 0.964499950409
                      return 0.0373546942004 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.00513479672372
                    if ( min_col_support <= 0.903499960899 ) {
                      return 0.0246505503532 < maxgini;
                    }
                    else {  // if min_col_support > 0.903499960899
                      return 0.0301455768983 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.510622143745
                  if ( min_col_support <= 0.610499978065 ) {
                    if ( max_col_coverage <= 0.858395338058 ) {
                      return 0.109960640377 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.858395338058
                      return 0.353719260038 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.610499978065
                    if ( median_col_coverage <= 0.542754173279 ) {
                      return 0.0200200181098 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.542754173279
                      return 0.0124709479617 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.990499973297
                if ( median_col_coverage <= 0.768891572952 ) {
                  if ( max_col_coverage <= 0.816263318062 ) {
                    if ( mean_col_coverage <= 0.76341676712 ) {
                      return 0.0378205376222 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.76341676712
                      return false;
                    }
                  }
                  else {  // if max_col_coverage > 0.816263318062
                    if ( max_col_coverage <= 0.987180829048 ) {
                      return 0.167718830852 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.987180829048
                      return 0.496527777778 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.768891572952
                  if ( mean_col_support <= 0.975852966309 ) {
                    if ( min_col_coverage <= 0.913067579269 ) {
                      return 0.477744204726 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.913067579269
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.975852966309
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.108642533652 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.981279194355
              if ( min_col_coverage <= 0.461146384478 ) {
                if ( median_col_support <= 0.980499982834 ) {
                  if ( max_col_coverage <= 0.427737146616 ) {
                    if ( median_col_coverage <= 0.00218610838056 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.00218610838056
                      return 0.0238017514392 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.427737146616
                    if ( median_col_coverage <= 0.154634058475 ) {
                      return 0.225328719723 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.154634058475
                      return 0.0170602813404 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.980499982834
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( max_col_coverage <= 0.441002517939 ) {
                      return 0.0156400725026 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.441002517939
                      return 0.0101704864932 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( mean_col_support <= 0.984919905663 ) {
                      return 0.0426969473972 < maxgini;
                    }
                    else {  // if mean_col_support > 0.984919905663
                      return 0.0153183779363 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.461146384478
                if ( max_col_coverage <= 0.756422936916 ) {
                  if ( max_col_coverage <= 0.756241321564 ) {
                    if ( mean_col_support <= 0.983617663383 ) {
                      return 0.0135006357882 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983617663383
                      return 0.00569970328717 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.756241321564
                    if ( mean_col_coverage <= 0.641087532043 ) {
                      return 0.277777777778 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.641087532043
                      return 0.0227242700489 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.756422936916
                  if ( min_col_coverage <= 0.974456846714 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.00279881149846 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.124444444444 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.974456846714
                    if ( min_col_support <= 0.810000002384 ) {
                      return 0.473372781065 < maxgini;
                    }
                    else {  // if min_col_support > 0.810000002384
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.983690023422
            if ( max_col_coverage <= 0.99780356884 ) {
              if ( median_col_coverage <= 0.995552480221 ) {
                if ( mean_col_support <= 0.979088187218 ) {
                  if ( mean_col_support <= 0.937852978706 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if mean_col_support > 0.937852978706
                    return false;
                  }
                }
                else {  // if mean_col_support > 0.979088187218
                  return 0.0 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.995552480221
                if ( max_col_coverage <= 0.996912837029 ) {
                  if ( mean_col_support <= 0.974676370621 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.974676370621
                    return 0.0 < maxgini;
                  }
                }
                else {  // if max_col_coverage > 0.996912837029
                  return false;
                }
              }
            }
            else {  // if max_col_coverage > 0.99780356884
              if ( min_col_support <= 0.703500032425 ) {
                if ( min_col_support <= 0.607500016689 ) {
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( min_col_coverage <= 0.988498449326 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.988498449326
                      return false;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    return 0.0 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.607500016689
                  if ( mean_col_support <= 0.974411725998 ) {
                    if ( median_col_coverage <= 0.998511910439 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.998511910439
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.974411725998
                    if ( min_col_support <= 0.633000016212 ) {
                      return 0.48 < maxgini;
                    }
                    else {  // if min_col_support > 0.633000016212
                      return 0.197530864198 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.703500032425
                if ( min_col_coverage <= 0.993855595589 ) {
                  if ( min_col_support <= 0.797999978065 ) {
                    if ( min_col_support <= 0.794499993324 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_support > 0.794499993324
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.797999978065
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.21875 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.993855595589
                  if ( median_col_support <= 0.99849998951 ) {
                    if ( mean_col_coverage <= 0.999335825443 ) {
                      return 0.244897959184 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.999335825443
                      return 0.083671875 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99849998951
                    if ( min_col_support <= 0.868000030518 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.868000030518
                      return 0.32 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.925500035286
        if ( mean_col_coverage <= 0.521750092506 ) {
          if ( max_col_coverage <= 0.443631887436 ) {
            if ( max_col_coverage <= 0.301287978888 ) {
              if ( median_col_support <= 0.989500045776 ) {
                if ( mean_col_support <= 0.986242711544 ) {
                  if ( mean_col_coverage <= 0.0598871670663 ) {
                    if ( min_col_coverage <= 0.00323627982289 ) {
                      return 0.0161279661577 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00323627982289
                      return 0.2371875 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.0598871670663
                    if ( max_col_coverage <= 0.180704116821 ) {
                      return 0.0135034504556 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.180704116821
                      return 0.0349259259873 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.986242711544
                  if ( median_col_support <= 0.949499964714 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.949499964714
                    if ( median_col_coverage <= 0.1013905406 ) {
                      return 0.0190982690711 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.1013905406
                      return 0.0290387024992 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.989500045776
                if ( min_col_support <= 0.984500050545 ) {
                  if ( min_col_coverage <= 0.0779738724232 ) {
                    if ( median_col_coverage <= 0.00233372533694 ) {
                      return 0.0968346522282 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00233372533694
                      return 0.00594977221427 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0779738724232
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0206881336242 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.00860933685538 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.984500050545
                  if ( min_col_support <= 0.991500020027 ) {
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.0296588631239 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.00919520611876 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.991500020027
                    if ( min_col_coverage <= 0.0952765643597 ) {
                      return 0.263671875 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0952765643597
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.301287978888
              if ( median_col_support <= 0.982499957085 ) {
                if ( median_col_coverage <= 0.00324730109423 ) {
                  if ( min_col_coverage <= 0.00318474555388 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.00318474555388
                    if ( mean_col_coverage <= 0.121504232287 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.121504232287
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.00324730109423
                  if ( median_col_coverage <= 0.251978933811 ) {
                    if ( mean_col_support <= 0.97979414463 ) {
                      return 0.0372058508309 < maxgini;
                    }
                    else {  // if mean_col_support > 0.97979414463
                      return 0.0244937665403 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.251978933811
                    if ( mean_col_support <= 0.975617647171 ) {
                      return 0.0456187968061 < maxgini;
                    }
                    else {  // if mean_col_support > 0.975617647171
                      return 0.0299232175581 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.982499957085
                if ( min_col_support <= 0.971500039101 ) {
                  if ( median_col_coverage <= 0.221836984158 ) {
                    if ( max_col_coverage <= 0.338365316391 ) {
                      return 0.0160939433823 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.338365316391
                      return 0.0110645219694 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.221836984158
                    if ( mean_col_support <= 0.993205904961 ) {
                      return 0.0196111407565 < maxgini;
                    }
                    else {  // if mean_col_support > 0.993205904961
                      return 0.0112942698498 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.971500039101
                  if ( median_col_support <= 0.991500020027 ) {
                    if ( mean_col_support <= 0.99426472187 ) {
                      return 0.025336535999 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99426472187
                      return 0.0175411008509 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.991500020027
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.0148742088214 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.00696420786723 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.443631887436
            if ( median_col_support <= 0.983500003815 ) {
              if ( median_col_support <= 0.971500039101 ) {
                if ( median_col_coverage <= 0.158604234457 ) {
                  if ( min_col_coverage <= 0.0885862857103 ) {
                    if ( min_col_coverage <= 0.0779848992825 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0779848992825
                      return 0.165289256198 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0885862857103
                    if ( mean_col_support <= 0.983852982521 ) {
                      return 0.197530864198 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983852982521
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.158604234457
                  if ( max_col_support <= 0.978999972343 ) {
                    return false;
                  }
                  else {  // if max_col_support > 0.978999972343
                    if ( mean_col_coverage <= 0.435604900122 ) {
                      return 0.0314828381794 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.435604900122
                      return 0.0262781852651 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.971500039101
                if ( median_col_support <= 0.979499995708 ) {
                  if ( min_col_support <= 0.949499964714 ) {
                    if ( median_col_support <= 0.978500008583 ) {
                      return 0.0198268588109 < maxgini;
                    }
                    else {  // if median_col_support > 0.978500008583
                      return 0.0140729997228 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.949499964714
                    if ( min_col_coverage <= 0.460298925638 ) {
                      return 0.0247152975347 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.460298925638
                      return 0.0830379533179 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.979499995708
                  if ( min_col_coverage <= 0.13513417542 ) {
                    if ( median_col_coverage <= 0.227262660861 ) {
                      return 0.0557679540304 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.227262660861
                      return 0.489795918367 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.13513417542
                    if ( max_col_coverage <= 0.639189898968 ) {
                      return 0.0190331477032 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.639189898968
                      return 0.00821903812279 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.983500003815
              if ( mean_col_support <= 0.995676517487 ) {
                if ( min_col_support <= 0.96850001812 ) {
                  if ( min_col_coverage <= 0.252293646336 ) {
                    if ( median_col_support <= 0.986500024796 ) {
                      return 0.00854831230754 < maxgini;
                    }
                    else {  // if median_col_support > 0.986500024796
                      return 0.00390026445337 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.252293646336
                    if ( min_col_support <= 0.953500032425 ) {
                      return 0.0100853565881 < maxgini;
                    }
                    else {  // if min_col_support > 0.953500032425
                      return 0.0131441342076 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.96850001812
                  if ( max_col_coverage <= 0.553653299809 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.0178286310701 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.0139369888154 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.553653299809
                    if ( min_col_coverage <= 0.467167377472 ) {
                      return 0.0137835625212 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.467167377472
                      return 0.0375449280007 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.995676517487
                if ( max_col_coverage <= 0.590113937855 ) {
                  if ( min_col_coverage <= 0.374637663364 ) {
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.00907754377387 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return 0.0038473393524 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.374637663364
                    if ( mean_col_support <= 0.997676491737 ) {
                      return 0.0110881729711 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997676491737
                      return 0.00619429862241 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.590113937855
                  if ( max_col_coverage <= 0.707015395164 ) {
                    if ( mean_col_coverage <= 0.507917165756 ) {
                      return 0.00493695498148 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.507917165756
                      return 0.00733026857258 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.707015395164
                    if ( min_col_support <= 0.985499978065 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_support > 0.985499978065
                      return 0.103140495868 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.521750092506
          if ( min_col_support <= 0.982499957085 ) {
            if ( mean_col_coverage <= 0.675523877144 ) {
              if ( median_col_support <= 0.986500024796 ) {
                if ( median_col_support <= 0.969500005245 ) {
                  if ( min_col_support <= 0.955500006676 ) {
                    if ( max_col_coverage <= 0.721166729927 ) {
                      return 0.0253088017974 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.721166729927
                      return 0.016266375614 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.955500006676
                    if ( median_col_coverage <= 0.438046783209 ) {
                      return 0.2112 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.438046783209
                      return 0.0156458558839 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.969500005245
                  if ( mean_col_support <= 0.995558857918 ) {
                    if ( mean_col_support <= 0.987735271454 ) {
                      return 0.0159587809834 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987735271454
                      return 0.0127085899673 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.995558857918
                    if ( median_col_coverage <= 0.574772715569 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.574772715569
                      return false;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.986500024796
                if ( median_col_coverage <= 0.540478825569 ) {
                  if ( min_col_support <= 0.956499993801 ) {
                    if ( min_col_coverage <= 0.523169755936 ) {
                      return 0.00486434024466 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.523169755936
                      return 0.0338882282996 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.956499993801
                    if ( mean_col_coverage <= 0.523044586182 ) {
                      return 0.0124939283549 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.523044586182
                      return 0.00768802950206 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.540478825569
                  if ( mean_col_coverage <= 0.675517857075 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.00793767399654 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.00419932329725 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.675517857075
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.21875 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.675523877144
              if ( mean_col_support <= 0.991558790207 ) {
                if ( max_col_coverage <= 0.879204034805 ) {
                  if ( min_col_support <= 0.966500043869 ) {
                    if ( max_col_coverage <= 0.772980749607 ) {
                      return 0.0112149360939 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.772980749607
                      return 0.00719734566215 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.966500043869
                    if ( mean_col_support <= 0.985735297203 ) {
                      return 0.0525303925628 < maxgini;
                    }
                    else {  // if mean_col_support > 0.985735297203
                      return 0.0114549838899 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.879204034805
                  if ( mean_col_support <= 0.983676433563 ) {
                    if ( mean_col_support <= 0.983617663383 ) {
                      return 0.00877700879415 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983617663383
                      return 0.0555102040816 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.983676433563
                    if ( median_col_support <= 0.977499961853 ) {
                      return 0.00539935697189 < maxgini;
                    }
                    else {  // if median_col_support > 0.977499961853
                      return 0.00240047662963 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.991558790207
                if ( max_col_coverage <= 0.832812428474 ) {
                  if ( mean_col_support <= 0.994852900505 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.00619329996492 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.0035277099038 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994852900505
                    if ( mean_col_support <= 0.996264696121 ) {
                      return 0.00214909982913 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996264696121
                      return 0.000753643933123 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.832812428474
                  if ( median_col_coverage <= 0.867941141129 ) {
                    if ( min_col_coverage <= 0.85985159874 ) {
                      return 0.00148608891 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.85985159874
                      return 0.0388197235565 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.867941141129
                    if ( min_col_coverage <= 0.820840120316 ) {
                      return 0.00257068982938 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.820840120316
                      return 0.000142068385838 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.982499957085
            if ( mean_col_coverage <= 0.69903922081 ) {
              if ( median_col_coverage <= 0.525996923447 ) {
                if ( median_col_coverage <= 0.525968551636 ) {
                  if ( max_col_coverage <= 0.56422406435 ) {
                    if ( median_col_coverage <= 0.520647287369 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.520647287369
                      return 0.48 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.56422406435
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.00948862336048 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.00453051569018 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.525968551636
                  if ( min_col_coverage <= 0.471404731274 ) {
                    if ( min_col_coverage <= 0.470863610506 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.470863610506
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.471404731274
                    if ( mean_col_support <= 0.996323525906 ) {
                      return 0.0734222947381 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996323525906
                      return 0.0155632449058 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.525996923447
                if ( median_col_support <= 0.993499994278 ) {
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( max_col_coverage <= 0.867381036282 ) {
                      return 0.00842617751851 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.867381036282
                      return 0.375 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( mean_col_support <= 0.995382368565 ) {
                      return 0.0058516872324 < maxgini;
                    }
                    else {  // if mean_col_support > 0.995382368565
                      return 0.00459207599148 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( max_col_coverage <= 0.895438790321 ) {
                    if ( mean_col_coverage <= 0.699038863182 ) {
                      return 0.00254786579397 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.699038863182
                      return 0.231111111111 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.895438790321
                    if ( max_col_coverage <= 0.896157264709 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.896157264709
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.69903922081
              if ( mean_col_support <= 0.996264696121 ) {
                if ( median_col_support <= 0.991500020027 ) {
                  if ( median_col_coverage <= 0.674355268478 ) {
                    if ( max_col_coverage <= 0.814201951027 ) {
                      return 0.00667913117961 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.814201951027
                      return 0.00934062864001 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.674355268478
                    if ( mean_col_support <= 0.994147062302 ) {
                      return 0.00686734921698 < maxgini;
                    }
                    else {  // if mean_col_support > 0.994147062302
                      return 0.00365871829042 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.991500020027
                  if ( min_col_coverage <= 0.520848631859 ) {
                    if ( mean_col_support <= 0.994794130325 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.994794130325
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.520848631859
                    if ( mean_col_support <= 0.99432349205 ) {
                      return 0.00814097662021 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99432349205
                      return 0.00216439845628 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.996264696121
                if ( median_col_support <= 0.993499994278 ) {
                  if ( min_col_coverage <= 0.641648769379 ) {
                    if ( min_col_support <= 0.991500020027 ) {
                      return 0.00377080460296 < maxgini;
                    }
                    else {  // if min_col_support > 0.991500020027
                      return 0.0144919814588 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.641648769379
                    if ( min_col_coverage <= 0.79197537899 ) {
                      return 0.00231844215512 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.79197537899
                      return 0.000898472415099 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( min_col_coverage <= 0.562571167946 ) {
                    if ( mean_col_support <= 0.997323513031 ) {
                      return 0.042149112426 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997323513031
                      return 0.00235848727853 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.562571167946
                    if ( median_col_coverage <= 0.70448154211 ) {
                      return 0.000919247067471 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.70448154211
                      return 0.000342830983932 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    bool shouldCorrect8(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( median_col_coverage <= 0.459621369839 ) {
        if ( mean_col_support <= 0.980939269066 ) {
          if ( mean_col_support <= 0.952064573765 ) {
            if ( max_col_coverage <= 0.341731667519 ) {
              if ( max_col_coverage <= 0.110066577792 ) {
                if ( median_col_support <= 0.969500005245 ) {
                  if ( max_col_coverage <= 0.0874817818403 ) {
                    if ( min_col_support <= 0.555500030518 ) {
                      return 0.0211914931692 < maxgini;
                    }
                    else {  // if min_col_support > 0.555500030518
                      return 0.0320716754611 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.0874817818403
                    if ( max_col_coverage <= 0.0875165760517 ) {
                      return 0.244897959184 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.0875165760517
                      return 0.0316667698808 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.969500005245
                  if ( median_col_coverage <= 0.0106952209026 ) {
                    if ( min_col_support <= 0.694999992847 ) {
                      return 0.0231807370032 < maxgini;
                    }
                    else {  // if min_col_support > 0.694999992847
                      return 0.32 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0106952209026
                    if ( max_col_coverage <= 0.0194548889995 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.0194548889995
                      return 0.0799939508507 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.110066577792
                if ( median_col_support <= 0.544499993324 ) {
                  if ( min_col_support <= 0.432500004768 ) {
                    if ( min_col_coverage <= 0.191737055779 ) {
                      return 0.0155280593236 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.191737055779
                      return 0.0777940102264 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.432500004768
                    if ( mean_col_support <= 0.725617647171 ) {
                      return 0.0198392777539 < maxgini;
                    }
                    else {  // if mean_col_support > 0.725617647171
                      return 0.0426643392774 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.544499993324
                  if ( min_col_coverage <= 0.00605145096779 ) {
                    if ( median_col_support <= 0.727499961853 ) {
                      return 0.0323471658732 < maxgini;
                    }
                    else {  // if median_col_support > 0.727499961853
                      return 0.056600853968 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00605145096779
                    if ( min_col_support <= 0.644500017166 ) {
                      return 0.0469178387674 < maxgini;
                    }
                    else {  // if min_col_support > 0.644500017166
                      return 0.0395513836697 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.341731667519
              if ( median_col_coverage <= 0.0752887874842 ) {
                if ( mean_col_support <= 0.923441171646 ) {
                  if ( min_col_coverage <= 0.00762735586613 ) {
                    if ( mean_col_support <= 0.908205866814 ) {
                      return 0.34139252213 < maxgini;
                    }
                    else {  // if mean_col_support > 0.908205866814
                      return 0.0788140364046 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00762735586613
                    if ( median_col_coverage <= 0.0314634554088 ) {
                      return 0.192189349112 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0314634554088
                      return 0.0596449704142 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.923441171646
                  if ( min_col_support <= 0.841499984264 ) {
                    if ( min_col_support <= 0.554499983788 ) {
                      return 0.336734693878 < maxgini;
                    }
                    else {  // if min_col_support > 0.554499983788
                      return 0.166876689313 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.841499984264
                    if ( max_col_coverage <= 0.346585839987 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.346585839987
                      return false;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.0752887874842
                if ( min_col_support <= 0.551499962807 ) {
                  if ( max_col_support <= 0.995499968529 ) {
                    if ( median_col_support <= 0.568500041962 ) {
                      return 0.0162608808872 < maxgini;
                    }
                    else {  // if median_col_support > 0.568500041962
                      return 0.0290865768661 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.995499968529
                    if ( mean_col_support <= 0.820911765099 ) {
                      return 0.0365138178732 < maxgini;
                    }
                    else {  // if mean_col_support > 0.820911765099
                      return 0.053953254956 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.551499962807
                  if ( max_col_coverage <= 0.581530094147 ) {
                    if ( median_col_support <= 0.891499996185 ) {
                      return 0.0379406942726 < maxgini;
                    }
                    else {  // if median_col_support > 0.891499996185
                      return 0.0330132362852 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.581530094147
                    if ( max_col_coverage <= 0.979933142662 ) {
                      return 0.027729973428 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.979933142662
                      return 0.489795918367 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.952064573765
            if ( max_col_coverage <= 0.416387856007 ) {
              if ( min_col_coverage <= 0.00357782887295 ) {
                if ( median_col_support <= 0.943500041962 ) {
                  if ( max_col_coverage <= 0.265767067671 ) {
                    if ( max_col_coverage <= 0.167229786515 ) {
                      return 0.0402514543066 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.167229786515
                      return 0.0742142550891 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.265767067671
                    if ( mean_col_support <= 0.961147069931 ) {
                      return 0.158049978734 < maxgini;
                    }
                    else {  // if mean_col_support > 0.961147069931
                      return 0.259155457086 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.943500041962
                  if ( max_col_coverage <= 0.29949682951 ) {
                    if ( max_col_coverage <= 0.232529073954 ) {
                      return 0.0334529586079 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.232529073954
                      return 0.062351915158 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.29949682951
                    if ( mean_col_coverage <= 0.108947701752 ) {
                      return 0.462809917355 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.108947701752
                      return 0.151871784106 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.00357782887295
                if ( mean_col_coverage <= 0.14224255085 ) {
                  if ( min_col_support <= 0.754500031471 ) {
                    if ( min_col_support <= 0.677500009537 ) {
                      return 0.0077784902405 < maxgini;
                    }
                    else {  // if min_col_support > 0.677500009537
                      return 0.0160465316963 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.754500031471
                    if ( median_col_coverage <= 0.00836821738631 ) {
                      return 0.096551644488 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00836821738631
                      return 0.0263684480098 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.14224255085
                  if ( mean_col_support <= 0.968735337257 ) {
                    if ( min_col_support <= 0.933500051498 ) {
                      return 0.0356346590792 < maxgini;
                    }
                    else {  // if min_col_support > 0.933500051498
                      return 0.108727810651 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.968735337257
                    if ( min_col_support <= 0.902500033379 ) {
                      return 0.0272449526066 < maxgini;
                    }
                    else {  // if min_col_support > 0.902500033379
                      return 0.0340418256926 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.416387856007
              if ( mean_col_support <= 0.969323515892 ) {
                if ( median_col_coverage <= 0.133436858654 ) {
                  if ( min_col_support <= 0.907000005245 ) {
                    if ( min_col_support <= 0.682000041008 ) {
                      return 0.0253122945431 < maxgini;
                    }
                    else {  // if min_col_support > 0.682000041008
                      return 0.20837116559 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.907000005245
                    if ( median_col_support <= 0.93649995327 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.93649995327
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.133436858654
                  if ( median_col_support <= 0.99849998951 ) {
                    if ( min_col_coverage <= 0.237936586142 ) {
                      return 0.0213175856921 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.237936586142
                      return 0.0316296774178 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99849998951
                    return false;
                  }
                }
              }
              else {  // if mean_col_support > 0.969323515892
                if ( min_col_support <= 0.890499949455 ) {
                  if ( min_col_coverage <= 0.0994796007872 ) {
                    if ( min_col_support <= 0.848500013351 ) {
                      return 0.0193218322427 < maxgini;
                    }
                    else {  // if min_col_support > 0.848500013351
                      return 0.221004499927 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0994796007872
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.0197337413678 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.489795918367 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.890499949455
                  if ( min_col_support <= 0.932500004768 ) {
                    if ( median_col_coverage <= 0.110715702176 ) {
                      return 0.359861591696 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.110715702176
                      return 0.0267642292798 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.932500004768
                    if ( max_col_coverage <= 0.580830097198 ) {
                      return 0.0343080129209 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.580830097198
                      return 0.024393540368 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.980939269066
          if ( max_col_coverage <= 0.44037425518 ) {
            if ( max_col_coverage <= 0.321080803871 ) {
              if ( max_col_coverage <= 0.212637633085 ) {
                if ( median_col_coverage <= 0.00243605719879 ) {
                  if ( min_col_support <= 0.910500049591 ) {
                    if ( median_col_support <= 0.975499987602 ) {
                      return 0.401234567901 < maxgini;
                    }
                    else {  // if median_col_support > 0.975499987602
                      return 0.132317507975 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.910500049591
                    if ( mean_col_coverage <= 0.0657465904951 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0657465904951
                      return 0.0568233194527 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.00243605719879
                  if ( min_col_support <= 0.930500030518 ) {
                    if ( mean_col_support <= 0.986566960812 ) {
                      return 0.0264628153566 < maxgini;
                    }
                    else {  // if mean_col_support > 0.986566960812
                      return 0.0103741567115 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.930500030518
                    if ( median_col_support <= 0.980499982834 ) {
                      return 0.0236355200041 < maxgini;
                    }
                    else {  // if median_col_support > 0.980499982834
                      return 0.00815736572541 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.212637633085
                if ( median_col_coverage <= 0.00355240888894 ) {
                  if ( median_col_support <= 0.978500008583 ) {
                    if ( mean_col_support <= 0.980970561504 ) {
                      return 0.456747404844 < maxgini;
                    }
                    else {  // if mean_col_support > 0.980970561504
                      return 0.111340639333 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.978500008583
                    if ( min_col_support <= 0.981500029564 ) {
                      return 0.035784347978 < maxgini;
                    }
                    else {  // if min_col_support > 0.981500029564
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.00355240888894
                  if ( median_col_coverage <= 0.095534928143 ) {
                    if ( mean_col_support <= 0.991303324699 ) {
                      return 0.020313692797 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991303324699
                      return 0.0110191170173 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.095534928143
                    if ( median_col_coverage <= 0.139605760574 ) {
                      return 0.027792765967 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.139605760574
                      return 0.0233317929173 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.321080803871
              if ( median_col_support <= 0.982499957085 ) {
                if ( min_col_support <= 0.941499948502 ) {
                  if ( min_col_coverage <= 0.00711752753705 ) {
                    if ( mean_col_coverage <= 0.187052994967 ) {
                      return 0.146211072664 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.187052994967
                      return 0.342653182629 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00711752753705
                    if ( max_col_coverage <= 0.435439229012 ) {
                      return 0.0223276657473 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.435439229012
                      return 0.0149462150189 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.941499948502
                  if ( mean_col_support <= 0.987735271454 ) {
                    if ( mean_col_support <= 0.983500003815 ) {
                      return 0.0360977154034 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983500003815
                      return 0.0295963514655 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.987735271454
                    if ( min_col_support <= 0.964499950409 ) {
                      return 0.0208280510109 < maxgini;
                    }
                    else {  // if min_col_support > 0.964499950409
                      return 0.0266951783801 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.982499957085
                if ( min_col_coverage <= 0.201198220253 ) {
                  if ( mean_col_coverage <= 0.289809167385 ) {
                    if ( min_col_support <= 0.771499991417 ) {
                      return 0.138115393183 < maxgini;
                    }
                    else {  // if min_col_support > 0.771499991417
                      return 0.0135159075886 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.289809167385
                    if ( mean_col_coverage <= 0.322706669569 ) {
                      return 0.00689990899018 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.322706669569
                      return 0.0223435155926 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.201198220253
                  if ( mean_col_support <= 0.99538230896 ) {
                    if ( min_col_support <= 0.971500039101 ) {
                      return 0.0193385685725 < maxgini;
                    }
                    else {  // if min_col_support > 0.971500039101
                      return 0.0247632964061 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.99538230896
                    if ( min_col_coverage <= 0.223739847541 ) {
                      return 0.0114545647015 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.223739847541
                      return 0.0168330870904 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.44037425518
            if ( median_col_support <= 0.986500024796 ) {
              if ( median_col_support <= 0.977499961853 ) {
                if ( mean_col_coverage <= 0.302022904158 ) {
                  if ( max_col_coverage <= 0.471077084541 ) {
                    if ( mean_col_support <= 0.987441182137 ) {
                      return 0.0384467512495 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987441182137
                      return 0.0920915712799 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.471077084541
                    if ( mean_col_support <= 0.987852931023 ) {
                      return 0.148101436872 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987852931023
                      return 0.444444444444 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.302022904158
                  if ( median_col_support <= 0.971500039101 ) {
                    if ( mean_col_support <= 0.981970608234 ) {
                      return 0.0213012282002 < maxgini;
                    }
                    else {  // if mean_col_support > 0.981970608234
                      return 0.0277348040019 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.971500039101
                    if ( max_col_coverage <= 0.563345372677 ) {
                      return 0.0238896488899 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.563345372677
                      return 0.019527452868 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.977499961853
                if ( min_col_coverage <= 0.339236795902 ) {
                  if ( median_col_support <= 0.981500029564 ) {
                    if ( min_col_coverage <= 0.0300111994147 ) {
                      return 0.375 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0300111994147
                      return 0.0194712179427 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.981500029564
                    if ( max_col_coverage <= 0.461118549109 ) {
                      return 0.0179852026012 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.461118549109
                      return 0.0143345921449 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.339236795902
                  if ( max_col_coverage <= 0.54917049408 ) {
                    if ( median_col_support <= 0.981500029564 ) {
                      return 0.0222627847731 < maxgini;
                    }
                    else {  // if median_col_support > 0.981500029564
                      return 0.0195303416378 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.54917049408
                    if ( median_col_support <= 0.982499957085 ) {
                      return 0.0202556464373 < maxgini;
                    }
                    else {  // if median_col_support > 0.982499957085
                      return 0.0165475643096 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.986500024796
              if ( min_col_support <= 0.987499952316 ) {
                if ( mean_col_support <= 0.995676517487 ) {
                  if ( max_col_coverage <= 0.581593573093 ) {
                    if ( median_col_coverage <= 0.281055420637 ) {
                      return 0.00803419395136 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.281055420637
                      return 0.0140096729368 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.581593573093
                    if ( min_col_support <= 0.976500034332 ) {
                      return 0.00853267457528 < maxgini;
                    }
                    else {  // if min_col_support > 0.976500034332
                      return 0.0131022306398 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.995676517487
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( median_col_coverage <= 0.336485445499 ) {
                      return 0.0112857947315 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.336485445499
                      return 0.00834697553719 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( min_col_support <= 0.974500000477 ) {
                      return 0.00236552751932 < maxgini;
                    }
                    else {  // if min_col_support > 0.974500000477
                      return 0.0065049350677 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.987499952316
                if ( median_col_support <= 0.994500041008 ) {
                  if ( mean_col_coverage <= 0.511976003647 ) {
                    if ( mean_col_support <= 0.996735334396 ) {
                      return 0.0138360819575 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996735334396
                      return 0.00981496724775 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.511976003647
                    if ( mean_col_support <= 0.996735334396 ) {
                      return 0.0109354218958 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996735334396
                      return 0.0040412946132 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.994500041008
                  if ( max_col_coverage <= 0.573739171028 ) {
                    if ( mean_col_support <= 0.997617661953 ) {
                      return 0.0106233672688 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997617661953
                      return 0.00653936643474 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.573739171028
                    if ( max_col_coverage <= 0.604524731636 ) {
                      return 0.00591254441983 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.604524731636
                      return 0.00273404624374 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if median_col_coverage > 0.459621369839
        if ( min_col_support <= 0.649500012398 ) {
          if ( min_col_coverage <= 0.813779115677 ) {
            if ( min_col_coverage <= 0.734561502934 ) {
              if ( mean_col_coverage <= 0.747076511383 ) {
                if ( mean_col_support <= 0.96532356739 ) {
                  if ( min_col_support <= 0.328999996185 ) {
                    if ( median_col_coverage <= 0.490549832582 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.490549832582
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.328999996185
                    if ( median_col_coverage <= 0.481600224972 ) {
                      return 0.0354913630425 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.481600224972
                      return 0.0503484454633 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.96532356739
                  if ( min_col_coverage <= 0.441265285015 ) {
                    if ( max_col_coverage <= 0.644944429398 ) {
                      return 0.165289256198 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.644944429398
                      return 0.48 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.441265285015
                    if ( min_col_support <= 0.5625 ) {
                      return 0.338779323754 < maxgini;
                    }
                    else {  // if min_col_support > 0.5625
                      return 0.083671875 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.747076511383
                if ( median_col_support <= 0.988499999046 ) {
                  if ( median_col_coverage <= 0.598701477051 ) {
                    if ( median_col_support <= 0.939999997616 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.939999997616
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.598701477051
                    if ( median_col_support <= 0.96749997139 ) {
                      return 0.0460765969131 < maxgini;
                    }
                    else {  // if median_col_support > 0.96749997139
                      return 0.199843310884 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.988499999046
                  if ( min_col_coverage <= 0.61243224144 ) {
                    if ( mean_col_support <= 0.974205851555 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.974205851555
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.61243224144
                    if ( max_col_coverage <= 0.987633824348 ) {
                      return 0.411125544804 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.987633824348
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.734561502934
              if ( median_col_support <= 0.988499999046 ) {
                if ( min_col_coverage <= 0.790500283241 ) {
                  if ( min_col_coverage <= 0.790341973305 ) {
                    if ( mean_col_coverage <= 0.80078369379 ) {
                      return 0.449071978271 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.80078369379
                      return 0.167468762224 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.790341973305
                    return false;
                  }
                }
                else {  // if min_col_coverage > 0.790500283241
                  if ( mean_col_coverage <= 0.874613285065 ) {
                    if ( median_col_coverage <= 0.824550628662 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.824550628662
                      return 0.375 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.874613285065
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if median_col_support > 0.988499999046
                if ( median_col_coverage <= 0.766075134277 ) {
                  if ( median_col_support <= 0.990000009537 ) {
                    if ( median_col_coverage <= 0.752414941788 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.752414941788
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990000009537
                    if ( mean_col_support <= 0.950705766678 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.950705766678
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.766075134277
                  if ( min_col_coverage <= 0.805818855762 ) {
                    if ( max_col_coverage <= 0.980290532112 ) {
                      return 0.499117293976 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.980290532112
                      return 0.328180737218 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.805818855762
                    if ( mean_col_coverage <= 0.863191962242 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.863191962242
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.813779115677
            if ( min_col_support <= 0.619500041008 ) {
              if ( min_col_coverage <= 0.932107329369 ) {
                if ( mean_col_support <= 0.966882348061 ) {
                  if ( median_col_coverage <= 0.831560254097 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.831560254097
                    if ( median_col_support <= 0.978500008583 ) {
                      return 0.251301939058 < maxgini;
                    }
                    else {  // if median_col_support > 0.978500008583
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.966882348061
                  if ( mean_col_support <= 0.971588253975 ) {
                    if ( mean_col_coverage <= 0.946306765079 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.946306765079
                      return 0.499479708637 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.971588253975
                    if ( mean_col_coverage <= 0.960005640984 ) {
                      return 0.4921875 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.960005640984
                      return false;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.932107329369
                if ( median_col_support <= 0.969500005245 ) {
                  if ( min_col_coverage <= 0.96369600296 ) {
                    if ( median_col_support <= 0.858999967575 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.858999967575
                      return 0.152777777778 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.96369600296
                    if ( min_col_coverage <= 0.996608257294 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.996608257294
                      return false;
                    }
                  }
                }
                else {  // if median_col_support > 0.969500005245
                  if ( min_col_support <= 0.5 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if min_col_support > 0.5
                    if ( median_col_support <= 0.99950003624 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.619500041008
              if ( max_col_coverage <= 0.990251600742 ) {
                if ( min_col_coverage <= 0.876868724823 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_coverage > 0.876868724823
                  if ( min_col_coverage <= 0.94563794136 ) {
                    if ( median_col_coverage <= 0.902814149857 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.902814149857
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.94563794136
                    return 0.0 < maxgini;
                  }
                }
              }
              else {  // if max_col_coverage > 0.990251600742
                if ( mean_col_coverage <= 0.995543122292 ) {
                  if ( mean_col_coverage <= 0.962236940861 ) {
                    if ( mean_col_support <= 0.976794064045 ) {
                      return 0.0783007080383 < maxgini;
                    }
                    else {  // if mean_col_support > 0.976794064045
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.962236940861
                    if ( min_col_support <= 0.629000008106 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_support > 0.629000008106
                      return 0.453686200378 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.995543122292
                  if ( min_col_coverage <= 0.997184157372 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.997184157372
                    if ( median_col_support <= 0.983500003815 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.983500003815
                      return false;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.649500012398
          if ( min_col_support <= 0.952499985695 ) {
            if ( min_col_coverage <= 0.540996551514 ) {
              if ( median_col_coverage <= 0.50325024128 ) {
                if ( min_col_support <= 0.901499986649 ) {
                  if ( median_col_support <= 0.953500032425 ) {
                    if ( max_col_coverage <= 0.700125932693 ) {
                      return 0.0312562424045 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.700125932693
                      return 0.0138169388353 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.953500032425
                    if ( max_col_coverage <= 0.513521611691 ) {
                      return 0.260355029586 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.513521611691
                      return 0.0156561371838 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.901499986649
                  if ( max_col_support <= 0.99849998951 ) {
                    if ( mean_col_coverage <= 0.487815320492 ) {
                      return 0.399524375743 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.487815320492
                      return 0.0236974413668 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.99849998951
                    if ( median_col_coverage <= 0.503218889236 ) {
                      return 0.0165493319003 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.503218889236
                      return 0.077339022394 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.50325024128
                if ( median_col_support <= 0.971500039101 ) {
                  if ( mean_col_support <= 0.865058779716 ) {
                    if ( median_col_support <= 0.776000022888 ) {
                      return 0.0873168439489 < maxgini;
                    }
                    else {  // if median_col_support > 0.776000022888
                      return 0.453686200378 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.865058779716
                    if ( median_col_support <= 0.918500006199 ) {
                      return 0.0297921639067 < maxgini;
                    }
                    else {  // if median_col_support > 0.918500006199
                      return 0.0197070396002 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.971500039101
                  if ( max_col_coverage <= 0.966987967491 ) {
                    if ( max_col_coverage <= 0.731729567051 ) {
                      return 0.00957595018411 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.731729567051
                      return 0.00530078419262 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.966987967491
                    if ( min_col_coverage <= 0.391490191221 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.391490191221
                      return 0.401234567901 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.540996551514
              if ( min_col_coverage <= 0.983823120594 ) {
                if ( mean_col_coverage <= 0.730562925339 ) {
                  if ( max_col_support <= 0.99849998951 ) {
                    if ( min_col_support <= 0.658499956131 ) {
                      return 0.139069478207 < maxgini;
                    }
                    else {  // if min_col_support > 0.658499956131
                      return 0.0193568577926 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.99849998951
                    if ( median_col_support <= 0.972499966621 ) {
                      return 0.0162614044086 < maxgini;
                    }
                    else {  // if median_col_support > 0.972499966621
                      return 0.00646402756074 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.730562925339
                  if ( median_col_support <= 0.713500022888 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.713500022888
                    if ( mean_col_support <= 0.963617682457 ) {
                      return 0.0258260732979 < maxgini;
                    }
                    else {  // if mean_col_support > 0.963617682457
                      return 0.003522338592 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.983823120594
                if ( mean_col_coverage <= 0.988440871239 ) {
                  if ( min_col_coverage <= 0.983875155449 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.983875155449
                    return 0.0 < maxgini;
                  }
                }
                else {  // if mean_col_coverage > 0.988440871239
                  if ( max_col_support <= 0.999000012875 ) {
                    return false;
                  }
                  else {  // if max_col_support > 0.999000012875
                    if ( mean_col_coverage <= 0.998140454292 ) {
                      return 0.0621925452815 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.998140454292
                      return 0.154872628034 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.952499985695
            if ( min_col_coverage <= 0.584799289703 ) {
              if ( median_col_support <= 0.990499973297 ) {
                if ( median_col_support <= 0.986500024796 ) {
                  if ( min_col_coverage <= 0.505933523178 ) {
                    if ( max_col_coverage <= 0.612264513969 ) {
                      return 0.0192683296289 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.612264513969
                      return 0.0158816350453 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.505933523178
                    if ( max_col_coverage <= 0.747516512871 ) {
                      return 0.0131116761166 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.747516512871
                      return 0.0106898461869 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.986500024796
                  if ( mean_col_coverage <= 0.520987093449 ) {
                    if ( min_col_coverage <= 0.460984259844 ) {
                      return 0.0156938544889 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.460984259844
                      return 0.0310344127912 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.520987093449
                    if ( median_col_coverage <= 0.539738178253 ) {
                      return 0.010471113161 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.539738178253
                      return 0.0080273460739 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.990499973297
                if ( median_col_support <= 0.993499994278 ) {
                  if ( median_col_coverage <= 0.509744763374 ) {
                    if ( min_col_support <= 0.986500024796 ) {
                      return 0.00756544304881 < maxgini;
                    }
                    else {  // if min_col_support > 0.986500024796
                      return 0.00966866832777 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.509744763374
                    if ( min_col_coverage <= 0.584555029869 ) {
                      return 0.00523611659164 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.584555029869
                      return 0.0194746837045 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( mean_col_coverage <= 0.56031358242 ) {
                    if ( max_col_coverage <= 0.598962783813 ) {
                      return 0.00702008149539 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.598962783813
                      return 0.00463208351837 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.56031358242
                    if ( max_col_coverage <= 0.677171230316 ) {
                      return 0.00348997472799 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.677171230316
                      return 0.00256836475575 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.584799289703
              if ( mean_col_coverage <= 0.740068972111 ) {
                if ( median_col_support <= 0.989500045776 ) {
                  if ( median_col_support <= 0.974500000477 ) {
                    if ( mean_col_support <= 0.985558748245 ) {
                      return 0.0277679358276 < maxgini;
                    }
                    else {  // if mean_col_support > 0.985558748245
                      return 0.012486128627 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.974500000477
                    if ( max_col_support <= 0.993499994278 ) {
                      return 0.42 < maxgini;
                    }
                    else {  // if max_col_support > 0.993499994278
                      return 0.00776579969003 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.989500045776
                  if ( median_col_support <= 0.992499947548 ) {
                    if ( min_col_support <= 0.981500029564 ) {
                      return 0.0034231654816 < maxgini;
                    }
                    else {  // if min_col_support > 0.981500029564
                      return 0.00534856499601 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.992499947548
                    if ( max_col_coverage <= 0.774363636971 ) {
                      return 0.00253128548635 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.774363636971
                      return 0.00125165099376 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.740068972111
                if ( min_col_support <= 0.987499952316 ) {
                  if ( median_col_coverage <= 0.780123710632 ) {
                    if ( min_col_support <= 0.983500003815 ) {
                      return 0.00288590306768 < maxgini;
                    }
                    else {  // if min_col_support > 0.983500003815
                      return 0.00186788847027 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.780123710632
                    if ( mean_col_coverage <= 0.894807696342 ) {
                      return 0.00147862905193 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.894807696342
                      return 0.000427444641649 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.987499952316
                  if ( min_col_support <= 0.993499994278 ) {
                    if ( median_col_coverage <= 0.836580634117 ) {
                      return 0.000829612454182 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.836580634117
                      return 0.000308618154828 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.993499994278
                    if ( mean_col_support <= 0.997323513031 ) {
                      return 0.00225515176543 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997323513031
                      return 0.000161675822749 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    bool shouldCorrect9(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( min_col_support <= 0.930500030518 ) {
        if ( max_col_coverage <= 0.51715362072 ) {
          if ( median_col_support <= 0.949499964714 ) {
            if ( median_col_coverage <= 0.00644124019891 ) {
              if ( max_col_coverage <= 0.177097529173 ) {
                if ( min_col_support <= 0.50049996376 ) {
                  if ( max_col_coverage <= 0.172460258007 ) {
                    if ( min_col_support <= 0.388500005007 ) {
                      return 0.0663117758662 < maxgini;
                    }
                    else {  // if min_col_support > 0.388500005007
                      return 0.0111884570572 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.172460258007
                    if ( mean_col_support <= 0.796088218689 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if mean_col_support > 0.796088218689
                      return 0.0677131425054 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.50049996376
                  if ( mean_col_coverage <= 0.0419211685658 ) {
                    if ( mean_col_coverage <= 0.0302275009453 ) {
                      return 0.0274880777993 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0302275009453
                      return 0.0419551900923 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.0419211685658
                    if ( mean_col_support <= 0.969393372536 ) {
                      return 0.0597438543817 < maxgini;
                    }
                    else {  // if mean_col_support > 0.969393372536
                      return 0.0293917116295 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.177097529173
                if ( mean_col_support <= 0.891558885574 ) {
                  if ( mean_col_coverage <= 0.106991678476 ) {
                    if ( min_col_coverage <= 0.00625979620963 ) {
                      return 0.0288826834814 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00625979620963
                      return 0.169921875 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.106991678476
                    if ( median_col_coverage <= 0.0035215197131 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.0035215197131
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.891558885574
                  if ( mean_col_support <= 0.891617596149 ) {
                    if ( min_col_support <= 0.526499986649 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.526499986649
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.891617596149
                    if ( max_col_coverage <= 0.250877261162 ) {
                      return 0.0974509039624 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.250877261162
                      return 0.221289490387 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.00644124019891
              if ( median_col_support <= 0.891499996185 ) {
                if ( max_col_support <= 0.992499947548 ) {
                  if ( min_col_support <= 0.278500020504 ) {
                    if ( min_col_coverage <= 0.0181041173637 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.0181041173637
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.278500020504
                    if ( max_col_support <= 0.984500050545 ) {
                      return 0.0181077164872 < maxgini;
                    }
                    else {  // if max_col_support > 0.984500050545
                      return 0.0279027539474 < maxgini;
                    }
                  }
                }
                else {  // if max_col_support > 0.992499947548
                  if ( median_col_support <= 0.827499985695 ) {
                    if ( min_col_coverage <= 0.283000826836 ) {
                      return 0.0458839447255 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.283000826836
                      return 0.0556070863505 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.827499985695
                    if ( min_col_support <= 0.776499986649 ) {
                      return 0.0402765004593 < maxgini;
                    }
                    else {  // if min_col_support > 0.776499986649
                      return 0.0440659409721 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.891499996185
                if ( mean_col_support <= 0.986294150352 ) {
                  if ( min_col_coverage <= 0.00299850758165 ) {
                    if ( median_col_coverage <= 0.0247388742864 ) {
                      return 0.0549501595628 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0247388742864
                      return 0.0971095965669 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00299850758165
                    if ( mean_col_coverage <= 0.103210195899 ) {
                      return 0.0274808725246 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.103210195899
                      return 0.0356324913969 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.986294150352
                  return false;
                }
              }
            }
          }
          else {  // if median_col_support > 0.949499964714
            if ( mean_col_support <= 0.984878540039 ) {
              if ( min_col_coverage <= 0.00303490832448 ) {
                if ( median_col_coverage <= 0.0745171383023 ) {
                  if ( median_col_support <= 0.977499961853 ) {
                    if ( max_col_coverage <= 0.274783015251 ) {
                      return 0.0568658998641 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.274783015251
                      return 0.186788257227 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.977499961853
                    if ( mean_col_support <= 0.934684514999 ) {
                      return 0.138995622974 < maxgini;
                    }
                    else {  // if mean_col_support > 0.934684514999
                      return 0.0415148668464 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.0745171383023
                  if ( mean_col_coverage <= 0.107486672699 ) {
                    if ( median_col_coverage <= 0.0756044536829 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.0756044536829
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.107486672699
                    if ( mean_col_support <= 0.970006465912 ) {
                      return 0.375 < maxgini;
                    }
                    else {  // if mean_col_support > 0.970006465912
                      return 0.0713305898491 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.00303490832448
                if ( mean_col_support <= 0.92191183567 ) {
                  if ( min_col_coverage <= 0.319353550673 ) {
                    if ( mean_col_support <= 0.921558499336 ) {
                      return 0.0530953833832 < maxgini;
                    }
                    else {  // if mean_col_support > 0.921558499336
                      return 0.18325697626 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.319353550673
                    if ( min_col_support <= 0.516999959946 ) {
                      return 0.18836565097 < maxgini;
                    }
                    else {  // if min_col_support > 0.516999959946
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.92191183567
                  if ( mean_col_support <= 0.980757236481 ) {
                    if ( mean_col_coverage <= 0.490298509598 ) {
                      return 0.0261968363348 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.490298509598
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.980757236481
                    if ( median_col_support <= 0.977499961853 ) {
                      return 0.0238050696329 < maxgini;
                    }
                    else {  // if median_col_support > 0.977499961853
                      return 0.0164122749937 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.984878540039
              if ( min_col_coverage <= 0.00279720826074 ) {
                if ( median_col_support <= 0.977499961853 ) {
                  if ( min_col_coverage <= 0.00278940564021 ) {
                    if ( min_col_coverage <= 0.00275103701279 ) {
                      return 0.110276160944 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00275103701279
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00278940564021
                    if ( mean_col_coverage <= 0.0978475213051 ) {
                      return 0.497041420118 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0978475213051
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.977499961853
                  if ( max_col_coverage <= 0.306398689747 ) {
                    if ( median_col_coverage <= 0.00220022257417 ) {
                      return 0.214910453978 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00220022257417
                      return 0.0317237929851 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.306398689747
                    if ( max_col_coverage <= 0.306769132614 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.306769132614
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.00279720826074
                if ( mean_col_support <= 0.986585736275 ) {
                  if ( mean_col_support <= 0.986577391624 ) {
                    if ( median_col_support <= 0.977499961853 ) {
                      return 0.0208670220741 < maxgini;
                    }
                    else {  // if median_col_support > 0.977499961853
                      return 0.0157184814331 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.986577391624
                    if ( max_col_coverage <= 0.213542968035 ) {
                      return 0.260355029586 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.213542968035
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.986585736275
                  if ( max_col_coverage <= 0.403866916895 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.0165198444167 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.0101079778258 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.403866916895
                    if ( median_col_support <= 0.983500003815 ) {
                      return 0.0156386488834 < maxgini;
                    }
                    else {  // if median_col_support > 0.983500003815
                      return 0.00957674476661 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.51715362072
          if ( min_col_coverage <= 0.983107328415 ) {
            if ( min_col_support <= 0.584499955177 ) {
              if ( min_col_coverage <= 0.795537471771 ) {
                if ( mean_col_coverage <= 0.751604795456 ) {
                  if ( median_col_support <= 0.979499995708 ) {
                    if ( max_col_coverage <= 0.982191801071 ) {
                      return 0.0449744144431 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.982191801071
                      return false;
                    }
                  }
                  else {  // if median_col_support > 0.979499995708
                    if ( max_col_coverage <= 0.638787031174 ) {
                      return 0.120511952347 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.638787031174
                      return 0.256016540914 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.751604795456
                  if ( median_col_coverage <= 0.612742781639 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.612742781639
                    if ( mean_col_support <= 0.962852954865 ) {
                      return 0.141705948689 < maxgini;
                    }
                    else {  // if mean_col_support > 0.962852954865
                      return 0.447028066128 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.795537471771
                if ( median_col_support <= 0.983500003815 ) {
                  if ( mean_col_support <= 0.932970523834 ) {
                    if ( mean_col_coverage <= 0.958909153938 ) {
                      return 0.326989619377 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.958909153938
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.932970523834
                    if ( max_col_coverage <= 0.95771241188 ) {
                      return 0.32 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.95771241188
                      return 0.0449200687013 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.983500003815
                  if ( median_col_support <= 0.996500015259 ) {
                    if ( mean_col_coverage <= 0.989816308022 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.989816308022
                      return 0.444444444444 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.996500015259
                    if ( mean_col_coverage <= 0.997402727604 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.997402727604
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.584499955177
              if ( min_col_coverage <= 0.491107165813 ) {
                if ( max_col_coverage <= 0.99876844883 ) {
                  if ( median_col_support <= 0.962499976158 ) {
                    if ( min_col_coverage <= 0.311082541943 ) {
                      return 0.0229232077028 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.311082541943
                      return 0.0296111701088 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.962499976158
                    if ( mean_col_coverage <= 0.712930381298 ) {
                      return 0.0151530588669 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.712930381298
                      return 0.308390022676 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.99876844883
                  if ( mean_col_coverage <= 0.785126924515 ) {
                    if ( median_col_coverage <= 0.426181674004 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.426181674004
                      return 0.387811634349 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.785126924515
                    return false;
                  }
                }
              }
              else {  // if min_col_coverage > 0.491107165813
                if ( max_col_coverage <= 0.787307500839 ) {
                  if ( median_col_support <= 0.965499997139 ) {
                    if ( mean_col_coverage <= 0.715144634247 ) {
                      return 0.0218638828238 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.715144634247
                      return 0.0510160232288 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.965499997139
                    if ( min_col_support <= 0.596500039101 ) {
                      return 0.246024236134 < maxgini;
                    }
                    else {  // if min_col_support > 0.596500039101
                      return 0.00870865755318 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.787307500839
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( min_col_support <= 0.712499976158 ) {
                      return 0.0576160990682 < maxgini;
                    }
                    else {  // if min_col_support > 0.712499976158
                      return 0.00685727833478 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( max_col_coverage <= 0.984265744686 ) {
                      return 0.0624349635796 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.984265744686
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.983107328415
            if ( max_col_support <= 0.99950003624 ) {
              return false;
            }
            else {  // if max_col_support > 0.99950003624
              if ( mean_col_coverage <= 0.99801158905 ) {
                if ( mean_col_support <= 0.972294151783 ) {
                  if ( mean_col_support <= 0.937852978706 ) {
                    if ( max_col_coverage <= 0.995798945427 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.995798945427
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.937852978706
                    if ( max_col_coverage <= 0.998727738857 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.998727738857
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.972294151783
                  if ( mean_col_support <= 0.977411746979 ) {
                    if ( median_col_coverage <= 0.990322470665 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.990322470665
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.977411746979
                    if ( median_col_coverage <= 0.993527412415 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.993527412415
                      return 0.0392 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.99801158905
                if ( min_col_support <= 0.675999999046 ) {
                  if ( min_col_coverage <= 0.990564584732 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.990564584732
                    if ( median_col_coverage <= 0.996666550636 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.996666550636
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.675999999046
                  if ( min_col_coverage <= 0.985598385334 ) {
                    if ( mean_col_support <= 0.97779405117 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.97779405117
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.985598385334
                    if ( min_col_coverage <= 0.988690495491 ) {
                      return 0.277777777778 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.988690495491
                      return 0.10416020832 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.930500030518
        if ( mean_col_coverage <= 0.521858215332 ) {
          if ( median_col_coverage <= 0.311667919159 ) {
            if ( mean_col_coverage <= 0.349960505962 ) {
              if ( median_col_coverage <= 0.0638933330774 ) {
                if ( median_col_support <= 0.96850001812 ) {
                  if ( min_col_coverage <= 0.0604448765516 ) {
                    if ( max_col_coverage <= 0.336023896933 ) {
                      return 0.0390214299571 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.336023896933
                      return 0.426035502959 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0604448765516
                    if ( min_col_coverage <= 0.0612308681011 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.0612308681011
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.96850001812
                  if ( mean_col_support <= 0.992193758488 ) {
                    if ( median_col_support <= 0.978500008583 ) {
                      return 0.0201169945368 < maxgini;
                    }
                    else {  // if median_col_support > 0.978500008583
                      return 0.012581195104 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992193758488
                    if ( median_col_support <= 0.984500050545 ) {
                      return 0.0153697832618 < maxgini;
                    }
                    else {  // if median_col_support > 0.984500050545
                      return 0.00537019756953 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.0638933330774
                if ( median_col_support <= 0.978500008583 ) {
                  if ( mean_col_coverage <= 0.34995174408 ) {
                    if ( min_col_support <= 0.965499997139 ) {
                      return 0.0303646123209 < maxgini;
                    }
                    else {  // if min_col_support > 0.965499997139
                      return 0.0344953121054 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.34995174408
                    if ( max_col_coverage <= 0.445048332214 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.445048332214
                      return false;
                    }
                  }
                }
                else {  // if median_col_support > 0.978500008583
                  if ( median_col_support <= 0.991500020027 ) {
                    if ( min_col_support <= 0.965499997139 ) {
                      return 0.0186759790371 < maxgini;
                    }
                    else {  // if min_col_support > 0.965499997139
                      return 0.0224529575631 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.991500020027
                    if ( median_col_coverage <= 0.254401564598 ) {
                      return 0.00928478872128 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.254401564598
                      return 0.0158828070957 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.349960505962
              if ( min_col_coverage <= 0.264223903418 ) {
                if ( min_col_coverage <= 0.199372023344 ) {
                  if ( median_col_support <= 0.977499961853 ) {
                    if ( mean_col_support <= 0.986970543861 ) {
                      return 0.0793341260404 < maxgini;
                    }
                    else {  // if mean_col_support > 0.986970543861
                      return 0.27173119065 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.977499961853
                    if ( min_col_coverage <= 0.199362009764 ) {
                      return 0.0198186842796 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.199362009764
                      return 0.375 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.199372023344
                  if ( median_col_support <= 0.977499961853 ) {
                    if ( mean_col_coverage <= 0.429803490639 ) {
                      return 0.0226173371574 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.429803490639
                      return 0.408163265306 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.977499961853
                    if ( max_col_coverage <= 0.629632949829 ) {
                      return 0.011263278252 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.629632949829
                      return 0.103852191764 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.264223903418
                if ( median_col_coverage <= 0.311665058136 ) {
                  if ( max_col_coverage <= 0.48742929101 ) {
                    if ( min_col_support <= 0.984500050545 ) {
                      return 0.0223013196296 < maxgini;
                    }
                    else {  // if min_col_support > 0.984500050545
                      return 0.0125594338838 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.48742929101
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.0209719839751 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.00998885110546 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.311665058136
                  if ( mean_col_coverage <= 0.3619607687 ) {
                    return false;
                  }
                  else {  // if mean_col_coverage > 0.3619607687
                    if ( max_col_coverage <= 0.476666688919 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.476666688919
                      return false;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.311667919159
            if ( min_col_support <= 0.975499987602 ) {
              if ( mean_col_support <= 0.990852952003 ) {
                if ( min_col_support <= 0.96850001812 ) {
                  if ( mean_col_coverage <= 0.329158127308 ) {
                    if ( max_col_coverage <= 0.384334146976 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.384334146976
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.329158127308
                    if ( median_col_support <= 0.977499961853 ) {
                      return 0.0267985668089 < maxgini;
                    }
                    else {  // if median_col_support > 0.977499961853
                      return 0.0173198443345 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.96850001812
                  if ( median_col_coverage <= 0.486445665359 ) {
                    if ( max_col_support <= 0.997500002384 ) {
                      return 0.0398375799897 < maxgini;
                    }
                    else {  // if max_col_support > 0.997500002384
                      return 0.0256495478005 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.486445665359
                    if ( min_col_coverage <= 0.46598482132 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.46598482132
                      return 0.252400548697 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.990852952003
                if ( min_col_support <= 0.96850001812 ) {
                  if ( mean_col_support <= 0.992500066757 ) {
                    if ( median_col_support <= 0.985499978065 ) {
                      return 0.0156739654462 < maxgini;
                    }
                    else {  // if median_col_support > 0.985499978065
                      return 0.0106111831169 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992500066757
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.00941528466639 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.00289016735744 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.96850001812
                  if ( median_col_support <= 0.989500045776 ) {
                    if ( mean_col_support <= 0.99326467514 ) {
                      return 0.0183330206412 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99326467514
                      return 0.012876589947 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.989500045776
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0119237492114 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.00651510853845 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.975499987602
              if ( mean_col_support <= 0.995676517487 ) {
                if ( mean_col_coverage <= 0.500835776329 ) {
                  if ( mean_col_support <= 0.993794083595 ) {
                    if ( max_col_coverage <= 0.513192951679 ) {
                      return 0.0216539361261 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.513192951679
                      return 0.017253525254 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.993794083595
                    if ( min_col_coverage <= 0.471247076988 ) {
                      return 0.0145553854267 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.471247076988
                      return 0.444444444444 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.500835776329
                  if ( mean_col_support <= 0.992029428482 ) {
                    if ( mean_col_coverage <= 0.521685838699 ) {
                      return 0.0211616002745 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.521685838699
                      return 0.0907029478458 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992029428482
                    if ( median_col_coverage <= 0.495481550694 ) {
                      return 0.0114498061937 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.495481550694
                      return 0.10833663628 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.995676517487
                if ( min_col_support <= 0.982499957085 ) {
                  if ( mean_col_coverage <= 0.521856486797 ) {
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.00893655618176 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.00547954067506 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.521856486797
                    if ( min_col_support <= 0.980000019073 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.980000019073
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.982499957085
                  if ( mean_col_support <= 0.997264742851 ) {
                    if ( max_col_coverage <= 0.601270914078 ) {
                      return 0.010931157812 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.601270914078
                      return 0.00730188214997 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.997264742851
                    if ( mean_col_support <= 0.997617661953 ) {
                      return 0.00757441406635 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997617661953
                      return 0.00568704809032 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.521858215332
          if ( min_col_coverage <= 0.581304609776 ) {
            if ( median_col_support <= 0.988499999046 ) {
              if ( median_col_support <= 0.979499995708 ) {
                if ( median_col_coverage <= 0.562678158283 ) {
                  if ( min_col_coverage <= 0.366772115231 ) {
                    if ( min_col_support <= 0.96850001812 ) {
                      return 0.0636253462604 < maxgini;
                    }
                    else {  // if min_col_support > 0.96850001812
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.366772115231
                    if ( mean_col_support <= 0.980617642403 ) {
                      return 0.0278572977619 < maxgini;
                    }
                    else {  // if mean_col_support > 0.980617642403
                      return 0.0176169195538 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.562678158283
                  if ( median_col_support <= 0.941499948502 ) {
                    if ( median_col_coverage <= 0.581790924072 ) {
                      return 0.231111111111 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.581790924072
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.941499948502
                    if ( max_col_support <= 0.994500041008 ) {
                      return 0.0956472232286 < maxgini;
                    }
                    else {  // if max_col_support > 0.994500041008
                      return 0.0142256829857 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.979499995708
                if ( min_col_support <= 0.957499980927 ) {
                  if ( mean_col_support <= 0.988735258579 ) {
                    if ( min_col_support <= 0.947499990463 ) {
                      return 0.00885693277712 < maxgini;
                    }
                    else {  // if min_col_support > 0.947499990463
                      return 0.011828261647 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.988735258579
                    if ( median_col_support <= 0.984500050545 ) {
                      return 0.00867951362738 < maxgini;
                    }
                    else {  // if median_col_support > 0.984500050545
                      return 0.00568776499615 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.957499980927
                  if ( mean_col_support <= 0.996676445007 ) {
                    if ( median_col_support <= 0.986500024796 ) {
                      return 0.014060643897 < maxgini;
                    }
                    else {  // if median_col_support > 0.986500024796
                      return 0.0104956675838 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.996676445007
                    if ( min_col_coverage <= 0.468151211739 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.468151211739
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.988499999046
              if ( median_col_support <= 0.992499947548 ) {
                if ( median_col_coverage <= 0.509534418583 ) {
                  if ( min_col_coverage <= 0.330410838127 ) {
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.197530864198 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.330410838127
                    if ( min_col_support <= 0.977499961853 ) {
                      return 0.00753178598893 < maxgini;
                    }
                    else {  // if min_col_support > 0.977499961853
                      return 0.0103013484566 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.509534418583
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( mean_col_support <= 0.996852934361 ) {
                      return 0.00782856745827 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996852934361
                      return 0.15572657311 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( mean_col_support <= 0.993147015572 ) {
                      return 0.00305595845072 < maxgini;
                    }
                    else {  // if mean_col_support > 0.993147015572
                      return 0.00592712383356 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.992499947548
                if ( median_col_support <= 0.994500041008 ) {
                  if ( median_col_coverage <= 0.345077693462 ) {
                    if ( min_col_support <= 0.960500001907 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.960500001907
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.345077693462
                    if ( min_col_coverage <= 0.473096251488 ) {
                      return 0.00642696296516 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.473096251488
                      return 0.00423532010616 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.994500041008
                  if ( median_col_coverage <= 0.513281702995 ) {
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.00443002496049 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.00232433084229 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.513281702995
                    if ( max_col_support <= 0.99849998951 ) {
                      return 0.0680968858131 < maxgini;
                    }
                    else {  // if max_col_support > 0.99849998951
                      return 0.00241108335766 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.581304609776
            if ( min_col_coverage <= 0.655306816101 ) {
              if ( median_col_support <= 0.989500045776 ) {
                if ( median_col_support <= 0.978500008583 ) {
                  if ( mean_col_coverage <= 0.752265274525 ) {
                    if ( min_col_coverage <= 0.655296564102 ) {
                      return 0.0129322965418 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.655296564102
                      return 0.260355029586 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.752265274525
                    if ( median_col_support <= 0.958500027657 ) {
                      return 0.046848 < maxgini;
                    }
                    else {  // if median_col_support > 0.958500027657
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.978500008583
                  if ( max_col_support <= 0.993499994278 ) {
                    if ( mean_col_support <= 0.983999967575 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983999967575
                      return false;
                    }
                  }
                  else {  // if max_col_support > 0.993499994278
                    if ( median_col_support <= 0.984500050545 ) {
                      return 0.00918799717942 < maxgini;
                    }
                    else {  // if median_col_support > 0.984500050545
                      return 0.00705658518247 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.989500045776
                if ( min_col_support <= 0.990499973297 ) {
                  if ( median_col_support <= 0.992499947548 ) {
                    if ( min_col_coverage <= 0.655299603939 ) {
                      return 0.00431889710112 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.655299603939
                      return 0.174817898023 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.992499947548
                    if ( median_col_coverage <= 0.660763263702 ) {
                      return 0.00253947979092 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.660763263702
                      return 0.00132648655274 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.990499973297
                  if ( max_col_coverage <= 0.781434178352 ) {
                    if ( max_col_coverage <= 0.781426072121 ) {
                      return 0.00229090070365 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.781426072121
                      return 0.15572657311 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.781434178352
                    if ( mean_col_support <= 0.996558785439 ) {
                      return 0.00336070743164 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996558785439
                      return 0.000808015381905 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.655306816101
              if ( min_col_support <= 0.986500024796 ) {
                if ( median_col_coverage <= 0.778110980988 ) {
                  if ( max_col_coverage <= 0.822970986366 ) {
                    if ( mean_col_support <= 0.992852926254 ) {
                      return 0.00949721156164 < maxgini;
                    }
                    else {  // if mean_col_support > 0.992852926254
                      return 0.00234954450958 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.822970986366
                    if ( median_col_coverage <= 0.778102278709 ) {
                      return 0.00240357156071 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.778102278709
                      return 0.14201183432 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.778110980988
                  if ( median_col_support <= 0.993499994278 ) {
                    if ( mean_col_support <= 0.990499973297 ) {
                      return 0.00318864287435 < maxgini;
                    }
                    else {  // if mean_col_support > 0.990499973297
                      return 0.00147626186316 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.993499994278
                    if ( min_col_support <= 0.976500034332 ) {
                      return 7.42822476784e-05 < maxgini;
                    }
                    else {  // if min_col_support > 0.976500034332
                      return 0.000517938999506 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.986500024796
                if ( mean_col_support <= 0.997558832169 ) {
                  if ( median_col_support <= 0.992499947548 ) {
                    if ( max_col_coverage <= 0.993231773376 ) {
                      return 0.00287927739363 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.993231773376
                      return 0.000720720627061 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.992499947548
                    if ( median_col_coverage <= 0.697379887104 ) {
                      return 0.00198408717486 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.697379887104
                      return 0.000754108645963 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.997558832169
                  if ( mean_col_coverage <= 0.739335954189 ) {
                    if ( min_col_coverage <= 0.677575945854 ) {
                      return 0.00170945263954 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.677575945854
                      return 0.0127763031728 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.739335954189
                    if ( min_col_coverage <= 0.983704447746 ) {
                      return 0.000213010989887 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.983704447746
                      return 0.00177078070482 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    std::pair<int, int> shouldCorrect_forest(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
        std::pair<int,int> result{0,0};
        bool b0 = shouldCorrect0(min_col_support, min_col_coverage, max_col_support, max_col_coverage, mean_col_support, mean_col_coverage, median_col_support, median_col_coverage, maxgini);
        result.first += !b0;
        result.second += b0;
        bool b1 = shouldCorrect1(min_col_support, min_col_coverage, max_col_support, max_col_coverage, mean_col_support, mean_col_coverage, median_col_support, median_col_coverage, maxgini);
        result.first += !b1;
        result.second += b1;
        bool b2 = shouldCorrect2(min_col_support, min_col_coverage, max_col_support, max_col_coverage, mean_col_support, mean_col_coverage, median_col_support, median_col_coverage, maxgini);
        result.first += !b2;
        result.second += b2;
        bool b3 = shouldCorrect3(min_col_support, min_col_coverage, max_col_support, max_col_coverage, mean_col_support, mean_col_coverage, median_col_support, median_col_coverage, maxgini);
        result.first += !b3;
        result.second += b3;
        bool b4 = shouldCorrect4(min_col_support, min_col_coverage, max_col_support, max_col_coverage, mean_col_support, mean_col_coverage, median_col_support, median_col_coverage, maxgini);
        result.first += !b4;
        result.second += b4;
        bool b5 = shouldCorrect5(min_col_support, min_col_coverage, max_col_support, max_col_coverage, mean_col_support, mean_col_coverage, median_col_support, median_col_coverage, maxgini);
        result.first += !b5;
        result.second += b5;
        bool b6 = shouldCorrect6(min_col_support, min_col_coverage, max_col_support, max_col_coverage, mean_col_support, mean_col_coverage, median_col_support, median_col_coverage, maxgini);
        result.first += !b6;
        result.second += b6;
        bool b7 = shouldCorrect7(min_col_support, min_col_coverage, max_col_support, max_col_coverage, mean_col_support, mean_col_coverage, median_col_support, median_col_coverage, maxgini);
        result.first += !b7;
        result.second += b7;
        bool b8 = shouldCorrect8(min_col_support, min_col_coverage, max_col_support, max_col_coverage, mean_col_support, mean_col_coverage, median_col_support, median_col_coverage, maxgini);
        result.first += !b8;
        result.second += b8;
        bool b9 = shouldCorrect9(min_col_support, min_col_coverage, max_col_support, max_col_coverage, mean_col_support, mean_col_coverage, median_col_support, median_col_coverage, maxgini);
        result.first += !b9;
        result.second += b9;
        return result;
    }










}
