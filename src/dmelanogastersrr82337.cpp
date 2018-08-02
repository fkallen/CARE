#include "../inc/dmelanogastersrr82337.hpp"

namespace dmelanogaster_srr82337{

    bool shouldCorrect0(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
      if ( mean_col_support <= 0.988468527794 ) {
        if ( median_col_coverage <= 0.606097817421 ) {
          if ( min_col_support <= 0.757500052452 ) {
            if ( mean_col_coverage <= 0.451627969742 ) {
              if ( min_col_coverage <= 0.243949741125 ) {
                if ( median_col_support <= 0.596500039101 ) {
                  if ( max_col_coverage <= 0.314324438572 ) {
                    if ( mean_col_coverage <= 0.206802368164 ) {
                      return 0.136183369507 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.206802368164
                      return 0.241077359663 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.314324438572
                    if ( min_col_coverage <= 0.182234510779 ) {
                      return 0.199353773378 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.182234510779
                      return 0.334679696868 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.596500039101
                  if ( mean_col_coverage <= 0.275632202625 ) {
                    if ( min_col_coverage <= 0.0512162223458 ) {
                      return 0.111824289124 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0512162223458
                      return 0.0899670857576 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.275632202625
                    if ( median_col_support <= 0.970499992371 ) {
                      return 0.110279852904 < maxgini;
                    }
                    else {  // if median_col_support > 0.970499992371
                      return 0.328871246637 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.243949741125
                if ( mean_col_coverage <= 0.400340139866 ) {
                  if ( median_col_coverage <= 0.305584430695 ) {
                    if ( median_col_support <= 0.981500029564 ) {
                      return 0.146305383667 < maxgini;
                    }
                    else {  // if median_col_support > 0.981500029564
                      return 0.36140953601 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.305584430695
                    if ( min_col_coverage <= 0.272930771112 ) {
                      return 0.166029477053 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.272930771112
                      return 0.213140470337 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.400340139866
                  if ( mean_col_support <= 0.970441102982 ) {
                    if ( min_col_support <= 0.640499949455 ) {
                      return 0.26914767798 < maxgini;
                    }
                    else {  // if min_col_support > 0.640499949455
                      return 0.133775785431 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.970441102982
                    if ( min_col_coverage <= 0.320737719536 ) {
                      return 0.372357395135 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.320737719536
                      return 0.395585243762 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.451627969742
              if ( median_col_coverage <= 0.487850725651 ) {
                if ( min_col_coverage <= 0.394911825657 ) {
                  if ( mean_col_coverage <= 0.567678451538 ) {
                    if ( min_col_support <= 0.630499958992 ) {
                      return 0.321385151002 < maxgini;
                    }
                    else {  // if min_col_support > 0.630499958992
                      return 0.200948947799 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.567678451538
                    if ( max_col_coverage <= 0.997929334641 ) {
                      return 0.299317522028 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.997929334641
                      return 0.408989194858 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.394911825657
                  if ( min_col_support <= 0.664499998093 ) {
                    if ( mean_col_coverage <= 0.481884777546 ) {
                      return 0.33615939626 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.481884777546
                      return 0.371724210283 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.664499998093
                    if ( min_col_support <= 0.705500006676 ) {
                      return 0.289763107516 < maxgini;
                    }
                    else {  // if min_col_support > 0.705500006676
                      return 0.223840371067 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.487850725651
                if ( min_col_support <= 0.669499993324 ) {
                  if ( mean_col_support <= 0.968735337257 ) {
                    if ( mean_col_support <= 0.941092133522 ) {
                      return 0.336463770503 < maxgini;
                    }
                    else {  // if mean_col_support > 0.941092133522
                      return 0.414228362417 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.968735337257
                    if ( median_col_coverage <= 0.559810042381 ) {
                      return 0.466349271833 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.559810042381
                      return 0.475455645623 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.669499993324
                  if ( median_col_support <= 0.987499952316 ) {
                    if ( max_col_coverage <= 0.617681324482 ) {
                      return 0.132300063083 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.617681324482
                      return 0.184441176886 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.987499952316
                    if ( min_col_coverage <= 0.573518753052 ) {
                      return 0.446805761225 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.573518753052
                      return 0.470268421073 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.757500052452
            if ( median_col_support <= 0.986500024796 ) {
              if ( mean_col_support <= 0.956710100174 ) {
                if ( median_col_support <= 0.836500048637 ) {
                  if ( mean_col_coverage <= 0.504279494286 ) {
                    if ( mean_col_coverage <= 0.419578999281 ) {
                      return 0.0993036160111 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.419578999281
                      return 0.132446702925 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.504279494286
                    if ( median_col_coverage <= 0.4893014431 ) {
                      return 0.169034142474 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.4893014431
                      return 0.2392361874 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.836500048637
                  if ( max_col_coverage <= 0.528263807297 ) {
                    if ( min_col_support <= 0.801499962807 ) {
                      return 0.0644015526396 < maxgini;
                    }
                    else {  // if min_col_support > 0.801499962807
                      return 0.0820882697383 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.528263807297
                    if ( min_col_coverage <= 0.5018004179 ) {
                      return 0.0819695908512 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.5018004179
                      return 0.105148282854 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.956710100174
                if ( max_col_coverage <= 0.998962640762 ) {
                  if ( median_col_coverage <= 0.0150848589838 ) {
                    if ( mean_col_coverage <= 0.10393165797 ) {
                      return 0.14005748738 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.10393165797
                      return 0.220734520792 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0150848589838
                    if ( median_col_coverage <= 0.0484100729227 ) {
                      return 0.0757969755561 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0484100729227
                      return 0.0448764142634 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.998962640762
                  if ( mean_col_support <= 0.985147118568 ) {
                    if ( median_col_support <= 0.985499978065 ) {
                      return 0.187247513973 < maxgini;
                    }
                    else {  // if median_col_support > 0.985499978065
                      return 0.491493383743 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.985147118568
                    if ( mean_col_coverage <= 0.821862518787 ) {
                      return 0.0621121564815 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.821862518787
                      return 0.3046875 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.986500024796
              if ( max_col_coverage <= 0.559038996696 ) {
                if ( min_col_support <= 0.804499983788 ) {
                  if ( mean_col_coverage <= 0.339164197445 ) {
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.203947122678 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.048052651348 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.339164197445
                    if ( min_col_coverage <= 0.333869874477 ) {
                      return 0.211404117633 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.333869874477
                      return 0.298794388782 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.804499983788
                  if ( median_col_coverage <= 0.343020021915 ) {
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.107996239723 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.0324039185096 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.343020021915
                    if ( min_col_support <= 0.84249997139 ) {
                      return 0.145545223081 < maxgini;
                    }
                    else {  // if min_col_support > 0.84249997139
                      return 0.0532315897701 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.559038996696
                if ( mean_col_coverage <= 0.552424132824 ) {
                  if ( median_col_coverage <= 0.441922098398 ) {
                    if ( min_col_coverage <= 0.0800943374634 ) {
                      return 0.340355992869 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0800943374634
                      return 0.115769633141 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.441922098398
                    if ( mean_col_support <= 0.977911710739 ) {
                      return 0.316342429247 < maxgini;
                    }
                    else {  // if mean_col_support > 0.977911710739
                      return 0.174926298421 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.552424132824
                  if ( mean_col_support <= 0.977205872536 ) {
                    if ( median_col_coverage <= 0.493800938129 ) {
                      return 0.264960548745 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.493800938129
                      return 0.381625032777 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.977205872536
                    if ( mean_col_support <= 0.987441182137 ) {
                      return 0.243524795965 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987441182137
                      return 0.208408832913 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.606097817421
          if ( max_col_coverage <= 0.86215698719 ) {
            if ( median_col_support <= 0.989500045776 ) {
              if ( mean_col_support <= 0.970911741257 ) {
                if ( max_col_coverage <= 0.75014936924 ) {
                  if ( max_col_coverage <= 0.722368478775 ) {
                    if ( min_col_support <= 0.664499998093 ) {
                      return 0.318444371653 < maxgini;
                    }
                    else {  // if min_col_support > 0.664499998093
                      return 0.117362919922 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.722368478775
                    if ( median_col_coverage <= 0.62835931778 ) {
                      return 0.270274564403 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.62835931778
                      return 0.220424661345 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.75014936924
                  if ( min_col_support <= 0.68850004673 ) {
                    if ( min_col_support <= 0.609500050545 ) {
                      return 0.415162227656 < maxgini;
                    }
                    else {  // if min_col_support > 0.609500050545
                      return 0.32892701935 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.68850004673
                    if ( min_col_support <= 0.755499958992 ) {
                      return 0.239741843594 < maxgini;
                    }
                    else {  // if min_col_support > 0.755499958992
                      return 0.121195593323 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.970911741257
                if ( median_col_support <= 0.979499995708 ) {
                  if ( min_col_support <= 0.786499977112 ) {
                    if ( min_col_coverage <= 0.565119862556 ) {
                      return 0.152546643005 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.565119862556
                      return 0.253495840546 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.786499977112
                    if ( min_col_support <= 0.871500015259 ) {
                      return 0.0759882966825 < maxgini;
                    }
                    else {  // if min_col_support > 0.871500015259
                      return 0.0330321530987 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.979499995708
                  if ( median_col_coverage <= 0.666835904121 ) {
                    if ( mean_col_support <= 0.980147004128 ) {
                      return 0.278045542255 < maxgini;
                    }
                    else {  // if mean_col_support > 0.980147004128
                      return 0.0657528393234 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.666835904121
                    if ( max_col_coverage <= 0.795682311058 ) {
                      return 0.108498861382 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.795682311058
                      return 0.17243090398 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.989500045776
              if ( median_col_support <= 0.99950003624 ) {
                if ( max_col_coverage <= 0.826372206211 ) {
                  if ( mean_col_support <= 0.97779417038 ) {
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.427204048426 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.449706753632 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.97779417038
                    if ( min_col_support <= 0.81149995327 ) {
                      return 0.385612084085 < maxgini;
                    }
                    else {  // if min_col_support > 0.81149995327
                      return 0.222958308462 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.826372206211
                  if ( max_col_coverage <= 0.858094990253 ) {
                    if ( median_col_support <= 0.991500020027 ) {
                      return 0.380338340057 < maxgini;
                    }
                    else {  // if median_col_support > 0.991500020027
                      return 0.432827891835 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.858094990253
                    if ( min_col_support <= 0.689499974251 ) {
                      return 0.442971618675 < maxgini;
                    }
                    else {  // if min_col_support > 0.689499974251
                      return 0.332340634266 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_coverage <= 0.67794406414 ) {
                  if ( median_col_coverage <= 0.659490466118 ) {
                    if ( mean_col_support <= 0.984382390976 ) {
                      return 0.471980628118 < maxgini;
                    }
                    else {  // if mean_col_support > 0.984382390976
                      return 0.370246252902 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.659490466118
                    if ( min_col_support <= 0.806499958038 ) {
                      return 0.480035509664 < maxgini;
                    }
                    else {  // if min_col_support > 0.806499958038
                      return 0.229786124206 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.67794406414
                  if ( min_col_support <= 0.81350004673 ) {
                    if ( mean_col_support <= 0.970911741257 ) {
                      return 0.473832572544 < maxgini;
                    }
                    else {  // if mean_col_support > 0.970911741257
                      return 0.482007247701 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.81350004673
                    if ( min_col_support <= 0.834499955177 ) {
                      return 0.335894401258 < maxgini;
                    }
                    else {  // if min_col_support > 0.834499955177
                      return 0.165858505137 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.86215698719
            if ( median_col_support <= 0.990499973297 ) {
              if ( min_col_support <= 0.771499991417 ) {
                if ( max_col_support <= 0.99849998951 ) {
                  if ( max_col_coverage <= 0.862565219402 ) {
                    return false;
                  }
                  else {  // if max_col_coverage > 0.862565219402
                    if ( max_col_coverage <= 0.914343357086 ) {
                      return 0.190577777778 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.914343357086
                      return 0.020173908105 < maxgini;
                    }
                  }
                }
                else {  // if max_col_support > 0.99849998951
                  if ( median_col_support <= 0.964499950409 ) {
                    if ( min_col_coverage <= 0.996778964996 ) {
                      return 0.374159628366 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.996778964996
                      return 0.182540581498 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.964499950409
                    if ( mean_col_support <= 0.95402944088 ) {
                      return 0.468609464791 < maxgini;
                    }
                    else {  // if mean_col_support > 0.95402944088
                      return 0.418408736845 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.771499991417
                if ( mean_col_coverage <= 0.902043640614 ) {
                  if ( median_col_coverage <= 0.764747321606 ) {
                    if ( min_col_support <= 0.855499982834 ) {
                      return 0.133762912409 < maxgini;
                    }
                    else {  // if min_col_support > 0.855499982834
                      return 0.0402480235258 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.764747321606
                    if ( min_col_coverage <= 0.727367997169 ) {
                      return 0.0704451381663 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.727367997169
                      return 0.11571545857 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.902043640614
                  if ( max_col_coverage <= 0.999430537224 ) {
                    if ( min_col_support <= 0.860499978065 ) {
                      return 0.30928437818 < maxgini;
                    }
                    else {  // if min_col_support > 0.860499978065
                      return 0.120697646687 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.999430537224
                    if ( median_col_support <= 0.981500029564 ) {
                      return 0.120089493669 < maxgini;
                    }
                    else {  // if median_col_support > 0.981500029564
                      return 0.203560684075 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.990499973297
              if ( median_col_support <= 0.99950003624 ) {
                if ( min_col_support <= 0.725499987602 ) {
                  if ( mean_col_coverage <= 0.946106255054 ) {
                    if ( min_col_support <= 0.597499966621 ) {
                      return 0.475163864148 < maxgini;
                    }
                    else {  // if min_col_support > 0.597499966621
                      return 0.443487768654 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.946106255054
                    if ( min_col_support <= 0.643499970436 ) {
                      return 0.478697196121 < maxgini;
                    }
                    else {  // if min_col_support > 0.643499970436
                      return 0.448642113472 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.725499987602
                  if ( min_col_support <= 0.841500043869 ) {
                    if ( min_col_support <= 0.790500044823 ) {
                      return 0.406348176892 < maxgini;
                    }
                    else {  // if min_col_support > 0.790500044823
                      return 0.355676302681 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.841500043869
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.165761304148 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.263217706643 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( mean_col_coverage <= 0.981377482414 ) {
                  if ( mean_col_coverage <= 0.822024941444 ) {
                    if ( mean_col_coverage <= 0.748407483101 ) {
                      return 0.437447062531 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.748407483101
                      return 0.465936565541 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.822024941444
                    if ( mean_col_coverage <= 0.880722761154 ) {
                      return 0.473657920094 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.880722761154
                      return 0.477485320678 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.981377482414
                  if ( min_col_coverage <= 0.97574287653 ) {
                    if ( min_col_coverage <= 0.962385773659 ) {
                      return 0.469360682117 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.962385773659
                      return 0.452782365112 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.97574287653
                    if ( mean_col_support <= 0.985323548317 ) {
                      return 0.423446429415 < maxgini;
                    }
                    else {  // if mean_col_support > 0.985323548317
                      return 0.341038757396 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.988468527794
        if ( mean_col_coverage <= 0.818223953247 ) {
          if ( median_col_support <= 0.992499947548 ) {
            if ( min_col_coverage <= 0.343378543854 ) {
              if ( median_col_support <= 0.966500043869 ) {
                if ( mean_col_support <= 0.990558862686 ) {
                  if ( median_col_coverage <= 0.00512477755547 ) {
                    if ( max_col_coverage <= 0.380858421326 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.380858421326
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.00512477755547
                    if ( median_col_support <= 0.960500001907 ) {
                      return 0.0872045007632 < maxgini;
                    }
                    else {  // if median_col_support > 0.960500001907
                      return 0.0444993661988 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.990558862686
                  if ( median_col_support <= 0.960500001907 ) {
                    if ( median_col_coverage <= 0.383066743612 ) {
                      return 0.287334593573 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.383066743612
                      return false;
                    }
                  }
                  else {  // if median_col_support > 0.960500001907
                    if ( median_col_coverage <= 0.18544973433 ) {
                      return 0.401234567901 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.18544973433
                      return 0.0925945550301 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.966500043869
                if ( mean_col_coverage <= 0.290810883045 ) {
                  if ( min_col_coverage <= 0.00366636412218 ) {
                    if ( mean_col_coverage <= 0.160121053457 ) {
                      return 0.136103929267 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.160121053457
                      return 0.322300295858 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00366636412218
                    if ( median_col_coverage <= 0.0170106571168 ) {
                      return 0.0848716672438 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0170106571168
                      return 0.0303627939423 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.290810883045
                  if ( median_col_support <= 0.972499966621 ) {
                    if ( min_col_support <= 0.963500022888 ) {
                      return 0.0260951107086 < maxgini;
                    }
                    else {  // if min_col_support > 0.963500022888
                      return 0.134749187838 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.972499966621
                    if ( max_col_coverage <= 0.690308511257 ) {
                      return 0.0182160431478 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.690308511257
                      return 0.0330788715172 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.343378543854
              if ( min_col_support <= 0.889500021935 ) {
                if ( median_col_support <= 0.989500045776 ) {
                  if ( median_col_coverage <= 0.567778944969 ) {
                    if ( min_col_support <= 0.844500005245 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.844500005245
                      return 0.0375197826152 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.567778944969
                    if ( min_col_support <= 0.867499947548 ) {
                      return 0.126885522555 < maxgini;
                    }
                    else {  // if min_col_support > 0.867499947548
                      return 0.0649526632685 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.989500045776
                  if ( median_col_support <= 0.991500020027 ) {
                    if ( mean_col_coverage <= 0.534307718277 ) {
                      return 0.0432480727321 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.534307718277
                      return 0.122476109709 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.991500020027
                    if ( min_col_support <= 0.846500039101 ) {
                      return 0.421092137362 < maxgini;
                    }
                    else {  // if min_col_support > 0.846500039101
                      return 0.175789099209 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.889500021935
                if ( median_col_support <= 0.971500039101 ) {
                  if ( min_col_coverage <= 0.478567063808 ) {
                    if ( min_col_support <= 0.958500027657 ) {
                      return 0.0266346899057 < maxgini;
                    }
                    else {  // if min_col_support > 0.958500027657
                      return 0.0534542628364 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.478567063808
                    if ( mean_col_support <= 0.989676475525 ) {
                      return 0.0192892238773 < maxgini;
                    }
                    else {  // if mean_col_support > 0.989676475525
                      return 0.0255160270732 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.971500039101
                  if ( mean_col_support <= 0.992617666721 ) {
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.0135471843247 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.0300712317971 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992617666721
                    if ( max_col_coverage <= 0.721756696701 ) {
                      return 0.0106162453341 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.721756696701
                      return 0.00694054572319 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.992499947548
            if ( median_col_support <= 0.99950003624 ) {
              if ( mean_col_support <= 0.992656886578 ) {
                if ( median_col_support <= 0.996500015259 ) {
                  if ( median_col_coverage <= 0.542330741882 ) {
                    if ( median_col_coverage <= 0.0180999189615 ) {
                      return 0.265196217372 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0180999189615
                      return 0.110048168398 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.542330741882
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.120575224895 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.205064565415 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.996500015259
                  if ( max_col_coverage <= 0.798786640167 ) {
                    if ( mean_col_support <= 0.991323530674 ) {
                      return 0.288047397402 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991323530674
                      return 0.211857138458 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.798786640167
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.23807848926 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.328180737218 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.992656886578
                if ( mean_col_support <= 0.99472796917 ) {
                  if ( min_col_support <= 0.920500040054 ) {
                    if ( median_col_coverage <= 0.596762299538 ) {
                      return 0.126059661899 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.596762299538
                      return 0.213478709561 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.920500040054
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.0247563297279 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return 0.123686969084 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.99472796917
                  if ( mean_col_support <= 0.995907902718 ) {
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.0183192798143 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.145525290006 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.995907902718
                    if ( min_col_support <= 0.950500011444 ) {
                      return 0.156167578215 < maxgini;
                    }
                    else {  // if min_col_support > 0.950500011444
                      return 0.0104081627713 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( mean_col_support <= 0.992060720921 ) {
                if ( median_col_coverage <= 0.57600158453 ) {
                  if ( median_col_coverage <= 0.488311350346 ) {
                    if ( min_col_support <= 0.852499961853 ) {
                      return 0.194225735269 < maxgini;
                    }
                    else {  // if min_col_support > 0.852499961853
                      return 0.0264196808631 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.488311350346
                    if ( median_col_coverage <= 0.516348898411 ) {
                      return 0.0835677168831 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.516348898411
                      return 0.115660958655 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.57600158453
                  if ( mean_col_support <= 0.990264713764 ) {
                    if ( median_col_coverage <= 0.65129327774 ) {
                      return 0.220201239676 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.65129327774
                      return 0.293897708283 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.990264713764
                    if ( min_col_support <= 0.870499968529 ) {
                      return 0.420607607529 < maxgini;
                    }
                    else {  // if min_col_support > 0.870499968529
                      return 0.0514712320669 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.992060720921
                if ( mean_col_support <= 0.993918836117 ) {
                  if ( median_col_coverage <= 0.606504499912 ) {
                    if ( mean_col_coverage <= 0.586513638496 ) {
                      return 0.0228538847256 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.586513638496
                      return 0.041002570325 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.606504499912
                    if ( min_col_support <= 0.895500004292 ) {
                      return 0.323857870093 < maxgini;
                    }
                    else {  // if min_col_support > 0.895500004292
                      return 0.0288342118687 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.993918836117
                  if ( mean_col_support <= 0.995766997337 ) {
                    if ( min_col_support <= 0.921499967575 ) {
                      return 0.0552157001253 < maxgini;
                    }
                    else {  // if min_col_support > 0.921499967575
                      return 0.013037558594 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.995766997337
                    if ( min_col_coverage <= 0.115978583694 ) {
                      return 0.0171223597171 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.115978583694
                      return 0.00790591211005 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.818223953247
          if ( min_col_support <= 0.888499975204 ) {
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_coverage <= 0.903531014919 ) {
                if ( min_col_support <= 0.860499978065 ) {
                  if ( median_col_coverage <= 0.902750730515 ) {
                    if ( median_col_coverage <= 0.902641177177 ) {
                      return 0.338199500373 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.902641177177
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.902750730515
                    if ( min_col_support <= 0.829499959946 ) {
                      return 0.21875 < maxgini;
                    }
                    else {  // if min_col_support > 0.829499959946
                      return 0.04875 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.860499978065
                  if ( mean_col_coverage <= 0.885266363621 ) {
                    if ( max_col_coverage <= 0.882533192635 ) {
                      return 0.0861552377184 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.882533192635
                      return 0.2052284333 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.885266363621
                    if ( mean_col_coverage <= 0.944681048393 ) {
                      return 0.235729165983 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.944681048393
                      return 0.0850769375117 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.903531014919
                if ( median_col_support <= 0.993499994278 ) {
                  if ( max_col_coverage <= 0.999228358269 ) {
                    if ( mean_col_support <= 0.990088224411 ) {
                      return 0.278824759701 < maxgini;
                    }
                    else {  // if mean_col_support > 0.990088224411
                      return 0.143359553277 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.999228358269
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.28270842505 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.176191151375 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( min_col_support <= 0.865499973297 ) {
                    if ( mean_col_coverage <= 0.98134291172 ) {
                      return 0.370110245667 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.98134291172
                      return 0.287587691327 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.865499973297
                    if ( min_col_coverage <= 0.902405023575 ) {
                      return 0.346867322949 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.902405023575
                      return 0.28768445606 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_coverage <= 0.976225793362 ) {
                if ( min_col_support <= 0.847499966621 ) {
                  if ( median_col_coverage <= 0.963691294193 ) {
                    if ( max_col_coverage <= 0.999181628227 ) {
                      return 0.467986365926 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.999181628227
                      return 0.477534717092 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.963691294193
                    if ( mean_col_coverage <= 0.99843108654 ) {
                      return 0.447352153056 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.99843108654
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.847499966621
                  if ( median_col_coverage <= 0.829207003117 ) {
                    if ( min_col_support <= 0.87450003624 ) {
                      return 0.389168057902 < maxgini;
                    }
                    else {  // if min_col_support > 0.87450003624
                      return 0.256121940528 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.829207003117
                    if ( min_col_coverage <= 0.883805274963 ) {
                      return 0.375115304783 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.883805274963
                      return 0.391513872997 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.976225793362
                if ( median_col_coverage <= 0.990719199181 ) {
                  if ( max_col_coverage <= 0.99035358429 ) {
                    if ( max_col_coverage <= 0.990171432495 ) {
                      return 0.2822 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.990171432495
                      return false;
                    }
                  }
                  else {  // if max_col_coverage > 0.99035358429
                    if ( median_col_coverage <= 0.987040996552 ) {
                      return 0.257369492552 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.987040996552
                      return 0.134012345679 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.990719199181
                  if ( median_col_coverage <= 0.990888357162 ) {
                    if ( max_col_coverage <= 0.995412826538 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.995412826538
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.990888357162
                    if ( min_col_coverage <= 0.989060878754 ) {
                      return 0.446090443306 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.989060878754
                      return 0.306831811295 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.888499975204
            if ( min_col_support <= 0.916499972343 ) {
              if ( min_col_coverage <= 0.853029310703 ) {
                if ( min_col_support <= 0.903499960899 ) {
                  if ( median_col_support <= 0.993499994278 ) {
                    if ( mean_col_coverage <= 0.82593691349 ) {
                      return 0.14201183432 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.82593691349
                      return 0.0562298798121 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.993499994278
                    if ( min_col_support <= 0.894500017166 ) {
                      return 0.230998668031 < maxgini;
                    }
                    else {  // if min_col_support > 0.894500017166
                      return 0.187658351725 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.903499960899
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( mean_col_support <= 0.991088271141 ) {
                      return 0.02736407184 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991088271141
                      return 0.0537008616025 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( mean_col_support <= 0.994147062302 ) {
                      return 0.0926359935466 < maxgini;
                    }
                    else {  // if mean_col_support > 0.994147062302
                      return 0.215250678516 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.853029310703
                if ( median_col_coverage <= 0.997863173485 ) {
                  if ( min_col_coverage <= 0.951291561127 ) {
                    if ( mean_col_coverage <= 0.918941020966 ) {
                      return 0.152547409748 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.918941020966
                      return 0.194453245116 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.951291561127
                    if ( min_col_coverage <= 0.997516036034 ) {
                      return 0.239949343974 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.997516036034
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.997863173485
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( mean_col_support <= 0.98920583725 ) {
                      return 0.122281421958 < maxgini;
                    }
                    else {  // if mean_col_support > 0.98920583725
                      return 0.0275175770687 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( mean_col_support <= 0.991323530674 ) {
                      return 0.0773154296546 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991323530674
                      return 0.148808525061 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.916499972343
              if ( median_col_coverage <= 0.95126914978 ) {
                if ( mean_col_support <= 0.996147036552 ) {
                  if ( median_col_support <= 0.993499994278 ) {
                    if ( mean_col_support <= 0.993088245392 ) {
                      return 0.0152013229753 < maxgini;
                    }
                    else {  // if mean_col_support > 0.993088245392
                      return 0.00771016620331 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.993499994278
                    if ( min_col_support <= 0.933500051498 ) {
                      return 0.0692667813469 < maxgini;
                    }
                    else {  // if min_col_support > 0.933500051498
                      return 0.0153223517181 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.996147036552
                  if ( min_col_support <= 0.953500032425 ) {
                    if ( mean_col_support <= 0.996558785439 ) {
                      return 0.0302387250532 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996558785439
                      return 0.0187371696508 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.953500032425
                    if ( min_col_support <= 0.969500005245 ) {
                      return 0.00966098605628 < maxgini;
                    }
                    else {  // if min_col_support > 0.969500005245
                      return 0.00568559849025 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.95126914978
                if ( min_col_coverage <= 0.947535932064 ) {
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( max_col_coverage <= 0.956270098686 ) {
                      return 0.270551508845 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.956270098686
                      return 0.0229518632557 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( max_col_coverage <= 0.977185964584 ) {
                      return 0.00566797041195 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.977185964584
                      return 0.0129343607998 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.947535932064
                  if ( mean_col_support <= 0.996264696121 ) {
                    if ( mean_col_support <= 0.993499994278 ) {
                      return 0.0681260239121 < maxgini;
                    }
                    else {  // if mean_col_support > 0.993499994278
                      return 0.038458966379 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.996264696121
                    if ( mean_col_support <= 0.997147083282 ) {
                      return 0.0199080704677 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997147083282
                      return 0.00951661993077 < maxgini;
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
      if ( max_col_coverage <= 0.829282343388 ) {
        if ( max_col_coverage <= 0.676523327827 ) {
          if ( min_col_support <= 0.773499965668 ) {
            if ( min_col_coverage <= 0.333748519421 ) {
              if ( min_col_support <= 0.631500005722 ) {
                if ( mean_col_coverage <= 0.318655192852 ) {
                  if ( mean_col_support <= 0.802656888962 ) {
                    if ( median_col_support <= 0.554499983788 ) {
                      return 0.234733413794 < maxgini;
                    }
                    else {  // if median_col_support > 0.554499983788
                      return 0.167261539007 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.802656888962
                    if ( min_col_coverage <= 0.184389472008 ) {
                      return 0.116138320737 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.184389472008
                      return 0.166415639012 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.318655192852
                  if ( min_col_support <= 0.547500014305 ) {
                    if ( median_col_support <= 0.577499985695 ) {
                      return 0.419793840395 < maxgini;
                    }
                    else {  // if median_col_support > 0.577499985695
                      return 0.215020176137 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.547500014305
                    if ( max_col_support <= 0.90649998188 ) {
                      return false;
                    }
                    else {  // if max_col_support > 0.90649998188
                      return 0.214994870943 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.631500005722
                if ( mean_col_support <= 0.978232979774 ) {
                  if ( mean_col_support <= 0.901594161987 ) {
                    if ( median_col_support <= 0.730499982834 ) {
                      return 0.156194788032 < maxgini;
                    }
                    else {  // if median_col_support > 0.730499982834
                      return 0.113935906786 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.901594161987
                    if ( max_col_coverage <= 0.382425874472 ) {
                      return 0.0697236120415 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.382425874472
                      return 0.0963682928395 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.978232979774
                  if ( min_col_coverage <= 0.156184077263 ) {
                    if ( median_col_coverage <= 0.156424790621 ) {
                      return 0.0624540941547 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.156424790621
                      return 0.202355380656 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.156184077263
                    if ( min_col_coverage <= 0.265261769295 ) {
                      return 0.285656214762 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.265261769295
                      return 0.380027986554 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.333748519421
              if ( min_col_coverage <= 0.425032287836 ) {
                if ( min_col_coverage <= 0.394784212112 ) {
                  if ( median_col_coverage <= 0.352990865707 ) {
                    if ( min_col_support <= 0.589499950409 ) {
                      return 0.282834094711 < maxgini;
                    }
                    else {  // if min_col_support > 0.589499950409
                      return 0.160505694854 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.352990865707
                    if ( median_col_coverage <= 0.440783083439 ) {
                      return 0.262179064731 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.440783083439
                      return 0.232706723683 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.394784212112
                  if ( min_col_support <= 0.662500023842 ) {
                    if ( min_col_support <= 0.581499993801 ) {
                      return 0.378926836219 < maxgini;
                    }
                    else {  // if min_col_support > 0.581499993801
                      return 0.326958833242 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.662500023842
                    if ( mean_col_support <= 0.978205859661 ) {
                      return 0.173459797486 < maxgini;
                    }
                    else {  // if mean_col_support > 0.978205859661
                      return 0.417958173033 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.425032287836
                if ( min_col_support <= 0.681499958038 ) {
                  if ( median_col_coverage <= 0.487907648087 ) {
                    if ( max_col_coverage <= 0.571823835373 ) {
                      return 0.33738012321 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.571823835373
                      return 0.380045873231 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.487907648087
                    if ( mean_col_support <= 0.969088196754 ) {
                      return 0.359581478488 < maxgini;
                    }
                    else {  // if mean_col_support > 0.969088196754
                      return 0.471818969482 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.681499958038
                  if ( min_col_coverage <= 0.48844408989 ) {
                    if ( min_col_support <= 0.736500024796 ) {
                      return 0.282017292627 < maxgini;
                    }
                    else {  // if min_col_support > 0.736500024796
                      return 0.223390197015 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.48844408989
                    if ( mean_col_support <= 0.979382395744 ) {
                      return 0.220134858836 < maxgini;
                    }
                    else {  // if mean_col_support > 0.979382395744
                      return 0.456062764439 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.773499965668
            if ( min_col_support <= 0.872500002384 ) {
              if ( mean_col_support <= 0.986644923687 ) {
                if ( median_col_support <= 0.861500024796 ) {
                  if ( mean_col_coverage <= 0.458594560623 ) {
                    if ( mean_col_support <= 0.923558831215 ) {
                      return 0.114373524785 < maxgini;
                    }
                    else {  // if mean_col_support > 0.923558831215
                      return 0.0868598400405 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.458594560623
                    if ( min_col_support <= 0.81350004673 ) {
                      return 0.148701243163 < maxgini;
                    }
                    else {  // if min_col_support > 0.81350004673
                      return 0.11376820133 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.861500024796
                  if ( max_col_coverage <= 0.545575499535 ) {
                    if ( median_col_coverage <= 0.00791035313159 ) {
                      return 0.173681749009 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00791035313159
                      return 0.0524830155362 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.545575499535
                    if ( min_col_coverage <= 0.394924998283 ) {
                      return 0.0622395781083 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.394924998283
                      return 0.0904687370331 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.986644923687
                if ( median_col_support <= 0.991500020027 ) {
                  if ( mean_col_coverage <= 0.45251262188 ) {
                    if ( min_col_support <= 0.817999958992 ) {
                      return 0.332409972299 < maxgini;
                    }
                    else {  // if min_col_support > 0.817999958992
                      return 0.0362817756514 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.45251262188
                    if ( mean_col_support <= 0.989264726639 ) {
                      return 0.083623218033 < maxgini;
                    }
                    else {  // if mean_col_support > 0.989264726639
                      return 0.131611681339 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.991500020027
                  if ( median_col_coverage <= 0.366179406643 ) {
                    if ( median_col_coverage <= 0.306278169155 ) {
                      return 0.0466128896232 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.306278169155
                      return 0.151191040707 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.366179406643
                    if ( mean_col_coverage <= 0.511199951172 ) {
                      return 0.230128232489 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.511199951172
                      return 0.329236692315 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.872500002384
              if ( mean_col_coverage <= 0.43729621172 ) {
                if ( min_col_support <= 0.916499972343 ) {
                  if ( max_col_support <= 0.977500021458 ) {
                    return false;
                  }
                  else {  // if max_col_support > 0.977500021458
                    if ( median_col_coverage <= 0.0156494900584 ) {
                      return 0.102922887507 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0156494900584
                      return 0.0367427573696 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.916499972343
                  if ( median_col_coverage <= 0.00838578119874 ) {
                    if ( median_col_coverage <= 0.00835076719522 ) {
                      return 0.087685255864 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00835076719522
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.00838578119874
                    if ( median_col_coverage <= 0.322004437447 ) {
                      return 0.0242422311496 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.322004437447
                      return 0.0188730612385 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.43729621172
                if ( mean_col_support <= 0.987382292747 ) {
                  if ( mean_col_support <= 0.974911808968 ) {
                    if ( min_col_support <= 0.919499993324 ) {
                      return 0.0636806087434 < maxgini;
                    }
                    else {  // if min_col_support > 0.919499993324
                      return 0.121089976148 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.974911808968
                    if ( min_col_coverage <= 0.483510226011 ) {
                      return 0.036514943393 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.483510226011
                      return 0.0316253624678 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.987382292747
                  if ( median_col_coverage <= 0.470409572124 ) {
                    if ( min_col_support <= 0.900499999523 ) {
                      return 0.0530731787227 < maxgini;
                    }
                    else {  // if min_col_support > 0.900499999523
                      return 0.01576267702 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.470409572124
                    if ( mean_col_coverage <= 0.524741292 ) {
                      return 0.0125425223044 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.524741292
                      return 0.0155609637926 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.676523327827
          if ( median_col_support <= 0.713500022888 ) {
            if ( mean_col_coverage <= 0.560588955879 ) {
              if ( median_col_support <= 0.538499951363 ) {
                if ( mean_col_support <= 0.81500005722 ) {
                  if ( min_col_support <= 0.480000019073 ) {
                    if ( max_col_coverage <= 0.720779240131 ) {
                      return 0.255 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.720779240131
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.480000019073
                    if ( max_col_coverage <= 0.695707082748 ) {
                      return 0.499178232286 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.695707082748
                      return false;
                    }
                  }
                }
                else {  // if mean_col_support > 0.81500005722
                  if ( median_col_coverage <= 0.392307698727 ) {
                    if ( mean_col_coverage <= 0.329723119736 ) {
                      return 0.408163265306 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.329723119736
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.392307698727
                    if ( median_col_support <= 0.524500012398 ) {
                      return 0.313271604938 < maxgini;
                    }
                    else {  // if median_col_support > 0.524500012398
                      return 0.497777777778 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.538499951363
                if ( mean_col_coverage <= 0.512852430344 ) {
                  if ( max_col_coverage <= 0.819347321987 ) {
                    if ( min_col_coverage <= 0.223611116409 ) {
                      return 0.241357376384 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.223611116409
                      return 0.145130055957 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.819347321987
                    return 0.0 < maxgini;
                  }
                }
                else {  // if mean_col_coverage > 0.512852430344
                  if ( mean_col_support <= 0.792794108391 ) {
                    if ( max_col_coverage <= 0.742396354675 ) {
                      return 0.495448813195 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.742396354675
                      return 0.124444444444 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.792794108391
                    if ( min_col_coverage <= 0.436130523682 ) {
                      return 0.294742973211 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.436130523682
                      return 0.409952540209 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.560588955879
              if ( max_col_support <= 0.99950003624 ) {
                if ( mean_col_coverage <= 0.754737019539 ) {
                  if ( min_col_support <= 0.53100001812 ) {
                    if ( mean_col_support <= 0.776294052601 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.776294052601
                      return 0.444444444444 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.53100001812
                    return 0.0 < maxgini;
                  }
                }
                else {  // if mean_col_coverage > 0.754737019539
                  return false;
                }
              }
              else {  // if max_col_support > 0.99950003624
                if ( min_col_coverage <= 0.518304228783 ) {
                  if ( median_col_support <= 0.597499966621 ) {
                    if ( min_col_coverage <= 0.515304803848 ) {
                      return 0.49779720695 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.515304803848
                      return 0.413194444444 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.597499966621
                    if ( min_col_coverage <= 0.445576906204 ) {
                      return 0.349635796046 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.445576906204
                      return 0.397436529491 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.518304228783
                  if ( min_col_coverage <= 0.558460593224 ) {
                    if ( mean_col_support <= 0.830205798149 ) {
                      return 0.499723963339 < maxgini;
                    }
                    else {  // if mean_col_support > 0.830205798149
                      return 0.466348408163 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.558460593224
                    if ( mean_col_coverage <= 0.662913918495 ) {
                      return 0.443599882937 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.662913918495
                      return 0.473132066132 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.713500022888
            if ( min_col_support <= 0.809499979019 ) {
              if ( median_col_coverage <= 0.558948934078 ) {
                if ( median_col_support <= 0.983500003815 ) {
                  if ( max_col_coverage <= 0.682802200317 ) {
                    if ( median_col_support <= 0.957499980927 ) {
                      return 0.213039485767 < maxgini;
                    }
                    else {  // if median_col_support > 0.957499980927
                      return 0.341026503012 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.682802200317
                    if ( min_col_coverage <= 0.241060018539 ) {
                      return 0.231090931643 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.241060018539
                      return 0.144967662103 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.983500003815
                  if ( mean_col_support <= 0.961088299751 ) {
                    if ( mean_col_coverage <= 0.3758636415 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.3758636415
                      return 0.468305587469 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.961088299751
                    if ( mean_col_coverage <= 0.543933033943 ) {
                      return 0.378413584691 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.543933033943
                      return 0.439038573465 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.558948934078
                if ( max_col_coverage <= 0.771445155144 ) {
                  if ( min_col_support <= 0.726500034332 ) {
                    if ( mean_col_coverage <= 0.681698203087 ) {
                      return 0.428845148567 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.681698203087
                      return 0.449588444993 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.726500034332
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.159278678761 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.435696615579 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.771445155144
                  if ( min_col_support <= 0.734500050545 ) {
                    if ( min_col_coverage <= 0.607727885246 ) {
                      return 0.430415399732 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.607727885246
                      return 0.457828451679 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.734500050545
                    if ( median_col_coverage <= 0.649101734161 ) {
                      return 0.323521673774 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.649101734161
                      return 0.384879938537 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.809499979019
              if ( min_col_support <= 0.881500005722 ) {
                if ( min_col_support <= 0.846500039101 ) {
                  if ( mean_col_coverage <= 0.647203564644 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.0733541029301 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.314081904686 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.647203564644
                    if ( mean_col_coverage <= 0.711292803288 ) {
                      return 0.2172064727 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.711292803288
                      return 0.268925700562 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.846500039101
                  if ( mean_col_support <= 0.989558815956 ) {
                    if ( min_col_coverage <= 0.529624760151 ) {
                      return 0.0591236062099 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.529624760151
                      return 0.0953378682035 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.989558815956
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.211116170388 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.311823688175 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.881500005722
                if ( mean_col_support <= 0.994735240936 ) {
                  if ( mean_col_support <= 0.980500042439 ) {
                    if ( median_col_support <= 0.927500009537 ) {
                      return 0.078988910501 < maxgini;
                    }
                    else {  // if median_col_support > 0.927500009537
                      return 0.0415762927109 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.980500042439
                    if ( mean_col_coverage <= 0.680501520634 ) {
                      return 0.0251393596687 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.680501520634
                      return 0.0301713081199 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.994735240936
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( max_col_coverage <= 0.709112763405 ) {
                      return 0.0081667392127 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.709112763405
                      return 0.00595768062974 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( max_col_coverage <= 0.681326985359 ) {
                      return 0.0123383871269 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.681326985359
                      return 0.00989972427715 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if max_col_coverage > 0.829282343388
        if ( mean_col_coverage <= 0.862769365311 ) {
          if ( min_col_support <= 0.831499993801 ) {
            if ( min_col_coverage <= 0.618061065674 ) {
              if ( mean_col_support <= 0.911332666874 ) {
                if ( max_col_support <= 0.997500002384 ) {
                  if ( mean_col_support <= 0.583941221237 ) {
                    if ( min_col_support <= 0.377499997616 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_support > 0.377499997616
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.583941221237
                    if ( max_col_coverage <= 0.923904061317 ) {
                      return 0.112274448652 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.923904061317
                      return 0.024594104608 < maxgini;
                    }
                  }
                }
                else {  // if max_col_support > 0.997500002384
                  if ( min_col_support <= 0.499500006437 ) {
                    if ( mean_col_support <= 0.78405880928 ) {
                      return 0.0751537093099 < maxgini;
                    }
                    else {  // if mean_col_support > 0.78405880928
                      return 0.17689100268 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.499500006437
                    if ( median_col_support <= 0.627499997616 ) {
                      return 0.420985442439 < maxgini;
                    }
                    else {  // if median_col_support > 0.627499997616
                      return 0.282916832605 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.911332666874
                if ( median_col_support <= 0.986500024796 ) {
                  if ( min_col_support <= 0.647500038147 ) {
                    if ( median_col_support <= 0.947499990463 ) {
                      return 0.324893946563 < maxgini;
                    }
                    else {  // if median_col_support > 0.947499990463
                      return 0.430870365592 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.647500038147
                    if ( max_col_coverage <= 0.981872916222 ) {
                      return 0.135002242097 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.981872916222
                      return 0.26268227557 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.986500024796
                  if ( min_col_support <= 0.71749997139 ) {
                    if ( min_col_support <= 0.607499957085 ) {
                      return 0.475709253806 < maxgini;
                    }
                    else {  // if min_col_support > 0.607499957085
                      return 0.456639866907 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.71749997139
                    if ( min_col_support <= 0.788499951363 ) {
                      return 0.41924732534 < maxgini;
                    }
                    else {  // if min_col_support > 0.788499951363
                      return 0.340068883788 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.618061065674
              if ( median_col_coverage <= 0.744157314301 ) {
                if ( median_col_support <= 0.989500045776 ) {
                  if ( mean_col_support <= 0.954852938652 ) {
                    if ( min_col_support <= 0.675500035286 ) {
                      return 0.415531264827 < maxgini;
                    }
                    else {  // if min_col_support > 0.675500035286
                      return 0.213464973189 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.954852938652
                    if ( median_col_support <= 0.972499966621 ) {
                      return 0.224180702968 < maxgini;
                    }
                    else {  // if median_col_support > 0.972499966621
                      return 0.336848259104 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.989500045776
                  if ( min_col_coverage <= 0.716968536377 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.44108524518 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.474232096964 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.716968536377
                    if ( max_col_coverage <= 0.838658630848 ) {
                      return 0.45154778438 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.838658630848
                      return 0.477741923037 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.744157314301
                if ( mean_col_support <= 0.970735311508 ) {
                  if ( median_col_support <= 0.970499992371 ) {
                    if ( median_col_support <= 0.716500043869 ) {
                      return 0.463582900292 < maxgini;
                    }
                    else {  // if median_col_support > 0.716500043869
                      return 0.318592970177 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.970499992371
                    if ( mean_col_coverage <= 0.819905877113 ) {
                      return 0.467710932379 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.819905877113
                      return 0.46287636987 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.970735311508
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.306614567143 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.432042204516 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( min_col_coverage <= 0.713895916939 ) {
                      return 0.475934560438 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.713895916939
                      return 0.480900712493 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.831499993801
            if ( min_col_support <= 0.889500021935 ) {
              if ( mean_col_coverage <= 0.743613421917 ) {
                if ( median_col_support <= 0.991500020027 ) {
                  if ( min_col_coverage <= 0.111960910261 ) {
                    if ( min_col_support <= 0.849500000477 ) {
                      return 0.105331661402 < maxgini;
                    }
                    else {  // if min_col_support > 0.849500000477
                      return 0.244265736277 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.111960910261
                    if ( median_col_support <= 0.892500042915 ) {
                      return 0.14201183432 < maxgini;
                    }
                    else {  // if median_col_support > 0.892500042915
                      return 0.049739653806 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.991500020027
                  if ( mean_col_support <= 0.990088224411 ) {
                    if ( median_col_coverage <= 0.107426099479 ) {
                      return 0.38829209789 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.107426099479
                      return 0.16701231988 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.990088224411
                    if ( min_col_coverage <= 0.464626908302 ) {
                      return 0.108283511374 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.464626908302
                      return 0.303015788649 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.743613421917
                if ( mean_col_support <= 0.989264667034 ) {
                  if ( min_col_coverage <= 0.666869163513 ) {
                    if ( median_col_coverage <= 0.471053600311 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.471053600311
                      return 0.104898439406 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.666869163513
                    if ( min_col_support <= 0.855499982834 ) {
                      return 0.213083684336 < maxgini;
                    }
                    else {  // if min_col_support > 0.855499982834
                      return 0.124874784671 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.989264667034
                  if ( mean_col_support <= 0.990088224411 ) {
                    if ( mean_col_coverage <= 0.752474069595 ) {
                      return 0.159050859516 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.752474069595
                      return 0.283610271987 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.990088224411
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.215797264329 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.380379930409 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.889500021935
              if ( mean_col_support <= 0.995029389858 ) {
                if ( mean_col_coverage <= 0.58719342947 ) {
                  if ( median_col_coverage <= 0.0793766826391 ) {
                    if ( mean_col_support <= 0.993676483631 ) {
                      return 0.32603730406 < maxgini;
                    }
                    else {  // if mean_col_support > 0.993676483631
                      return 0.119808 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0793766826391
                    if ( min_col_support <= 0.900499999523 ) {
                      return 0.122994074351 < maxgini;
                    }
                    else {  // if min_col_support > 0.900499999523
                      return 0.0461534222854 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.58719342947
                  if ( min_col_coverage <= 0.666851639748 ) {
                    if ( mean_col_support <= 0.98006516695 ) {
                      return 0.0530226249664 < maxgini;
                    }
                    else {  // if mean_col_support > 0.98006516695
                      return 0.0240274937135 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.666851639748
                    if ( median_col_coverage <= 0.810703992844 ) {
                      return 0.0367298147699 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.810703992844
                      return 0.030194483565 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.995029389858
                if ( mean_col_support <= 0.996323525906 ) {
                  if ( median_col_coverage <= 0.0064341085963 ) {
                    if ( median_col_coverage <= 0.00562602747232 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00562602747232
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.0064341085963
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.00562928989357 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.017688916062 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.996323525906
                  if ( max_col_coverage <= 0.982122898102 ) {
                    if ( mean_col_support <= 0.99749994278 ) {
                      return 0.00939768914512 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99749994278
                      return 0.00591085573678 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.982122898102
                    if ( mean_col_coverage <= 0.373435586691 ) {
                      return 0.396694214876 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.373435586691
                      return 0.015788058429 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.862769365311
          if ( mean_col_coverage <= 0.951565623283 ) {
            if ( min_col_coverage <= 0.889012217522 ) {
              if ( mean_col_support <= 0.98914706707 ) {
                if ( min_col_coverage <= 0.795130372047 ) {
                  if ( median_col_support <= 0.991500020027 ) {
                    if ( mean_col_support <= 0.977029442787 ) {
                      return 0.330247461004 < maxgini;
                    }
                    else {  // if mean_col_support > 0.977029442787
                      return 0.0978649771499 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.991500020027
                    if ( median_col_coverage <= 0.829169392586 ) {
                      return 0.452687294728 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.829169392586
                      return 0.467349672266 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.795130372047
                  if ( min_col_support <= 0.826499998569 ) {
                    if ( min_col_support <= 0.776499986649 ) {
                      return 0.472937168404 < maxgini;
                    }
                    else {  // if min_col_support > 0.776499986649
                      return 0.422959982073 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.826499998569
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0933316186557 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.283138138728 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.98914706707
                if ( mean_col_support <= 0.992852926254 ) {
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0226437733336 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.11973020527 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( min_col_support <= 0.878499984741 ) {
                      return 0.42859294099 < maxgini;
                    }
                    else {  // if min_col_support > 0.878499984741
                      return 0.0934320529403 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.992852926254
                  if ( median_col_coverage <= 0.816312432289 ) {
                    if ( min_col_support <= 0.909500002861 ) {
                      return 0.253531553318 < maxgini;
                    }
                    else {  // if min_col_support > 0.909500002861
                      return 0.00940827095663 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.816312432289
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.00846870331928 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.0197374943815 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.889012217522
              if ( max_col_coverage <= 0.998592495918 ) {
                if ( min_col_coverage <= 0.919016242027 ) {
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( mean_col_support <= 0.985970616341 ) {
                      return 0.417248405442 < maxgini;
                    }
                    else {  // if mean_col_support > 0.985970616341
                      return 0.0867313835282 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( max_col_coverage <= 0.98933327198 ) {
                      return 0.244802811252 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.98933327198
                      return 0.398812087995 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.919016242027
                  if ( mean_col_support <= 0.990970551968 ) {
                    if ( mean_col_support <= 0.985558867455 ) {
                      return 0.442737127368 < maxgini;
                    }
                    else {  // if mean_col_support > 0.985558867455
                      return 0.344801056597 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.990970551968
                    if ( median_col_coverage <= 0.929672241211 ) {
                      return 0.0336333565181 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.929672241211
                      return 0.0503171436546 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.998592495918
                if ( max_col_coverage <= 0.999377369881 ) {
                  return 0.0 < maxgini;
                }
                else {  // if max_col_coverage > 0.999377369881
                  if ( median_col_support <= 0.993499994278 ) {
                    if ( mean_col_support <= 0.984676420689 ) {
                      return 0.373753878255 < maxgini;
                    }
                    else {  // if mean_col_support > 0.984676420689
                      return 0.0379117188655 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.993499994278
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.285400098517 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.217874922999 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.951565623283
            if ( mean_col_support <= 0.990323543549 ) {
              if ( min_col_support <= 0.837499976158 ) {
                if ( min_col_support <= 0.78149998188 ) {
                  if ( min_col_coverage <= 0.975300610065 ) {
                    if ( max_col_support <= 0.99950003624 ) {
                      return 0.0296113904022 < maxgini;
                    }
                    else {  // if max_col_support > 0.99950003624
                      return 0.468796100177 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.975300610065
                    if ( min_col_coverage <= 0.997527837753 ) {
                      return 0.413167374586 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.997527837753
                      return 0.337560154971 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.78149998188
                  if ( mean_col_support <= 0.985676407814 ) {
                    if ( max_col_coverage <= 0.99843621254 ) {
                      return 0.358149565708 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.99843621254
                      return 0.329171451825 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.985676407814
                    if ( min_col_coverage <= 0.972242474556 ) {
                      return 0.444175304774 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.972242474556
                      return 0.368841297836 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.837499976158
                if ( min_col_coverage <= 0.914335429668 ) {
                  if ( min_col_support <= 0.859500050545 ) {
                    if ( min_col_support <= 0.846500039101 ) {
                      return 0.346675858523 < maxgini;
                    }
                    else {  // if min_col_support > 0.846500039101
                      return 0.271843512114 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.859500050545
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.05697199381 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.168459546182 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.914335429668
                  if ( min_col_coverage <= 0.997903585434 ) {
                    if ( median_col_coverage <= 0.947603225708 ) {
                      return 0.188431555349 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.947603225708
                      return 0.229179503832 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.997903585434
                    if ( min_col_coverage <= 0.999555945396 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.999555945396
                      return 0.127173520711 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.990323543549
              if ( min_col_support <= 0.897500038147 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( min_col_support <= 0.882500052452 ) {
                      return 0.291486312913 < maxgini;
                    }
                    else {  // if min_col_support > 0.882500052452
                      return 0.11706334124 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( mean_col_support <= 0.993088126183 ) {
                      return 0.309753262481 < maxgini;
                    }
                    else {  // if mean_col_support > 0.993088126183
                      return 0.376567382619 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.99950003624
                  if ( min_col_support <= 0.866500020027 ) {
                    if ( median_col_coverage <= 0.960860908031 ) {
                      return 0.466286132902 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.960860908031
                      return 0.394616111101 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.866500020027
                    if ( median_col_coverage <= 0.980487763882 ) {
                      return 0.346501131266 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.980487763882
                      return 0.231708199503 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.897500038147
                if ( min_col_support <= 0.929499983788 ) {
                  if ( median_col_coverage <= 0.997980952263 ) {
                    if ( min_col_support <= 0.913499951363 ) {
                      return 0.19956006869 < maxgini;
                    }
                    else {  // if min_col_support > 0.913499951363
                      return 0.112736290007 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.997980952263
                    if ( min_col_support <= 0.911499977112 ) {
                      return 0.114918375236 < maxgini;
                    }
                    else {  // if min_col_support > 0.911499977112
                      return 0.0653619568239 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.929499983788
                  if ( mean_col_support <= 0.996911764145 ) {
                    if ( min_col_coverage <= 0.953503847122 ) {
                      return 0.0155845668041 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.953503847122
                      return 0.0290274492705 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.996911764145
                    if ( min_col_coverage <= 0.989455163479 ) {
                      return 0.00816630524851 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.989455163479
                      return 0.0146938997796 < maxgini;
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
      if ( min_col_support <= 0.807500004768 ) {
        if ( median_col_support <= 0.979499995708 ) {
          if ( max_col_coverage <= 0.714480876923 ) {
            if ( median_col_coverage <= 0.342882812023 ) {
              if ( min_col_coverage <= 0.243951946497 ) {
                if ( median_col_coverage <= 0.0487567149103 ) {
                  if ( min_col_coverage <= 0.00740056112409 ) {
                    if ( min_col_support <= 0.685500025749 ) {
                      return 0.100345140654 < maxgini;
                    }
                    else {  // if min_col_support > 0.685500025749
                      return 0.148108982231 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00740056112409
                    if ( max_col_coverage <= 0.364313393831 ) {
                      return 0.129293942765 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.364313393831
                      return 0.209268604577 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.0487567149103
                  if ( mean_col_support <= 0.828878641129 ) {
                    if ( median_col_coverage <= 0.184271156788 ) {
                      return 0.142151984522 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.184271156788
                      return 0.244839607911 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.828878641129
                    if ( mean_col_support <= 0.923060774803 ) {
                      return 0.106425967485 < maxgini;
                    }
                    else {  // if mean_col_support > 0.923060774803
                      return 0.0674030524865 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.243951946497
                if ( mean_col_support <= 0.845323503017 ) {
                  if ( min_col_support <= 0.488499999046 ) {
                    if ( min_col_coverage <= 0.28128695488 ) {
                      return 0.203532600799 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.28128695488
                      return 0.258297288433 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.488499999046
                    if ( median_col_support <= 0.577499985695 ) {
                      return 0.46704364848 < maxgini;
                    }
                    else {  // if median_col_support > 0.577499985695
                      return 0.283930738399 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.845323503017
                  if ( mean_col_coverage <= 0.317335158587 ) {
                    if ( min_col_support <= 0.703500032425 ) {
                      return 0.114675838429 < maxgini;
                    }
                    else {  // if min_col_support > 0.703500032425
                      return 0.0756638482745 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.317335158587
                    if ( min_col_support <= 0.641499996185 ) {
                      return 0.150529117001 < maxgini;
                    }
                    else {  // if min_col_support > 0.641499996185
                      return 0.0894368844617 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.342882812023
              if ( mean_col_coverage <= 0.508748292923 ) {
                if ( median_col_support <= 0.65750002861 ) {
                  if ( min_col_coverage <= 0.324569642544 ) {
                    if ( min_col_coverage <= 0.274754911661 ) {
                      return 0.330185227205 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.274754911661
                      return 0.380678179116 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.324569642544
                    if ( min_col_coverage <= 0.363952517509 ) {
                      return 0.421128733652 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.363952517509
                      return 0.445976377902 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.65750002861
                  if ( mean_col_support <= 0.883323550224 ) {
                    if ( min_col_coverage <= 0.33412116766 ) {
                      return 0.178608343264 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.33412116766
                      return 0.222678859394 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.883323550224
                    if ( median_col_coverage <= 0.347768813372 ) {
                      return 0.167914146353 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.347768813372
                      return 0.110632796585 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.508748292923
                if ( mean_col_coverage <= 0.56600022316 ) {
                  if ( min_col_support <= 0.611500024796 ) {
                    if ( median_col_coverage <= 0.524357259274 ) {
                      return 0.306798702508 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.524357259274
                      return 0.246173797347 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.611500024796
                    if ( min_col_coverage <= 0.405458599329 ) {
                      return 0.102962565212 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.405458599329
                      return 0.141795522196 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.56600022316
                  if ( median_col_support <= 0.682500004768 ) {
                    if ( max_col_coverage <= 0.611678004265 ) {
                      return 0.405043767313 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.611678004265
                      return 0.474502508062 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.682500004768
                    if ( min_col_coverage <= 0.463727653027 ) {
                      return 0.122815811501 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.463727653027
                      return 0.171056181886 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.714480876923
            if ( min_col_support <= 0.685500025749 ) {
              if ( mean_col_coverage <= 0.733799695969 ) {
                if ( mean_col_support <= 0.86414706707 ) {
                  if ( min_col_coverage <= 0.379655182362 ) {
                    if ( max_col_support <= 0.995499968529 ) {
                      return 0.0294052127423 < maxgini;
                    }
                    else {  // if max_col_support > 0.995499968529
                      return 0.177767510436 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.379655182362
                    if ( min_col_coverage <= 0.441600620747 ) {
                      return 0.377295033164 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.441600620747
                      return 0.469302419809 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.86414706707
                  if ( median_col_support <= 0.943500041962 ) {
                    if ( max_col_coverage <= 0.99670445919 ) {
                      return 0.241858116033 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.99670445919
                      return 0.355476935592 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.943500041962
                    if ( median_col_support <= 0.964499950409 ) {
                      return 0.350369552118 < maxgini;
                    }
                    else {  // if median_col_support > 0.964499950409
                      return 0.403326546621 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.733799695969
                if ( median_col_support <= 0.931499958038 ) {
                  if ( max_col_support <= 0.99950003624 ) {
                    if ( mean_col_support <= 0.946705877781 ) {
                      return 0.0202156209485 < maxgini;
                    }
                    else {  // if mean_col_support > 0.946705877781
                      return false;
                    }
                  }
                  else {  // if max_col_support > 0.99950003624
                    if ( median_col_coverage <= 0.996677398682 ) {
                      return 0.384799116983 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.996677398682
                      return 0.170235652372 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.931499958038
                  if ( max_col_coverage <= 0.878815293312 ) {
                    if ( min_col_support <= 0.59350001812 ) {
                      return 0.44343883762 < maxgini;
                    }
                    else {  // if min_col_support > 0.59350001812
                      return 0.368624279413 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.878815293312
                    if ( mean_col_coverage <= 0.995185375214 ) {
                      return 0.444647804807 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.995185375214
                      return 0.32444013621 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.685500025749
              if ( min_col_coverage <= 0.781277537346 ) {
                if ( mean_col_coverage <= 0.704812526703 ) {
                  if ( mean_col_support <= 0.909852921963 ) {
                    if ( mean_col_support <= 0.891852974892 ) {
                      return 0.403723216098 < maxgini;
                    }
                    else {  // if mean_col_support > 0.891852974892
                      return 0.280242537591 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.909852921963
                    if ( max_col_coverage <= 0.979379296303 ) {
                      return 0.130598176957 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.979379296303
                      return 0.263462224866 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.704812526703
                  if ( median_col_support <= 0.807500004768 ) {
                    if ( min_col_coverage <= 0.60968285799 ) {
                      return 0.320956173608 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.60968285799
                      return 0.431283487984 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.807500004768
                    if ( min_col_coverage <= 0.666863441467 ) {
                      return 0.158412644196 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.666863441467
                      return 0.21260315196 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.781277537346
                if ( min_col_coverage <= 0.997925043106 ) {
                  if ( median_col_support <= 0.808500051498 ) {
                    if ( median_col_support <= 0.805500030518 ) {
                      return 0.413862060293 < maxgini;
                    }
                    else {  // if median_col_support > 0.805500030518
                      return false;
                    }
                  }
                  else {  // if median_col_support > 0.808500051498
                    if ( median_col_support <= 0.926499962807 ) {
                      return 0.265376117836 < maxgini;
                    }
                    else {  // if median_col_support > 0.926499962807
                      return 0.338580918044 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.997925043106
                  if ( mean_col_coverage <= 0.999944090843 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if mean_col_coverage > 0.999944090843
                    if ( mean_col_support <= 0.910705864429 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.910705864429
                      return 0.173351422392 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.979499995708
          if ( mean_col_coverage <= 0.372110396624 ) {
            if ( max_col_coverage <= 0.371543705463 ) {
              if ( min_col_coverage <= 0.152330815792 ) {
                if ( median_col_support <= 0.999000012875 ) {
                  if ( median_col_coverage <= 0.137757956982 ) {
                    if ( mean_col_coverage <= 0.104401402175 ) {
                      return 0.0792256203509 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.104401402175
                      return 0.164058703031 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.137757956982
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.182296185528 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.302303239265 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.999000012875
                  if ( min_col_support <= 0.550500035286 ) {
                    if ( max_col_coverage <= 0.306058049202 ) {
                      return 0.112461869248 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.306058049202
                      return 0.235374363024 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.550500035286
                    if ( mean_col_support <= 0.956029415131 ) {
                      return 0.0739694515619 < maxgini;
                    }
                    else {  // if mean_col_support > 0.956029415131
                      return 0.0439427447373 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.152330815792
                if ( mean_col_support <= 0.946205794811 ) {
                  if ( mean_col_coverage <= 0.302416473627 ) {
                    if ( min_col_coverage <= 0.177512705326 ) {
                      return 0.310741043084 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.177512705326
                      return 0.41795183076 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.302416473627
                    if ( median_col_support <= 0.983500003815 ) {
                      return 0.197530864198 < maxgini;
                    }
                    else {  // if median_col_support > 0.983500003815
                      return 0.497383431417 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.946205794811
                  if ( min_col_coverage <= 0.213025167584 ) {
                    if ( mean_col_support <= 0.954676449299 ) {
                      return 0.277777777778 < maxgini;
                    }
                    else {  // if mean_col_support > 0.954676449299
                      return 0.155326550873 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.213025167584
                    if ( mean_col_support <= 0.95926463604 ) {
                      return 0.301083277924 < maxgini;
                    }
                    else {  // if mean_col_support > 0.95926463604
                      return 0.21793858341 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.371543705463
              if ( max_col_coverage <= 0.607921600342 ) {
                if ( max_col_coverage <= 0.394794970751 ) {
                  if ( min_col_coverage <= 0.18246242404 ) {
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.272624969894 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.133408530307 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.18246242404
                    if ( mean_col_support <= 0.955382406712 ) {
                      return 0.407590182549 < maxgini;
                    }
                    else {  // if mean_col_support > 0.955382406712
                      return 0.254398166978 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.394794970751
                  if ( max_col_coverage <= 0.529833614826 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.329875378822 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.262124448387 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.529833614826
                    if ( min_col_support <= 0.661499977112 ) {
                      return 0.422380610223 < maxgini;
                    }
                    else {  // if min_col_support > 0.661499977112
                      return 0.227763206987 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.607921600342
                if ( min_col_coverage <= 0.030418690294 ) {
                  if ( median_col_support <= 0.985499978065 ) {
                    if ( min_col_support <= 0.570500016212 ) {
                      return 0.371124260355 < maxgini;
                    }
                    else {  // if min_col_support > 0.570500016212
                      return 0.162752376641 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.985499978065
                    if ( median_col_coverage <= 0.0453403368592 ) {
                      return 0.402721680778 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0453403368592
                      return 0.259100346021 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.030418690294
                  if ( mean_col_coverage <= 0.22292932868 ) {
                    if ( max_col_coverage <= 0.662393212318 ) {
                      return 0.424822990421 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.662393212318
                      return 0.0713305898491 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.22292932868
                    if ( min_col_coverage <= 0.122878924012 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.122878924012
                      return 0.29702055801 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.372110396624
            if ( min_col_coverage <= 0.519220232964 ) {
              if ( mean_col_coverage <= 0.469076693058 ) {
                if ( min_col_support <= 0.708500027657 ) {
                  if ( median_col_coverage <= 0.167184263468 ) {
                    if ( median_col_coverage <= 0.0594427473843 ) {
                      return 0.39381808915 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0594427473843
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.167184263468
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.382548909382 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return 0.441199876354 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.708500027657
                  if ( max_col_coverage <= 0.989444732666 ) {
                    if ( mean_col_support <= 0.98291182518 ) {
                      return 0.231921921829 < maxgini;
                    }
                    else {  // if mean_col_support > 0.98291182518
                      return 0.379010951152 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.989444732666
                    if ( min_col_coverage <= 0.0138489548117 ) {
                      return 0.21875 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0138489548117
                      return false;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.469076693058
                if ( mean_col_coverage <= 0.512700140476 ) {
                  if ( mean_col_support <= 0.957676410675 ) {
                    if ( median_col_coverage <= 0.179295837879 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.179295837879
                      return 0.461247863681 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.957676410675
                    if ( mean_col_support <= 0.987676501274 ) {
                      return 0.402674100834 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987676501274
                      return 0.473826002046 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.512700140476
                  if ( min_col_coverage <= 0.255769580603 ) {
                    if ( min_col_support <= 0.682500004768 ) {
                      return 0.498421916425 < maxgini;
                    }
                    else {  // if min_col_support > 0.682500004768
                      return 0.432100480066 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.255769580603
                    if ( min_col_coverage <= 0.431200355291 ) {
                      return 0.420469066788 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.431200355291
                      return 0.439049278988 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.519220232964
              if ( median_col_support <= 0.99950003624 ) {
                if ( max_col_coverage <= 0.826487064362 ) {
                  if ( min_col_support <= 0.677500009537 ) {
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.432990924645 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.458589874677 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.677500009537
                    if ( mean_col_coverage <= 0.702449560165 ) {
                      return 0.336271666949 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.702449560165
                      return 0.370502127878 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.826487064362
                  if ( max_col_coverage <= 0.972097992897 ) {
                    if ( mean_col_support <= 0.976970553398 ) {
                      return 0.451739273751 < maxgini;
                    }
                    else {  // if mean_col_support > 0.976970553398
                      return 0.406129040197 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.972097992897
                    if ( min_col_coverage <= 0.980591058731 ) {
                      return 0.450438090151 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.980591058731
                      return 0.415285369135 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_coverage <= 0.962335944176 ) {
                  if ( mean_col_support <= 0.92044121027 ) {
                    if ( max_col_coverage <= 0.585144877434 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.585144877434
                      return 0.499194221949 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.92044121027
                    if ( mean_col_coverage <= 0.749634742737 ) {
                      return 0.475210699551 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.749634742737
                      return 0.481661849697 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.962335944176
                  if ( median_col_coverage <= 0.975114047527 ) {
                    if ( mean_col_coverage <= 0.994042217731 ) {
                      return 0.465000994303 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.994042217731
                      return false;
                    }
                  }
                  else {  // if median_col_coverage > 0.975114047527
                    if ( mean_col_coverage <= 0.987660169601 ) {
                      return 0.400685818939 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.987660169601
                      return 0.436804771387 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.807500004768
        if ( mean_col_support <= 0.992899537086 ) {
          if ( min_col_coverage <= 0.757639706135 ) {
            if ( min_col_support <= 0.878499984741 ) {
              if ( median_col_coverage <= 0.575826764107 ) {
                if ( mean_col_coverage <= 0.507170319557 ) {
                  if ( min_col_support <= 0.839499950409 ) {
                    if ( mean_col_coverage <= 0.401713013649 ) {
                      return 0.0581694563182 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.401713013649
                      return 0.0845591079966 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.839499950409
                    if ( median_col_coverage <= 0.012698540464 ) {
                      return 0.151085591427 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.012698540464
                      return 0.0510752490631 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.507170319557
                  if ( mean_col_support <= 0.988441109657 ) {
                    if ( median_col_coverage <= 0.133974373341 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.133974373341
                      return 0.0817030994816 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.988441109657
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.105840547726 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.296412574029 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.575826764107
                if ( median_col_support <= 0.991500020027 ) {
                  if ( median_col_support <= 0.989500045776 ) {
                    if ( min_col_support <= 0.827499985695 ) {
                      return 0.109764797114 < maxgini;
                    }
                    else {  // if min_col_support > 0.827499985695
                      return 0.0710617759042 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.989500045776
                    if ( max_col_coverage <= 0.97748208046 ) {
                      return 0.173472583434 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.97748208046
                      return 0.269770408163 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.991500020027
                  if ( mean_col_support <= 0.988676428795 ) {
                    if ( mean_col_support <= 0.980499982834 ) {
                      return 0.362532291977 < maxgini;
                    }
                    else {  // if mean_col_support > 0.980499982834
                      return 0.263073196697 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.988676428795
                    if ( min_col_support <= 0.845499992371 ) {
                      return 0.448379492751 < maxgini;
                    }
                    else {  // if min_col_support > 0.845499992371
                      return 0.323177212696 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.878499984741
              if ( median_col_support <= 0.950500011444 ) {
                if ( median_col_support <= 0.912500023842 ) {
                  if ( median_col_coverage <= 0.254342854023 ) {
                    if ( min_col_support <= 0.886500000954 ) {
                      return 0.112517505953 < maxgini;
                    }
                    else {  // if min_col_support > 0.886500000954
                      return 0.092712295722 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.254342854023
                    if ( max_col_coverage <= 0.971008419991 ) {
                      return 0.0819269821463 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.971008419991
                      return 0.293367346939 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.912500023842
                  if ( median_col_support <= 0.93850004673 ) {
                    if ( min_col_support <= 0.912500023842 ) {
                      return 0.0518506521425 < maxgini;
                    }
                    else {  // if min_col_support > 0.912500023842
                      return 0.0716747946291 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.93850004673
                    if ( max_col_coverage <= 0.971825361252 ) {
                      return 0.0446755548471 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.971825361252
                      return 0.10885063014 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.950500011444
                if ( min_col_support <= 0.901499986649 ) {
                  if ( mean_col_coverage <= 0.634662806988 ) {
                    if ( min_col_support <= 0.886500000954 ) {
                      return 0.0399180544204 < maxgini;
                    }
                    else {  // if min_col_support > 0.886500000954
                      return 0.0317767191584 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.634662806988
                    if ( min_col_coverage <= 0.659205317497 ) {
                      return 0.0663419080385 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.659205317497
                      return 0.0966490160786 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.901499986649
                  if ( mean_col_support <= 0.987667322159 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.0275406276115 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.0481455848831 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.987667322159
                    if ( min_col_coverage <= 0.0185471847653 ) {
                      return 0.108535365217 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0185471847653
                      return 0.0207835693436 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.757639706135
            if ( min_col_coverage <= 0.853671967983 ) {
              if ( mean_col_support <= 0.990970551968 ) {
                if ( mean_col_coverage <= 0.829179763794 ) {
                  if ( min_col_coverage <= 0.763084411621 ) {
                    if ( median_col_support <= 0.991500020027 ) {
                      return 0.0692374946065 < maxgini;
                    }
                    else {  // if median_col_support > 0.991500020027
                      return 0.316034737376 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.763084411621
                    if ( median_col_coverage <= 0.775426328182 ) {
                      return 0.100329822978 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.775426328182
                      return 0.146410819085 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.829179763794
                  if ( mean_col_support <= 0.987911820412 ) {
                    if ( mean_col_support <= 0.981323480606 ) {
                      return 0.20612721395 < maxgini;
                    }
                    else {  // if mean_col_support > 0.981323480606
                      return 0.15662659205 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.987911820412
                    if ( mean_col_support <= 0.990029394627 ) {
                      return 0.241909942491 < maxgini;
                    }
                    else {  // if mean_col_support > 0.990029394627
                      return 0.188270535657 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.990970551968
                if ( median_col_support <= 0.993499994278 ) {
                  if ( min_col_coverage <= 0.757823586464 ) {
                    if ( median_col_support <= 0.988999962807 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.988999962807
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.757823586464
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0137353073365 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.0667314801982 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( median_col_support <= 0.995499968529 ) {
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.11478988964 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.161363064032 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.995499968529
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.254023239703 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.196287607302 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.853671967983
              if ( median_col_coverage <= 0.997837841511 ) {
                if ( mean_col_support <= 0.99120593071 ) {
                  if ( median_col_coverage <= 0.886347293854 ) {
                    if ( min_col_support <= 0.866500020027 ) {
                      return 0.366225182973 < maxgini;
                    }
                    else {  // if min_col_support > 0.866500020027
                      return 0.080237912824 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.886347293854
                    if ( min_col_support <= 0.866500020027 ) {
                      return 0.378843624897 < maxgini;
                    }
                    else {  // if min_col_support > 0.866500020027
                      return 0.128358228114 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.99120593071
                  if ( mean_col_support <= 0.992323517799 ) {
                    if ( median_col_coverage <= 0.914857387543 ) {
                      return 0.165558668196 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.914857387543
                      return 0.20760285558 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992323517799
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.0177228535227 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.209007385958 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.997837841511
                if ( max_col_support <= 0.99950003624 ) {
                  return 0.0 < maxgini;
                }
                else {  // if max_col_support > 0.99950003624
                  if ( mean_col_support <= 0.991088271141 ) {
                    if ( median_col_coverage <= 0.99789249897 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.99789249897
                      return 0.159369273186 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.991088271141
                    if ( min_col_support <= 0.885499954224 ) {
                      return 0.333489053377 < maxgini;
                    }
                    else {  // if min_col_support > 0.885499954224
                      return 0.0401330426405 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.992899537086
          if ( mean_col_coverage <= 0.882315576077 ) {
            if ( min_col_support <= 0.910500049591 ) {
              if ( median_col_coverage <= 0.564128696918 ) {
                if ( min_col_coverage <= 0.395274430513 ) {
                  if ( max_col_coverage <= 0.464230060577 ) {
                    if ( mean_col_coverage <= 0.446330547333 ) {
                      return 0.0168613718583 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.446330547333
                      return 0.375 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.464230060577
                    if ( median_col_coverage <= 0.0404876098037 ) {
                      return 0.169447113285 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0404876098037
                      return 0.0399850367095 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.395274430513
                  if ( min_col_coverage <= 0.488192886114 ) {
                    if ( min_col_support <= 0.892500042915 ) {
                      return 0.168714405083 < maxgini;
                    }
                    else {  // if min_col_support > 0.892500042915
                      return 0.0670752769555 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.488192886114
                    if ( mean_col_support <= 0.994205951691 ) {
                      return 0.137703001626 < maxgini;
                    }
                    else {  // if mean_col_support > 0.994205951691
                      return 0.0701985097011 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.564128696918
                if ( min_col_coverage <= 0.622177243233 ) {
                  if ( mean_col_support <= 0.994147062302 ) {
                    if ( max_col_coverage <= 0.688827514648 ) {
                      return 0.145780104566 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.688827514648
                      return 0.19954697188 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994147062302
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.404296875 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.123662423115 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.622177243233
                  if ( median_col_coverage <= 0.745065569878 ) {
                    if ( min_col_support <= 0.890499949455 ) {
                      return 0.355004565016 < maxgini;
                    }
                    else {  // if min_col_support > 0.890499949455
                      return 0.191256268165 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.745065569878
                    if ( min_col_support <= 0.895500004292 ) {
                      return 0.35812829452 < maxgini;
                    }
                    else {  // if min_col_support > 0.895500004292
                      return 0.229740196002 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.910500049591
              if ( min_col_support <= 0.929499983788 ) {
                if ( min_col_coverage <= 0.58887887001 ) {
                  if ( median_col_coverage <= 0.512170433998 ) {
                    if ( min_col_support <= 0.921499967575 ) {
                      return 0.0206471571317 < maxgini;
                    }
                    else {  // if min_col_support > 0.921499967575
                      return 0.0165320027979 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.512170433998
                    if ( median_col_coverage <= 0.560282945633 ) {
                      return 0.0307635416806 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.560282945633
                      return 0.040458736114 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.58887887001
                  if ( mean_col_coverage <= 0.738004565239 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.09350248152 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.0482324050794 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.738004565239
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.109052785969 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.070460400904 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.929499983788
                if ( median_col_coverage <= 0.465924561024 ) {
                  if ( mean_col_support <= 0.995852828026 ) {
                    if ( min_col_support <= 0.96749997139 ) {
                      return 0.0138006050944 < maxgini;
                    }
                    else {  // if min_col_support > 0.96749997139
                      return 0.0292391765385 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.995852828026
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.0178993569347 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return 0.00856315870652 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.465924561024
                  if ( max_col_coverage <= 0.983002662659 ) {
                    if ( mean_col_support <= 0.997088193893 ) {
                      return 0.010839326224 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997088193893
                      return 0.00642656073825 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.983002662659
                    if ( mean_col_coverage <= 0.800475597382 ) {
                      return 0.030367396524 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.800475597382
                      return 0.0110505238156 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.882315576077
            if ( max_col_coverage <= 0.998722076416 ) {
              if ( mean_col_coverage <= 0.96359783411 ) {
                if ( min_col_support <= 0.912500023842 ) {
                  if ( mean_col_support <= 0.994735240936 ) {
                    if ( mean_col_coverage <= 0.961395025253 ) {
                      return 0.315860041234 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.961395025253
                      return 0.127066115702 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994735240936
                    if ( median_col_coverage <= 0.948412656784 ) {
                      return 0.1638 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.948412656784
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.912500023842
                  if ( mean_col_support <= 0.996147036552 ) {
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.0105216814408 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.0363554919813 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.996147036552
                    if ( min_col_support <= 0.954499959946 ) {
                      return 0.0276249780134 < maxgini;
                    }
                    else {  // if min_col_support > 0.954499959946
                      return 0.00765700548033 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.96359783411
                if ( mean_col_coverage <= 0.963633656502 ) {
                  if ( mean_col_support <= 0.997058868408 ) {
                    if ( max_col_coverage <= 0.989090919495 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.989090919495
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.997058868408
                    return 0.0 < maxgini;
                  }
                }
                else {  // if mean_col_coverage > 0.963633656502
                  if ( min_col_support <= 0.935500025749 ) {
                    if ( min_col_support <= 0.90750002861 ) {
                      return 0.340207825186 < maxgini;
                    }
                    else {  // if min_col_support > 0.90750002861
                      return 0.211558879315 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.935500025749
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.0305159243474 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.0130818289886 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.998722076416
              if ( mean_col_support <= 0.994911789894 ) {
                if ( mean_col_support <= 0.993676424026 ) {
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( mean_col_coverage <= 0.992098927498 ) {
                      return 0.0142160776066 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.992098927498
                      return 0.0452676977989 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( min_col_coverage <= 0.883315443993 ) {
                      return 0.0970799540601 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.883315443993
                      return 0.152046548915 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.993676424026
                  if ( mean_col_coverage <= 0.929844379425 ) {
                    if ( min_col_support <= 0.914499998093 ) {
                      return 0.210851470842 < maxgini;
                    }
                    else {  // if min_col_support > 0.914499998093
                      return 0.0179766834773 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.929844379425
                    if ( min_col_support <= 0.917500019073 ) {
                      return 0.261252759378 < maxgini;
                    }
                    else {  // if min_col_support > 0.917500019073
                      return 0.0245021956741 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.994911789894
                if ( median_col_coverage <= 0.983405590057 ) {
                  if ( min_col_support <= 0.932500004768 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.287124890239 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.120894044015 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.932500004768
                    if ( min_col_support <= 0.943500041962 ) {
                      return 0.0339016142191 < maxgini;
                    }
                    else {  // if min_col_support > 0.943500041962
                      return 0.00830604857251 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.983405590057
                  if ( median_col_coverage <= 0.983410537243 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.983410537243
                    if ( median_col_coverage <= 0.996683239937 ) {
                      return 0.0435681047943 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.996683239937
                      return 0.0170500066751 < maxgini;
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
      if ( min_col_support <= 0.807500004768 ) {
        if ( median_col_support <= 0.979499995708 ) {
          if ( max_col_coverage <= 0.714480876923 ) {
            if ( min_col_support <= 0.634500026703 ) {
              if ( median_col_support <= 0.597499966621 ) {
                if ( median_col_coverage <= 0.274029910564 ) {
                  if ( median_col_coverage <= 0.18432277441 ) {
                    if ( mean_col_support <= 0.7206569314 ) {
                      return 0.259939772394 < maxgini;
                    }
                    else {  // if mean_col_support > 0.7206569314
                      return 0.145976445643 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.18432277441
                    if ( mean_col_coverage <= 0.281061857939 ) {
                      return 0.270307120279 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.281061857939
                      return 0.316732940924 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.274029910564
                  if ( mean_col_coverage <= 0.421648025513 ) {
                    if ( min_col_coverage <= 0.265225172043 ) {
                      return 0.375713624602 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.265225172043
                      return 0.430542792596 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.421648025513
                    if ( median_col_support <= 0.559499979019 ) {
                      return 0.499005887071 < maxgini;
                    }
                    else {  // if median_col_support > 0.559499979019
                      return 0.456075396097 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.597499966621
                if ( min_col_coverage <= 0.305634021759 ) {
                  if ( max_col_coverage <= 0.372200906277 ) {
                    if ( median_col_coverage <= 0.0512557774782 ) {
                      return 0.128811251365 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0512557774782
                      return 0.0978439022598 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.372200906277
                    if ( median_col_support <= 0.951499998569 ) {
                      return 0.132748460765 < maxgini;
                    }
                    else {  // if median_col_support > 0.951499998569
                      return 0.291005484946 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.305634021759
                  if ( median_col_coverage <= 0.488592982292 ) {
                    if ( median_col_support <= 0.677500009537 ) {
                      return 0.331397184996 < maxgini;
                    }
                    else {  // if median_col_support > 0.677500009537
                      return 0.164456801724 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.488592982292
                    if ( median_col_support <= 0.65649998188 ) {
                      return 0.436866811198 < maxgini;
                    }
                    else {  // if median_col_support > 0.65649998188
                      return 0.23667746543 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.634500026703
              if ( min_col_coverage <= 0.333745956421 ) {
                if ( median_col_support <= 0.815500020981 ) {
                  if ( min_col_coverage <= 0.22231374681 ) {
                    if ( max_col_coverage <= 0.279088318348 ) {
                      return 0.0844490800865 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.279088318348
                      return 0.10159533244 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.22231374681
                    if ( min_col_support <= 0.725499987602 ) {
                      return 0.154411725407 < maxgini;
                    }
                    else {  // if min_col_support > 0.725499987602
                      return 0.122935439334 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.815500020981
                  if ( mean_col_coverage <= 0.116503000259 ) {
                    if ( median_col_coverage <= 0.0101738041267 ) {
                      return 0.142032859775 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0101738041267
                      return 0.0665965920464 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.116503000259
                    if ( mean_col_support <= 0.939029455185 ) {
                      return 0.0732236495027 < maxgini;
                    }
                    else {  // if mean_col_support > 0.939029455185
                      return 0.0581202501364 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.333745956421
                if ( min_col_support <= 0.729499995708 ) {
                  if ( min_col_coverage <= 0.457224786282 ) {
                    if ( min_col_support <= 0.684499979019 ) {
                      return 0.17182750336 < maxgini;
                    }
                    else {  // if min_col_support > 0.684499979019
                      return 0.133044705899 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.457224786282
                    if ( min_col_support <= 0.661499977112 ) {
                      return 0.235831422931 < maxgini;
                    }
                    else {  // if min_col_support > 0.661499977112
                      return 0.181346289895 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.729499995708
                  if ( min_col_coverage <= 0.488526254892 ) {
                    if ( min_col_support <= 0.783499956131 ) {
                      return 0.0997610997507 < maxgini;
                    }
                    else {  // if min_col_support > 0.783499956131
                      return 0.0789987193661 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.488526254892
                    if ( median_col_coverage <= 0.574878334999 ) {
                      return 0.116793554952 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.574878334999
                      return 0.0997438939925 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.714480876923
            if ( min_col_support <= 0.664499998093 ) {
              if ( median_col_coverage <= 0.667340755463 ) {
                if ( median_col_support <= 0.674499988556 ) {
                  if ( min_col_coverage <= 0.37939876318 ) {
                    if ( mean_col_coverage <= 0.617166519165 ) {
                      return 0.203647512864 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.617166519165
                      return 0.130352797639 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.37939876318
                    if ( median_col_support <= 0.574499964714 ) {
                      return 0.499033625313 < maxgini;
                    }
                    else {  // if median_col_support > 0.574499964714
                      return 0.450124005799 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.674499988556
                  if ( max_col_coverage <= 0.992856800556 ) {
                    if ( median_col_support <= 0.943500041962 ) {
                      return 0.205635393023 < maxgini;
                    }
                    else {  // if median_col_support > 0.943500041962
                      return 0.383718100039 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.992856800556
                    if ( median_col_support <= 0.959499955177 ) {
                      return 0.363198236849 < maxgini;
                    }
                    else {  // if median_col_support > 0.959499955177
                      return 0.470923232046 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.667340755463
                if ( median_col_support <= 0.926499962807 ) {
                  if ( min_col_coverage <= 0.978672862053 ) {
                    if ( median_col_support <= 0.694499969482 ) {
                      return 0.444271665835 < maxgini;
                    }
                    else {  // if median_col_support > 0.694499969482
                      return 0.317907619622 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.978672862053
                    if ( median_col_support <= 0.824499964714 ) {
                      return 0.119019482133 < maxgini;
                    }
                    else {  // if median_col_support > 0.824499964714
                      return 0.267798289242 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.926499962807
                  if ( median_col_coverage <= 0.996219277382 ) {
                    if ( median_col_support <= 0.964499950409 ) {
                      return 0.426745191775 < maxgini;
                    }
                    else {  // if median_col_support > 0.964499950409
                      return 0.454003418366 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.996219277382
                    if ( min_col_coverage <= 0.789103031158 ) {
                      return 0.0555102040816 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.789103031158
                      return 0.314110243688 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.664499998093
              if ( mean_col_support <= 0.909029364586 ) {
                if ( max_col_support <= 0.999000012875 ) {
                  if ( median_col_coverage <= 0.533119678497 ) {
                    if ( max_col_coverage <= 0.746980667114 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.746980667114
                      return 0.110726643599 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.533119678497
                    return 0.0 < maxgini;
                  }
                }
                else {  // if max_col_support > 0.999000012875
                  if ( median_col_support <= 0.75150001049 ) {
                    if ( mean_col_coverage <= 0.554041326046 ) {
                      return 0.09652398736 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.554041326046
                      return 0.457048487907 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.75150001049
                    if ( min_col_support <= 0.708500027657 ) {
                      return 0.183095768968 < maxgini;
                    }
                    else {  // if min_col_support > 0.708500027657
                      return 0.333697264914 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.909029364586
                if ( mean_col_coverage <= 0.802302420139 ) {
                  if ( median_col_support <= 0.802500009537 ) {
                    if ( median_col_support <= 0.74849998951 ) {
                      return 0.466701352758 < maxgini;
                    }
                    else {  // if median_col_support > 0.74849998951
                      return 0.307777147922 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.802500009537
                    if ( median_col_coverage <= 0.575929403305 ) {
                      return 0.10879961354 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.575929403305
                      return 0.172436803753 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.802302420139
                  if ( mean_col_coverage <= 0.905756294727 ) {
                    if ( median_col_support <= 0.964499950409 ) {
                      return 0.235903134938 < maxgini;
                    }
                    else {  // if median_col_support > 0.964499950409
                      return 0.319354006253 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.905756294727
                    if ( max_col_coverage <= 0.999401926994 ) {
                      return 0.376018747696 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.999401926994
                      return 0.299065750178 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.979499995708
          if ( min_col_support <= 0.716500043869 ) {
            if ( mean_col_support <= 0.970585763454 ) {
              if ( mean_col_coverage <= 0.29120528698 ) {
                if ( max_col_coverage <= 0.474081933498 ) {
                  if ( min_col_coverage <= 0.128601074219 ) {
                    if ( max_col_coverage <= 0.334584206343 ) {
                      return 0.0732028081177 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.334584206343
                      return 0.221142834762 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.128601074219
                    if ( min_col_coverage <= 0.181886538863 ) {
                      return 0.218234046495 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.181886538863
                      return 0.29147683343 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.474081933498
                  if ( max_col_coverage <= 0.52280151844 ) {
                    if ( mean_col_support <= 0.960441112518 ) {
                      return 0.45110823457 < maxgini;
                    }
                    else {  // if mean_col_support > 0.960441112518
                      return 0.256090637944 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.52280151844
                    if ( median_col_coverage <= 0.0189276337624 ) {
                      return 0.309539794922 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0189276337624
                      return 0.494837408949 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.29120528698
                if ( mean_col_support <= 0.955323457718 ) {
                  if ( mean_col_coverage <= 0.482701838017 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.404931367008 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.448569356841 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.482701838017
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.462922155161 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.481385261114 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.955323457718
                  if ( max_col_coverage <= 0.614857077599 ) {
                    if ( max_col_coverage <= 0.472713083029 ) {
                      return 0.329731318318 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.472713083029
                      return 0.401595082295 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.614857077599
                    if ( median_col_coverage <= 0.208483219147 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.208483219147
                      return 0.457370399584 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.970585763454
              if ( mean_col_coverage <= 0.354706048965 ) {
                if ( median_col_coverage <= 0.152009502053 ) {
                  if ( mean_col_coverage <= 0.18154913187 ) {
                    if ( median_col_coverage <= 0.0622234493494 ) {
                      return 0.0520832031788 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0622234493494
                      return 0.102849068646 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.18154913187
                    if ( median_col_support <= 0.985499978065 ) {
                      return 0.0654984199943 < maxgini;
                    }
                    else {  // if median_col_support > 0.985499978065
                      return 0.216771783452 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.152009502053
                  if ( median_col_coverage <= 0.223201334476 ) {
                    if ( mean_col_support <= 0.977911770344 ) {
                      return 0.255982993714 < maxgini;
                    }
                    else {  // if mean_col_support > 0.977911770344
                      return 0.335823362652 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.223201334476
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.25859352888 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.371326151406 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.354706048965
                if ( min_col_coverage <= 0.518307089806 ) {
                  if ( median_col_support <= 0.997500002384 ) {
                    if ( mean_col_support <= 0.971852898598 ) {
                      return 0.420328397249 < maxgini;
                    }
                    else {  // if mean_col_support > 0.971852898598
                      return 0.374585231609 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.997500002384
                    if ( min_col_coverage <= 0.358471333981 ) {
                      return 0.442211803377 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.358471333981
                      return 0.470161602851 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.518307089806
                  if ( min_col_support <= 0.644500017166 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.470148173105 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.48608125168 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.644500017166
                    if ( mean_col_coverage <= 0.73931312561 ) {
                      return 0.46236954467 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.73931312561
                      return 0.475772104318 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.716500043869
            if ( max_col_coverage <= 0.545568585396 ) {
              if ( max_col_coverage <= 0.405955374241 ) {
                if ( median_col_coverage <= 0.212230801582 ) {
                  if ( min_col_coverage <= 0.122280597687 ) {
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.0899912136658 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.0322558854282 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.122280597687
                    if ( min_col_support <= 0.753499984741 ) {
                      return 0.0952922604992 < maxgini;
                    }
                    else {  // if min_col_support > 0.753499984741
                      return 0.054462531004 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.212230801582
                  if ( min_col_coverage <= 0.194758355618 ) {
                    if ( mean_col_support <= 0.983382344246 ) {
                      return 0.0645944678845 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983382344246
                      return 0.16406879871 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.194758355618
                    if ( max_col_coverage <= 0.335822165012 ) {
                      return 0.0802609577588 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.335822165012
                      return 0.169994668191 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.405955374241
                if ( mean_col_support <= 0.983147025108 ) {
                  if ( max_col_coverage <= 0.487837761641 ) {
                    if ( min_col_coverage <= 0.273171067238 ) {
                      return 0.110250529278 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.273171067238
                      return 0.210702274656 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.487837761641
                    if ( median_col_coverage <= 0.312751293182 ) {
                      return 0.126035057626 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.312751293182
                      return 0.254224024651 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.983147025108
                  if ( min_col_coverage <= 0.299918949604 ) {
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.207228273436 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.289527240181 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.299918949604
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.278766843825 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.404350497705 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.545568585396
              if ( median_col_support <= 0.99950003624 ) {
                if ( mean_col_coverage <= 0.701585292816 ) {
                  if ( min_col_coverage <= 0.488401979208 ) {
                    if ( median_col_coverage <= 0.121045619249 ) {
                      return 0.443121739435 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.121045619249
                      return 0.259385162199 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.488401979208
                    if ( mean_col_support <= 0.982323527336 ) {
                      return 0.304188372881 < maxgini;
                    }
                    else {  // if mean_col_support > 0.982323527336
                      return 0.344618098895 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.701585292816
                  if ( mean_col_coverage <= 0.899394869804 ) {
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.314429174939 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.394242257393 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.899394869804
                    if ( mean_col_coverage <= 0.998960614204 ) {
                      return 0.40505078102 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.998960614204
                      return 0.304896894249 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( mean_col_coverage <= 0.634934425354 ) {
                  if ( median_col_coverage <= 0.441539466381 ) {
                    if ( mean_col_support <= 0.983323454857 ) {
                      return 0.262546898466 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983323454857
                      return 0.415733135856 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.441539466381
                    if ( min_col_support <= 0.775499999523 ) {
                      return 0.443717431027 < maxgini;
                    }
                    else {  // if min_col_support > 0.775499999523
                      return 0.38533959479 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.634934425354
                  if ( mean_col_coverage <= 0.814692854881 ) {
                    if ( min_col_coverage <= 0.573379933834 ) {
                      return 0.448823707231 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.573379933834
                      return 0.468112178702 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.814692854881
                    if ( mean_col_support <= 0.983323574066 ) {
                      return 0.460673802501 < maxgini;
                    }
                    else {  // if mean_col_support > 0.983323574066
                      return 0.482782944739 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.807500004768
        if ( max_col_coverage <= 0.889013528824 ) {
          if ( min_col_support <= 0.887500047684 ) {
            if ( median_col_coverage <= 0.575826764107 ) {
              if ( min_col_coverage <= 0.424316376448 ) {
                if ( min_col_coverage <= 0.314443051815 ) {
                  if ( min_col_support <= 0.839499950409 ) {
                    if ( mean_col_support <= 0.952902793884 ) {
                      return 0.0870088464715 < maxgini;
                    }
                    else {  // if mean_col_support > 0.952902793884
                      return 0.0505110266372 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.839499950409
                    if ( mean_col_support <= 0.972364068031 ) {
                      return 0.0630316758456 < maxgini;
                    }
                    else {  // if mean_col_support > 0.972364068031
                      return 0.0363096744815 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.314443051815
                  if ( mean_col_support <= 0.988558828831 ) {
                    if ( mean_col_support <= 0.962441205978 ) {
                      return 0.08263915427 < maxgini;
                    }
                    else {  // if mean_col_support > 0.962441205978
                      return 0.0535839495562 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.988558828831
                    if ( mean_col_coverage <= 0.468124359846 ) {
                      return 0.120735626489 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.468124359846
                      return 0.193522220442 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.424316376448
                if ( min_col_support <= 0.851500034332 ) {
                  if ( median_col_support <= 0.989500045776 ) {
                    if ( min_col_support <= 0.821500003338 ) {
                      return 0.0833560315279 < maxgini;
                    }
                    else {  // if min_col_support > 0.821500003338
                      return 0.0657650200066 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.989500045776
                    if ( mean_col_support <= 0.988676428795 ) {
                      return 0.223158865355 < maxgini;
                    }
                    else {  // if mean_col_support > 0.988676428795
                      return 0.402079007021 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.851500034332
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( min_col_coverage <= 0.424448877573 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.424448877573
                      return 0.050317278155 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( mean_col_support <= 0.99126470089 ) {
                      return 0.114506740834 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99126470089
                      return 0.239495616725 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.575826764107
              if ( mean_col_support <= 0.98808825016 ) {
                if ( min_col_support <= 0.828500032425 ) {
                  if ( median_col_support <= 0.989500045776 ) {
                    if ( mean_col_support <= 0.984382390976 ) {
                      return 0.103123740064 < maxgini;
                    }
                    else {  // if mean_col_support > 0.984382390976
                      return 0.216203611901 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.989500045776
                    if ( median_col_coverage <= 0.613063573837 ) {
                      return 0.300184833965 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.613063573837
                      return 0.331488714432 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.828500032425
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( median_col_support <= 0.884500026703 ) {
                      return 0.157429061123 < maxgini;
                    }
                    else {  // if median_col_support > 0.884500026703
                      return 0.0634025089668 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( min_col_coverage <= 0.617959141731 ) {
                      return 0.181407084611 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.617959141731
                      return 0.239378343396 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.98808825016
                if ( median_col_coverage <= 0.730666160583 ) {
                  if ( mean_col_support <= 0.988676428795 ) {
                    if ( median_col_coverage <= 0.576648533344 ) {
                      return 0.494341330919 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.576648533344
                      return 0.23824263263 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.988676428795
                    if ( min_col_support <= 0.856500029564 ) {
                      return 0.419319675047 < maxgini;
                    }
                    else {  // if min_col_support > 0.856500029564
                      return 0.235984731676 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.730666160583
                  if ( min_col_support <= 0.854499995708 ) {
                    if ( mean_col_support <= 0.988499999046 ) {
                      return 0.364639023678 < maxgini;
                    }
                    else {  // if mean_col_support > 0.988499999046
                      return 0.44486531835 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.854499995708
                    if ( min_col_support <= 0.869500041008 ) {
                      return 0.336194080651 < maxgini;
                    }
                    else {  // if min_col_support > 0.869500041008
                      return 0.244581264438 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.887500047684
            if ( min_col_support <= 0.928499996662 ) {
              if ( mean_col_support <= 0.974970579147 ) {
                if ( min_col_support <= 0.911499977112 ) {
                  if ( median_col_coverage <= 0.315982967615 ) {
                    if ( min_col_support <= 0.903499960899 ) {
                      return 0.0692656005145 < maxgini;
                    }
                    else {  // if min_col_support > 0.903499960899
                      return 0.105780922638 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.315982967615
                    if ( mean_col_coverage <= 0.818483233452 ) {
                      return 0.0598400930872 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.818483233452
                      return 0.189164201183 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.911499977112
                  if ( median_col_support <= 0.93850004673 ) {
                    if ( median_col_coverage <= 0.441982865334 ) {
                      return 0.12347159301 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.441982865334
                      return 0.0809085697117 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.93850004673
                    if ( max_col_coverage <= 0.887626290321 ) {
                      return 0.060910590268 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.887626290321
                      return 0.345679012346 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.974970579147
                if ( mean_col_coverage <= 0.683683991432 ) {
                  if ( min_col_coverage <= 0.51523566246 ) {
                    if ( min_col_support <= 0.903499960899 ) {
                      return 0.0350979297708 < maxgini;
                    }
                    else {  // if min_col_support > 0.903499960899
                      return 0.0275904460805 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.51523566246
                    if ( min_col_support <= 0.902500033379 ) {
                      return 0.0645855635907 < maxgini;
                    }
                    else {  // if min_col_support > 0.902500033379
                      return 0.0341955768175 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.683683991432
                  if ( median_col_coverage <= 0.667011916637 ) {
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0261972685738 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.0806464987641 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.667011916637
                    if ( mean_col_support <= 0.992441177368 ) {
                      return 0.0472599206029 < maxgini;
                    }
                    else {  // if mean_col_support > 0.992441177368
                      return 0.110697759459 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.928499996662
              if ( mean_col_coverage <= 0.519837856293 ) {
                if ( median_col_support <= 0.966500043869 ) {
                  if ( min_col_coverage <= 0.00161289481912 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.00161289481912
                    if ( mean_col_support <= 0.977205872536 ) {
                      return 0.123978547909 < maxgini;
                    }
                    else {  // if mean_col_support > 0.977205872536
                      return 0.0490076692917 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.966500043869
                  if ( median_col_coverage <= 0.238944172859 ) {
                    if ( min_col_coverage <= 0.00758534716442 ) {
                      return 0.0592451894079 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00758534716442
                      return 0.0219717810532 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.238944172859
                    if ( min_col_coverage <= 0.385562002659 ) {
                      return 0.0153947794272 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.385562002659
                      return 0.0123398053296 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.519837856293
                if ( mean_col_support <= 0.98908829689 ) {
                  if ( min_col_coverage <= 0.469718813896 ) {
                    if ( median_col_support <= 0.954499959946 ) {
                      return 0.0643534049807 < maxgini;
                    }
                    else {  // if median_col_support > 0.954499959946
                      return 0.0295933201558 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.469718813896
                    if ( min_col_coverage <= 0.876893937588 ) {
                      return 0.0255611244987 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.876893937588
                      return 0.444444444444 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.98908829689
                  if ( max_col_coverage <= 0.705723881721 ) {
                    if ( min_col_coverage <= 0.482155591249 ) {
                      return 0.0119296952906 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.482155591249
                      return 0.0102136623453 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.705723881721
                    if ( median_col_coverage <= 0.575704693794 ) {
                      return 0.0107333951783 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.575704693794
                      return 0.00939158572117 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.889013528824
          if ( median_col_support <= 0.99950003624 ) {
            if ( median_col_coverage <= 0.923202872276 ) {
              if ( min_col_coverage <= 0.73690444231 ) {
                if ( median_col_coverage <= 0.133974373341 ) {
                  if ( min_col_support <= 0.943500041962 ) {
                    if ( min_col_support <= 0.924499988556 ) {
                      return 0.333107433013 < maxgini;
                    }
                    else {  // if min_col_support > 0.924499988556
                      return 0.466128729593 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.943500041962
                    if ( mean_col_coverage <= 0.314085036516 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.314085036516
                      return 0.038080874729 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.133974373341
                  if ( mean_col_support <= 0.990556359291 ) {
                    if ( mean_col_support <= 0.939823508263 ) {
                      return 0.322356766726 < maxgini;
                    }
                    else {  // if mean_col_support > 0.939823508263
                      return 0.0733257914077 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.990556359291
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.0142963407192 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.102159531601 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.73690444231
                if ( max_col_coverage <= 0.998861670494 ) {
                  if ( median_col_support <= 0.995499968529 ) {
                    if ( mean_col_support <= 0.989323437214 ) {
                      return 0.125929772175 < maxgini;
                    }
                    else {  // if mean_col_support > 0.989323437214
                      return 0.0186624507744 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.995499968529
                    if ( min_col_support <= 0.916499972343 ) {
                      return 0.307651844623 < maxgini;
                    }
                    else {  // if min_col_support > 0.916499972343
                      return 0.052644915378 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.998861670494
                  if ( mean_col_support <= 0.989970564842 ) {
                    if ( min_col_support <= 0.857499957085 ) {
                      return 0.225966147881 < maxgini;
                    }
                    else {  // if min_col_support > 0.857499957085
                      return 0.0538217164213 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.989970564842
                    if ( mean_col_coverage <= 0.94908452034 ) {
                      return 0.0216538647681 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.94908452034
                      return 0.0398263211185 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.923202872276
              if ( min_col_support <= 0.895500004292 ) {
                if ( median_col_support <= 0.993499994278 ) {
                  if ( median_col_coverage <= 0.997871756554 ) {
                    if ( median_col_coverage <= 0.924860656261 ) {
                      return 0.373402572112 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.924860656261
                      return 0.238017484096 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.997871756554
                    if ( mean_col_coverage <= 0.995854854584 ) {
                      return 0.0368194474267 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.995854854584
                      return 0.146132527085 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( mean_col_coverage <= 0.92727637291 ) {
                    if ( median_col_coverage <= 0.923297643661 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.923297643661
                      return 0.462809917355 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.92727637291
                    if ( median_col_coverage <= 0.979807257652 ) {
                      return 0.333862834611 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.979807257652
                      return 0.29945942721 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.895500004292
                if ( min_col_support <= 0.931499958038 ) {
                  if ( mean_col_support <= 0.993088245392 ) {
                    if ( min_col_support <= 0.90649998188 ) {
                      return 0.146132527085 < maxgini;
                    }
                    else {  // if min_col_support > 0.90649998188
                      return 0.0886294644184 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.993088245392
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.0814060220623 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.239018897711 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.931499958038
                  if ( min_col_support <= 0.948500037193 ) {
                    if ( min_col_coverage <= 0.946041584015 ) {
                      return 0.0355229842671 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.946041584015
                      return 0.0704779741567 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.948500037193
                    if ( mean_col_coverage <= 0.998864769936 ) {
                      return 0.0110588290867 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.998864769936
                      return 0.0378531432634 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.99950003624
            if ( mean_col_coverage <= 0.882756590843 ) {
              if ( min_col_coverage <= 0.150396823883 ) {
                if ( max_col_coverage <= 0.994222402573 ) {
                  if ( min_col_support <= 0.81200003624 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.81200003624
                    if ( min_col_coverage <= 0.117869749665 ) {
                      return 0.12220274493 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.117869749665
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.994222402573
                  if ( min_col_coverage <= 0.0224747471511 ) {
                    if ( median_col_coverage <= 0.0186050534248 ) {
                      return 0.276140787197 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0186050534248
                      return 0.0962666137671 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0224747471511
                    if ( mean_col_support <= 0.989264786243 ) {
                      return 0.475610674094 < maxgini;
                    }
                    else {  // if mean_col_support > 0.989264786243
                      return 0.140315545287 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.150396823883
                if ( min_col_support <= 0.885499954224 ) {
                  if ( mean_col_support <= 0.988676428795 ) {
                    if ( min_col_support <= 0.833500027657 ) {
                      return 0.387108763676 < maxgini;
                    }
                    else {  // if min_col_support > 0.833500027657
                      return 0.216454053672 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.988676428795
                    if ( min_col_support <= 0.844500005245 ) {
                      return 0.469833123127 < maxgini;
                    }
                    else {  // if min_col_support > 0.844500005245
                      return 0.355737025901 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.885499954224
                  if ( mean_col_support <= 0.994676470757 ) {
                    if ( max_col_coverage <= 0.889430522919 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.889430522919
                      return 0.0512156819797 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994676470757
                    if ( min_col_coverage <= 0.429658561945 ) {
                      return 0.040684189785 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.429658561945
                      return 0.0102066320351 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.882756590843
              if ( median_col_coverage <= 0.911845505238 ) {
                if ( max_col_coverage <= 0.998629927635 ) {
                  if ( mean_col_support <= 0.992558836937 ) {
                    if ( mean_col_coverage <= 0.913407921791 ) {
                      return 0.332918866538 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.913407921791
                      return 0.357241997668 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992558836937
                    if ( max_col_coverage <= 0.998142957687 ) {
                      return 0.0201041189347 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.998142957687
                      return 0.497448979592 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.998629927635
                  if ( median_col_coverage <= 0.872516095638 ) {
                    if ( mean_col_support <= 0.991911768913 ) {
                      return 0.270132567386 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991911768913
                      return 0.0214740792675 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.872516095638
                    if ( min_col_support <= 0.888499975204 ) {
                      return 0.413896416282 < maxgini;
                    }
                    else {  // if min_col_support > 0.888499975204
                      return 0.0188592300754 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.911845505238
                if ( median_col_coverage <= 0.997950792313 ) {
                  if ( min_col_support <= 0.899500012398 ) {
                    if ( median_col_coverage <= 0.963691294193 ) {
                      return 0.416265842654 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.963691294193
                      return 0.355250845832 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.899500012398
                    if ( median_col_coverage <= 0.997842490673 ) {
                      return 0.0171776317579 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.997842490673
                      return 0.497041420118 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.997950792313
                  if ( mean_col_coverage <= 0.9985268116 ) {
                    if ( min_col_support <= 0.87549996376 ) {
                      return 0.313993259896 < maxgini;
                    }
                    else {  // if min_col_support > 0.87549996376
                      return 0.0195601351724 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.9985268116
                    if ( min_col_coverage <= 0.998334288597 ) {
                      return 0.0786787441394 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.998334288597
                      return 0.0522958150648 < maxgini;
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
      if ( mean_col_coverage <= 0.749627590179 ) {
        if ( mean_col_support <= 0.986531376839 ) {
          if ( min_col_coverage <= 0.488405942917 ) {
            if ( mean_col_coverage <= 0.469170033932 ) {
              if ( max_col_coverage <= 0.73747587204 ) {
                if ( mean_col_support <= 0.835861861706 ) {
                  if ( min_col_coverage <= 0.212165266275 ) {
                    if ( mean_col_support <= 0.730205893517 ) {
                      return 0.31512782534 < maxgini;
                    }
                    else {  // if mean_col_support > 0.730205893517
                      return 0.153717406535 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.212165266275
                    if ( min_col_support <= 0.499500006437 ) {
                      return 0.254116265806 < maxgini;
                    }
                    else {  // if min_col_support > 0.499500006437
                      return 0.378266979376 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.835861861706
                  if ( max_col_coverage <= 0.461581885815 ) {
                    if ( mean_col_support <= 0.949414253235 ) {
                      return 0.104542007625 < maxgini;
                    }
                    else {  // if mean_col_support > 0.949414253235
                      return 0.0649518177491 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.461581885815
                    if ( min_col_support <= 0.713500022888 ) {
                      return 0.214629507482 < maxgini;
                    }
                    else {  // if min_col_support > 0.713500022888
                      return 0.0681800285735 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.73747587204
                if ( median_col_coverage <= 0.164631426334 ) {
                  if ( min_col_coverage <= 0.0353877916932 ) {
                    if ( mean_col_support <= 0.950323462486 ) {
                      return 0.19893443138 < maxgini;
                    }
                    else {  // if mean_col_support > 0.950323462486
                      return 0.342090523429 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0353877916932
                    if ( median_col_support <= 0.955500006676 ) {
                      return 0.244956398457 < maxgini;
                    }
                    else {  // if median_col_support > 0.955500006676
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.164631426334
                  if ( median_col_coverage <= 0.254687666893 ) {
                    if ( min_col_support <= 0.662500023842 ) {
                      return 0.343411132429 < maxgini;
                    }
                    else {  // if min_col_support > 0.662500023842
                      return 0.1128 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.254687666893
                    if ( min_col_support <= 0.611500024796 ) {
                      return 0.228598144496 < maxgini;
                    }
                    else {  // if min_col_support > 0.611500024796
                      return 0.0699833680194 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.469170033932
              if ( median_col_support <= 0.987499952316 ) {
                if ( min_col_support <= 0.684499979019 ) {
                  if ( max_col_coverage <= 0.996082723141 ) {
                    if ( median_col_support <= 0.642500042915 ) {
                      return 0.46010508898 < maxgini;
                    }
                    else {  // if median_col_support > 0.642500042915
                      return 0.212494787803 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.996082723141
                    if ( median_col_support <= 0.877499997616 ) {
                      return 0.265657478607 < maxgini;
                    }
                    else {  // if median_col_support > 0.877499997616
                      return 0.457051540683 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.684499979019
                  if ( median_col_coverage <= 0.212443590164 ) {
                    if ( median_col_support <= 0.982499957085 ) {
                      return 0.308878979356 < maxgini;
                    }
                    else {  // if median_col_support > 0.982499957085
                      return 0.49429729475 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.212443590164
                    if ( median_col_support <= 0.802500009537 ) {
                      return 0.265331110894 < maxgini;
                    }
                    else {  // if median_col_support > 0.802500009537
                      return 0.0596484519646 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.987499952316
                if ( median_col_coverage <= 0.268619120121 ) {
                  if ( mean_col_coverage <= 0.616489648819 ) {
                    if ( max_col_coverage <= 0.996033430099 ) {
                      return 0.405968042349 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.996033430099
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.616489648819
                    if ( min_col_coverage <= 0.04923325032 ) {
                      return 0.264513648597 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.04923325032
                      return false;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.268619120121
                  if ( mean_col_support <= 0.981029391289 ) {
                    if ( mean_col_support <= 0.961088240147 ) {
                      return 0.470265154209 < maxgini;
                    }
                    else {  // if mean_col_support > 0.961088240147
                      return 0.417443052306 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.981029391289
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.157648627015 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.280425248581 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.488405942917
            if ( median_col_support <= 0.988499999046 ) {
              if ( mean_col_support <= 0.95191180706 ) {
                if ( min_col_support <= 0.62349998951 ) {
                  if ( min_col_coverage <= 0.617724061012 ) {
                    if ( median_col_support <= 0.648499965668 ) {
                      return 0.485178129776 < maxgini;
                    }
                    else {  // if median_col_support > 0.648499965668
                      return 0.286478689775 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.617724061012
                    if ( mean_col_coverage <= 0.691263914108 ) {
                      return 0.347851564039 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.691263914108
                      return 0.41494994664 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.62349998951
                  if ( max_col_coverage <= 0.676525235176 ) {
                    if ( max_col_coverage <= 0.676279783249 ) {
                      return 0.172469276685 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.676279783249
                      return 0.0727317815459 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.676525235176
                    if ( min_col_support <= 0.699499964714 ) {
                      return 0.268277341849 < maxgini;
                    }
                    else {  // if min_col_support > 0.699499964714
                      return 0.178285178816 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.95191180706
                if ( min_col_support <= 0.725499987602 ) {
                  if ( max_col_coverage <= 0.714470028877 ) {
                    if ( min_col_support <= 0.631500005722 ) {
                      return 0.365845767051 < maxgini;
                    }
                    else {  // if min_col_support > 0.631500005722
                      return 0.225668460224 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.714470028877
                    if ( median_col_support <= 0.970499992371 ) {
                      return 0.263243543295 < maxgini;
                    }
                    else {  // if median_col_support > 0.970499992371
                      return 0.39816433867 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.725499987602
                  if ( mean_col_coverage <= 0.691188573837 ) {
                    if ( max_col_coverage <= 0.667444586754 ) {
                      return 0.0503016992209 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.667444586754
                      return 0.0657299631373 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.691188573837
                    if ( mean_col_coverage <= 0.721068680286 ) {
                      return 0.0753612371161 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.721068680286
                      return 0.0855555368727 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.988499999046
              if ( mean_col_support <= 0.981029391289 ) {
                if ( min_col_support <= 0.707499980927 ) {
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( median_col_coverage <= 0.591688752174 ) {
                      return 0.439261737951 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.591688752174
                      return 0.45743291074 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( mean_col_support <= 0.922500014305 ) {
                      return 0.499833328299 < maxgini;
                    }
                    else {  // if mean_col_support > 0.922500014305
                      return 0.479675167371 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.707499980927
                  if ( mean_col_coverage <= 0.594724476337 ) {
                    if ( max_col_coverage <= 0.588760495186 ) {
                      return 0.25623672878 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.588760495186
                      return 0.360412873232 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.594724476337
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.353380191927 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.422635429443 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.981029391289
                if ( median_col_support <= 0.997500002384 ) {
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( median_col_coverage <= 0.564174175262 ) {
                      return 0.220121949174 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.564174175262
                      return 0.181761584895 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( median_col_coverage <= 0.624293446541 ) {
                      return 0.297592252897 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.624293446541
                      return 0.329220367327 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.997500002384
                  if ( median_col_support <= 0.99849998951 ) {
                    if ( mean_col_coverage <= 0.650370359421 ) {
                      return 0.333686033858 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.650370359421
                      return 0.382474489429 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99849998951
                    if ( median_col_coverage <= 0.606419861317 ) {
                      return 0.386593296465 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.606419861317
                      return 0.442811174832 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.986531376839
          if ( min_col_support <= 0.850499987602 ) {
            if ( median_col_coverage <= 0.369528353214 ) {
              if ( min_col_support <= 0.808500051498 ) {
                if ( mean_col_coverage <= 0.313538730145 ) {
                  if ( median_col_coverage <= 0.215412855148 ) {
                    if ( min_col_coverage <= 0.00290995230898 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00290995230898
                      return 0.0394016805177 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.215412855148
                    if ( min_col_support <= 0.805500030518 ) {
                      return 0.213877479156 < maxgini;
                    }
                    else {  // if min_col_support > 0.805500030518
                      return 0.0392 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.313538730145
                  if ( min_col_support <= 0.790500044823 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.498456790123 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.415349186754 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.790500044823
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.102788279773 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return 0.337392358088 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.808500051498
                if ( median_col_support <= 0.99950003624 ) {
                  if ( median_col_support <= 0.99849998951 ) {
                    if ( mean_col_support <= 0.989382386208 ) {
                      return 0.141182198061 < maxgini;
                    }
                    else {  // if mean_col_support > 0.989382386208
                      return 0.308390022676 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99849998951
                    if ( min_col_coverage <= 0.2759988904 ) {
                      return 0.0997229916898 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.2759988904
                      return false;
                    }
                  }
                }
                else {  // if median_col_support > 0.99950003624
                  if ( mean_col_coverage <= 0.345846712589 ) {
                    if ( mean_col_coverage <= 0.301380932331 ) {
                      return 0.0265710349915 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.301380932331
                      return 0.089211955375 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.345846712589
                    if ( mean_col_support <= 0.988558828831 ) {
                      return 0.0939404844291 < maxgini;
                    }
                    else {  // if mean_col_support > 0.988558828831
                      return 0.22101708113 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.369528353214
              if ( median_col_support <= 0.99950003624 ) {
                if ( min_col_support <= 0.827499985695 ) {
                  if ( max_col_coverage <= 0.808433353901 ) {
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.208170981058 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.332080249572 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.808433353901
                    if ( max_col_coverage <= 0.810974419117 ) {
                      return 0.48553843306 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.810974419117
                      return 0.397087921063 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.827499985695
                  if ( mean_col_support <= 0.988558769226 ) {
                    if ( mean_col_coverage <= 0.675922095776 ) {
                      return 0.182227868445 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.675922095776
                      return 0.240064833019 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.988558769226
                    if ( median_col_support <= 0.991500020027 ) {
                      return 0.18 < maxgini;
                    }
                    else {  // if median_col_support > 0.991500020027
                      return 0.310731466807 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_coverage <= 0.48819476366 ) {
                  if ( min_col_coverage <= 0.425043553114 ) {
                    if ( mean_col_support <= 0.990323424339 ) {
                      return 0.33356364047 < maxgini;
                    }
                    else {  // if mean_col_support > 0.990323424339
                      return 0.301391144693 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.425043553114
                    if ( median_col_coverage <= 0.467372626066 ) {
                      return 0.346093660603 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.467372626066
                      return 0.387619876735 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.48819476366
                  if ( mean_col_coverage <= 0.648172199726 ) {
                    if ( mean_col_support <= 0.989147007465 ) {
                      return 0.416363302602 < maxgini;
                    }
                    else {  // if mean_col_support > 0.989147007465
                      return 0.442656472807 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.648172199726
                    if ( max_col_coverage <= 0.84663271904 ) {
                      return 0.443452745778 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.84663271904
                      return 0.413243800836 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.850499987602
            if ( mean_col_support <= 0.993834912777 ) {
              if ( mean_col_coverage <= 0.633734822273 ) {
                if ( min_col_support <= 0.883499979973 ) {
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( median_col_coverage <= 0.47979593277 ) {
                      return 0.0276100037773 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.47979593277
                      return 0.0558124548933 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( mean_col_support <= 0.989932119846 ) {
                      return 0.0579076933255 < maxgini;
                    }
                    else {  // if mean_col_support > 0.989932119846
                      return 0.133467501442 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.883499979973
                  if ( median_col_support <= 0.96749997139 ) {
                    if ( mean_col_coverage <= 0.396118462086 ) {
                      return 0.0521496255809 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.396118462086
                      return 0.0335063173117 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.96749997139
                    if ( median_col_coverage <= 0.024721916765 ) {
                      return 0.0998455364086 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.024721916765
                      return 0.0214328444233 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.633734822273
                if ( mean_col_support <= 0.992794156075 ) {
                  if ( mean_col_coverage <= 0.705989003181 ) {
                    if ( min_col_coverage <= 0.558881878853 ) {
                      return 0.0385268684443 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.558881878853
                      return 0.048116624233 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.705989003181
                    if ( mean_col_coverage <= 0.707008183002 ) {
                      return 0.185898491084 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.707008183002
                      return 0.0552300435533 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.992794156075
                  if ( median_col_coverage <= 0.155939310789 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.155939310789
                    if ( min_col_coverage <= 0.606179535389 ) {
                      return 0.0308266414163 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.606179535389
                      return 0.0388089778136 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.993834912777
              if ( min_col_coverage <= 0.146099805832 ) {
                if ( min_col_coverage <= 0.146091341972 ) {
                  if ( median_col_coverage <= 0.0164158977568 ) {
                    if ( mean_col_coverage <= 0.164203137159 ) {
                      return 0.0137608073007 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.164203137159
                      return 0.0993525934909 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0164158977568
                    if ( median_col_coverage <= 0.347421884537 ) {
                      return 0.0175595501717 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.347421884537
                      return 0.132653061224 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.146091341972
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.146099805832
                if ( median_col_support <= 0.993499994278 ) {
                  if ( mean_col_coverage <= 0.52595937252 ) {
                    if ( min_col_coverage <= 0.236737340689 ) {
                      return 0.0281361801631 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.236737340689
                      return 0.0144135361906 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.52595937252
                    if ( mean_col_coverage <= 0.652881979942 ) {
                      return 0.00876662550385 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.652881979942
                      return 0.00593649523836 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( min_col_support <= 0.919499993324 ) {
                    if ( mean_col_coverage <= 0.603881537914 ) {
                      return 0.0348206016878 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.603881537914
                      return 0.115807639679 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.919499993324
                    if ( median_col_coverage <= 0.466499686241 ) {
                      return 0.0110270945847 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.466499686241
                      return 0.00966946703751 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_coverage > 0.749627590179
        if ( min_col_support <= 0.855499982834 ) {
          if ( median_col_support <= 0.990499973297 ) {
            if ( max_col_support <= 0.99849998951 ) {
              if ( median_col_support <= 0.466000020504 ) {
                if ( min_col_coverage <= 0.2515822649 ) {
                  if ( max_col_support <= 0.773000001907 ) {
                    return false;
                  }
                  else {  // if max_col_support > 0.773000001907
                    return 0.0 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.2515822649
                  return false;
                }
              }
              else {  // if median_col_support > 0.466000020504
                if ( max_col_coverage <= 0.899664402008 ) {
                  if ( min_col_support <= 0.503000020981 ) {
                    if ( min_col_coverage <= 0.839629411697 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.839629411697
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.503000020981
                    return 0.0 < maxgini;
                  }
                }
                else {  // if max_col_coverage > 0.899664402008
                  if ( median_col_support <= 0.528499960899 ) {
                    if ( median_col_support <= 0.520500004292 ) {
                      return 0.0222194167403 < maxgini;
                    }
                    else {  // if median_col_support > 0.520500004292
                      return 0.149013878744 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.528499960899
                    if ( mean_col_support <= 0.83208823204 ) {
                      return 0.00839491256396 < maxgini;
                    }
                    else {  // if mean_col_support > 0.83208823204
                      return 0.025988273062 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_support > 0.99849998951
              if ( min_col_support <= 0.737499952316 ) {
                if ( max_col_coverage <= 0.857350468636 ) {
                  if ( min_col_support <= 0.620499968529 ) {
                    if ( mean_col_coverage <= 0.826364338398 ) {
                      return 0.426594366978 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.826364338398
                      return 0.294476781998 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.620499968529
                    if ( median_col_support <= 0.972499966621 ) {
                      return 0.266286159917 < maxgini;
                    }
                    else {  // if median_col_support > 0.972499966621
                      return 0.397176032559 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.857350468636
                  if ( min_col_support <= 0.62349998951 ) {
                    if ( min_col_coverage <= 0.976688563824 ) {
                      return 0.449996741613 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.976688563824
                      return 0.288058473036 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.62349998951
                    if ( median_col_support <= 0.972499966621 ) {
                      return 0.338062307859 < maxgini;
                    }
                    else {  // if median_col_support > 0.972499966621
                      return 0.421999534597 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.737499952316
                if ( median_col_coverage <= 0.838801860809 ) {
                  if ( min_col_support <= 0.795500040054 ) {
                    if ( min_col_coverage <= 0.708300828934 ) {
                      return 0.197279435168 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.708300828934
                      return 0.268352828206 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.795500040054
                    if ( max_col_coverage <= 0.833580136299 ) {
                      return 0.0902159793557 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.833580136299
                      return 0.142504893893 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.838801860809
                  if ( min_col_coverage <= 0.997871756554 ) {
                    if ( median_col_coverage <= 0.921080708504 ) {
                      return 0.274970686162 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.921080708504
                      return 0.331122659994 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.997871756554
                    if ( min_col_support <= 0.753499984741 ) {
                      return 0.195665502378 < maxgini;
                    }
                    else {  // if min_col_support > 0.753499984741
                      return 0.152990830218 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.990499973297
            if ( mean_col_support <= 0.983382344246 ) {
              if ( min_col_support <= 0.738499999046 ) {
                if ( mean_col_coverage <= 0.981289982796 ) {
                  if ( min_col_coverage <= 0.945966780186 ) {
                    if ( min_col_support <= 0.71749997139 ) {
                      return 0.478920245519 < maxgini;
                    }
                    else {  // if min_col_support > 0.71749997139
                      return 0.45716982247 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.945966780186
                    if ( max_col_coverage <= 0.986544072628 ) {
                      return 0.43642045532 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.986544072628
                      return 0.470144901293 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.981289982796
                  if ( mean_col_coverage <= 0.999882280827 ) {
                    if ( max_col_coverage <= 0.99509203434 ) {
                      return 0.413109110898 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.99509203434
                      return 0.467672156714 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.999882280827
                    if ( min_col_coverage <= 0.999058365822 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.999058365822
                      return 0.440915619681 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.738499999046
                if ( min_col_coverage <= 0.758647918701 ) {
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( min_col_support <= 0.779500007629 ) {
                      return 0.406276349413 < maxgini;
                    }
                    else {  // if min_col_support > 0.779500007629
                      return 0.328775340025 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( mean_col_coverage <= 0.875265479088 ) {
                      return 0.417873988816 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.875265479088
                      return 0.327954807902 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.758647918701
                  if ( mean_col_support <= 0.976558804512 ) {
                    if ( min_col_support <= 0.800500035286 ) {
                      return 0.456037023572 < maxgini;
                    }
                    else {  // if min_col_support > 0.800500035286
                      return 0.416434778646 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.976558804512
                    if ( min_col_coverage <= 0.839119553566 ) {
                      return 0.377048542036 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.839119553566
                      return 0.402292964454 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.983382344246
              if ( min_col_support <= 0.793500006199 ) {
                if ( min_col_support <= 0.762500047684 ) {
                  if ( min_col_coverage <= 0.945534944534 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.410599144408 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.490096817839 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.945534944534
                    if ( min_col_coverage <= 0.961640357971 ) {
                      return 0.454747252712 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.961640357971
                      return 0.428372485198 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.762500047684
                  if ( min_col_coverage <= 0.947624981403 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.403166203248 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.480668323404 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.947624981403
                    if ( median_col_coverage <= 0.966086566448 ) {
                      return 0.406436642986 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.966086566448
                      return 0.438045099863 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.793500006199
                if ( median_col_support <= 0.99950003624 ) {
                  if ( mean_col_coverage <= 0.948320269585 ) {
                    if ( median_col_coverage <= 0.697894513607 ) {
                      return 0.282439933884 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.697894513607
                      return 0.335918924597 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.948320269585
                    if ( max_col_coverage <= 0.993284106255 ) {
                      return 0.381673474499 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.993284106255
                      return 0.34047573691 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.99950003624
                  if ( mean_col_support <= 0.986676454544 ) {
                    if ( min_col_support <= 0.809499979019 ) {
                      return 0.371131997387 < maxgini;
                    }
                    else {  // if min_col_support > 0.809499979019
                      return 0.183135406266 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.986676454544
                    if ( min_col_coverage <= 0.970392286777 ) {
                      return 0.454756902157 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.970392286777
                      return 0.353662643301 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.855499982834
          if ( mean_col_support <= 0.993911743164 ) {
            if ( min_col_support <= 0.896499991417 ) {
              if ( median_col_support <= 0.993499994278 ) {
                if ( min_col_coverage <= 0.853921055794 ) {
                  if ( min_col_coverage <= 0.424927175045 ) {
                    if ( min_col_coverage <= 0.397802203894 ) {
                      return 0.225328719723 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.397802203894
                      return 0.457856399584 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.424927175045
                    if ( median_col_support <= 0.991500020027 ) {
                      return 0.0708754644394 < maxgini;
                    }
                    else {  // if median_col_support > 0.991500020027
                      return 0.154051572023 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.853921055794
                  if ( min_col_support <= 0.886500000954 ) {
                    if ( median_col_support <= 0.981500029564 ) {
                      return 0.150894899486 < maxgini;
                    }
                    else {  // if median_col_support > 0.981500029564
                      return 0.198367242166 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.886500000954
                    if ( median_col_coverage <= 0.925895452499 ) {
                      return 0.0865851852201 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.925895452499
                      return 0.149776690636 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.993499994278
                if ( min_col_support <= 0.87650001049 ) {
                  if ( mean_col_coverage <= 0.852313280106 ) {
                    if ( mean_col_support <= 0.991029500961 ) {
                      return 0.232344876637 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991029500961
                      return 0.398004765973 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.852313280106
                    if ( min_col_coverage <= 0.966056346893 ) {
                      return 0.354876357755 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.966056346893
                      return 0.294321704178 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.87650001049
                  if ( mean_col_support <= 0.991852939129 ) {
                    if ( median_col_coverage <= 0.879283666611 ) {
                      return 0.167514644683 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.879283666611
                      return 0.230601675471 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.991852939129
                    if ( min_col_support <= 0.886500000954 ) {
                      return 0.368160791894 < maxgini;
                    }
                    else {  // if min_col_support > 0.886500000954
                      return 0.292513691344 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.896499991417
              if ( median_col_coverage <= 0.923433899879 ) {
                if ( mean_col_support <= 0.987088263035 ) {
                  if ( median_col_coverage <= 0.182682782412 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.182682782412
                    if ( median_col_coverage <= 0.861219644547 ) {
                      return 0.0392394903118 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.861219644547
                      return 0.0536089666917 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.987088263035
                  if ( min_col_support <= 0.912500023842 ) {
                    if ( min_col_support <= 0.904500007629 ) {
                      return 0.118880227449 < maxgini;
                    }
                    else {  // if min_col_support > 0.904500007629
                      return 0.0778517139569 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.912500023842
                    if ( mean_col_coverage <= 0.914198875427 ) {
                      return 0.0194583138793 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.914198875427
                      return 0.0252363379037 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.923433899879
                if ( max_col_coverage <= 0.998801350594 ) {
                  if ( min_col_coverage <= 0.944598734379 ) {
                    if ( max_col_coverage <= 0.998336076736 ) {
                      return 0.0792081551652 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.998336076736
                      return 0.444444444444 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.944598734379
                    if ( median_col_coverage <= 0.944803237915 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.944803237915
                      return 0.170091749907 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.998801350594
                  if ( mean_col_coverage <= 0.831243753433 ) {
                    return false;
                  }
                  else {  // if mean_col_coverage > 0.831243753433
                    if ( mean_col_coverage <= 0.97479981184 ) {
                      return 0.0487793531254 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.97479981184
                      return 0.0672949318797 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.993911743164
            if ( median_col_support <= 0.995499968529 ) {
              if ( mean_col_coverage <= 0.998596072197 ) {
                if ( mean_col_support <= 0.994911789894 ) {
                  if ( min_col_support <= 0.921499967575 ) {
                    if ( mean_col_support <= 0.99426472187 ) {
                      return 0.128418549346 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99426472187
                      return 0.408163265306 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.921499967575
                    if ( min_col_support <= 0.933500051498 ) {
                      return 0.033215801698 < maxgini;
                    }
                    else {  // if min_col_support > 0.933500051498
                      return 0.0088357231675 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.994911789894
                  if ( min_col_support <= 0.935500025749 ) {
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.375 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.0806648199446 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.935500025749
                    if ( median_col_coverage <= 0.476978212595 ) {
                      return 0.408163265306 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.476978212595
                      return 0.00610597754927 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.998596072197
                if ( mean_col_coverage <= 0.998605012894 ) {
                  if ( min_col_support <= 0.976499974728 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if min_col_support > 0.976499974728
                    if ( mean_col_support <= 0.996205806732 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996205806732
                      return false;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.998605012894
                  if ( median_col_support <= 0.993499994278 ) {
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0242681815163 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.0919508039861 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.993499994278
                    if ( mean_col_coverage <= 0.998920559883 ) {
                      return 0.060546875 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.998920559883
                      return 0.00647242368022 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.995499968529
              if ( min_col_support <= 0.921499967575 ) {
                if ( min_col_support <= 0.904500007629 ) {
                  if ( median_col_support <= 0.99849998951 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if median_col_support > 0.99849998951
                    if ( min_col_coverage <= 0.828919887543 ) {
                      return 0.275100425751 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.828919887543
                      return 0.349067811242 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.904500007629
                  if ( median_col_coverage <= 0.859357833862 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.201665449381 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.132798085676 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.859357833862
                    if ( min_col_support <= 0.916499972343 ) {
                      return 0.239685562016 < maxgini;
                    }
                    else {  // if min_col_support > 0.916499972343
                      return 0.1712374693 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.921499967575
                if ( median_col_coverage <= 0.917033791542 ) {
                  if ( mean_col_support <= 0.996264696121 ) {
                    if ( mean_col_coverage <= 0.88769119978 ) {
                      return 0.0165739550393 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.88769119978
                      return 0.0225079169347 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.996264696121
                    if ( min_col_coverage <= 0.412141680717 ) {
                      return 0.0913386255634 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.412141680717
                      return 0.00789625678727 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.917033791542
                  if ( mean_col_support <= 0.996264696121 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.133880394327 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.0252175061862 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.996264696121
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.0178616266544 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.00907463060905 < maxgini;
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
      if ( max_col_coverage <= 0.83341896534 ) {
        if ( mean_col_support <= 0.987171530724 ) {
          if ( max_col_coverage <= 0.667074799538 ) {
            if ( min_col_support <= 0.723500013351 ) {
              if ( max_col_coverage <= 0.515201210976 ) {
                if ( median_col_coverage <= 0.272887587547 ) {
                  if ( min_col_coverage <= 0.181886538863 ) {
                    if ( median_col_support <= 0.601500034332 ) {
                      return 0.164127735896 < maxgini;
                    }
                    else {  // if median_col_support > 0.601500034332
                      return 0.101516984142 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.181886538863
                    if ( mean_col_support <= 0.809617638588 ) {
                      return 0.298490063408 < maxgini;
                    }
                    else {  // if mean_col_support > 0.809617638588
                      return 0.142425514248 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.272887587547
                  if ( median_col_coverage <= 0.333729952574 ) {
                    if ( mean_col_coverage <= 0.336823582649 ) {
                      return 0.174979253136 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.336823582649
                      return 0.209155462977 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.333729952574
                    if ( median_col_support <= 0.983500003815 ) {
                      return 0.193009407852 < maxgini;
                    }
                    else {  // if median_col_support > 0.983500003815
                      return 0.430088136575 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.515201210976
                if ( median_col_coverage <= 0.395422339439 ) {
                  if ( mean_col_coverage <= 0.422626823187 ) {
                    if ( min_col_coverage <= 0.110404968262 ) {
                      return 0.291608901095 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.110404968262
                      return 0.187271248764 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.422626823187
                    if ( mean_col_support <= 0.970205903053 ) {
                      return 0.223352510638 < maxgini;
                    }
                    else {  // if mean_col_support > 0.970205903053
                      return 0.423433445466 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.395422339439
                  if ( min_col_coverage <= 0.428866028786 ) {
                    if ( median_col_support <= 0.983500003815 ) {
                      return 0.230197101849 < maxgini;
                    }
                    else {  // if median_col_support > 0.983500003815
                      return 0.45257073715 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.428866028786
                    if ( min_col_coverage <= 0.488122224808 ) {
                      return 0.364301623286 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.488122224808
                      return 0.400141819753 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.723500013351
              if ( median_col_support <= 0.985499978065 ) {
                if ( mean_col_support <= 0.952060699463 ) {
                  if ( mean_col_support <= 0.921667337418 ) {
                    if ( median_col_support <= 0.810500025749 ) {
                      return 0.135018978305 < maxgini;
                    }
                    else {  // if median_col_support > 0.810500025749
                      return 0.109496302895 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.921667337418
                    if ( min_col_support <= 0.771499991417 ) {
                      return 0.072968621833 < maxgini;
                    }
                    else {  // if min_col_support > 0.771499991417
                      return 0.0892125135924 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.952060699463
                  if ( mean_col_support <= 0.977060675621 ) {
                    if ( mean_col_support <= 0.962710142136 ) {
                      return 0.0650179103883 < maxgini;
                    }
                    else {  // if mean_col_support > 0.962710142136
                      return 0.0543604051587 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.977060675621
                    if ( median_col_coverage <= 0.00749074202031 ) {
                      return 0.201052540554 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00749074202031
                      return 0.0354621354523 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.985499978065
                if ( min_col_support <= 0.787500023842 ) {
                  if ( mean_col_coverage <= 0.383650779724 ) {
                    if ( mean_col_coverage <= 0.290916055441 ) {
                      return 0.0529156346251 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.290916055441
                      return 0.203671904452 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.383650779724
                    if ( min_col_coverage <= 0.40284666419 ) {
                      return 0.319872742809 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.40284666419
                      return 0.402380050253 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.787500023842
                  if ( median_col_coverage <= 0.372152626514 ) {
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.136355738834 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.0370823265352 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.372152626514
                    if ( median_col_coverage <= 0.486565053463 ) {
                      return 0.10836119941 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.486565053463
                      return 0.168725354746 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.667074799538
            if ( mean_col_coverage <= 0.656637072563 ) {
              if ( mean_col_coverage <= 0.590083777905 ) {
                if ( max_col_coverage <= 0.673899173737 ) {
                  if ( min_col_support <= 0.787500023842 ) {
                    if ( max_col_coverage <= 0.671913027763 ) {
                      return 0.387520198271 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.671913027763
                      return 0.411545270891 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.787500023842
                    if ( median_col_support <= 0.991500020027 ) {
                      return 0.0893995311793 < maxgini;
                    }
                    else {  // if median_col_support > 0.991500020027
                      return 0.284749019669 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.673899173737
                  if ( min_col_coverage <= 0.117358915508 ) {
                    if ( median_col_support <= 0.964499950409 ) {
                      return 0.213105671785 < maxgini;
                    }
                    else {  // if median_col_support > 0.964499950409
                      return 0.431857910949 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.117358915508
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.103915201774 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.380576686625 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.590083777905
                if ( min_col_coverage <= 0.515169203281 ) {
                  if ( median_col_coverage <= 0.515175759792 ) {
                    if ( median_col_support <= 0.988499999046 ) {
                      return 0.0987132999654 < maxgini;
                    }
                    else {  // if median_col_support > 0.988499999046
                      return 0.403570767437 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.515175759792
                    if ( mean_col_support <= 0.846382379532 ) {
                      return 0.484210396705 < maxgini;
                    }
                    else {  // if mean_col_support > 0.846382379532
                      return 0.211166190578 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.515169203281
                  if ( min_col_support <= 0.78149998188 ) {
                    if ( median_col_coverage <= 0.559098184109 ) {
                      return 0.387826450073 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.559098184109
                      return 0.413534404562 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.78149998188
                    if ( median_col_support <= 0.989500045776 ) {
                      return 0.0583768255094 < maxgini;
                    }
                    else {  // if median_col_support > 0.989500045776
                      return 0.246863658173 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.656637072563
              if ( min_col_coverage <= 0.606115341187 ) {
                if ( median_col_coverage <= 0.589767813683 ) {
                  if ( mean_col_coverage <= 0.657350182533 ) {
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.179698610801 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.474802165764 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.657350182533
                    if ( max_col_coverage <= 0.817917346954 ) {
                      return 0.234205585329 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.817917346954
                      return 0.204081069873 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.589767813683
                  if ( min_col_support <= 0.78149998188 ) {
                    if ( median_col_support <= 0.985499978065 ) {
                      return 0.287972524234 < maxgini;
                    }
                    else {  // if median_col_support > 0.985499978065
                      return 0.46799971535 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.78149998188
                    if ( max_col_coverage <= 0.78783261776 ) {
                      return 0.100615734832 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.78783261776
                      return 0.0813027682778 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.606115341187
                if ( median_col_support <= 0.989500045776 ) {
                  if ( min_col_support <= 0.733500003815 ) {
                    if ( min_col_support <= 0.636500000954 ) {
                      return 0.403348529197 < maxgini;
                    }
                    else {  // if min_col_support > 0.636500000954
                      return 0.299541540904 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.733500003815
                    if ( mean_col_coverage <= 0.699717640877 ) {
                      return 0.0698910743893 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.699717640877
                      return 0.084392372787 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.989500045776
                  if ( mean_col_support <= 0.982558846474 ) {
                    if ( max_col_coverage <= 0.833160102367 ) {
                      return 0.468521434551 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.833160102367
                      return 0.480344703829 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.982558846474
                    if ( median_col_coverage <= 0.69467228651 ) {
                      return 0.404951732659 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.69467228651
                      return 0.433902677601 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.987171530724
          if ( min_col_support <= 0.856500029564 ) {
            if ( max_col_coverage <= 0.538018703461 ) {
              if ( mean_col_coverage <= 0.345853865147 ) {
                if ( median_col_coverage <= 0.218618690968 ) {
                  if ( max_col_coverage <= 0.274754911661 ) {
                    if ( mean_col_coverage <= 0.0149646103382 ) {
                      return 0.224765868887 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0149646103382
                      return 0.0106773104511 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.274754911661
                    if ( mean_col_coverage <= 0.0947133302689 ) {
                      return 0.455 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0947133302689
                      return 0.0432414039967 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.218618690968
                  if ( min_col_support <= 0.815500020981 ) {
                    if ( max_col_coverage <= 0.460407525301 ) {
                      return 0.214981683329 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.460407525301
                      return 0.414779900682 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.815500020981
                    if ( mean_col_coverage <= 0.301395535469 ) {
                      return 0.0368873311097 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.301395535469
                      return 0.0868469141455 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.345853865147
                if ( mean_col_coverage <= 0.413753122091 ) {
                  if ( min_col_support <= 0.824499964714 ) {
                    if ( mean_col_coverage <= 0.384050399065 ) {
                      return 0.279244587118 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.384050399065
                      return 0.36694931442 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.824499964714
                    if ( min_col_support <= 0.841500043869 ) {
                      return 0.180492551189 < maxgini;
                    }
                    else {  // if min_col_support > 0.841500043869
                      return 0.10017566568 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.413753122091
                  if ( max_col_coverage <= 0.513265967369 ) {
                    if ( mean_col_support <= 0.987852871418 ) {
                      return 0.226234987062 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987852871418
                      return 0.319912589536 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.513265967369
                    if ( median_col_coverage <= 0.419677406549 ) {
                      return 0.233467038998 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.419677406549
                      return 0.28863005678 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.538018703461
              if ( median_col_support <= 0.99950003624 ) {
                if ( mean_col_coverage <= 0.596818685532 ) {
                  if ( min_col_support <= 0.831499993801 ) {
                    if ( mean_col_coverage <= 0.588391184807 ) {
                      return 0.301381545599 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.588391184807
                      return 0.18673938185 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.831499993801
                    if ( median_col_coverage <= 0.46930167079 ) {
                      return 0.17473539416 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.46930167079
                      return 0.238057711108 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.596818685532
                  if ( median_col_support <= 0.991500020027 ) {
                    if ( min_col_support <= 0.823500037193 ) {
                      return 0.483237452886 < maxgini;
                    }
                    else {  // if min_col_support > 0.823500037193
                      return 0.144767980889 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.991500020027
                    if ( mean_col_coverage <= 0.598194241524 ) {
                      return 0.499927420525 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.598194241524
                      return 0.318143970625 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_support <= 0.828500032425 ) {
                  if ( min_col_coverage <= 0.487287223339 ) {
                    if ( min_col_support <= 0.810500025749 ) {
                      return 0.456531209766 < maxgini;
                    }
                    else {  // if min_col_support > 0.810500025749
                      return 0.379504132231 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.487287223339
                    if ( min_col_support <= 0.81350004673 ) {
                      return 0.478219030529 < maxgini;
                    }
                    else {  // if min_col_support > 0.81350004673
                      return 0.443885297875 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.828500032425
                  if ( median_col_coverage <= 0.515528559685 ) {
                    if ( median_col_coverage <= 0.384412944317 ) {
                      return 0.140825151285 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.384412944317
                      return 0.256029843333 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.515528559685
                    if ( min_col_coverage <= 0.599040448666 ) {
                      return 0.33953363578 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.599040448666
                      return 0.388455930882 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.856500029564
            if ( mean_col_support <= 0.994182050228 ) {
              if ( min_col_support <= 0.884500026703 ) {
                if ( min_col_coverage <= 0.457204848528 ) {
                  if ( median_col_coverage <= 0.39484333992 ) {
                    if ( median_col_coverage <= 0.305510878563 ) {
                      return 0.0255165577995 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.305510878563
                      return 0.0572447967529 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.39484333992
                    if ( mean_col_coverage <= 0.479529589415 ) {
                      return 0.0721898187575 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.479529589415
                      return 0.113521989068 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.457204848528
                  if ( median_col_support <= 0.992499947548 ) {
                    if ( min_col_coverage <= 0.555937886238 ) {
                      return 0.0581371515377 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.555937886238
                      return 0.0848424924509 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.992499947548
                    if ( min_col_support <= 0.871500015259 ) {
                      return 0.253856477633 < maxgini;
                    }
                    else {  // if min_col_support > 0.871500015259
                      return 0.18123608842 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.884500026703
                if ( median_col_support <= 0.991500020027 ) {
                  if ( min_col_coverage <= 0.00692049786448 ) {
                    if ( mean_col_coverage <= 0.24904564023 ) {
                      return 0.128774664066 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.24904564023
                      return 0.41022694628 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.00692049786448
                    if ( median_col_coverage <= 0.383830964565 ) {
                      return 0.0271615236553 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.383830964565
                      return 0.0155415799754 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.991500020027
                  if ( mean_col_support <= 0.989303350449 ) {
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.165243963045 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.0315291005283 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.989303350449
                    if ( median_col_coverage <= 0.559305310249 ) {
                      return 0.0222217479356 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.559305310249
                      return 0.0398241186583 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.994182050228
              if ( min_col_support <= 0.928499996662 ) {
                if ( median_col_coverage <= 0.575296878815 ) {
                  if ( mean_col_support <= 0.994911789894 ) {
                    if ( min_col_coverage <= 0.412917077541 ) {
                      return 0.018612396226 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.412917077541
                      return 0.0558095861616 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994911789894
                    if ( median_col_coverage <= 0.424621224403 ) {
                      return 0.0131759489975 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.424621224403
                      return 0.0281562244118 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.575296878815
                  if ( median_col_coverage <= 0.575449943542 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.575449943542
                    if ( median_col_coverage <= 0.658788621426 ) {
                      return 0.0675929752013 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.658788621426
                      return 0.098984653828 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.928499996662
                if ( mean_col_support <= 0.997424662113 ) {
                  if ( min_col_support <= 0.942499995232 ) {
                    if ( min_col_coverage <= 0.487050324678 ) {
                      return 0.0121416210447 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.487050324678
                      return 0.0210924821037 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.942499995232
                    if ( mean_col_coverage <= 0.52801668644 ) {
                      return 0.0124794583716 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.52801668644
                      return 0.00930320814594 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.997424662113
                  if ( median_col_support <= 0.99849998951 ) {
                    if ( mean_col_coverage <= 0.515292406082 ) {
                      return 0.013958766341 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.515292406082
                      return 0.00821609998893 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99849998951
                    if ( mean_col_coverage <= 0.169727772474 ) {
                      return 0.0204457247352 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.169727772474
                      return 0.00579744967569 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if max_col_coverage > 0.83341896534
        if ( min_col_coverage <= 0.833538889885 ) {
          if ( min_col_coverage <= 0.409805238247 ) {
            if ( median_col_support <= 0.978500008583 ) {
              if ( min_col_support <= 0.738499999046 ) {
                if ( max_col_support <= 0.99849998951 ) {
                  if ( max_col_support <= 0.787000000477 ) {
                    if ( mean_col_support <= 0.605470538139 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.605470538139
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.787000000477
                    if ( mean_col_coverage <= 0.391154199839 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.391154199839
                      return 0.020367515151 < maxgini;
                    }
                  }
                }
                else {  // if max_col_support > 0.99849998951
                  if ( min_col_coverage <= 0.0640512853861 ) {
                    if ( mean_col_coverage <= 0.628489494324 ) {
                      return 0.192568154635 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.628489494324
                      return 0.071539990547 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0640512853861
                    if ( max_col_coverage <= 0.998281776905 ) {
                      return 0.290379439184 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.998281776905
                      return 0.419423133529 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.738499999046
                if ( min_col_support <= 0.87349998951 ) {
                  if ( min_col_coverage <= 0.132237106562 ) {
                    if ( median_col_coverage <= 0.136602878571 ) {
                      return 0.320298383703 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.136602878571
                      return 0.116701234568 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.132237106562
                    if ( mean_col_coverage <= 0.667350113392 ) {
                      return 0.0808525889695 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.667350113392
                      return 0.184200475004 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.87349998951
                  if ( median_col_coverage <= 0.995085954666 ) {
                    if ( max_col_coverage <= 0.990846276283 ) {
                      return 0.0405585162755 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.990846276283
                      return 0.0845798418044 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.995085954666
                    if ( mean_col_coverage <= 0.956751346588 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.956751346588
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.978500008583
              if ( max_col_coverage <= 0.994160592556 ) {
                if ( min_col_coverage <= 0.146247655153 ) {
                  if ( min_col_support <= 0.708500027657 ) {
                    if ( mean_col_support <= 0.964970588684 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.964970588684
                      return 0.453078506249 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.708500027657
                    if ( mean_col_support <= 0.984676420689 ) {
                      return 0.371688885042 < maxgini;
                    }
                    else {  // if mean_col_support > 0.984676420689
                      return 0.145429362881 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.146247655153
                  if ( mean_col_support <= 0.98091173172 ) {
                    if ( min_col_coverage <= 0.408782064915 ) {
                      return 0.411878001323 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.408782064915
                      return false;
                    }
                  }
                  else {  // if mean_col_support > 0.98091173172
                    if ( mean_col_support <= 0.992382287979 ) {
                      return 0.121553745148 < maxgini;
                    }
                    else {  // if mean_col_support > 0.992382287979
                      return 0.0213527485572 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.994160592556
                if ( mean_col_coverage <= 0.579393982887 ) {
                  if ( mean_col_support <= 0.980558753014 ) {
                    if ( mean_col_support <= 0.969117641449 ) {
                      return false;
                    }
                    else {  // if mean_col_support > 0.969117641449
                      return 0.484514914977 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.980558753014
                    if ( median_col_coverage <= 0.150630265474 ) {
                      return 0.330989724175 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.150630265474
                      return 0.0782202656155 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.579393982887
                  if ( mean_col_support <= 0.980676472187 ) {
                    if ( min_col_support <= 0.630499958992 ) {
                      return 0.490627360304 < maxgini;
                    }
                    else {  // if min_col_support > 0.630499958992
                      return 0.445733891168 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.980676472187
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.102749292536 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.187014497123 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.409805238247
            if ( mean_col_support <= 0.987735271454 ) {
              if ( mean_col_support <= 0.982676446438 ) {
                if ( min_col_coverage <= 0.65794801712 ) {
                  if ( min_col_coverage <= 0.606281638145 ) {
                    if ( max_col_coverage <= 0.949185609818 ) {
                      return 0.287359972246 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.949185609818
                      return 0.346723292356 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.606281638145
                    if ( min_col_coverage <= 0.657109379768 ) {
                      return 0.368564767232 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.657109379768
                      return 0.296608593217 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.65794801712
                  if ( min_col_support <= 0.745499968529 ) {
                    if ( max_col_support <= 0.997500002384 ) {
                      return 0.0397187728269 < maxgini;
                    }
                    else {  // if max_col_support > 0.997500002384
                      return 0.465089326459 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.745499968529
                    if ( median_col_coverage <= 0.794967532158 ) {
                      return 0.212587794625 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.794967532158
                      return 0.26947776199 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.982676446438
                if ( max_col_coverage <= 0.88589066267 ) {
                  if ( min_col_support <= 0.807500004768 ) {
                    if ( median_col_coverage <= 0.754364192486 ) {
                      return 0.459393031314 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.754364192486
                      return 0.475986245532 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.807500004768
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.0470571809981 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.218331317512 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.88589066267
                  if ( median_col_support <= 0.992499947548 ) {
                    if ( median_col_coverage <= 0.744116842747 ) {
                      return 0.0524761070271 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.744116842747
                      return 0.101353125011 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.992499947548
                    if ( min_col_support <= 0.803499996662 ) {
                      return 0.474627118102 < maxgini;
                    }
                    else {  // if min_col_support > 0.803499996662
                      return 0.259911335466 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.987735271454
              if ( mean_col_support <= 0.991676449776 ) {
                if ( min_col_support <= 0.862499952316 ) {
                  if ( median_col_coverage <= 0.80326205492 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.315057683856 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.436835143469 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.80326205492
                    if ( mean_col_support <= 0.991088211536 ) {
                      return 0.434053727869 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991088211536
                      return 0.473934482721 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.862499952316
                  if ( mean_col_support <= 0.990382373333 ) {
                    if ( min_col_coverage <= 0.669597089291 ) {
                      return 0.0301639950184 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.669597089291
                      return 0.0534542275782 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.990382373333
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0203539262245 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.118760938486 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.991676449776
                if ( median_col_coverage <= 0.703808307648 ) {
                  if ( median_col_support <= 0.993499994278 ) {
                    if ( min_col_support <= 0.885499954224 ) {
                      return 0.444444444444 < maxgini;
                    }
                    else {  // if min_col_support > 0.885499954224
                      return 0.00728992378341 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.993499994278
                    if ( max_col_coverage <= 0.835739076138 ) {
                      return 0.0744579403954 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.835739076138
                      return 0.0182175086333 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.703808307648
                  if ( median_col_coverage <= 0.703817009926 ) {
                    if ( mean_col_coverage <= 0.752975702286 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.752975702286
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.703817009926
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.00963159851906 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.0238591393479 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.833538889885
          if ( mean_col_support <= 0.990029394627 ) {
            if ( mean_col_support <= 0.9860881567 ) {
              if ( median_col_support <= 0.990499973297 ) {
                if ( median_col_coverage <= 0.997755289078 ) {
                  if ( median_col_support <= 0.979499995708 ) {
                    if ( max_col_support <= 0.997500002384 ) {
                      return 0.0151162810932 < maxgini;
                    }
                    else {  // if max_col_support > 0.997500002384
                      return 0.350167436243 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.979499995708
                    if ( min_col_coverage <= 0.882750988007 ) {
                      return 0.392863190629 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.882750988007
                      return 0.415310165031 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.997755289078
                  if ( median_col_coverage <= 0.997933745384 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.997933745384
                    if ( mean_col_coverage <= 0.999875187874 ) {
                      return 0.257639118997 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.999875187874
                      return 0.194056108152 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.990499973297
                if ( median_col_support <= 0.99950003624 ) {
                  if ( max_col_coverage <= 0.983901619911 ) {
                    if ( mean_col_support <= 0.978911757469 ) {
                      return 0.451155118129 < maxgini;
                    }
                    else {  // if mean_col_support > 0.978911757469
                      return 0.390131976569 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.983901619911
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.434045719318 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.452401639252 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.99950003624
                  if ( mean_col_coverage <= 0.983210682869 ) {
                    if ( mean_col_support <= 0.970617651939 ) {
                      return 0.472547189878 < maxgini;
                    }
                    else {  // if mean_col_support > 0.970617651939
                      return 0.481958194135 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.983210682869
                    if ( min_col_support <= 0.775499999523 ) {
                      return 0.453143347275 < maxgini;
                    }
                    else {  // if min_col_support > 0.775499999523
                      return 0.330471013717 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.9860881567
              if ( min_col_coverage <= 0.969311773777 ) {
                if ( median_col_support <= 0.99849998951 ) {
                  if ( min_col_support <= 0.862499952316 ) {
                    if ( min_col_coverage <= 0.966442584991 ) {
                      return 0.337916237595 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.966442584991
                      return 0.46200146092 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.862499952316
                    if ( median_col_coverage <= 0.951761305332 ) {
                      return 0.0810876048702 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.951761305332
                      return 0.150437549795 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.99849998951
                  if ( min_col_support <= 0.845499992371 ) {
                    if ( mean_col_coverage <= 0.954386115074 ) {
                      return 0.472993613331 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.954386115074
                      return 0.462720910786 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.845499992371
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.289395201965 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.17144786625 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.969311773777
                if ( min_col_support <= 0.837499976158 ) {
                  if ( min_col_support <= 0.801499962807 ) {
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.291555583111 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.446316217363 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.801499962807
                    if ( min_col_coverage <= 0.970078468323 ) {
                      return 0.48535021508 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.970078468323
                      return 0.364953179424 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.837499976158
                  if ( median_col_support <= 0.991500020027 ) {
                    if ( median_col_coverage <= 0.975074887276 ) {
                      return 0.0251136762873 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.975074887276
                      return 0.128062990296 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.991500020027
                    if ( max_col_coverage <= 0.998435020447 ) {
                      return 0.283892417311 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.998435020447
                      return 0.184534088841 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.990029394627
            if ( median_col_support <= 0.994500041008 ) {
              if ( min_col_support <= 0.908499956131 ) {
                if ( median_col_support <= 0.993499994278 ) {
                  if ( min_col_support <= 0.862499952316 ) {
                    if ( min_col_support <= 0.861500024796 ) {
                      return 0.324864639733 < maxgini;
                    }
                    else {  // if min_col_support > 0.861500024796
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.862499952316
                    if ( mean_col_coverage <= 0.96512979269 ) {
                      return 0.0989645808956 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.96512979269
                      return 0.149274296455 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( median_col_coverage <= 0.86996781826 ) {
                    if ( min_col_support <= 0.894500017166 ) {
                      return 0.398502596305 < maxgini;
                    }
                    else {  // if min_col_support > 0.894500017166
                      return 0.18 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.86996781826
                    if ( min_col_support <= 0.870499968529 ) {
                      return 0.399818594104 < maxgini;
                    }
                    else {  // if min_col_support > 0.870499968529
                      return 0.181988602125 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.908499956131
                if ( mean_col_coverage <= 0.991145730019 ) {
                  if ( median_col_support <= 0.959499955177 ) {
                    if ( max_col_coverage <= 0.965909123421 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.965909123421
                      return 0.297520661157 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.959499955177
                    if ( min_col_support <= 0.931499958038 ) {
                      return 0.0472202697336 < maxgini;
                    }
                    else {  // if min_col_support > 0.931499958038
                      return 0.00917895717965 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.991145730019
                  if ( median_col_coverage <= 0.996666073799 ) {
                    if ( mean_col_coverage <= 0.998882055283 ) {
                      return 0.0456025072295 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.998882055283
                      return 0.497777777778 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.996666073799
                    if ( mean_col_support <= 0.997676491737 ) {
                      return 0.0277165730099 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997676491737
                      return 0.0896885813149 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.994500041008
              if ( median_col_coverage <= 0.948791146278 ) {
                if ( median_col_coverage <= 0.882519483566 ) {
                  if ( min_col_support <= 0.889500021935 ) {
                    if ( median_col_coverage <= 0.875313818455 ) {
                      return 0.407374455596 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.875313818455
                      return 0.375100341156 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.889500021935
                    if ( mean_col_support <= 0.995029389858 ) {
                      return 0.0662852635932 < maxgini;
                    }
                    else {  // if mean_col_support > 0.995029389858
                      return 0.0103444063953 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.882519483566
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( min_col_support <= 0.916499972343 ) {
                      return 0.28809249431 < maxgini;
                    }
                    else {  // if min_col_support > 0.916499972343
                      return 0.0384499086912 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( min_col_support <= 0.900499999523 ) {
                      return 0.406697674045 < maxgini;
                    }
                    else {  // if min_col_support > 0.900499999523
                      return 0.0163974705148 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.948791146278
                if ( min_col_support <= 0.905499994755 ) {
                  if ( mean_col_coverage <= 0.98217189312 ) {
                    if ( median_col_coverage <= 0.96205496788 ) {
                      return 0.379416166299 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.96205496788
                      return 0.342497805601 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.98217189312
                    if ( min_col_support <= 0.870499968529 ) {
                      return 0.355544552574 < maxgini;
                    }
                    else {  // if min_col_support > 0.870499968529
                      return 0.25539433087 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.905499994755
                  if ( mean_col_support <= 0.995735287666 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.159044115193 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.0412737612591 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.995735287666
                    if ( min_col_support <= 0.942499995232 ) {
                      return 0.0655471466387 < maxgini;
                    }
                    else {  // if min_col_support > 0.942499995232
                      return 0.0101682475865 < maxgini;
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
      if ( min_col_support <= 0.807500004768 ) {
        if ( median_col_support <= 0.979499995708 ) {
          if ( min_col_coverage <= 0.515256226063 ) {
            if ( median_col_support <= 0.640499949455 ) {
              if ( median_col_coverage <= 0.272976756096 ) {
                if ( min_col_coverage <= 0.151547044516 ) {
                  if ( mean_col_coverage <= 0.208680570126 ) {
                    if ( mean_col_coverage <= 0.16969011724 ) {
                      return 0.119001464484 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.16969011724
                      return 0.141064077429 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.208680570126
                    if ( median_col_support <= 0.547500014305 ) {
                      return 0.234376239292 < maxgini;
                    }
                    else {  // if median_col_support > 0.547500014305
                      return 0.162624955074 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.151547044516
                  if ( median_col_coverage <= 0.212732195854 ) {
                    if ( min_col_support <= 0.436500012875 ) {
                      return 0.125213631248 < maxgini;
                    }
                    else {  // if min_col_support > 0.436500012875
                      return 0.210890694217 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.212732195854
                    if ( mean_col_support <= 0.763264775276 ) {
                      return 0.374470900062 < maxgini;
                    }
                    else {  // if mean_col_support > 0.763264775276
                      return 0.262569426681 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.272976756096
                if ( mean_col_coverage <= 0.422246336937 ) {
                  if ( median_col_support <= 0.577499985695 ) {
                    if ( median_col_coverage <= 0.323754489422 ) {
                      return 0.410012296255 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.323754489422
                      return 0.46109709202 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.577499985695
                    if ( median_col_support <= 0.595499992371 ) {
                      return 0.347388926993 < maxgini;
                    }
                    else {  // if median_col_support > 0.595499992371
                      return 0.275407424174 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.422246336937
                  if ( mean_col_coverage <= 0.689116835594 ) {
                    if ( min_col_support <= 0.499500006437 ) {
                      return 0.31995478665 < maxgini;
                    }
                    else {  // if min_col_support > 0.499500006437
                      return 0.46968054976 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.689116835594
                    if ( max_col_coverage <= 0.914589643478 ) {
                      return 0.457328620124 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.914589643478
                      return 0.111844263313 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.640499949455
              if ( mean_col_coverage <= 0.460549384356 ) {
                if ( max_col_coverage <= 0.658595204353 ) {
                  if ( min_col_coverage <= 0.0487209260464 ) {
                    if ( max_col_coverage <= 0.41202712059 ) {
                      return 0.11257675056 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.41202712059
                      return 0.182945945789 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0487209260464
                    if ( min_col_support <= 0.703500032425 ) {
                      return 0.107042103232 < maxgini;
                    }
                    else {  // if min_col_support > 0.703500032425
                      return 0.0736427776317 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.658595204353
                  if ( min_col_support <= 0.633499979973 ) {
                    if ( median_col_support <= 0.947499990463 ) {
                      return 0.213772514143 < maxgini;
                    }
                    else {  // if median_col_support > 0.947499990463
                      return 0.453309283929 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.633499979973
                    if ( max_col_coverage <= 0.796261370182 ) {
                      return 0.121245880303 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.796261370182
                      return 0.231326793773 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.460549384356
                if ( max_col_coverage <= 0.997938036919 ) {
                  if ( min_col_support <= 0.685500025749 ) {
                    if ( min_col_coverage <= 0.394761353731 ) {
                      return 0.167141604781 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.394761353731
                      return 0.212584802407 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.685500025749
                    if ( min_col_coverage <= 0.38239094615 ) {
                      return 0.0871411925641 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.38239094615
                      return 0.110936434664 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.997938036919
                  if ( min_col_support <= 0.640499949455 ) {
                    if ( mean_col_coverage <= 0.857516050339 ) {
                      return 0.397268036921 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.857516050339
                      return 0.176895733611 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.640499949455
                    if ( mean_col_support <= 0.976470589638 ) {
                      return 0.244697051894 < maxgini;
                    }
                    else {  // if mean_col_support > 0.976470589638
                      return 0.440832 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.515256226063
            if ( min_col_coverage <= 0.697024405003 ) {
              if ( median_col_support <= 0.710500001907 ) {
                if ( median_col_support <= 0.598500013351 ) {
                  if ( min_col_support <= 0.497500002384 ) {
                    if ( max_col_coverage <= 0.859858691692 ) {
                      return 0.423511227471 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.859858691692
                      return 0.166004068047 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.497500002384
                    if ( max_col_support <= 0.998000025749 ) {
                      return 0.0509205860994 < maxgini;
                    }
                    else {  // if max_col_support > 0.998000025749
                      return 0.498249083102 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.598500013351
                  if ( min_col_support <= 0.567499995232 ) {
                    if ( mean_col_coverage <= 0.92214679718 ) {
                      return 0.29132483344 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.92214679718
                      return 0.0361867952176 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.567499995232
                    if ( mean_col_support <= 0.830205917358 ) {
                      return 0.499970990846 < maxgini;
                    }
                    else {  // if mean_col_support > 0.830205917358
                      return 0.475620744923 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.710500001907
                if ( median_col_coverage <= 0.667083859444 ) {
                  if ( max_col_coverage <= 0.677521526814 ) {
                    if ( median_col_support <= 0.970499992371 ) {
                      return 0.12599636905 < maxgini;
                    }
                    else {  // if median_col_support > 0.970499992371
                      return 0.251830947092 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.677521526814
                    if ( median_col_support <= 0.960500001907 ) {
                      return 0.185193800526 < maxgini;
                    }
                    else {  // if median_col_support > 0.960500001907
                      return 0.299472043672 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.667083859444
                  if ( median_col_coverage <= 0.672033667564 ) {
                    if ( min_col_coverage <= 0.6277115345 ) {
                      return 0.383256882265 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.6277115345
                      return 0.450959954885 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.672033667564
                    if ( mean_col_support <= 0.952852964401 ) {
                      return 0.28801351392 < maxgini;
                    }
                    else {  // if mean_col_support > 0.952852964401
                      return 0.243543924515 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.697024405003
              if ( max_col_support <= 0.99950003624 ) {
                if ( mean_col_coverage <= 0.811192989349 ) {
                  if ( mean_col_support <= 0.942382335663 ) {
                    if ( max_col_support <= 0.986500024796 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_support > 0.986500024796
                      return 0.244897959184 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.942382335663
                    if ( median_col_coverage <= 0.736602306366 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.736602306366
                      return false;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.811192989349
                  if ( mean_col_support <= 0.948029458523 ) {
                    if ( min_col_support <= 0.611500024796 ) {
                      return 0.00879630973694 < maxgini;
                    }
                    else {  // if min_col_support > 0.611500024796
                      return 0.0345465523488 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.948029458523
                    if ( min_col_support <= 0.660500049591 ) {
                      return 0.48 < maxgini;
                    }
                    else {  // if min_col_support > 0.660500049591
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_support > 0.99950003624
                if ( median_col_coverage <= 0.997931599617 ) {
                  if ( median_col_support <= 0.719500005245 ) {
                    if ( min_col_coverage <= 0.9064874053 ) {
                      return 0.473764917449 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.9064874053
                      return 0.382040816327 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.719500005245
                    if ( min_col_support <= 0.665500044823 ) {
                      return 0.426719517773 < maxgini;
                    }
                    else {  // if min_col_support > 0.665500044823
                      return 0.300301315053 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.997931599617
                  if ( min_col_coverage <= 0.996655166149 ) {
                    if ( mean_col_coverage <= 0.998332023621 ) {
                      return 0.214663836735 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.998332023621
                      return 0.367773309441 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.996655166149
                    if ( mean_col_coverage <= 0.999683260918 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.999683260918
                      return 0.202042013915 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.979499995708
          if ( mean_col_coverage <= 0.367368072271 ) {
            if ( median_col_support <= 0.99950003624 ) {
              if ( mean_col_coverage <= 0.149799361825 ) {
                if ( min_col_support <= 0.540500044823 ) {
                  if ( mean_col_coverage <= 0.102715104818 ) {
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.0794480967454 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.102715104818
                    if ( median_col_coverage <= 0.045179516077 ) {
                      return 0.390497221481 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.045179516077
                      return 0.171523545706 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.540500044823
                  if ( mean_col_coverage <= 0.116402111948 ) {
                    if ( mean_col_coverage <= 0.0783930718899 ) {
                      return 0.04570666446 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0783930718899
                      return 0.107996650737 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.116402111948
                    if ( min_col_support <= 0.605499982834 ) {
                      return 0.212247324614 < maxgini;
                    }
                    else {  // if min_col_support > 0.605499982834
                      return 0.157730747922 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.149799361825
                if ( min_col_coverage <= 0.0789747238159 ) {
                  if ( mean_col_support <= 0.966558814049 ) {
                    if ( median_col_coverage <= 0.0776819884777 ) {
                      return 0.483851422709 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0776819884777
                      return 0.408039662716 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.966558814049
                    if ( mean_col_coverage <= 0.225938752294 ) {
                      return 0.170618959972 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.225938752294
                      return 0.313306353186 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.0789747238159
                  if ( median_col_coverage <= 0.224253565073 ) {
                    if ( min_col_support <= 0.635499954224 ) {
                      return 0.319374457465 < maxgini;
                    }
                    else {  // if min_col_support > 0.635499954224
                      return 0.189969427525 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.224253565073
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.292189349599 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.39084343941 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( max_col_coverage <= 0.363992333412 ) {
                if ( min_col_coverage <= 0.151741161942 ) {
                  if ( max_col_coverage <= 0.273056209087 ) {
                    if ( min_col_support <= 0.551499962807 ) {
                      return 0.106555128076 < maxgini;
                    }
                    else {  // if min_col_support > 0.551499962807
                      return 0.0463237623411 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.273056209087
                    if ( median_col_coverage <= 0.0215635001659 ) {
                      return 0.197745267264 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0215635001659
                      return 0.0775314849961 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.151741161942
                  if ( min_col_support <= 0.676499962807 ) {
                    if ( mean_col_support <= 0.974500060081 ) {
                      return 0.279989648082 < maxgini;
                    }
                    else {  // if mean_col_support > 0.974500060081
                      return 0.426446883512 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.676499962807
                    if ( min_col_support <= 0.74849998951 ) {
                      return 0.150760567158 < maxgini;
                    }
                    else {  // if min_col_support > 0.74849998951
                      return 0.0685661026296 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.363992333412
                if ( min_col_support <= 0.664499998093 ) {
                  if ( max_col_coverage <= 0.57593280077 ) {
                    if ( max_col_coverage <= 0.395287156105 ) {
                      return 0.302467156485 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.395287156105
                      return 0.360308571683 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.57593280077
                    if ( median_col_coverage <= 0.0270728357136 ) {
                      return 0.343743971836 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0270728357136
                      return 0.497034614876 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.664499998093
                  if ( median_col_coverage <= 0.244308933616 ) {
                    if ( min_col_coverage <= 0.0258844029158 ) {
                      return 0.239551857933 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0258844029158
                      return 0.107779791556 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.244308933616
                    if ( mean_col_coverage <= 0.351440519094 ) {
                      return 0.178656290194 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.351440519094
                      return 0.244726375505 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.367368072271
            if ( median_col_support <= 0.99950003624 ) {
              if ( mean_col_support <= 0.974382340908 ) {
                if ( max_col_coverage <= 0.71033847332 ) {
                  if ( median_col_coverage <= 0.450181216002 ) {
                    if ( mean_col_support <= 0.956970572472 ) {
                      return 0.414757004756 < maxgini;
                    }
                    else {  // if mean_col_support > 0.956970572472
                      return 0.359741921219 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.450181216002
                    if ( min_col_coverage <= 0.429859161377 ) {
                      return 0.388729913568 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.429859161377
                      return 0.419306642997 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.71033847332
                  if ( min_col_coverage <= 0.676512956619 ) {
                    if ( median_col_coverage <= 0.208941057324 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.208941057324
                      return 0.43709655858 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.676512956619
                    if ( median_col_support <= 0.990499973297 ) {
                      return 0.443042913857 < maxgini;
                    }
                    else {  // if median_col_support > 0.990499973297
                      return 0.463490884322 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.974382340908
                if ( median_col_coverage <= 0.639657855034 ) {
                  if ( min_col_support <= 0.719500005245 ) {
                    if ( median_col_coverage <= 0.410367965698 ) {
                      return 0.337499899311 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.410367965698
                      return 0.413940547188 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.719500005245
                    if ( median_col_support <= 0.988499999046 ) {
                      return 0.178028553861 < maxgini;
                    }
                    else {  // if median_col_support > 0.988499999046
                      return 0.338059546696 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.639657855034
                  if ( min_col_support <= 0.732499957085 ) {
                    if ( max_col_coverage <= 0.782515645027 ) {
                      return 0.413808684506 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.782515645027
                      return 0.446715189543 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.732499957085
                    if ( mean_col_support <= 0.981794118881 ) {
                      return 0.338242323927 < maxgini;
                    }
                    else {  // if mean_col_support > 0.981794118881
                      return 0.393063858273 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( max_col_coverage <= 0.611789524555 ) {
                if ( min_col_support <= 0.725499987602 ) {
                  if ( min_col_support <= 0.653499960899 ) {
                    if ( max_col_coverage <= 0.50529563427 ) {
                      return 0.44508382531 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.50529563427
                      return 0.467432479276 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.653499960899
                    if ( mean_col_support <= 0.979676485062 ) {
                      return 0.391960378309 < maxgini;
                    }
                    else {  // if mean_col_support > 0.979676485062
                      return 0.470690854264 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.725499987602
                  if ( mean_col_coverage <= 0.455170571804 ) {
                    if ( min_col_support <= 0.772500038147 ) {
                      return 0.313423955436 < maxgini;
                    }
                    else {  // if min_col_support > 0.772500038147
                      return 0.213351202941 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.455170571804
                    if ( min_col_support <= 0.767500042915 ) {
                      return 0.41497089884 < maxgini;
                    }
                    else {  // if min_col_support > 0.767500042915
                      return 0.341685828909 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.611789524555
                if ( min_col_support <= 0.762500047684 ) {
                  if ( median_col_coverage <= 0.516635894775 ) {
                    if ( mean_col_support <= 0.952676415443 ) {
                      return 0.481577282283 < maxgini;
                    }
                    else {  // if mean_col_support > 0.952676415443
                      return 0.458335337886 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.516635894775
                    if ( min_col_coverage <= 0.946037471294 ) {
                      return 0.482263730152 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.946037471294
                      return 0.453505704551 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.762500047684
                  if ( min_col_coverage <= 0.559668838978 ) {
                    if ( median_col_coverage <= 0.476144969463 ) {
                      return 0.322431921201 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.476144969463
                      return 0.410663297443 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.559668838978
                    if ( min_col_coverage <= 0.701705932617 ) {
                      return 0.453926887474 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.701705932617
                      return 0.470557216719 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.807500004768
        if ( min_col_coverage <= 0.794910669327 ) {
          if ( mean_col_support <= 0.992710113525 ) {
            if ( median_col_coverage <= 0.648664116859 ) {
              if ( median_col_support <= 0.989500045776 ) {
                if ( mean_col_coverage <= 0.397447288036 ) {
                  if ( mean_col_support <= 0.962355136871 ) {
                    if ( mean_col_support <= 0.952902793884 ) {
                      return 0.0920619963394 < maxgini;
                    }
                    else {  // if mean_col_support > 0.952902793884
                      return 0.0673148450858 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.962355136871
                    if ( min_col_coverage <= 0.00725163519382 ) {
                      return 0.153485967357 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00725163519382
                      return 0.0421069763501 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.397447288036
                  if ( mean_col_support <= 0.974911749363 ) {
                    if ( max_col_coverage <= 0.667252063751 ) {
                      return 0.0661725105991 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.667252063751
                      return 0.0712634696897 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.974911749363
                    if ( median_col_support <= 0.965499997139 ) {
                      return 0.0382288255104 < maxgini;
                    }
                    else {  // if median_col_support > 0.965499997139
                      return 0.02336430667 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.989500045776
                if ( min_col_support <= 0.862499952316 ) {
                  if ( min_col_coverage <= 0.400123149157 ) {
                    if ( min_col_coverage <= 0.303244382143 ) {
                      return 0.0458851914871 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.303244382143
                      return 0.139148582624 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.400123149157
                    if ( mean_col_support <= 0.988676428795 ) {
                      return 0.218647715608 < maxgini;
                    }
                    else {  // if mean_col_support > 0.988676428795
                      return 0.37314880914 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.862499952316
                  if ( min_col_support <= 0.884500026703 ) {
                    if ( median_col_coverage <= 0.457473129034 ) {
                      return 0.0417769913885 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.457473129034
                      return 0.151115285869 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.884500026703
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.0746089477303 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.0269484429429 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.648664116859
              if ( mean_col_support <= 0.991303324699 ) {
                if ( median_col_support <= 0.992499947548 ) {
                  if ( min_col_support <= 0.869500041008 ) {
                    if ( mean_col_support <= 0.985205888748 ) {
                      return 0.104159834137 < maxgini;
                    }
                    else {  // if mean_col_support > 0.985205888748
                      return 0.165147544454 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.869500041008
                    if ( max_col_coverage <= 0.800359725952 ) {
                      return 0.0254978817266 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.800359725952
                      return 0.0334248140233 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.992499947548
                  if ( min_col_support <= 0.855499982834 ) {
                    if ( min_col_coverage <= 0.701729476452 ) {
                      return 0.370778276978 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.701729476452
                      return 0.394380243101 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.855499982834
                    if ( min_col_support <= 0.881500005722 ) {
                      return 0.227157115258 < maxgini;
                    }
                    else {  // if min_col_support > 0.881500005722
                      return 0.0763179626298 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.991303324699
                if ( max_col_coverage <= 0.795888185501 ) {
                  if ( min_col_coverage <= 0.674510896206 ) {
                    if ( mean_col_coverage <= 0.747102737427 ) {
                      return 0.055671457501 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.747102737427
                      return 0.0178556935314 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.674510896206
                    if ( min_col_coverage <= 0.677835166454 ) {
                      return 0.0215936707461 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.677835166454
                      return 0.0434714839368 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.795888185501
                  if ( median_col_support <= 0.993499994278 ) {
                    if ( min_col_support <= 0.902500033379 ) {
                      return 0.0911955389354 < maxgini;
                    }
                    else {  // if min_col_support > 0.902500033379
                      return 0.0116846130742 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.993499994278
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.212244892313 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.13912741081 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.992710113525
            if ( min_col_support <= 0.908499956131 ) {
              if ( median_col_coverage <= 0.559109687805 ) {
                if ( min_col_coverage <= 0.417962163687 ) {
                  if ( min_col_coverage <= 0.369164764881 ) {
                    if ( max_col_coverage <= 0.438810437918 ) {
                      return 0.0141769365666 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.438810437918
                      return 0.032842114725 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.369164764881
                    if ( min_col_support <= 0.892500042915 ) {
                      return 0.111034727266 < maxgini;
                    }
                    else {  // if min_col_support > 0.892500042915
                      return 0.0436329431954 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.417962163687
                  if ( mean_col_coverage <= 0.547465145588 ) {
                    if ( max_col_coverage <= 0.539230823517 ) {
                      return 0.0439338467365 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.539230823517
                      return 0.0972558225811 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.547465145588
                    if ( min_col_support <= 0.894500017166 ) {
                      return 0.208588934711 < maxgini;
                    }
                    else {  // if min_col_support > 0.894500017166
                      return 0.0930693387252 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.559109687805
                if ( min_col_support <= 0.890499949455 ) {
                  if ( median_col_coverage <= 0.719172954559 ) {
                    if ( min_col_support <= 0.885499954224 ) {
                      return 0.339068949218 < maxgini;
                    }
                    else {  // if min_col_support > 0.885499954224
                      return 0.272793054617 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.719172954559
                    if ( mean_col_coverage <= 0.75551712513 ) {
                      return 0.259791159262 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.75551712513
                      return 0.38839541567 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.890499949455
                  if ( mean_col_coverage <= 0.699365258217 ) {
                    if ( mean_col_coverage <= 0.635174036026 ) {
                      return 0.10366410617 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.635174036026
                      return 0.156637283503 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.699365258217
                    if ( median_col_coverage <= 0.744786143303 ) {
                      return 0.196373980759 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.744786143303
                      return 0.242781909763 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.908499956131
              if ( mean_col_coverage <= 0.519875049591 ) {
                if ( max_col_coverage <= 0.985776901245 ) {
                  if ( min_col_support <= 0.916499972343 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.100854973371 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.019362234869 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.916499972343
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.0185424014497 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.011211366778 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.985776901245
                  if ( min_col_coverage <= 0.0121216569096 ) {
                    if ( min_col_coverage <= 0.00985245592892 ) {
                      return 0.18 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00985245592892
                      return 0.444444444444 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0121216569096
                    if ( min_col_coverage <= 0.0313725508749 ) {
                      return 0.145429362881 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0313725508749
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.519875049591
                if ( median_col_coverage <= 0.128598883748 ) {
                  if ( median_col_coverage <= 0.127932354808 ) {
                    if ( mean_col_coverage <= 0.564467608929 ) {
                      return 0.0165277777778 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.564467608929
                      return 0.226843100189 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.127932354808
                    return false;
                  }
                }
                else {  // if median_col_coverage > 0.128598883748
                  if ( max_col_coverage <= 0.857329010963 ) {
                    if ( median_col_coverage <= 0.465921580791 ) {
                      return 0.0122890263953 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.465921580791
                      return 0.0104553737924 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.857329010963
                    if ( min_col_support <= 0.929499983788 ) {
                      return 0.0778646288672 < maxgini;
                    }
                    else {  // if min_col_support > 0.929499983788
                      return 0.00967920931997 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.794910669327
          if ( min_col_coverage <= 0.947460651398 ) {
            if ( min_col_support <= 0.888499975204 ) {
              if ( min_col_coverage <= 0.853759169579 ) {
                if ( min_col_support <= 0.84249997139 ) {
                  if ( min_col_support <= 0.818500041962 ) {
                    if ( min_col_coverage <= 0.819386065006 ) {
                      return 0.39275002888 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.819386065006
                      return 0.423708781476 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.818500041962
                    if ( max_col_coverage <= 0.898175835609 ) {
                      return 0.339154443408 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.898175835609
                      return 0.37616393678 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.84249997139
                  if ( mean_col_coverage <= 0.843205094337 ) {
                    if ( min_col_support <= 0.865499973297 ) {
                      return 0.258072069934 < maxgini;
                    }
                    else {  // if min_col_support > 0.865499973297
                      return 0.176157965603 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.843205094337
                    if ( min_col_coverage <= 0.852870702744 ) {
                      return 0.279343452615 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.852870702744
                      return 0.214963620834 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.853759169579
                if ( mean_col_support <= 0.988441109657 ) {
                  if ( median_col_support <= 0.993499994278 ) {
                    if ( max_col_coverage <= 0.99841016531 ) {
                      return 0.255798932248 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.99841016531
                      return 0.194597449291 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.993499994278
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.328304466415 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.381957175286 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.988441109657
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.217195419819 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.324501815756 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( median_col_coverage <= 0.966735839844 ) {
                      return 0.438839463209 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.966735839844
                      return 0.364778343745 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.888499975204
              if ( min_col_support <= 0.916499972343 ) {
                if ( min_col_support <= 0.904500007629 ) {
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( mean_col_coverage <= 0.940652430058 ) {
                      return 0.071590350128 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.940652430058
                      return 0.108325031375 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( min_col_support <= 0.895500004292 ) {
                      return 0.256636125338 < maxgini;
                    }
                    else {  // if min_col_support > 0.895500004292
                      return 0.220375647077 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.904500007629
                  if ( min_col_coverage <= 0.834211707115 ) {
                    if ( max_col_coverage <= 0.993609666824 ) {
                      return 0.0958556620642 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.993609666824
                      return 0.0652562741493 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.834211707115
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.0563828754136 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.168324700562 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.916499972343
                if ( median_col_support <= 0.99950003624 ) {
                  if ( median_col_support <= 0.996500015259 ) {
                    if ( mean_col_support <= 0.987970590591 ) {
                      return 0.0385531916491 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987970590591
                      return 0.0113698954327 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.996500015259
                    if ( min_col_support <= 0.949499964714 ) {
                      return 0.159175030085 < maxgini;
                    }
                    else {  // if min_col_support > 0.949499964714
                      return 0.0191004204138 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.99950003624
                  if ( min_col_coverage <= 0.861860275269 ) {
                    if ( min_col_coverage <= 0.824528217316 ) {
                      return 0.0109932300774 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.824528217316
                      return 0.0117396349495 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.861860275269
                    if ( max_col_coverage <= 0.897843241692 ) {
                      return 0.00446028101769 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.897843241692
                      return 0.0130341001518 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.947460651398
            if ( mean_col_coverage <= 0.992641866207 ) {
              if ( median_col_support <= 0.99950003624 ) {
                if ( mean_col_support <= 0.993499994278 ) {
                  if ( max_col_coverage <= 0.998676300049 ) {
                    if ( min_col_support <= 0.909500002861 ) {
                      return 0.31743345658 < maxgini;
                    }
                    else {  // if min_col_support > 0.909500002861
                      return 0.138144706262 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.998676300049
                    if ( mean_col_coverage <= 0.967015147209 ) {
                      return 0.0980975029727 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.967015147209
                      return 0.235029089452 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.993499994278
                  if ( min_col_support <= 0.935500025749 ) {
                    if ( mean_col_coverage <= 0.992627620697 ) {
                      return 0.228423719269 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.992627620697
                      return false;
                    }
                  }
                  else {  // if min_col_support > 0.935500025749
                    if ( min_col_coverage <= 0.947546720505 ) {
                      return 0.32 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.947546720505
                      return 0.0168477014539 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( median_col_coverage <= 0.983418881893 ) {
                  if ( mean_col_support <= 0.992852926254 ) {
                    if ( mean_col_support <= 0.990499973297 ) {
                      return 0.373113234213 < maxgini;
                    }
                    else {  // if mean_col_support > 0.990499973297
                      return 0.290900772587 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992852926254
                    if ( min_col_coverage <= 0.981452822685 ) {
                      return 0.0197012408984 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.981452822685
                      return 0.0688775510204 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.983418881893
                  if ( median_col_coverage <= 0.983448266983 ) {
                    if ( min_col_support <= 0.976500034332 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.976500034332
                      return 0.0 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.983448266983
                    if ( mean_col_support <= 0.991794109344 ) {
                      return 0.354099047952 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991794109344
                      return 0.0550522032919 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.992641866207
              if ( median_col_support <= 0.99950003624 ) {
                if ( min_col_support <= 0.911499977112 ) {
                  if ( max_col_coverage <= 0.99741601944 ) {
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.349454306377 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return false;
                    }
                  }
                  else {  // if max_col_coverage > 0.99741601944
                    if ( min_col_support <= 0.828500032425 ) {
                      return 0.237309130219 < maxgini;
                    }
                    else {  // if min_col_support > 0.828500032425
                      return 0.166673787264 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.911499977112
                  if ( min_col_support <= 0.942499995232 ) {
                    if ( max_col_coverage <= 0.997867703438 ) {
                      return 0.139551441794 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.997867703438
                      return 0.075552987701 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.942499995232
                    if ( min_col_support <= 0.965499997139 ) {
                      return 0.044521570756 < maxgini;
                    }
                    else {  // if min_col_support > 0.965499997139
                      return 0.0188244351467 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_support <= 0.894500017166 ) {
                  if ( median_col_coverage <= 0.980147778988 ) {
                    if ( min_col_support <= 0.81350004673 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.81350004673
                      return 0.336297863822 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.980147778988
                    if ( median_col_coverage <= 0.985781669617 ) {
                      return 0.129799891833 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.985781669617
                      return 0.293103060142 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.894500017166
                  if ( mean_col_support <= 0.995735347271 ) {
                    if ( max_col_coverage <= 0.99898993969 ) {
                      return 0.3046875 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.99898993969
                      return 0.0469575054124 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.995735347271
                    if ( mean_col_support <= 0.996970653534 ) {
                      return 0.0213865896444 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996970653534
                      return 0.010283020015 < maxgini;
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
      if ( min_col_support <= 0.807500004768 ) {
        if ( median_col_support <= 0.979499995708 ) {
          if ( max_col_coverage <= 0.714480876923 ) {
            if ( median_col_coverage <= 0.34290060401 ) {
              if ( mean_col_support <= 0.828355133533 ) {
                if ( median_col_coverage <= 0.243471622467 ) {
                  if ( mean_col_coverage <= 0.243887484074 ) {
                    if ( min_col_support <= 0.50049996376 ) {
                      return 0.112387995983 < maxgini;
                    }
                    else {  // if min_col_support > 0.50049996376
                      return 0.145922073706 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.243887484074
                    if ( median_col_coverage <= 0.18430159986 ) {
                      return 0.176626084248 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.18430159986
                      return 0.230379225478 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.243471622467
                  if ( median_col_support <= 0.577499985695 ) {
                    if ( min_col_support <= 0.499500006437 ) {
                      return 0.298707914879 < maxgini;
                    }
                    else {  // if min_col_support > 0.499500006437
                      return 0.443610459364 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.577499985695
                    if ( min_col_support <= 0.536499977112 ) {
                      return 0.1875822075 < maxgini;
                    }
                    else {  // if min_col_support > 0.536499977112
                      return 0.322267693897 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.828355133533
                if ( median_col_support <= 0.741500020027 ) {
                  if ( mean_col_coverage <= 0.295885413885 ) {
                    if ( min_col_coverage <= 0.0524845644832 ) {
                      return 0.129647675082 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0524845644832
                      return 0.111522945991 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.295885413885
                    if ( median_col_coverage <= 0.285836607218 ) {
                      return 0.161904445873 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.285836607218
                      return 0.212040145743 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.741500020027
                  if ( min_col_coverage <= 0.0487059503794 ) {
                    if ( min_col_support <= 0.667500019073 ) {
                      return 0.122082338275 < maxgini;
                    }
                    else {  // if min_col_support > 0.667500019073
                      return 0.0945369284677 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0487059503794
                    if ( max_col_coverage <= 0.366110950708 ) {
                      return 0.0659982642146 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.366110950708
                      return 0.0842350940566 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.34290060401
              if ( min_col_coverage <= 0.395413577557 ) {
                if ( min_col_support <= 0.62349998951 ) {
                  if ( median_col_support <= 0.62450003624 ) {
                    if ( median_col_support <= 0.558500051498 ) {
                      return 0.492715955591 < maxgini;
                    }
                    else {  // if median_col_support > 0.558500051498
                      return 0.395922740861 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.62450003624
                    if ( mean_col_support <= 0.795823514462 ) {
                      return 0.417791603939 < maxgini;
                    }
                    else {  // if mean_col_support > 0.795823514462
                      return 0.167985473345 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.62349998951
                  if ( min_col_support <= 0.723500013351 ) {
                    if ( median_col_coverage <= 0.34979429841 ) {
                      return 0.184055974551 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.34979429841
                      return 0.137771682771 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.723500013351
                    if ( min_col_support <= 0.740499973297 ) {
                      return 0.101903954326 < maxgini;
                    }
                    else {  // if min_col_support > 0.740499973297
                      return 0.0813726088394 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.395413577557
                if ( max_col_coverage <= 0.550325036049 ) {
                  if ( min_col_support <= 0.590499997139 ) {
                    if ( mean_col_support <= 0.835852980614 ) {
                      return 0.463119075626 < maxgini;
                    }
                    else {  // if mean_col_support > 0.835852980614
                      return 0.212155332888 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.590499997139
                    if ( mean_col_support <= 0.896264731884 ) {
                      return 0.31093077205 < maxgini;
                    }
                    else {  // if mean_col_support > 0.896264731884
                      return 0.0953481708673 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.550325036049
                  if ( mean_col_support <= 0.869029402733 ) {
                    if ( median_col_coverage <= 0.474588572979 ) {
                      return 0.417267738489 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.474588572979
                      return 0.442945327146 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.869029402733
                    if ( median_col_support <= 0.709499955177 ) {
                      return 0.419424526975 < maxgini;
                    }
                    else {  // if median_col_support > 0.709499955177
                      return 0.149493341486 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.714480876923
            if ( min_col_support <= 0.673500001431 ) {
              if ( median_col_support <= 0.943500041962 ) {
                if ( mean_col_coverage <= 0.626937031746 ) {
                  if ( min_col_coverage <= 0.0810029655695 ) {
                    if ( mean_col_support <= 0.922794103622 ) {
                      return 0.130599691992 < maxgini;
                    }
                    else {  // if mean_col_support > 0.922794103622
                      return 0.252923028364 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.0810029655695
                    if ( max_col_coverage <= 0.991701126099 ) {
                      return 0.248595041322 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.991701126099
                      return 0.370905381761 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.626937031746
                  if ( median_col_coverage <= 0.978213429451 ) {
                    if ( min_col_coverage <= 0.705945491791 ) {
                      return 0.320587151297 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.705945491791
                      return 0.408561874132 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.978213429451
                    if ( median_col_support <= 0.851500034332 ) {
                      return 0.124430258752 < maxgini;
                    }
                    else {  // if median_col_support > 0.851500034332
                      return 0.290006892773 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.943500041962
                if ( min_col_coverage <= 0.63892185688 ) {
                  if ( mean_col_support <= 0.947676479816 ) {
                    if ( min_col_coverage <= 0.0480110645294 ) {
                      return 0.182158509861 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.0480110645294
                      return 0.437792763293 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.947676479816
                    if ( max_col_coverage <= 0.946187257767 ) {
                      return 0.347274419782 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.946187257767
                      return 0.433691540473 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.63892185688
                  if ( mean_col_coverage <= 0.839202880859 ) {
                    if ( median_col_coverage <= 0.664292097092 ) {
                      return 0.349124667446 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.664292097092
                      return 0.428159783111 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.839202880859
                    if ( mean_col_support <= 0.967970609665 ) {
                      return 0.456661116802 < maxgini;
                    }
                    else {  // if mean_col_support > 0.967970609665
                      return 0.396581538151 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.673500001431
              if ( median_col_support <= 0.776499986649 ) {
                if ( min_col_support <= 0.736500024796 ) {
                  if ( mean_col_coverage <= 0.5658608675 ) {
                    if ( max_col_coverage <= 0.717047274113 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.717047274113
                      return 0.103795649704 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.5658608675
                    if ( median_col_coverage <= 0.988731622696 ) {
                      return 0.446273307966 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.988731622696
                      return 0.111237836395 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.736500024796
                  if ( median_col_support <= 0.740499973297 ) {
                    if ( min_col_support <= 0.739500045776 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_support > 0.739500045776
                      return false;
                    }
                  }
                  else {  // if median_col_support > 0.740499973297
                    if ( mean_col_support <= 0.941323518753 ) {
                      return 0.349052046854 < maxgini;
                    }
                    else {  // if mean_col_support > 0.941323518753
                      return 0.476846659189 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.776499986649
                if ( median_col_coverage <= 0.795477747917 ) {
                  if ( median_col_support <= 0.963500022888 ) {
                    if ( mean_col_coverage <= 0.728535294533 ) {
                      return 0.117690583067 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.728535294533
                      return 0.168701634732 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.963500022888
                    if ( mean_col_support <= 0.962499976158 ) {
                      return 0.359147406815 < maxgini;
                    }
                    else {  // if mean_col_support > 0.962499976158
                      return 0.213510806144 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.795477747917
                  if ( min_col_coverage <= 0.825112104416 ) {
                    if ( max_col_coverage <= 0.939274072647 ) {
                      return 0.270551508845 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.939274072647
                      return 0.231674066001 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.825112104416
                    if ( mean_col_support <= 0.959970533848 ) {
                      return 0.309259345711 < maxgini;
                    }
                    else {  // if mean_col_support > 0.959970533848
                      return 0.354548717171 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.979499995708
          if ( mean_col_coverage <= 0.372109353542 ) {
            if ( min_col_support <= 0.709499955177 ) {
              if ( mean_col_coverage <= 0.219833940268 ) {
                if ( max_col_coverage <= 0.395199716091 ) {
                  if ( median_col_support <= 0.99849998951 ) {
                    if ( mean_col_coverage <= 0.104472756386 ) {
                      return 0.0931939821727 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.104472756386
                      return 0.216873962954 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99849998951
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.076966982883 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.395199716091
                  if ( max_col_coverage <= 0.46062964201 ) {
                    if ( mean_col_coverage <= 0.19415307045 ) {
                      return 0.315584920916 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.19415307045
                      return 0.194750751754 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.46062964201
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.451350896389 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.368183052143 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.219833940268
                if ( mean_col_support <= 0.948828458786 ) {
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( median_col_support <= 0.982499957085 ) {
                      return 0.402181952663 < maxgini;
                    }
                    else {  // if median_col_support > 0.982499957085
                      return 0.43771099485 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( mean_col_coverage <= 0.270289182663 ) {
                      return 0.352092311367 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.270289182663
                      return 0.433267686133 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.948828458786
                  if ( median_col_coverage <= 0.107609599829 ) {
                    if ( mean_col_support <= 0.9663823843 ) {
                      return 0.494827598802 < maxgini;
                    }
                    else {  // if mean_col_support > 0.9663823843
                      return 0.365399814406 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.107609599829
                    if ( min_col_support <= 0.642500042915 ) {
                      return 0.353248349738 < maxgini;
                    }
                    else {  // if min_col_support > 0.642500042915
                      return 0.262190133438 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_support > 0.709499955177
              if ( min_col_coverage <= 0.189248144627 ) {
                if ( median_col_support <= 0.99849998951 ) {
                  if ( max_col_coverage <= 0.756996154785 ) {
                    if ( mean_col_support <= 0.970382332802 ) {
                      return 0.241529848766 < maxgini;
                    }
                    else {  // if mean_col_support > 0.970382332802
                      return 0.11788264298 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.756996154785
                    if ( median_col_support <= 0.991500020027 ) {
                      return 0.365199704142 < maxgini;
                    }
                    else {  // if median_col_support > 0.991500020027
                      return false;
                    }
                  }
                }
                else {  // if median_col_support > 0.99849998951
                  if ( max_col_coverage <= 0.526572227478 ) {
                    if ( min_col_coverage <= 0.151860505342 ) {
                      return 0.0368455860576 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.151860505342
                      return 0.0813188665616 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.526572227478
                    if ( max_col_coverage <= 0.677430510521 ) {
                      return 0.210889976316 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.677430510521
                      return 0.343057895937 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.189248144627
                if ( mean_col_coverage <= 0.298630177975 ) {
                  if ( mean_col_support <= 0.983029425144 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.19859355687 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.0624076825465 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.983029425144
                    if ( min_col_support <= 0.778499960899 ) {
                      return 0.282147822962 < maxgini;
                    }
                    else {  // if min_col_support > 0.778499960899
                      return 0.116203476843 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.298630177975
                  if ( min_col_support <= 0.760499954224 ) {
                    if ( mean_col_support <= 0.982794046402 ) {
                      return 0.206119993754 < maxgini;
                    }
                    else {  // if mean_col_support > 0.982794046402
                      return 0.408852900424 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.760499954224
                    if ( min_col_support <= 0.790500044823 ) {
                      return 0.167182397707 < maxgini;
                    }
                    else {  // if min_col_support > 0.790500044823
                      return 0.126082654737 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.372109353542
            if ( median_col_support <= 0.99950003624 ) {
              if ( mean_col_coverage <= 0.683682560921 ) {
                if ( median_col_support <= 0.987499952316 ) {
                  if ( median_col_support <= 0.983500003815 ) {
                    if ( min_col_support <= 0.690500020981 ) {
                      return 0.38675901683 < maxgini;
                    }
                    else {  // if min_col_support > 0.690500020981
                      return 0.172195974123 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.983500003815
                    if ( median_col_support <= 0.985499978065 ) {
                      return 0.332314881639 < maxgini;
                    }
                    else {  // if median_col_support > 0.985499978065
                      return 0.363393730523 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.987499952316
                  if ( mean_col_support <= 0.977205872536 ) {
                    if ( median_col_coverage <= 0.150187969208 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.150187969208
                      return 0.428390413938 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.977205872536
                    if ( mean_col_support <= 0.980323433876 ) {
                      return 0.383766586146 < maxgini;
                    }
                    else {  // if mean_col_support > 0.980323433876
                      return 0.34315922065 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.683682560921
                if ( mean_col_support <= 0.977852880955 ) {
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( mean_col_support <= 0.956205844879 ) {
                      return 0.467985181258 < maxgini;
                    }
                    else {  // if mean_col_support > 0.956205844879
                      return 0.41271755 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( max_col_coverage <= 0.983284831047 ) {
                      return 0.45703764213 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.983284831047
                      return 0.466909452952 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.977852880955
                  if ( mean_col_coverage <= 0.840762376785 ) {
                    if ( min_col_coverage <= 0.602696120739 ) {
                      return 0.347000106216 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.602696120739
                      return 0.389235124192 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.840762376785
                    if ( mean_col_support <= 0.982794046402 ) {
                      return 0.426271687628 < maxgini;
                    }
                    else {  // if mean_col_support > 0.982794046402
                      return 0.398929205622 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( max_col_coverage <= 0.607713580132 ) {
                if ( min_col_support <= 0.726500034332 ) {
                  if ( min_col_support <= 0.690500020981 ) {
                    if ( min_col_support <= 0.649500012398 ) {
                      return 0.463081218337 < maxgini;
                    }
                    else {  // if min_col_support > 0.649500012398
                      return 0.446451210798 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.690500020981
                    if ( mean_col_coverage <= 0.463886022568 ) {
                      return 0.380396182855 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.463886022568
                      return 0.44472917578 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.726500034332
                  if ( mean_col_support <= 0.983852863312 ) {
                    if ( min_col_coverage <= 0.371996939182 ) {
                      return 0.194403487036 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.371996939182
                      return 0.309585947056 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.983852863312
                    if ( min_col_support <= 0.757500052452 ) {
                      return 0.479289429875 < maxgini;
                    }
                    else {  // if min_col_support > 0.757500052452
                      return 0.389622995922 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.607713580132
                if ( mean_col_support <= 0.920617640018 ) {
                  if ( min_col_coverage <= 0.903977870941 ) {
                    if ( median_col_coverage <= 0.941742062569 ) {
                      return 0.49892715812 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.941742062569
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.903977870941
                    if ( mean_col_support <= 0.918441176414 ) {
                      return 0.486043771142 < maxgini;
                    }
                    else {  // if mean_col_support > 0.918441176414
                      return 0.324417009602 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.920617640018
                  if ( min_col_coverage <= 0.530240833759 ) {
                    if ( mean_col_support <= 0.96132349968 ) {
                      return 0.47801531812 < maxgini;
                    }
                    else {  // if mean_col_support > 0.96132349968
                      return 0.45116560013 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.530240833759
                    if ( min_col_coverage <= 0.94744515419 ) {
                      return 0.48020061822 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.94744515419
                      return 0.451469769146 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.807500004768
        if ( mean_col_support <= 0.993029356003 ) {
          if ( median_col_coverage <= 0.795068860054 ) {
            if ( median_col_support <= 0.990499973297 ) {
              if ( mean_col_coverage <= 0.44677978754 ) {
                if ( median_col_support <= 0.915500044823 ) {
                  if ( mean_col_support <= 0.948735296726 ) {
                    if ( mean_col_coverage <= 0.287809073925 ) {
                      return 0.110616177492 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.287809073925
                      return 0.0889941625838 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.948735296726
                    if ( mean_col_support <= 0.95349997282 ) {
                      return 0.0772328467672 < maxgini;
                    }
                    else {  // if mean_col_support > 0.95349997282
                      return 0.0648791881318 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.915500044823
                  if ( median_col_coverage <= 0.0192492976785 ) {
                    if ( max_col_coverage <= 0.220894575119 ) {
                      return 0.118729197729 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.220894575119
                      return 0.202634362412 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0192492976785
                    if ( mean_col_support <= 0.982907891273 ) {
                      return 0.0430087877791 < maxgini;
                    }
                    else {  // if mean_col_support > 0.982907891273
                      return 0.0289319014035 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.44677978754
                if ( min_col_coverage <= 0.111375033855 ) {
                  if ( mean_col_coverage <= 0.568574130535 ) {
                    if ( median_col_coverage <= 0.0291986353695 ) {
                      return 0.3078 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0291986353695
                      return 0.0461635620833 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.568574130535
                    if ( median_col_coverage <= 0.133974373341 ) {
                      return 0.460223537147 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.133974373341
                      return 0.15572657311 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.111375033855
                  if ( median_col_coverage <= 0.419900864363 ) {
                    if ( median_col_support <= 0.933500051498 ) {
                      return 0.0707656467114 < maxgini;
                    }
                    else {  // if median_col_support > 0.933500051498
                      return 0.0336063927547 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.419900864363
                    if ( min_col_coverage <= 0.296473503113 ) {
                      return 0.0845904974967 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.296473503113
                      return 0.0379471410999 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.990499973297
              if ( min_col_coverage <= 0.515205264091 ) {
                if ( max_col_coverage <= 0.545950889587 ) {
                  if ( mean_col_coverage <= 0.384144574404 ) {
                    if ( min_col_support <= 0.838500022888 ) {
                      return 0.0461800225258 < maxgini;
                    }
                    else {  // if min_col_support > 0.838500022888
                      return 0.0259662054453 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.384144574404
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.178684207993 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.041545331458 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.545950889587
                  if ( mean_col_support <= 0.990556359291 ) {
                    if ( median_col_coverage <= 0.487879097462 ) {
                      return 0.0840977730398 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.487879097462
                      return 0.13802532243 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.990556359291
                    if ( min_col_support <= 0.880499958992 ) {
                      return 0.244897959184 < maxgini;
                    }
                    else {  // if min_col_support > 0.880499958992
                      return 0.0261818505038 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.515205264091
                if ( min_col_coverage <= 0.648687005043 ) {
                  if ( mean_col_support <= 0.991029381752 ) {
                    if ( median_col_coverage <= 0.617944478989 ) {
                      return 0.172823540019 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.617944478989
                      return 0.219349041364 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.991029381752
                    if ( mean_col_support <= 0.991676449776 ) {
                      return 0.115631723268 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991676449776
                      return 0.073124557099 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.648687005043
                  if ( mean_col_support <= 0.990852952003 ) {
                    if ( mean_col_coverage <= 0.750977277756 ) {
                      return 0.237477309627 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.750977277756
                      return 0.272946007159 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.990852952003
                    if ( min_col_support <= 0.885499954224 ) {
                      return 0.375789184154 < maxgini;
                    }
                    else {  // if min_col_support > 0.885499954224
                      return 0.0532121695181 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.795068860054
            if ( mean_col_support <= 0.991147100925 ) {
              if ( min_col_coverage <= 0.853720188141 ) {
                if ( min_col_support <= 0.862499952316 ) {
                  if ( mean_col_support <= 0.98808825016 ) {
                    if ( mean_col_support <= 0.986441135406 ) {
                      return 0.242151254448 < maxgini;
                    }
                    else {  // if mean_col_support > 0.986441135406
                      return 0.313513072718 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.98808825016
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.33290904756 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.444985726524 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.862499952316
                  if ( min_col_coverage <= 0.764766037464 ) {
                    if ( max_col_coverage <= 0.975900113583 ) {
                      return 0.0414869780207 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.975900113583
                      return 0.0595492445286 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.764766037464
                    if ( mean_col_coverage <= 0.838712453842 ) {
                      return 0.0536723885681 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.838712453842
                      return 0.0765035133029 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.853720188141
                if ( min_col_support <= 0.861500024796 ) {
                  if ( mean_col_coverage <= 0.982759356499 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.315483185761 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.444841285245 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.982759356499
                    if ( mean_col_coverage <= 0.99984139204 ) {
                      return 0.326319512016 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.99984139204
                      return 0.228542021303 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.861500024796
                  if ( median_col_support <= 0.993499994278 ) {
                    if ( median_col_coverage <= 0.946006894112 ) {
                      return 0.0727774763946 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.946006894112
                      return 0.110749398807 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.993499994278
                    if ( median_col_coverage <= 0.997193694115 ) {
                      return 0.204220261181 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.997193694115
                      return 0.118684321636 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.991147100925
              if ( min_col_coverage <= 0.857241749763 ) {
                if ( min_col_support <= 0.887500047684 ) {
                  if ( mean_col_coverage <= 0.810602545738 ) {
                    if ( max_col_coverage <= 0.808608055115 ) {
                      return 0.486111111111 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.808608055115
                      return 0.190131281123 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.810602545738
                    if ( min_col_support <= 0.87349998951 ) {
                      return 0.449171434507 < maxgini;
                    }
                    else {  // if min_col_support > 0.87349998951
                      return 0.334381086666 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.887500047684
                  if ( min_col_support <= 0.903499960899 ) {
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.115630825902 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.193770924842 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.903499960899
                    if ( min_col_coverage <= 0.593333363533 ) {
                      return 0.2112 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.593333363533
                      return 0.0249628801347 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.857241749763
                if ( min_col_coverage <= 0.888967931271 ) {
                  if ( min_col_support <= 0.889500021935 ) {
                    if ( min_col_support <= 0.879500031471 ) {
                      return 0.436925255922 < maxgini;
                    }
                    else {  // if min_col_support > 0.879500031471
                      return 0.318842446712 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.889500021935
                    if ( median_col_support <= 0.994500041008 ) {
                      return 0.0219520231831 < maxgini;
                    }
                    else {  // if median_col_support > 0.994500041008
                      return 0.077748271149 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.888967931271
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( mean_col_support <= 0.99120593071 ) {
                      return 0.0891898361889 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99120593071
                      return 0.0275651453207 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.196230870103 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.255681700556 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.993029356003
          if ( min_col_support <= 0.915500044823 ) {
            if ( mean_col_support <= 0.994382381439 ) {
              if ( mean_col_coverage <= 0.699374258518 ) {
                if ( mean_col_coverage <= 0.510182261467 ) {
                  if ( median_col_coverage <= 0.406739115715 ) {
                    if ( mean_col_coverage <= 0.374274671078 ) {
                      return 0.0159216583927 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.374274671078
                      return 0.0358053814948 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.406739115715
                    if ( min_col_support <= 0.896499991417 ) {
                      return 0.113937707514 < maxgini;
                    }
                    else {  // if min_col_support > 0.896499991417
                      return 0.0426932260675 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.510182261467
                  if ( median_col_coverage <= 0.559109687805 ) {
                    if ( max_col_coverage <= 0.56840467453 ) {
                      return 0.0314205607083 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.56840467453
                      return 0.102786372103 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.559109687805
                    if ( min_col_support <= 0.896499991417 ) {
                      return 0.231962492151 < maxgini;
                    }
                    else {  // if min_col_support > 0.896499991417
                      return 0.105948824314 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.699374258518
                if ( min_col_support <= 0.904500007629 ) {
                  if ( min_col_support <= 0.893499970436 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.294403747327 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.375344352454 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.893499970436
                    if ( min_col_coverage <= 0.829238057137 ) {
                      return 0.253381417093 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.829238057137
                      return 0.315931474143 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.904500007629
                  if ( median_col_coverage <= 0.891852378845 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.19026175033 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.110794032737 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.891852378845
                    if ( min_col_coverage <= 0.975111484528 ) {
                      return 0.207943234133 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.975111484528
                      return 0.113736962754 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.994382381439
              if ( median_col_support <= 0.99950003624 ) {
                if ( mean_col_support <= 0.994558811188 ) {
                  if ( min_col_support <= 0.914499998093 ) {
                    if ( max_col_coverage <= 0.991675496101 ) {
                      return 0.222148760331 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.991675496101
                      return 0.493469387755 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.914499998093
                    if ( mean_col_support <= 0.994441151619 ) {
                      return 0.1472 < maxgini;
                    }
                    else {  // if mean_col_support > 0.994441151619
                      return 0.240129799892 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.994558811188
                  if ( min_col_coverage <= 0.923970222473 ) {
                    if ( min_col_support <= 0.913499951363 ) {
                      return 0.0570934256055 < maxgini;
                    }
                    else {  // if min_col_support > 0.913499951363
                      return 0.160512 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.923970222473
                    if ( mean_col_coverage <= 0.97841334343 ) {
                      return 0.48347107438 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.97841334343
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_coverage <= 0.639672160149 ) {
                  if ( mean_col_support <= 0.994735240936 ) {
                    if ( mean_col_coverage <= 0.58135509491 ) {
                      return 0.0296173476524 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.58135509491
                      return 0.111851291514 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994735240936
                    if ( max_col_coverage <= 0.604957163334 ) {
                      return 0.0206618585066 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.604957163334
                      return 0.0674767685535 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.639672160149
                  if ( mean_col_support <= 0.994676470757 ) {
                    if ( max_col_coverage <= 0.901613295078 ) {
                      return 0.157680647907 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.901613295078
                      return 0.262482865595 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994676470757
                    if ( min_col_support <= 0.911499977112 ) {
                      return 0.19080603071 < maxgini;
                    }
                    else {  // if min_col_support > 0.911499977112
                      return 0.168158326268 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.915500044823
            if ( max_col_coverage <= 0.954646110535 ) {
              if ( min_col_support <= 0.933500051498 ) {
                if ( mean_col_coverage <= 0.675925731659 ) {
                  if ( min_col_support <= 0.923500001431 ) {
                    if ( mean_col_coverage <= 0.572780787945 ) {
                      return 0.0186214085041 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.572780787945
                      return 0.0433173601751 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.923500001431
                    if ( min_col_coverage <= 0.430478751659 ) {
                      return 0.0126329475482 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.430478751659
                      return 0.0229021996772 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.675925731659
                  if ( mean_col_support <= 0.994617640972 ) {
                    if ( median_col_coverage <= 0.800218939781 ) {
                      return 0.0373179702804 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.800218939781
                      return 0.0558361809114 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994617640972
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.188480820497 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.0612257129433 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.933500051498
                if ( min_col_support <= 0.953500032425 ) {
                  if ( max_col_coverage <= 0.658141136169 ) {
                    if ( mean_col_support <= 0.994436740875 ) {
                      return 0.0125505932045 < maxgini;
                    }
                    else {  // if mean_col_support > 0.994436740875
                      return 0.00984587490933 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.658141136169
                    if ( median_col_coverage <= 0.839110374451 ) {
                      return 0.0150563339085 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.839110374451
                      return 0.0224482967905 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.953500032425
                  if ( median_col_support <= 0.978500008583 ) {
                    if ( mean_col_support <= 0.994323551655 ) {
                      return 0.0193481153858 < maxgini;
                    }
                    else {  // if mean_col_support > 0.994323551655
                      return 0.0385084150101 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.978500008583
                    if ( median_col_coverage <= 0.440315723419 ) {
                      return 0.0127344253482 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.440315723419
                      return 0.00739217259878 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.954646110535
              if ( min_col_support <= 0.935500025749 ) {
                if ( min_col_support <= 0.922500014305 ) {
                  if ( mean_col_coverage <= 0.922288775444 ) {
                    if ( max_col_coverage <= 0.969588160515 ) {
                      return 0.120828558551 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.969588160515
                      return 0.0664344627883 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.922288775444
                    if ( min_col_coverage <= 0.997929334641 ) {
                      return 0.150610313118 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.997929334641
                      return 0.0763497273892 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.922500014305
                  if ( min_col_coverage <= 0.830008268356 ) {
                    if ( median_col_coverage <= 0.0110809179023 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.0110809179023
                      return 0.0457153656972 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.830008268356
                    if ( max_col_coverage <= 0.998342990875 ) {
                      return 0.0982926490503 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.998342990875
                      return 0.0727650395899 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.935500025749
                if ( max_col_coverage <= 0.954651236534 ) {
                  if ( median_col_coverage <= 0.797052145004 ) {
                    return 0.0 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.797052145004
                    if ( mean_col_coverage <= 0.87855142355 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.87855142355
                      return 0.0 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.954651236534
                  if ( min_col_coverage <= 0.308101475239 ) {
                    if ( mean_col_coverage <= 0.319915652275 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.319915652275
                      return 0.0671700376041 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.308101475239
                    if ( max_col_coverage <= 0.955185413361 ) {
                      return 0.0508128544423 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.955185413361
                      return 0.0101038900703 < maxgini;
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
      if ( median_col_coverage <= 0.705911517143 ) {
        if ( mean_col_support <= 0.987171530724 ) {
          if ( mean_col_support <= 0.814656853676 ) {
            if ( max_col_coverage <= 0.472548007965 ) {
              if ( mean_col_coverage <= 0.292505741119 ) {
                if ( median_col_support <= 0.558500051498 ) {
                  if ( mean_col_coverage <= 0.231979459524 ) {
                    if ( max_col_coverage <= 0.212233453989 ) {
                      return 0.119120893375 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.212233453989
                      return 0.167681262082 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.231979459524
                    if ( median_col_coverage <= 0.178968295455 ) {
                      return 0.21394794745 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.178968295455
                      return 0.319276572307 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.558500051498
                  if ( min_col_coverage <= 0.151804238558 ) {
                    if ( min_col_support <= 0.50049996376 ) {
                      return 0.086965458093 < maxgini;
                    }
                    else {  // if min_col_support > 0.50049996376
                      return 0.144627091015 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.151804238558
                    if ( max_col_coverage <= 0.183473393321 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.183473393321
                      return 0.197767503285 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.292505741119
                if ( median_col_support <= 0.547500014305 ) {
                  if ( min_col_support <= 0.481499999762 ) {
                    if ( min_col_coverage <= 0.243436902761 ) {
                      return 0.274524926124 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.243436902761
                      return 0.360210791767 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.481499999762
                    if ( mean_col_support <= 0.72932356596 ) {
                      return 0.499197709378 < maxgini;
                    }
                    else {  // if mean_col_support > 0.72932356596
                      return 0.45482412473 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.547500014305
                  if ( min_col_support <= 0.523499965668 ) {
                    if ( median_col_support <= 0.577499985695 ) {
                      return 0.305108961844 < maxgini;
                    }
                    else {  // if median_col_support > 0.577499985695
                      return 0.183202005239 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.523499965668
                    if ( min_col_support <= 0.524500012398 ) {
                      return 0.474129197443 < maxgini;
                    }
                    else {  // if min_col_support > 0.524500012398
                      return 0.353458441539 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.472548007965
              if ( median_col_coverage <= 0.365049183369 ) {
                if ( max_col_coverage <= 0.669398903847 ) {
                  if ( min_col_coverage <= 0.212443590164 ) {
                    if ( mean_col_support <= 0.762676477432 ) {
                      return 0.379767517356 < maxgini;
                    }
                    else {  // if mean_col_support > 0.762676477432
                      return 0.190504497769 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.212443590164
                    if ( median_col_coverage <= 0.305839002132 ) {
                      return 0.349969439838 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.305839002132
                      return 0.407672160648 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.669398903847
                  if ( mean_col_coverage <= 0.532995045185 ) {
                    if ( min_col_support <= 0.513499975204 ) {
                      return 0.0751201604261 < maxgini;
                    }
                    else {  // if min_col_support > 0.513499975204
                      return 0.2289058219 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.532995045185
                    if ( max_col_coverage <= 0.870899438858 ) {
                      return 0.277777777778 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.870899438858
                      return 0.029893444884 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.365049183369
                if ( min_col_support <= 0.499500006437 ) {
                  if ( max_col_support <= 0.995000004768 ) {
                    if ( median_col_support <= 0.466000020504 ) {
                      return 0.456747404844 < maxgini;
                    }
                    else {  // if median_col_support > 0.466000020504
                      return 0.0663667174455 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.995000004768
                    if ( mean_col_support <= 0.756676495075 ) {
                      return 0.452169028607 < maxgini;
                    }
                    else {  // if mean_col_support > 0.756676495075
                      return 0.327088503639 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.499500006437
                  if ( max_col_coverage <= 0.914889335632 ) {
                    if ( median_col_support <= 0.566499948502 ) {
                      return false;
                    }
                    else {  // if median_col_support > 0.566499948502
                      return 0.467289562755 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.914889335632
                    if ( median_col_coverage <= 0.696002602577 ) {
                      return 0.0493043043801 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.696002602577
                      return 0.414039262344 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.814656853676
            if ( max_col_coverage <= 0.667120456696 ) {
              if ( min_col_coverage <= 0.395377904177 ) {
                if ( min_col_support <= 0.71749997139 ) {
                  if ( mean_col_support <= 0.970585763454 ) {
                    if ( max_col_coverage <= 0.42878562212 ) {
                      return 0.107974097376 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.42878562212
                      return 0.185644590892 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.970585763454
                    if ( mean_col_coverage <= 0.279508858919 ) {
                      return 0.140159878091 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.279508858919
                      return 0.410525828705 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.71749997139
                  if ( min_col_support <= 0.81350004673 ) {
                    if ( min_col_support <= 0.779500007629 ) {
                      return 0.0919881930547 < maxgini;
                    }
                    else {  // if min_col_support > 0.779500007629
                      return 0.0717780295445 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.81350004673
                    if ( min_col_coverage <= 0.00725163519382 ) {
                      return 0.150832559049 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00725163519382
                      return 0.0483264006696 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.395377904177
                if ( mean_col_coverage <= 0.529488563538 ) {
                  if ( min_col_support <= 0.737499952316 ) {
                    if ( min_col_support <= 0.660500049591 ) {
                      return 0.357693620472 < maxgini;
                    }
                    else {  // if min_col_support > 0.660500049591
                      return 0.265741432088 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.737499952316
                    if ( min_col_support <= 0.799499988556 ) {
                      return 0.171920934026 < maxgini;
                    }
                    else {  // if min_col_support > 0.799499988556
                      return 0.0539741693336 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.529488563538
                  if ( mean_col_support <= 0.981147050858 ) {
                    if ( min_col_support <= 0.705500006676 ) {
                      return 0.385192489042 < maxgini;
                    }
                    else {  // if min_col_support > 0.705500006676
                      return 0.100853992731 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.981147050858
                    if ( min_col_support <= 0.793500006199 ) {
                      return 0.446790202695 < maxgini;
                    }
                    else {  // if min_col_support > 0.793500006199
                      return 0.0505412002287 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.667120456696
              if ( mean_col_support <= 0.981323599815 ) {
                if ( min_col_support <= 0.723500013351 ) {
                  if ( min_col_support <= 0.682500004768 ) {
                    if ( min_col_support <= 0.499500006437 ) {
                      return 0.273213215195 < maxgini;
                    }
                    else {  // if min_col_support > 0.499500006437
                      return 0.431902583545 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.682500004768
                    if ( mean_col_support <= 0.96226477623 ) {
                      return 0.220713320907 < maxgini;
                    }
                    else {  // if mean_col_support > 0.96226477623
                      return 0.398890869548 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.723500013351
                  if ( median_col_coverage <= 0.606084942818 ) {
                    if ( mean_col_support <= 0.916441202164 ) {
                      return 0.280088747133 < maxgini;
                    }
                    else {  // if mean_col_support > 0.916441202164
                      return 0.103018042048 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.606084942818
                    if ( min_col_support <= 0.803499996662 ) {
                      return 0.26508926198 < maxgini;
                    }
                    else {  // if min_col_support > 0.803499996662
                      return 0.0891221690064 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.981323599815
                if ( median_col_support <= 0.991500020027 ) {
                  if ( min_col_support <= 0.808500051498 ) {
                    if ( mean_col_coverage <= 0.687447428703 ) {
                      return 0.264424398512 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.687447428703
                      return 0.310318161126 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.808500051498
                    if ( min_col_support <= 0.850499987602 ) {
                      return 0.112126716014 < maxgini;
                    }
                    else {  // if min_col_support > 0.850499987602
                      return 0.0314649819334 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.991500020027
                  if ( median_col_coverage <= 0.559215426445 ) {
                    if ( min_col_coverage <= 0.442120879889 ) {
                      return 0.23778216409 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.442120879889
                      return 0.339897301224 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.559215426445
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.32623062479 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.426003534634 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.987171530724
          if ( median_col_support <= 0.992499947548 ) {
            if ( median_col_support <= 0.970499992371 ) {
              if ( min_col_coverage <= 0.353101015091 ) {
                if ( mean_col_coverage <= 0.0555707328022 ) {
                  if ( max_col_coverage <= 0.0569358170033 ) {
                    if ( mean_col_coverage <= 0.0459079295397 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.0459079295397
                      return false;
                    }
                  }
                  else {  // if max_col_coverage > 0.0569358170033
                    if ( min_col_support <= 0.949999988079 ) {
                      return 0.104938271605 < maxgini;
                    }
                    else {  // if min_col_support > 0.949999988079
                      return false;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.0555707328022
                  if ( mean_col_support <= 0.992323517799 ) {
                    if ( min_col_coverage <= 0.00738258706406 ) {
                      return 0.3046875 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.00738258706406
                      return 0.0457454600365 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992323517799
                    if ( min_col_coverage <= 0.349418580532 ) {
                      return 0.157413978618 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.349418580532
                      return 0.486111111111 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.353101015091
                if ( mean_col_support <= 0.992617666721 ) {
                  if ( median_col_coverage <= 0.511369526386 ) {
                    if ( min_col_support <= 0.953500032425 ) {
                      return 0.0259416714224 < maxgini;
                    }
                    else {  // if min_col_support > 0.953500032425
                      return 0.0429572909944 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.511369526386
                    if ( mean_col_coverage <= 0.786679208279 ) {
                      return 0.0246989176005 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.786679208279
                      return 0.0575723359425 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.992617666721
                  if ( min_col_coverage <= 0.451680660248 ) {
                    if ( min_col_coverage <= 0.447088479996 ) {
                      return 0.289627465303 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.447088479996
                      return false;
                    }
                  }
                  else {  // if min_col_coverage > 0.451680660248
                    if ( mean_col_coverage <= 0.686747610569 ) {
                      return 0.058769513315 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.686747610569
                      return 0.0 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.970499992371
              if ( median_col_support <= 0.990499973297 ) {
                if ( mean_col_support <= 0.990656971931 ) {
                  if ( min_col_support <= 0.871500015259 ) {
                    if ( mean_col_coverage <= 0.64239013195 ) {
                      return 0.069030168239 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.64239013195
                      return 0.132158484453 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.871500015259
                    if ( median_col_support <= 0.988499999046 ) {
                      return 0.0170384138206 < maxgini;
                    }
                    else {  // if median_col_support > 0.988499999046
                      return 0.0343503272664 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.990656971931
                  if ( median_col_coverage <= 0.382230997086 ) {
                    if ( mean_col_support <= 0.9952647686 ) {
                      return 0.021305650039 < maxgini;
                    }
                    else {  // if mean_col_support > 0.9952647686
                      return 0.0398837156602 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.382230997086
                    if ( min_col_coverage <= 0.524602770805 ) {
                      return 0.0114606913393 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.524602770805
                      return 0.00885773068267 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.990499973297
                if ( min_col_support <= 0.87349998951 ) {
                  if ( mean_col_coverage <= 0.319004774094 ) {
                    if ( median_col_coverage <= 0.00253173452802 ) {
                      return 0.21875 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.00253173452802
                      return 0.00909071954296 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.319004774094
                    if ( min_col_support <= 0.827499985695 ) {
                      return 0.413622584633 < maxgini;
                    }
                    else {  // if min_col_support > 0.827499985695
                      return 0.190983528358 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.87349998951
                  if ( mean_col_coverage <= 0.571602463722 ) {
                    if ( mean_col_coverage <= 0.571590423584 ) {
                      return 0.024937651357 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.571590423584
                      return false;
                    }
                  }
                  else {  // if mean_col_coverage > 0.571602463722
                    if ( min_col_coverage <= 0.125525206327 ) {
                      return 0.329861111111 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.125525206327
                      return 0.0121030062072 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.992499947548
            if ( mean_col_coverage <= 0.591997385025 ) {
              if ( median_col_support <= 0.99950003624 ) {
                if ( mean_col_support <= 0.991970539093 ) {
                  if ( min_col_support <= 0.871500015259 ) {
                    if ( median_col_coverage <= 0.308098733425 ) {
                      return 0.128203108591 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.308098733425
                      return 0.260418506652 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.871500015259
                    if ( min_col_coverage <= 0.482788234949 ) {
                      return 0.119173553719 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.482788234949
                      return 0.208635058196 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.991970539093
                  if ( mean_col_support <= 0.994617581367 ) {
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.0416070140467 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return 0.158401685654 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994617581367
                    if ( max_col_coverage <= 0.341645956039 ) {
                      return 0.0313045370347 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.341645956039
                      return 0.0153815782857 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_coverage <= 0.372641146183 ) {
                  if ( max_col_coverage <= 0.929479241371 ) {
                    if ( mean_col_support <= 0.991656899452 ) {
                      return 0.0337216893153 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991656899452
                      return 0.0142092454521 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.929479241371
                    if ( mean_col_support <= 0.992794156075 ) {
                      return 0.243448518686 < maxgini;
                    }
                    else {  // if mean_col_support > 0.992794156075
                      return 0.0627143347051 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.372641146183
                  if ( max_col_coverage <= 0.551677286625 ) {
                    if ( median_col_coverage <= 0.372699022293 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.372699022293
                      return 0.0209654232793 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.551677286625
                    if ( min_col_support <= 0.855499982834 ) {
                      return 0.368273547041 < maxgini;
                    }
                    else {  // if min_col_support > 0.855499982834
                      return 0.0171025101633 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.591997385025
              if ( max_col_coverage <= 0.757623195648 ) {
                if ( mean_col_support <= 0.991029381752 ) {
                  if ( min_col_support <= 0.849500000477 ) {
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.312339576822 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.437631409179 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.849500000477
                    if ( min_col_coverage <= 0.488861322403 ) {
                      return 0.0497855705136 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.488861322403
                      return 0.0819414059437 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.991029381752
                  if ( min_col_support <= 0.889500021935 ) {
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.161934201419 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.306590440033 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.889500021935
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.033347284388 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.0125346787016 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.757623195648
                if ( min_col_support <= 0.872500002384 ) {
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( min_col_support <= 0.835500001907 ) {
                      return 0.354603120655 < maxgini;
                    }
                    else {  // if min_col_support > 0.835500001907
                      return 0.2774217951 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( mean_col_coverage <= 0.650669455528 ) {
                      return 0.324652468769 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.650669455528
                      return 0.403438674237 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.872500002384
                  if ( mean_col_support <= 0.994242727757 ) {
                    if ( min_col_support <= 0.904500007629 ) {
                      return 0.157832929834 < maxgini;
                    }
                    else {  // if min_col_support > 0.904500007629
                      return 0.028603058724 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994242727757
                    if ( min_col_support <= 0.927500009537 ) {
                      return 0.0782643316022 < maxgini;
                    }
                    else {  // if min_col_support > 0.927500009537
                      return 0.00895304470806 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if median_col_coverage > 0.705911517143
        if ( median_col_coverage <= 0.829306364059 ) {
          if ( median_col_coverage <= 0.76477086544 ) {
            if ( mean_col_support <= 0.987264692783 ) {
              if ( median_col_support <= 0.990499973297 ) {
                if ( min_col_support <= 0.747500002384 ) {
                  if ( mean_col_support <= 0.900794148445 ) {
                    if ( max_col_coverage <= 0.960769176483 ) {
                      return 0.457009628395 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.960769176483
                      return 0.343833635764 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.900794148445
                    if ( mean_col_coverage <= 0.74564152956 ) {
                      return 0.308925619835 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.74564152956
                      return 0.383529677426 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.747500002384
                  if ( max_col_coverage <= 0.830036282539 ) {
                    if ( min_col_support <= 0.827499985695 ) {
                      return 0.143836051313 < maxgini;
                    }
                    else {  // if min_col_support > 0.827499985695
                      return 0.0413468953893 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.830036282539
                    if ( min_col_coverage <= 0.667006850243 ) {
                      return 0.0633368356388 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.667006850243
                      return 0.113088018679 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.990499973297
                if ( mean_col_support <= 0.982558846474 ) {
                  if ( max_col_coverage <= 0.981776714325 ) {
                    if ( min_col_coverage <= 0.748903512955 ) {
                      return 0.469029424998 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.748903512955
                      return 0.481133827879 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.981776714325
                    if ( max_col_coverage <= 0.993071556091 ) {
                      return 0.376802427741 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.993071556091
                      return 0.443330865005 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.982558846474
                  if ( median_col_support <= 0.99849998951 ) {
                    if ( mean_col_support <= 0.986264705658 ) {
                      return 0.357763806686 < maxgini;
                    }
                    else {  // if mean_col_support > 0.986264705658
                      return 0.313407771693 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99849998951
                    if ( min_col_coverage <= 0.626992166042 ) {
                      return 0.376218910322 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.626992166042
                      return 0.4530410779 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.987264692783
              if ( min_col_support <= 0.879500031471 ) {
                if ( median_col_coverage <= 0.743860960007 ) {
                  if ( mean_col_support <= 0.989911794662 ) {
                    if ( max_col_coverage <= 0.804915785789 ) {
                      return 0.395362236893 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.804915785789
                      return 0.362394058566 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.989911794662
                    if ( mean_col_support <= 0.992852926254 ) {
                      return 0.396735235687 < maxgini;
                    }
                    else {  // if mean_col_support > 0.992852926254
                      return 0.46820415879 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.743860960007
                  if ( max_col_coverage <= 0.957640886307 ) {
                    if ( min_col_support <= 0.848500013351 ) {
                      return 0.458394393346 < maxgini;
                    }
                    else {  // if min_col_support > 0.848500013351
                      return 0.309089529762 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.957640886307
                    if ( min_col_coverage <= 0.734253406525 ) {
                      return 0.32362794176 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.734253406525
                      return 0.415873038691 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.879500031471
                if ( median_col_support <= 0.99950003624 ) {
                  if ( mean_col_coverage <= 0.756859540939 ) {
                    if ( mean_col_support <= 0.993558824062 ) {
                      return 0.019347705799 < maxgini;
                    }
                    else {  // if mean_col_support > 0.993558824062
                      return 0.005771032144 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.756859540939
                    if ( median_col_support <= 0.996500015259 ) {
                      return 0.0159350255896 < maxgini;
                    }
                    else {  // if median_col_support > 0.996500015259
                      return 0.0991771003212 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.99950003624
                  if ( max_col_coverage <= 0.763022243977 ) {
                    if ( min_col_coverage <= 0.634380817413 ) {
                      return 0.0190718011689 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.634380817413
                      return 0.0103029324372 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.763022243977
                    if ( mean_col_coverage <= 0.716413497925 ) {
                      return 0.0328266721316 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.716413497925
                      return 0.0165118519581 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.76477086544
            if ( max_col_coverage <= 0.886394619942 ) {
              if ( min_col_support <= 0.831499993801 ) {
                if ( mean_col_coverage <= 0.812476754189 ) {
                  if ( max_col_coverage <= 0.813779115677 ) {
                    if ( mean_col_support <= 0.97620588541 ) {
                      return 0.383990091232 < maxgini;
                    }
                    else {  // if mean_col_support > 0.97620588541
                      return 0.455900038917 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.813779115677
                    if ( min_col_support <= 0.768499970436 ) {
                      return 0.46624929529 < maxgini;
                    }
                    else {  // if min_col_support > 0.768499970436
                      return 0.376706464628 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.812476754189
                  if ( min_col_coverage <= 0.80490309 ) {
                    if ( min_col_support <= 0.779500007629 ) {
                      return 0.468140668231 < maxgini;
                    }
                    else {  // if min_col_support > 0.779500007629
                      return 0.384636711573 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.80490309
                    if ( min_col_coverage <= 0.828759431839 ) {
                      return 0.43805202375 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.828759431839
                      return 0.492694119068 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.831499993801
                if ( mean_col_support <= 0.993852853775 ) {
                  if ( min_col_support <= 0.895500004292 ) {
                    if ( median_col_support <= 0.992499947548 ) {
                      return 0.0758585679236 < maxgini;
                    }
                    else {  // if median_col_support > 0.992499947548
                      return 0.312574265553 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.895500004292
                    if ( min_col_support <= 0.914499998093 ) {
                      return 0.0630996331649 < maxgini;
                    }
                    else {  // if min_col_support > 0.914499998093
                      return 0.0148663036927 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.993852853775
                  if ( min_col_coverage <= 0.730142116547 ) {
                    if ( max_col_coverage <= 0.770032048225 ) {
                      return 0.044095200407 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.770032048225
                      return 0.00911162796231 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.730142116547
                    if ( median_col_support <= 0.974500000477 ) {
                      return 0.290657439446 < maxgini;
                    }
                    else {  // if median_col_support > 0.974500000477
                      return 0.0113240458414 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.886394619942
              if ( mean_col_support <= 0.987970590591 ) {
                if ( median_col_support <= 0.990499973297 ) {
                  if ( mean_col_support <= 0.973735332489 ) {
                    if ( median_col_support <= 0.970499992371 ) {
                      return 0.327858916211 < maxgini;
                    }
                    else {  // if median_col_support > 0.970499992371
                      return 0.443788927336 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.973735332489
                    if ( min_col_support <= 0.798500001431 ) {
                      return 0.351185364448 < maxgini;
                    }
                    else {  // if min_col_support > 0.798500001431
                      return 0.0784259112154 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.990499973297
                  if ( min_col_coverage <= 0.799799084663 ) {
                    if ( mean_col_support <= 0.982676446438 ) {
                      return 0.470166882457 < maxgini;
                    }
                    else {  // if mean_col_support > 0.982676446438
                      return 0.42654269935 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.799799084663
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.437106473456 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.481051814088 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.987970590591
                if ( mean_col_support <= 0.992205858231 ) {
                  if ( max_col_coverage <= 0.965432524681 ) {
                    if ( median_col_support <= 0.993499994278 ) {
                      return 0.0351109144024 < maxgini;
                    }
                    else {  // if median_col_support > 0.993499994278
                      return 0.304882012335 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.965432524681
                    if ( max_col_coverage <= 0.973663806915 ) {
                      return 0.0931384220908 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.973663806915
                      return 0.156435333265 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.992205858231
                  if ( mean_col_support <= 0.994617640972 ) {
                    if ( min_col_coverage <= 0.736904501915 ) {
                      return 0.0430915875254 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.736904501915
                      return 0.0648445557016 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994617640972
                    if ( mean_col_support <= 0.996088266373 ) {
                      return 0.0220085319104 < maxgini;
                    }
                    else {  // if mean_col_support > 0.996088266373
                      return 0.00826575999441 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.829306364059
          if ( median_col_support <= 0.99950003624 ) {
            if ( mean_col_coverage <= 0.95099568367 ) {
              if ( min_col_coverage <= 0.866752922535 ) {
                if ( min_col_coverage <= 0.789607167244 ) {
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( mean_col_support <= 0.980676472187 ) {
                      return 0.303044875364 < maxgini;
                    }
                    else {  // if mean_col_support > 0.980676472187
                      return 0.0414089010704 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( mean_col_support <= 0.991147041321 ) {
                      return 0.452572684295 < maxgini;
                    }
                    else {  // if mean_col_support > 0.991147041321
                      return 0.0500061414436 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.789607167244
                  if ( mean_col_support <= 0.983911752701 ) {
                    if ( max_col_coverage <= 0.889154016972 ) {
                      return 0.364284723007 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.889154016972
                      return 0.407758893901 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.983911752701
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.0459415682718 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.196156708075 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.866752922535
                if ( min_col_support <= 0.845499992371 ) {
                  if ( max_col_coverage <= 0.941266179085 ) {
                    if ( min_col_coverage <= 0.870050311089 ) {
                      return 0.456239153093 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.870050311089
                      return 0.404605574055 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.941266179085
                    if ( max_col_support <= 0.99849998951 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_support > 0.99849998951
                      return 0.432765451847 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.845499992371
                  if ( median_col_support <= 0.996500015259 ) {
                    if ( min_col_coverage <= 0.92321228981 ) {
                      return 0.0460815162274 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.92321228981
                      return 0.0918484586963 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.996500015259
                    if ( median_col_support <= 0.997500002384 ) {
                      return 0.127589556287 < maxgini;
                    }
                    else {  // if median_col_support > 0.997500002384
                      return 0.170949567666 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_coverage > 0.95099568367
              if ( mean_col_support <= 0.987323522568 ) {
                if ( median_col_support <= 0.981500029564 ) {
                  if ( median_col_coverage <= 0.997973382473 ) {
                    if ( max_col_support <= 0.99950003624 ) {
                      return 0.0275428357417 < maxgini;
                    }
                    else {  // if max_col_support > 0.99950003624
                      return 0.350742094016 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.997973382473
                    if ( median_col_support <= 0.818500041962 ) {
                      return 0.0950982830792 < maxgini;
                    }
                    else {  // if median_col_support > 0.818500041962
                      return 0.207735675149 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.981500029564
                  if ( min_col_support <= 0.725499987602 ) {
                    if ( mean_col_support <= 0.964794099331 ) {
                      return 0.455747108957 < maxgini;
                    }
                    else {  // if mean_col_support > 0.964794099331
                      return 0.47041355861 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.725499987602
                    if ( mean_col_coverage <= 0.999842405319 ) {
                      return 0.360620812336 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.999842405319
                      return 0.234103142228 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.987323522568
                if ( mean_col_support <= 0.99173527956 ) {
                  if ( max_col_coverage <= 0.998792886734 ) {
                    if ( min_col_coverage <= 0.942954361439 ) {
                      return 0.253046483797 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.942954361439
                      return 0.303212727217 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.998792886734
                    if ( min_col_coverage <= 0.916722595692 ) {
                      return 0.136305203641 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.916722595692
                      return 0.185136107539 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.99173527956
                  if ( min_col_support <= 0.929499983788 ) {
                    if ( median_col_support <= 0.995499968529 ) {
                      return 0.0792225213785 < maxgini;
                    }
                    else {  // if median_col_support > 0.995499968529
                      return 0.25868913997 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.929499983788
                    if ( median_col_coverage <= 0.959316134453 ) {
                      return 0.014816438292 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.959316134453
                      return 0.0276919222166 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.99950003624
            if ( min_col_coverage <= 0.885790348053 ) {
              if ( min_col_coverage <= 0.810960412025 ) {
                if ( median_col_coverage <= 0.833079099655 ) {
                  if ( mean_col_coverage <= 0.869816243649 ) {
                    if ( min_col_support <= 0.889500021935 ) {
                      return 0.469935410627 < maxgini;
                    }
                    else {  // if min_col_support > 0.889500021935
                      return 0.016265681312 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.869816243649
                    if ( mean_col_support <= 0.992617666721 ) {
                      return 0.473639928147 < maxgini;
                    }
                    else {  // if mean_col_support > 0.992617666721
                      return 0.0207144157536 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.833079099655
                  if ( median_col_coverage <= 0.847637653351 ) {
                    if ( mean_col_coverage <= 0.842394948006 ) {
                      return 0.16418805364 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.842394948006
                      return 0.207543592435 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.847637653351
                    if ( min_col_support <= 0.87450003624 ) {
                      return 0.474083953536 < maxgini;
                    }
                    else {  // if min_col_support > 0.87450003624
                      return 0.0185922361999 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.810960412025
                if ( min_col_coverage <= 0.885606467724 ) {
                  if ( mean_col_coverage <= 0.939382612705 ) {
                    if ( min_col_support <= 0.880499958992 ) {
                      return 0.477322782149 < maxgini;
                    }
                    else {  // if min_col_support > 0.880499958992
                      return 0.0215635776489 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.939382612705
                    if ( median_col_coverage <= 0.883820593357 ) {
                      return 0.186096376563 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.883820593357
                      return 0.212147420497 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.885606467724
                  if ( mean_col_support <= 0.989088177681 ) {
                    if ( min_col_coverage <= 0.885682225227 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.885682225227
                      return 0.464909268029 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.989088177681
                    if ( mean_col_support <= 0.99279409647 ) {
                      return 0.189003959652 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99279409647
                      return 0.0129053246891 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.885790348053
              if ( min_col_coverage <= 0.995464801788 ) {
                if ( mean_col_support <= 0.991205811501 ) {
                  if ( median_col_coverage <= 0.962285041809 ) {
                    if ( min_col_support <= 0.854499995708 ) {
                      return 0.479618151411 < maxgini;
                    }
                    else {  // if min_col_support > 0.854499995708
                      return 0.216793779993 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.962285041809
                    if ( min_col_coverage <= 0.961679160595 ) {
                      return 0.460786131516 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.961679160595
                      return 0.42904575638 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.991205811501
                  if ( max_col_coverage <= 0.998617827892 ) {
                    if ( mean_col_coverage <= 0.978664577007 ) {
                      return 0.0333258084234 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.978664577007
                      return 0.126324098929 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.998617827892
                    if ( mean_col_support <= 0.993617653847 ) {
                      return 0.211604753693 < maxgini;
                    }
                    else {  // if mean_col_support > 0.993617653847
                      return 0.01575106679 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.995464801788
                if ( mean_col_support <= 0.987029373646 ) {
                  if ( min_col_support <= 0.766499996185 ) {
                    if ( mean_col_coverage <= 0.999238491058 ) {
                      return 0.244897959184 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.999238491058
                      return 0.434916521182 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.766499996185
                    if ( min_col_support <= 0.881500005722 ) {
                      return 0.33449052355 < maxgini;
                    }
                    else {  // if min_col_support > 0.881500005722
                      return 0.0721875 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.987029373646
                  if ( mean_col_support <= 0.992323517799 ) {
                    if ( median_col_coverage <= 0.99615675211 ) {
                      return 0.426035502959 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.99615675211
                      return 0.216790563276 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992323517799
                    if ( min_col_support <= 0.900499999523 ) {
                      return 0.271140139929 < maxgini;
                    }
                    else {  // if min_col_support > 0.900499999523
                      return 0.0193209990265 < maxgini;
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
      if ( min_col_support <= 0.807500004768 ) {
        if ( median_col_coverage <= 0.51519882679 ) {
          if ( median_col_support <= 0.980499982834 ) {
            if ( max_col_coverage <= 0.515201210976 ) {
              if ( mean_col_support <= 0.826303362846 ) {
                if ( mean_col_support <= 0.748558759689 ) {
                  if ( max_col_coverage <= 0.367399305105 ) {
                    if ( mean_col_support <= 0.727676510811 ) {
                      return 0.283078521601 < maxgini;
                    }
                    else {  // if mean_col_support > 0.727676510811
                      return 0.195746389391 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.367399305105
                    if ( mean_col_support <= 0.70655888319 ) {
                      return 0.465183613538 < maxgini;
                    }
                    else {  // if mean_col_support > 0.70655888319
                      return 0.403763319647 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.748558759689
                  if ( median_col_coverage <= 0.243374869227 ) {
                    if ( median_col_coverage <= 0.184271156788 ) {
                      return 0.136572168081 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.184271156788
                      return 0.200686464179 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.243374869227
                    if ( mean_col_coverage <= 0.38168412447 ) {
                      return 0.299228578278 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.38168412447
                      return 0.393570419653 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.826303362846
                if ( median_col_support <= 0.739500045776 ) {
                  if ( mean_col_coverage <= 0.322800934315 ) {
                    if ( median_col_support <= 0.597499966621 ) {
                      return 0.165198953565 < maxgini;
                    }
                    else {  // if median_col_support > 0.597499966621
                      return 0.117621812827 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.322800934315
                    if ( mean_col_support <= 0.917617619038 ) {
                      return 0.218374979107 < maxgini;
                    }
                    else {  // if mean_col_support > 0.917617619038
                      return 0.35453995908 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.739500045776
                  if ( max_col_coverage <= 0.428820967674 ) {
                    if ( mean_col_support <= 0.934379816055 ) {
                      return 0.0925853441398 < maxgini;
                    }
                    else {  // if mean_col_support > 0.934379816055
                      return 0.0682631837179 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.428820967674
                    if ( mean_col_coverage <= 0.229195743799 ) {
                      return 0.183593272803 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.229195743799
                      return 0.0904597079582 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.515201210976
              if ( median_col_support <= 0.65649998188 ) {
                if ( max_col_coverage <= 0.85239648819 ) {
                  if ( min_col_coverage <= 0.316869914532 ) {
                    if ( min_col_coverage <= 0.265183180571 ) {
                      return 0.251985723182 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.265183180571
                      return 0.360710029641 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.316869914532
                    if ( mean_col_coverage <= 0.487904727459 ) {
                      return 0.423775978245 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.487904727459
                      return 0.464133697327 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.85239648819
                  if ( max_col_support <= 0.995499968529 ) {
                    if ( max_col_support <= 0.742500007153 ) {
                      return false;
                    }
                    else {  // if max_col_support > 0.742500007153
                      return 0.0228842568841 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.995499968529
                    if ( mean_col_support <= 0.851146996021 ) {
                      return 0.105602906703 < maxgini;
                    }
                    else {  // if mean_col_support > 0.851146996021
                      return 0.42181468577 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.65649998188
                if ( median_col_support <= 0.734500050545 ) {
                  if ( min_col_coverage <= 0.363803476095 ) {
                    if ( min_col_support <= 0.59249997139 ) {
                      return 0.151348638132 < maxgini;
                    }
                    else {  // if min_col_support > 0.59249997139
                      return 0.245168849395 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.363803476095
                    if ( min_col_support <= 0.601500034332 ) {
                      return 0.156908201748 < maxgini;
                    }
                    else {  // if min_col_support > 0.601500034332
                      return 0.390123461846 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.734500050545
                  if ( median_col_support <= 0.954499959946 ) {
                    if ( min_col_coverage <= 0.141629785299 ) {
                      return 0.194741928446 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.141629785299
                      return 0.102601564682 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.954499959946
                    if ( min_col_support <= 0.634500026703 ) {
                      return 0.374166768168 < maxgini;
                    }
                    else {  // if min_col_support > 0.634500026703
                      return 0.136781381285 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.980499982834
            if ( median_col_coverage <= 0.272865891457 ) {
              if ( median_col_support <= 0.99950003624 ) {
                if ( min_col_support <= 0.707499980927 ) {
                  if ( max_col_coverage <= 0.563504993916 ) {
                    if ( mean_col_coverage <= 0.136251926422 ) {
                      return 0.136887948208 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.136251926422
                      return 0.333642045402 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.563504993916
                    if ( min_col_support <= 0.641499996185 ) {
                      return 0.499278961705 < maxgini;
                    }
                    else {  // if min_col_support > 0.641499996185
                      return 0.473921958065 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.707499980927
                  if ( median_col_support <= 0.991500020027 ) {
                    if ( max_col_coverage <= 0.737726330757 ) {
                      return 0.128588906753 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.737726330757
                      return 0.403528692462 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.991500020027
                    if ( mean_col_coverage <= 0.377298146486 ) {
                      return 0.251774925412 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.377298146486
                      return 0.4771003346 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( max_col_coverage <= 0.529857933521 ) {
                  if ( min_col_support <= 0.551499962807 ) {
                    if ( mean_col_support <= 0.96897059679 ) {
                      return 0.175050380418 < maxgini;
                    }
                    else {  // if mean_col_support > 0.96897059679
                      return 0.334725740472 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.551499962807
                    if ( median_col_coverage <= 0.182120800018 ) {
                      return 0.0600798579132 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.182120800018
                      return 0.173405454397 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.529857933521
                  if ( mean_col_coverage <= 0.492026567459 ) {
                    if ( max_col_coverage <= 0.650163412094 ) {
                      return 0.340046503574 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.650163412094
                      return 0.459757318988 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.492026567459
                    if ( median_col_coverage <= 0.111805558205 ) {
                      return 0.464832673261 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.111805558205
                      return false;
                    }
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.272865891457
              if ( median_col_support <= 0.987499952316 ) {
                if ( max_col_coverage <= 0.579179227352 ) {
                  if ( max_col_coverage <= 0.578721940517 ) {
                    if ( median_col_support <= 0.983500003815 ) {
                      return 0.240662366444 < maxgini;
                    }
                    else {  // if median_col_support > 0.983500003815
                      return 0.322675832156 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.578721940517
                    if ( median_col_support <= 0.983500003815 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if median_col_support > 0.983500003815
                      return 0.21029182446 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.579179227352
                  if ( mean_col_support <= 0.960970520973 ) {
                    if ( max_col_coverage <= 0.591319322586 ) {
                      return 0.471531282245 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.591319322586
                      return 0.426460585395 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.960970520973
                    if ( min_col_support <= 0.700500011444 ) {
                      return 0.360673471777 < maxgini;
                    }
                    else {  // if min_col_support > 0.700500011444
                      return 0.188440574329 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.987499952316
                if ( min_col_support <= 0.708500027657 ) {
                  if ( min_col_coverage <= 0.358385384083 ) {
                    if ( mean_col_support <= 0.946911752224 ) {
                      return 0.461649998113 < maxgini;
                    }
                    else {  // if mean_col_support > 0.946911752224
                      return 0.413457506068 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.358385384083
                    if ( min_col_coverage <= 0.485697239637 ) {
                      return 0.456603568873 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.485697239637
                      return 0.477496379128 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.708500027657
                  if ( median_col_coverage <= 0.372454553843 ) {
                    if ( min_col_support <= 0.770500004292 ) {
                      return 0.293932720445 < maxgini;
                    }
                    else {  // if min_col_support > 0.770500004292
                      return 0.188887489944 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.372454553843
                    if ( mean_col_support <= 0.982852876186 ) {
                      return 0.325356140139 < maxgini;
                    }
                    else {  // if mean_col_support > 0.982852876186
                      return 0.420536468324 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.51519882679
          if ( max_col_coverage <= 0.795415878296 ) {
            if ( mean_col_support <= 0.969852924347 ) {
              if ( mean_col_support <= 0.846382379532 ) {
                if ( min_col_support <= 0.493499994278 ) {
                  if ( median_col_support <= 0.553499996662 ) {
                    if ( median_col_coverage <= 0.596483469009 ) {
                      return 0.491993408017 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.596483469009
                      return 0.396475462613 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.553499996662
                    if ( mean_col_coverage <= 0.526816606522 ) {
                      return false;
                    }
                    else {  // if mean_col_coverage > 0.526816606522
                      return 0.287282418547 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.493499994278
                  if ( mean_col_support <= 0.817264676094 ) {
                    if ( min_col_support <= 0.511500000954 ) {
                      return false;
                    }
                    else {  // if min_col_support > 0.511500000954
                      return 0.496931201004 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.817264676094
                    if ( min_col_support <= 0.580500006676 ) {
                      return 0.439176802329 < maxgini;
                    }
                    else {  // if min_col_support > 0.580500006676
                      return 0.486263767365 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.846382379532
                if ( min_col_coverage <= 0.545605540276 ) {
                  if ( max_col_coverage <= 0.639186739922 ) {
                    if ( mean_col_coverage <= 0.550675272942 ) {
                      return 0.209072838057 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.550675272942
                      return 0.255144729866 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.639186739922
                    if ( median_col_coverage <= 0.523642480373 ) {
                      return 0.364851042041 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.523642480373
                      return 0.296636636314 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.545605540276
                  if ( max_col_coverage <= 0.714502811432 ) {
                    if ( mean_col_coverage <= 0.609006881714 ) {
                      return 0.274502598086 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.609006881714
                      return 0.331237099196 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.714502811432
                    if ( median_col_coverage <= 0.575888216496 ) {
                      return 0.297374468892 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.575888216496
                      return 0.381737660376 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.969852924347
              if ( max_col_coverage <= 0.716657459736 ) {
                if ( min_col_coverage <= 0.517776489258 ) {
                  if ( median_col_support <= 0.987499952316 ) {
                    if ( max_col_coverage <= 0.714944064617 ) {
                      return 0.170422103594 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.714944064617
                      return 0.470416362308 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.987499952316
                    if ( mean_col_coverage <= 0.610015153885 ) {
                      return 0.443445628182 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.610015153885
                      return 0.468160869457 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.517776489258
                  if ( median_col_support <= 0.989500045776 ) {
                    if ( min_col_support <= 0.714499950409 ) {
                      return 0.370171967513 < maxgini;
                    }
                    else {  // if min_col_support > 0.714499950409
                      return 0.14402668064 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.989500045776
                    if ( max_col_coverage <= 0.714547157288 ) {
                      return 0.462898218375 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.714547157288
                      return 0.363423683112 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.716657459736
                if ( min_col_support <= 0.728500008583 ) {
                  if ( median_col_support <= 0.99950003624 ) {
                    if ( min_col_support <= 0.650499999523 ) {
                      return 0.456852674279 < maxgini;
                    }
                    else {  // if min_col_support > 0.650499999523
                      return 0.386729520572 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.99950003624
                    if ( min_col_support <= 0.655499994755 ) {
                      return 0.486648692889 < maxgini;
                    }
                    else {  // if min_col_support > 0.655499994755
                      return 0.477070386229 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.728500008583
                  if ( mean_col_support <= 0.98291182518 ) {
                    if ( median_col_support <= 0.987499952316 ) {
                      return 0.164146402509 < maxgini;
                    }
                    else {  // if median_col_support > 0.987499952316
                      return 0.376099042159 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.98291182518
                    if ( min_col_support <= 0.775499999523 ) {
                      return 0.472670158315 < maxgini;
                    }
                    else {  // if min_col_support > 0.775499999523
                      return 0.427281144832 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.795415878296
            if ( median_col_support <= 0.989500045776 ) {
              if ( median_col_coverage <= 0.757600069046 ) {
                if ( min_col_coverage <= 0.617723822594 ) {
                  if ( median_col_coverage <= 0.606186151505 ) {
                    if ( mean_col_support <= 0.90154594183 ) {
                      return 0.3312104565 < maxgini;
                    }
                    else {  // if mean_col_support > 0.90154594183
                      return 0.220530544562 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.606186151505
                    if ( min_col_coverage <= 0.211032390594 ) {
                      return 0.0744425172035 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.211032390594
                      return 0.276273290724 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.617723822594
                  if ( median_col_support <= 0.678499996662 ) {
                    if ( min_col_support <= 0.499500006437 ) {
                      return 0.402821135391 < maxgini;
                    }
                    else {  // if min_col_support > 0.499500006437
                      return 0.484022241366 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.678499996662
                    if ( median_col_support <= 0.96749997139 ) {
                      return 0.273736361188 < maxgini;
                    }
                    else {  // if median_col_support > 0.96749997139
                      return 0.383388608143 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_coverage > 0.757600069046
                if ( mean_col_coverage <= 0.99982625246 ) {
                  if ( max_col_support <= 0.99849998951 ) {
                    if ( max_col_coverage <= 0.862565219402 ) {
                      return 0.287334593573 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.862565219402
                      return 0.0158196929394 < maxgini;
                    }
                  }
                  else {  // if max_col_support > 0.99849998951
                    if ( median_col_support <= 0.971500039101 ) {
                      return 0.379745232498 < maxgini;
                    }
                    else {  // if median_col_support > 0.971500039101
                      return 0.425434330842 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_coverage > 0.99982625246
                  if ( min_col_support <= 0.753499984741 ) {
                    if ( mean_col_support <= 0.909029364586 ) {
                      return 0.0822882354951 < maxgini;
                    }
                    else {  // if mean_col_support > 0.909029364586
                      return 0.294562998469 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.753499984741
                    if ( max_col_support <= 0.999000012875 ) {
                      return 0.0 < maxgini;
                    }
                    else {  // if max_col_support > 0.999000012875
                      return 0.119907296007 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.989500045776
              if ( median_col_support <= 0.99950003624 ) {
                if ( max_col_coverage <= 0.988272249699 ) {
                  if ( median_col_support <= 0.994500041008 ) {
                    if ( max_col_coverage <= 0.985760033131 ) {
                      return 0.452001769045 < maxgini;
                    }
                    else {  // if max_col_coverage > 0.985760033131
                      return 0.388028005303 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.994500041008
                    if ( min_col_support <= 0.652500033379 ) {
                      return 0.46765527168 < maxgini;
                    }
                    else {  // if min_col_support > 0.652500033379
                      return 0.405718370444 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.988272249699
                  if ( mean_col_support <= 0.978323459625 ) {
                    if ( min_col_support <= 0.644500017166 ) {
                      return 0.475572698266 < maxgini;
                    }
                    else {  // if min_col_support > 0.644500017166
                      return 0.430660989098 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.978323459625
                    if ( median_col_coverage <= 0.994968533516 ) {
                      return 0.43026554048 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.994968533516
                      return 0.355873662108 < maxgini;
                    }
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( max_col_coverage <= 0.982203483582 ) {
                  if ( median_col_coverage <= 0.641370296478 ) {
                    if ( median_col_coverage <= 0.552779793739 ) {
                      return 0.442775875329 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.552779793739
                      return 0.470434454717 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.641370296478
                    if ( mean_col_support <= 0.970558822155 ) {
                      return 0.476515475319 < maxgini;
                    }
                    else {  // if mean_col_support > 0.970558822155
                      return 0.483391555735 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.982203483582
                  if ( min_col_coverage <= 0.949175238609 ) {
                    if ( median_col_coverage <= 0.773983240128 ) {
                      return 0.439771851085 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.773983240128
                      return 0.482753470201 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.949175238609
                    if ( min_col_support <= 0.708500027657 ) {
                      return 0.459449803828 < maxgini;
                    }
                    else {  // if min_col_support > 0.708500027657
                      return 0.436468427758 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.807500004768
        if ( max_col_coverage <= 0.886421382427 ) {
          if ( median_col_support <= 0.927500009537 ) {
            if ( mean_col_support <= 0.954313755035 ) {
              if ( max_col_coverage <= 0.738296031952 ) {
                if ( min_col_support <= 0.869500041008 ) {
                  if ( mean_col_support <= 0.944537520409 ) {
                    if ( mean_col_support <= 0.88535284996 ) {
                      return 0.4992 < maxgini;
                    }
                    else {  // if mean_col_support > 0.88535284996
                      return 0.098415868171 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.944537520409
                    if ( min_col_coverage <= 0.486702561378 ) {
                      return 0.0801671023946 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.486702561378
                      return 0.107973441797 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.869500041008
                  if ( mean_col_coverage <= 0.356453806162 ) {
                    if ( min_col_support <= 0.889500021935 ) {
                      return 0.173900749848 < maxgini;
                    }
                    else {  // if min_col_support > 0.889500021935
                      return 0.274058641975 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.356453806162
                    if ( min_col_support <= 0.902500033379 ) {
                      return 0.121355682259 < maxgini;
                    }
                    else {  // if min_col_support > 0.902500033379
                      return 0.265118040651 < maxgini;
                    }
                  }
                }
              }
              else {  // if max_col_coverage > 0.738296031952
                if ( min_col_support <= 0.824499964714 ) {
                  if ( median_col_coverage <= 0.704124569893 ) {
                    if ( median_col_support <= 0.855499982834 ) {
                      return 0.262037333761 < maxgini;
                    }
                    else {  // if median_col_support > 0.855499982834
                      return 0.102431034849 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.704124569893
                    if ( min_col_coverage <= 0.520733654499 ) {
                      return false;
                    }
                    else {  // if min_col_coverage > 0.520733654499
                      return 0.252212331713 < maxgini;
                    }
                  }
                }
                else {  // if min_col_support > 0.824499964714
                  if ( mean_col_support <= 0.936205863953 ) {
                    if ( mean_col_support <= 0.92041182518 ) {
                      return 0.351239669421 < maxgini;
                    }
                    else {  // if mean_col_support > 0.92041182518
                      return 0.181857760952 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.936205863953
                    if ( max_col_coverage <= 0.738416790962 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.738416790962
                      return 0.112767445478 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.954313755035
              if ( mean_col_support <= 0.961905479431 ) {
                if ( min_col_coverage <= 0.433279037476 ) {
                  if ( median_col_coverage <= 0.0976236760616 ) {
                    if ( median_col_coverage <= 0.0951397120953 ) {
                      return 0.0862361853489 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.0951397120953
                      return 0.235898354684 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.0976236760616
                    if ( median_col_support <= 0.87549996376 ) {
                      return 0.0831134993243 < maxgini;
                    }
                    else {  // if median_col_support > 0.87549996376
                      return 0.0620038521768 < maxgini;
                    }
                  }
                }
                else {  // if min_col_coverage > 0.433279037476
                  if ( max_col_coverage <= 0.794448971748 ) {
                    if ( min_col_support <= 0.834499955177 ) {
                      return 0.0696009860611 < maxgini;
                    }
                    else {  // if min_col_support > 0.834499955177
                      return 0.090649957835 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.794448971748
                    if ( max_col_coverage <= 0.794486999512 ) {
                      return false;
                    }
                    else {  // if max_col_coverage > 0.794486999512
                      return 0.110300103663 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_support > 0.961905479431
                if ( median_col_support <= 0.90649998188 ) {
                  if ( max_col_coverage <= 0.669171333313 ) {
                    if ( mean_col_support <= 0.968150734901 ) {
                      return 0.0664659129057 < maxgini;
                    }
                    else {  // if mean_col_support > 0.968150734901
                      return 0.0786733873851 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.669171333313
                    if ( median_col_coverage <= 0.735993385315 ) {
                      return 0.100450775938 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.735993385315
                      return 0.21348818701 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.90649998188
                  if ( mean_col_coverage <= 0.586745858192 ) {
                    if ( min_col_coverage <= 0.114363566041 ) {
                      return 0.0672655784162 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.114363566041
                      return 0.052684963965 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.586745858192
                    if ( median_col_coverage <= 0.710325717926 ) {
                      return 0.076118587888 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.710325717926
                      return 0.051234502197 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.927500009537
            if ( min_col_coverage <= 0.617665410042 ) {
              if ( min_col_support <= 0.87549996376 ) {
                if ( mean_col_support <= 0.988029420376 ) {
                  if ( median_col_coverage <= 0.515169918537 ) {
                    if ( min_col_support <= 0.828500032425 ) {
                      return 0.0688806129461 < maxgini;
                    }
                    else {  // if min_col_support > 0.828500032425
                      return 0.0434114500169 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.515169918537
                    if ( min_col_coverage <= 0.545575618744 ) {
                      return 0.0881269916317 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.545575618744
                      return 0.118092436433 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.988029420376
                  if ( min_col_coverage <= 0.382544696331 ) {
                    if ( mean_col_coverage <= 0.373909175396 ) {
                      return 0.0401721121882 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.373909175396
                      return 0.159554938013 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.382544696331
                    if ( mean_col_support <= 0.988676428795 ) {
                      return 0.21014925244 < maxgini;
                    }
                    else {  // if mean_col_support > 0.988676428795
                      return 0.322936641647 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.87549996376
                if ( mean_col_support <= 0.993899583817 ) {
                  if ( median_col_support <= 0.959499955177 ) {
                    if ( median_col_coverage <= 0.318935215473 ) {
                      return 0.0507199512067 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.318935215473
                      return 0.0417354789575 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.959499955177
                    if ( mean_col_support <= 0.988719701767 ) {
                      return 0.0293035499789 < maxgini;
                    }
                    else {  // if mean_col_support > 0.988719701767
                      return 0.023379379123 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.993899583817
                  if ( min_col_support <= 0.925500035286 ) {
                    if ( median_col_coverage <= 0.516594171524 ) {
                      return 0.0248140213672 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.516594171524
                      return 0.0773417379097 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.925500035286
                    if ( mean_col_support <= 0.997029423714 ) {
                      return 0.0117097649167 < maxgini;
                    }
                    else {  // if mean_col_support > 0.997029423714
                      return 0.00673792485132 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.617665410042
              if ( mean_col_coverage <= 0.74170589447 ) {
                if ( max_col_coverage <= 0.76748919487 ) {
                  if ( min_col_coverage <= 0.627467215061 ) {
                    if ( median_col_support <= 0.991500020027 ) {
                      return 0.0260505853775 < maxgini;
                    }
                    else {  // if median_col_support > 0.991500020027
                      return 0.0423319395691 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.627467215061
                    if ( median_col_support <= 0.991500020027 ) {
                      return 0.0224194627481 < maxgini;
                    }
                    else {  // if median_col_support > 0.991500020027
                      return 0.0327621357914 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.76748919487
                  if ( min_col_coverage <= 0.666499137878 ) {
                    if ( median_col_coverage <= 0.651178240776 ) {
                      return 0.0354866569671 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.651178240776
                      return 0.0468634522382 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.666499137878
                    if ( mean_col_support <= 0.99279409647 ) {
                      return 0.0853131971668 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99279409647
                      return 0.0122530482249 < maxgini;
                    }
                  }
                }
              }
              else {  // if mean_col_coverage > 0.74170589447
                if ( max_col_coverage <= 0.825031161308 ) {
                  if ( mean_col_support <= 0.992029428482 ) {
                    if ( mean_col_support <= 0.987500011921 ) {
                      return 0.0821168200961 < maxgini;
                    }
                    else {  // if mean_col_support > 0.987500011921
                      return 0.116007470283 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992029428482
                    if ( min_col_coverage <= 0.648313760757 ) {
                      return 0.0126577143089 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.648313760757
                      return 0.0155057543085 < maxgini;
                    }
                  }
                }
                else {  // if max_col_coverage > 0.825031161308
                  if ( mean_col_support <= 0.992088198662 ) {
                    if ( min_col_support <= 0.865499973297 ) {
                      return 0.29647659432 < maxgini;
                    }
                    else {  // if min_col_support > 0.865499973297
                      return 0.0537050097334 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992088198662
                    if ( min_col_coverage <= 0.652371942997 ) {
                      return 0.0143130972395 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.652371942997
                      return 0.0182681891008 < maxgini;
                    }
                  }
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.886421382427
          if ( median_col_coverage <= 0.91679161787 ) {
            if ( min_col_coverage <= 0.111288040876 ) {
              if ( min_col_coverage <= 0.00347852380946 ) {
                return 0.0 < maxgini;
              }
              else {  // if min_col_coverage > 0.00347852380946
                if ( min_col_support <= 0.895500004292 ) {
                  if ( median_col_coverage <= 0.139211893082 ) {
                    if ( mean_col_coverage <= 0.498027771711 ) {
                      return 0.335204181978 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.498027771711
                      return 0.45822893787 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.139211893082
                    if ( min_col_support <= 0.893499970436 ) {
                      return 0.178163561876 < maxgini;
                    }
                    else {  // if min_col_support > 0.893499970436
                      return false;
                    }
                  }
                }
                else {  // if min_col_support > 0.895500004292
                  if ( median_col_support <= 0.944499969482 ) {
                    if ( median_col_coverage <= 0.0234567895532 ) {
                      return false;
                    }
                    else {  // if median_col_coverage > 0.0234567895532
                      return 0.4608 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.944499969482
                    if ( median_col_support <= 0.980499982834 ) {
                      return 0.0403055089841 < maxgini;
                    }
                    else {  // if median_col_support > 0.980499982834
                      return 0.154637063445 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.111288040876
              if ( min_col_support <= 0.886500000954 ) {
                if ( median_col_support <= 0.993499994278 ) {
                  if ( median_col_support <= 0.990499973297 ) {
                    if ( min_col_coverage <= 0.794321417809 ) {
                      return 0.0990422880669 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.794321417809
                      return 0.168256979041 < maxgini;
                    }
                  }
                  else {  // if median_col_support > 0.990499973297
                    if ( min_col_coverage <= 0.234610021114 ) {
                      return 0.499131944444 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.234610021114
                      return 0.230626158624 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( median_col_coverage <= 0.807004213333 ) {
                    if ( mean_col_coverage <= 0.74343585968 ) {
                      return 0.20333168025 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.74343585968
                      return 0.352686983043 < maxgini;
                    }
                  }
                  else {  // if median_col_coverage > 0.807004213333
                    if ( median_col_support <= 0.99950003624 ) {
                      return 0.3241705408 < maxgini;
                    }
                    else {  // if median_col_support > 0.99950003624
                      return 0.410684401499 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_support > 0.886500000954
                if ( mean_col_support <= 0.994970560074 ) {
                  if ( min_col_coverage <= 0.737091183662 ) {
                    if ( median_col_coverage <= 0.727483272552 ) {
                      return 0.0253495144388 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.727483272552
                      return 0.0352822370551 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.737091183662
                    if ( mean_col_support <= 0.994382381439 ) {
                      return 0.0502716607554 < maxgini;
                    }
                    else {  // if mean_col_support > 0.994382381439
                      return 0.0361749297345 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.994970560074
                  if ( min_col_support <= 0.935500025749 ) {
                    if ( mean_col_support <= 0.995852947235 ) {
                      return 0.0946826460915 < maxgini;
                    }
                    else {  // if mean_col_support > 0.995852947235
                      return 0.0514912414111 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.935500025749
                    if ( min_col_support <= 0.944499969482 ) {
                      return 0.0306236756348 < maxgini;
                    }
                    else {  // if min_col_support > 0.944499969482
                      return 0.00820840384098 < maxgini;
                    }
                  }
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.91679161787
            if ( median_col_support <= 0.99950003624 ) {
              if ( min_col_coverage <= 0.944552540779 ) {
                if ( median_col_support <= 0.995499968529 ) {
                  if ( min_col_coverage <= 0.909284949303 ) {
                    if ( median_col_coverage <= 0.917997717857 ) {
                      return 0.223921222773 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.917997717857
                      return 0.0529072080695 < maxgini;
                    }
                  }
                  else {  // if min_col_coverage > 0.909284949303
                    if ( mean_col_coverage <= 0.954082667828 ) {
                      return 0.111236781903 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.954082667828
                      return 0.0705316064006 < maxgini;
                    }
                  }
                }
                else {  // if median_col_support > 0.995499968529
                  if ( mean_col_support <= 0.994441151619 ) {
                    if ( mean_col_support <= 0.99020588398 ) {
                      return 0.327449203131 < maxgini;
                    }
                    else {  // if mean_col_support > 0.99020588398
                      return 0.241950000175 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.994441151619
                    if ( min_col_support <= 0.948500037193 ) {
                      return 0.15078837685 < maxgini;
                    }
                    else {  // if min_col_support > 0.948500037193
                      return 0.0203922168324 < maxgini;
                    }
                  }
                }
              }
              else {  // if min_col_coverage > 0.944552540779
                if ( mean_col_support <= 0.993617653847 ) {
                  if ( mean_col_coverage <= 0.992048025131 ) {
                    if ( min_col_support <= 0.894500017166 ) {
                      return 0.316934971164 < maxgini;
                    }
                    else {  // if min_col_support > 0.894500017166
                      return 0.120279694904 < maxgini;
                    }
                  }
                  else {  // if mean_col_coverage > 0.992048025131
                    if ( median_col_coverage <= 0.997903585434 ) {
                      return 0.210840124645 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.997903585434
                      return 0.103387170583 < maxgini;
                    }
                  }
                }
                else {  // if mean_col_support > 0.993617653847
                  if ( max_col_coverage <= 0.99855697155 ) {
                    if ( median_col_support <= 0.99849998951 ) {
                      return 0.0374905648805 < maxgini;
                    }
                    else {  // if median_col_support > 0.99849998951
                      return 0.151983673469 < maxgini;
                    }
                  }
                  else {  // if max_col_coverage > 0.99855697155
                    if ( median_col_coverage <= 0.961317777634 ) {
                      return 0.0114363622362 < maxgini;
                    }
                    else {  // if median_col_coverage > 0.961317777634
                      return 0.0391659907375 < maxgini;
                    }
                  }
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( median_col_coverage <= 0.916846692562 ) {
                return false;
              }
              else {  // if median_col_coverage > 0.916846692562
                if ( median_col_coverage <= 0.998146355152 ) {
                  if ( min_col_support <= 0.898499965668 ) {
                    if ( min_col_coverage <= 0.961957454681 ) {
                      return 0.410207487852 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.961957454681
                      return 0.331844784663 < maxgini;
                    }
                  }
                  else {  // if min_col_support > 0.898499965668
                    if ( min_col_support <= 0.925500035286 ) {
                      return 0.151280056055 < maxgini;
                    }
                    else {  // if min_col_support > 0.925500035286
                      return 0.0114841461381 < maxgini;
                    }
                  }
                }
                else {  // if median_col_coverage > 0.998146355152
                  if ( mean_col_support <= 0.992300868034 ) {
                    if ( min_col_coverage <= 0.822021126747 ) {
                      return 0.438276113952 < maxgini;
                    }
                    else {  // if min_col_coverage > 0.822021126747
                      return 0.215040666534 < maxgini;
                    }
                  }
                  else {  // if mean_col_support > 0.992300868034
                    if ( mean_col_coverage <= 0.98988699913 ) {
                      return 0.002775844742 < maxgini;
                    }
                    else {  // if mean_col_coverage > 0.98988699913
                      return 0.0211024348844 < maxgini;
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
