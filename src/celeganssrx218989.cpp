#include "../inc/celeganssrx218989.hpp"

namespace celegans_srx218989{

#if 1
    //coverage is normalized to number of reads in msa
    bool shouldCorrect(double min_col_support, double min_col_coverage,
        double max_col_support, double max_col_coverage,
        double mean_col_support, double mean_col_coverage,
        double median_col_support, double median_col_coverage,
        double maxgini) {
    if ( min_col_support <= 0.766499996185 ) {
      if ( min_col_coverage <= 0.454648315907 ) {
        if ( median_col_coverage <= 0.350115209818 ) {
          if ( mean_col_support <= 0.914069056511 ) {
            if ( mean_col_support <= 0.855355024338 ) {
              if ( median_col_coverage <= 0.195742756128 ) {
                if ( median_col_support <= 0.777500033379 ) {
                  if ( min_col_support <= 0.425000011921 ) {
                    return 0.132653061224 < maxgini;
                  }
                  else {  // if min_col_support > 0.425000011921
                    return 0.468372815725 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.777500033379
                  if ( min_col_coverage <= 0.0504237264395 ) {
                    return 0.398710193207 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.0504237264395
                    return 0.308525376298 < maxgini;
                  }
                }
              }
              else {  // if median_col_coverage > 0.195742756128
                if ( median_col_support <= 0.96850001812 ) {
                  if ( median_col_support <= 0.827749967575 ) {
                    return 0.465155361638 < maxgini;
                  }
                  else {  // if median_col_support > 0.827749967575
                    return 0.359419613006 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.96850001812
                  if ( min_col_support <= 0.62650001049 ) {
                    return 0.487412312579 < maxgini;
                  }
                  else {  // if min_col_support > 0.62650001049
                    return 0.324779169633 < maxgini;
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.855355024338
              if ( min_col_coverage <= 0.0506038144231 ) {
                if ( median_col_coverage <= 0.15331196785 ) {
                  if ( min_col_support <= 0.533499956131 ) {
                    return 0.394646303399 < maxgini;
                  }
                  else {  // if min_col_support > 0.533499956131
                    return 0.290670662195 < maxgini;
                  }
                }
                else {  // if median_col_coverage > 0.15331196785
                  if ( median_col_coverage <= 0.268761754036 ) {
                    return 0.409621294067 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.268761754036
                    return 0.475818855541 < maxgini;
                  }
                }
              }
              else {  // if min_col_coverage > 0.0506038144231
                if ( median_col_coverage <= 0.238242655993 ) {
                  if ( median_col_coverage <= 0.150558397174 ) {
                    return 0.159472853678 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.150558397174
                    return 0.262153573055 < maxgini;
                  }
                }
                else {  // if median_col_coverage > 0.238242655993
                  if ( mean_col_support <= 0.888179123402 ) {
                    return 0.385909255488 < maxgini;
                  }
                  else {  // if mean_col_support > 0.888179123402
                    return 0.313974148623 < maxgini;
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.914069056511
            if ( mean_col_support <= 0.935433030128 ) {
              if ( min_col_coverage <= 0.0701195672154 ) {
                if ( median_col_coverage <= 0.101357549429 ) {
                  if ( min_col_support <= 0.579499959946 ) {
                    return 0.139329656549 < maxgini;
                  }
                  else {  // if min_col_support > 0.579499959946
                    return 0.24957556603 < maxgini;
                  }
                }
                else {  // if median_col_coverage > 0.101357549429
                  if ( mean_col_support <= 0.926925837994 ) {
                    return 0.331175351109 < maxgini;
                  }
                  else {  // if mean_col_support > 0.926925837994
                    return 0.265223850968 < maxgini;
                  }
                }
              }
              else {  // if min_col_coverage > 0.0701195672154
                if ( median_col_coverage <= 0.200694441795 ) {
                  if ( min_col_support <= 0.648499965668 ) {
                    return 0.102292914997 < maxgini;
                  }
                  else {  // if min_col_support > 0.648499965668
                    return 0.16395626252 < maxgini;
                  }
                }
                else {  // if median_col_coverage > 0.200694441795
                  if ( min_col_support <= 0.551499962807 ) {
                    return 0.291193324425 < maxgini;
                  }
                  else {  // if min_col_support > 0.551499962807
                    return 0.222394324407 < maxgini;
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.935433030128
              if ( min_col_coverage <= 0.200290709734 ) {
                if ( mean_col_support <= 0.954594135284 ) {
                  if ( min_col_support <= 0.550500035286 ) {
                    return 0.210413375024 < maxgini;
                  }
                  else {  // if min_col_support > 0.550500035286
                    return 0.135454640142 < maxgini;
                  }
                }
                else {  // if mean_col_support > 0.954594135284
                  if ( min_col_support <= 0.550500035286 ) {
                    return 0.228663898341 < maxgini;
                  }
                  else {  // if min_col_support > 0.550500035286
                    return 0.073505436511 < maxgini;
                  }
                }
              }
              else {  // if min_col_coverage > 0.200290709734
                if ( min_col_support <= 0.611500024796 ) {
                  if ( mean_col_support <= 0.969794154167 ) {
                    return 0.449112179974 < maxgini;
                  }
                  else {  // if mean_col_support > 0.969794154167
                    return false;
                  }
                }
                else {  // if min_col_support > 0.611500024796
                  if ( min_col_support <= 0.659500002861 ) {
                    return 0.321790509644 < maxgini;
                  }
                  else {  // if min_col_support > 0.659500002861
                    return 0.1433390355 < maxgini;
                  }
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.350115209818
          if ( min_col_support <= 0.62450003624 ) {
            if ( min_col_coverage <= 0.350090265274 ) {
              if ( mean_col_support <= 0.847323536873 ) {
                if ( median_col_support <= 0.676499962807 ) {
                  if ( mean_col_support <= 0.800764679909 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.800764679909
                    return 0.457618971319 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.676499962807
                  if ( median_col_support <= 0.969500005245 ) {
                    return 0.494071165773 < maxgini;
                  }
                  else {  // if median_col_support > 0.969500005245
                    return false;
                  }
                }
              }
              else {  // if mean_col_support > 0.847323536873
                if ( mean_col_support <= 0.96850001812 ) {
                  if ( median_col_support <= 0.97025001049 ) {
                    return 0.289230737078 < maxgini;
                  }
                  else {  // if median_col_support > 0.97025001049
                    return 0.450947269665 < maxgini;
                  }
                }
                else {  // if mean_col_support > 0.96850001812
                  if ( min_col_coverage <= 0.131006866693 ) {
                    return 0.0809363221583 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.131006866693
                    return false;
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.350090265274
              if ( mean_col_support <= 0.939852952957 ) {
                if ( mean_col_support <= 0.845735311508 ) {
                  if ( median_col_support <= 0.981000006199 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.981000006199
                    return false;
                  }
                }
                else {  // if mean_col_support > 0.845735311508
                  if ( median_col_support <= 0.975499987602 ) {
                    return 0.317024843774 < maxgini;
                  }
                  else {  // if median_col_support > 0.975499987602
                    return 0.495863579815 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.939852952957
                if ( min_col_support <= 0.580500006676 ) {
                  if ( mean_col_support <= 0.970205903053 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.970205903053
                    return false;
                  }
                }
                else {  // if min_col_support > 0.580500006676
                  if ( mean_col_support <= 0.974852919579 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.974852919579
                    return false;
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.62450003624
            if ( mean_col_support <= 0.910558819771 ) {
              if ( median_col_coverage <= 0.440095692873 ) {
                if ( min_col_support <= 0.696500003338 ) {
                  if ( median_col_support <= 0.965499997139 ) {
                    return 0.383915013865 < maxgini;
                  }
                  else {  // if median_col_support > 0.965499997139
                    return 0.449488336422 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.696500003338
                  if ( median_col_support <= 0.789499998093 ) {
                    return 0.240050522833 < maxgini;
                  }
                  else {  // if median_col_support > 0.789499998093
                    return 0.383997080061 < maxgini;
                  }
                }
              }
              else {  // if median_col_coverage > 0.440095692873
                if ( mean_col_support <= 0.873029410839 ) {
                  if ( min_col_support <= 0.676499962807 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.676499962807
                    return 0.460325332275 < maxgini;
                  }
                }
                else {  // if mean_col_support > 0.873029410839
                  if ( median_col_support <= 0.971500039101 ) {
                    return 0.387778290852 < maxgini;
                  }
                  else {  // if median_col_support > 0.971500039101
                    return 0.4557585051 < maxgini;
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.910558819771
              if ( mean_col_support <= 0.976382374763 ) {
                if ( min_col_coverage <= 0.36373308301 ) {
                  if ( mean_col_support <= 0.931970596313 ) {
                    return 0.343920209022 < maxgini;
                  }
                  else {  // if mean_col_support > 0.931970596313
                    return 0.222232797827 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.36373308301
                  if ( min_col_support <= 0.695500016212 ) {
                    return 0.426192109328 < maxgini;
                  }
                  else {  // if min_col_support > 0.695500016212
                    return 0.305670848189 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.976382374763
                if ( min_col_support <= 0.710500001907 ) {
                  if ( min_col_support <= 0.655499994755 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.655499994755
                    return 0.493501765832 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.710500001907
                  if ( min_col_coverage <= 0.301026165485 ) {
                    return 0.180246485645 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.301026165485
                    return 0.394448502557 < maxgini;
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_coverage > 0.454648315907
        if ( min_col_support <= 0.679499983788 ) {
          if ( min_col_coverage <= 0.600547790527 ) {
            if ( mean_col_support <= 0.940441131592 ) {
              if ( min_col_support <= 0.565500020981 ) {
                if ( median_col_support <= 0.984500050545 ) {
                  if ( mean_col_support <= 0.822441220284 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.822441220284
                    return 0.408163265306 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.984500050545
                  if ( mean_col_support <= 0.845529437065 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.845529437065
                    return false;
                  }
                }
              }
              else {  // if min_col_support > 0.565500020981
                if ( mean_col_support <= 0.876941204071 ) {
                  if ( median_col_support <= 0.983500003815 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.983500003815
                    return false;
                  }
                }
                else {  // if mean_col_support > 0.876941204071
                  if ( median_col_support <= 0.984500050545 ) {
                    return 0.279629621558 < maxgini;
                  }
                  else {  // if median_col_support > 0.984500050545
                    return 0.48829377425 < maxgini;
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.940441131592
              if ( min_col_support <= 0.611500024796 ) {
                if ( mean_col_support <= 0.974970579147 ) {
                  if ( min_col_support <= 0.571500003338 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.571500003338
                    return false;
                  }
                }
                else {  // if mean_col_support > 0.974970579147
                  if ( max_col_coverage <= 0.800568223 ) {
                    return false;
                  }
                  else {  // if max_col_coverage > 0.800568223
                    return false;
                  }
                }
              }
              else {  // if min_col_support > 0.611500024796
                if ( mean_col_support <= 0.953441143036 ) {
                  if ( max_col_coverage <= 0.625748336315 ) {
                    return 0.350374534109 < maxgini;
                  }
                  else {  // if max_col_coverage > 0.625748336315
                    return 0.497936544001 < maxgini;
                  }
                }
                else {  // if mean_col_support > 0.953441143036
                  if ( max_col_coverage <= 0.619292140007 ) {
                    return 0.493339284675 < maxgini;
                  }
                  else {  // if max_col_coverage > 0.619292140007
                    return false;
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.600547790527
            if ( median_col_coverage <= 0.9947437644 ) {
              if ( median_col_support <= 0.992499947548 ) {
                if ( mean_col_support <= 0.84035295248 ) {
                  if ( min_col_coverage <= 0.739935576916 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.739935576916
                    return false;
                  }
                }
                else {  // if mean_col_support > 0.84035295248
                  if ( min_col_coverage <= 0.823978424072 ) {
                    return 0.463241295437 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.823978424072
                    return false;
                  }
                }
              }
              else {  // if median_col_support > 0.992499947548
                if ( min_col_support <= 0.614500045776 ) {
                  if ( max_col_coverage <= 0.820243060589 ) {
                    return false;
                  }
                  else {  // if max_col_coverage > 0.820243060589
                    return false;
                  }
                }
                else {  // if min_col_support > 0.614500045776
                  if ( min_col_coverage <= 0.800362348557 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.800362348557
                    return false;
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.9947437644
              if ( median_col_support <= 0.924499988556 ) {
                if ( mean_col_coverage <= 0.963693141937 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.963693141937
                  if ( min_col_support <= 0.554499983788 ) {
                    return 0.430160187736 < maxgini;
                  }
                  else {  // if min_col_support > 0.554499983788
                    return 0.169921875 < maxgini;
                  }
                }
              }
              else {  // if median_col_support > 0.924499988556
                if ( min_col_support <= 0.613499999046 ) {
                  if ( mean_col_support <= 0.970323562622 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.970323562622
                    return false;
                  }
                }
                else {  // if min_col_support > 0.613499999046
                  if ( mean_col_support <= 0.976970553398 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.976970553398
                    return false;
                  }
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.679499983788
          if ( min_col_coverage <= 0.600196838379 ) {
            if ( mean_col_support <= 0.964205861092 ) {
              if ( mean_col_support <= 0.916558861732 ) {
                if ( mean_col_support <= 0.901499986649 ) {
                  if ( min_col_support <= 0.736500024796 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.736500024796
                    return 0.461094182825 < maxgini;
                  }
                }
                else {  // if mean_col_support > 0.901499986649
                  if ( median_col_support <= 0.991500020027 ) {
                    return 0.314098750744 < maxgini;
                  }
                  else {  // if median_col_support > 0.991500020027
                    return 0.487274599161 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.916558861732
                if ( min_col_support <= 0.693500041962 ) {
                  if ( mean_col_support <= 0.958382368088 ) {
                    return 0.387715169723 < maxgini;
                  }
                  else {  // if mean_col_support > 0.958382368088
                    return 0.49729202976 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.693500041962
                  if ( median_col_support <= 0.985499978065 ) {
                    return 0.137434371347 < maxgini;
                  }
                  else {  // if median_col_support > 0.985499978065
                    return 0.324237062916 < maxgini;
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.964205861092
              if ( max_col_coverage <= 0.60092818737 ) {
                if ( max_col_coverage <= 0.572370648384 ) {
                  if ( min_col_support <= 0.693500041962 ) {
                    return 0.32 < maxgini;
                  }
                  else {  // if min_col_support > 0.693500041962
                    return 0.0622855602392 < maxgini;
                  }
                }
                else {  // if max_col_coverage > 0.572370648384
                  if ( min_col_coverage <= 0.478843390942 ) {
                    return 0.466068780453 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.478843390942
                    return 0.232158644136 < maxgini;
                  }
                }
              }
              else {  // if max_col_coverage > 0.60092818737
                if ( min_col_support <= 0.710500001907 ) {
                  if ( mean_col_support <= 0.980617642403 ) {
                    return 0.499649039299 < maxgini;
                  }
                  else {  // if mean_col_support > 0.980617642403
                    return false;
                  }
                }
                else {  // if min_col_support > 0.710500001907
                  if ( median_col_support <= 0.99950003624 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.99950003624
                    return 0.468825739893 < maxgini;
                  }
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.600196838379
            if ( max_col_coverage <= 0.751114249229 ) {
              if ( min_col_coverage <= 0.631392300129 ) {
                if ( max_col_coverage <= 0.714655816555 ) {
                  if ( max_col_coverage <= 0.697056293488 ) {
                    return 0.20471252365 < maxgini;
                  }
                  else {  // if max_col_coverage > 0.697056293488
                    return 0.421264802217 < maxgini;
                  }
                }
                else {  // if max_col_coverage > 0.714655816555
                  if ( min_col_support <= 0.727499961853 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.727499961853
                    return 0.453174626216 < maxgini;
                  }
                }
              }
              else {  // if min_col_coverage > 0.631392300129
                if ( min_col_support <= 0.71850001812 ) {
                  if ( min_col_coverage <= 0.647897958755 ) {
                    return 0.475308641975 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.647897958755
                    return 0.346160888672 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.71850001812
                  if ( mean_col_support <= 0.917176485062 ) {
                    return 0.489795918367 < maxgini;
                  }
                  else {  // if mean_col_support > 0.917176485062
                    return 0.196064349325 < maxgini;
                  }
                }
              }
            }
            else {  // if max_col_coverage > 0.751114249229
              if ( mean_col_support <= 0.962029457092 ) {
                if ( min_col_coverage <= 0.80031645298 ) {
                  if ( min_col_support <= 0.727499961853 ) {
                    return 0.494351491628 < maxgini;
                  }
                  else {  // if min_col_support > 0.727499961853
                    return 0.403724660281 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.80031645298
                  if ( max_col_coverage <= 0.996979117393 ) {
                    return false;
                  }
                  else {  // if max_col_coverage > 0.996979117393
                    return false;
                  }
                }
              }
              else {  // if mean_col_support > 0.962029457092
                if ( min_col_support <= 0.713500022888 ) {
                  if ( mean_col_support <= 0.980264663696 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.980264663696
                    return false;
                  }
                }
                else {  // if min_col_support > 0.713500022888
                  if ( mean_col_support <= 0.982852935791 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.982852935791
                    return false;
                  }
                }
              }
            }
          }
        }
      }
    }
    else {  // if min_col_support > 0.766499996185
      if ( min_col_support <= 0.850499987602 ) {
        if ( min_col_coverage <= 0.600261092186 ) {
          if ( mean_col_support <= 0.95297062397 ) {
            if ( mean_col_support <= 0.938563346863 ) {
              if ( median_col_coverage <= 0.408249139786 ) {
                if ( median_col_support <= 0.911499977112 ) {
                  if ( median_col_support <= 0.810500025749 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.810500025749
                    return 0.462809917355 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.911499977112
                  if ( mean_col_support <= 0.93850004673 ) {
                    return 0.253550964256 < maxgini;
                  }
                  else {  // if mean_col_support > 0.93850004673
                    return 0.4872 < maxgini;
                  }
                }
              }
              else {  // if median_col_coverage > 0.408249139786
                if ( mean_col_support <= 0.880411744118 ) {
                  if ( median_col_support <= 0.805500030518 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.805500030518
                    return 0.476370510397 < maxgini;
                  }
                }
                else {  // if mean_col_support > 0.880411744118
                  if ( median_col_support <= 0.978500008583 ) {
                    return 0.322249758191 < maxgini;
                  }
                  else {  // if median_col_support > 0.978500008583
                    return 0.387274302955 < maxgini;
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.938563346863
              if ( median_col_coverage <= 0.401923060417 ) {
                if ( min_col_coverage <= 0.180652678013 ) {
                  if ( mean_col_support <= 0.946677088737 ) {
                    return 0.258059042979 < maxgini;
                  }
                  else {  // if mean_col_support > 0.946677088737
                    return 0.190715853066 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.180652678013
                  if ( median_col_support <= 0.932500004768 ) {
                    return 0.452662721893 < maxgini;
                  }
                  else {  // if median_col_support > 0.932500004768
                    return 0.161418078707 < maxgini;
                  }
                }
              }
              else {  // if median_col_coverage > 0.401923060417
                if ( median_col_support <= 0.983500003815 ) {
                  if ( median_col_coverage <= 0.902127623558 ) {
                    return 0.118508377157 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.902127623558
                    return 0.485582057672 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.983500003815
                  if ( median_col_coverage <= 0.552177846432 ) {
                    return 0.263763374682 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.552177846432
                    return 0.319693877606 < maxgini;
                  }
                }
              }
            }
          }
          else {  // if mean_col_support > 0.95297062397
            if ( median_col_coverage <= 0.500572085381 ) {
              if ( mean_col_support <= 0.96571958065 ) {
                if ( median_col_coverage <= 0.3053201437 ) {
                  if ( min_col_coverage <= 0.037235096097 ) {
                    return 0.239883788356 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.037235096097
                    return 0.105029420414 < maxgini;
                  }
                }
                else {  // if median_col_coverage > 0.3053201437
                  if ( max_col_coverage <= 0.598076939583 ) {
                    return 0.157121103418 < maxgini;
                  }
                  else {  // if max_col_coverage > 0.598076939583
                    return 0.0975082885932 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.96571958065
                if ( median_col_support <= 0.99950003624 ) {
                  if ( median_col_support <= 0.997500002384 ) {
                    return 0.345026837741 < maxgini;
                  }
                  else {  // if median_col_support > 0.997500002384
                    return false;
                  }
                }
                else {  // if median_col_support > 0.99950003624
                  if ( mean_col_support <= 0.97471010685 ) {
                    return 0.0805740883492 < maxgini;
                  }
                  else {  // if mean_col_support > 0.97471010685
                    return 0.038857758259 < maxgini;
                  }
                }
              }
            }
            else {  // if median_col_coverage > 0.500572085381
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_support <= 0.993499994278 ) {
                  if ( max_col_coverage <= 0.955050468445 ) {
                    return 0.0852640541042 < maxgini;
                  }
                  else {  // if max_col_coverage > 0.955050468445
                    return 0.444444444444 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.993499994278
                  if ( median_col_support <= 0.997500002384 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.997500002384
                    return false;
                  }
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_support <= 0.797500014305 ) {
                  if ( mean_col_support <= 0.972676515579 ) {
                    return 0.162970828234 < maxgini;
                  }
                  else {  // if mean_col_support > 0.972676515579
                    return 0.333263643974 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.797500014305
                  if ( mean_col_support <= 0.961441159248 ) {
                    return 0.240224639467 < maxgini;
                  }
                  else {  // if mean_col_support > 0.961441159248
                    return 0.135318185514 < maxgini;
                  }
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.600261092186
          if ( min_col_support <= 0.808500051498 ) {
            if ( min_col_coverage <= 0.857724070549 ) {
              if ( mean_col_support <= 0.973558783531 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  if ( median_col_support <= 0.992499947548 ) {
                    return 0.350151846129 < maxgini;
                  }
                  else {  // if median_col_support > 0.992499947548
                    return false;
                  }
                }
                else {  // if median_col_support > 0.99950003624
                  if ( min_col_support <= 0.78250002861 ) {
                    return 0.38299994825 < maxgini;
                  }
                  else {  // if min_col_support > 0.78250002861
                    return 0.274170751351 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.973558783531
                if ( median_col_support <= 0.99950003624 ) {
                  if ( median_col_support <= 0.997500002384 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.997500002384
                    return false;
                  }
                }
                else {  // if median_col_support > 0.99950003624
                  if ( mean_col_support <= 0.985676527023 ) {
                    return 0.440666211787 < maxgini;
                  }
                  else {  // if mean_col_support > 0.985676527023
                    return 0.499404480444 < maxgini;
                  }
                }
              }
            }
            else {  // if min_col_coverage > 0.857724070549
              if ( median_col_coverage <= 0.998701334 ) {
                if ( min_col_coverage <= 0.913142740726 ) {
                  if ( max_col_coverage <= 0.998721241951 ) {
                    return false;
                  }
                  else {  // if max_col_coverage > 0.998721241951
                    return 0.482972016838 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.913142740726
                  if ( median_col_coverage <= 0.964724779129 ) {
                    return false;
                  }
                  else {  // if median_col_coverage > 0.964724779129
                    return false;
                  }
                }
              }
              else {  // if median_col_coverage > 0.998701334
                if ( min_col_coverage <= 0.960676312447 ) {
                  if ( mean_col_coverage <= 0.985892057419 ) {
                    return 0.219590178769 < maxgini;
                  }
                  else {  // if mean_col_coverage > 0.985892057419
                    return 0.40030062102 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.960676312447
                  if ( min_col_coverage <= 0.996280968189 ) {
                    return false;
                  }
                  else {  // if min_col_coverage > 0.996280968189
                    return 0.479312089324 < maxgini;
                  }
                }
              }
            }
          }
          else {  // if min_col_support > 0.808500051498
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_support <= 0.993499994278 ) {
                if ( min_col_coverage <= 0.880789458752 ) {
                  if ( median_col_support <= 0.962499976158 ) {
                    return 0.435424804688 < maxgini;
                  }
                  else {  // if median_col_support > 0.962499976158
                    return 0.121174159226 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.880789458752
                  if ( median_col_support <= 0.954499959946 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.954499959946
                    return 0.452060089155 < maxgini;
                  }
                }
              }
              else {  // if median_col_support > 0.993499994278
                if ( median_col_support <= 0.996500015259 ) {
                  if ( min_col_support <= 0.840499997139 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.840499997139
                    return 0.415224913495 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.996500015259
                  if ( mean_col_support <= 0.985558867455 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.985558867455
                    return false;
                  }
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_coverage <= 0.889069855213 ) {
                if ( min_col_support <= 0.831499993801 ) {
                  if ( mean_col_support <= 0.987735271454 ) {
                    return 0.281759219546 < maxgini;
                  }
                  else {  // if mean_col_support > 0.987735271454
                    return 0.412718145994 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.831499993801
                  if ( mean_col_support <= 0.98914706707 ) {
                    return 0.181114720707 < maxgini;
                  }
                  else {  // if mean_col_support > 0.98914706707
                    return 0.306674755031 < maxgini;
                  }
                }
              }
              else {  // if min_col_coverage > 0.889069855213
                if ( median_col_coverage <= 0.998777508736 ) {
                  if ( median_col_coverage <= 0.958411931992 ) {
                    return 0.432844615678 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.958411931992
                    return false;
                  }
                }
                else {  // if median_col_coverage > 0.998777508736
                  if ( min_col_coverage <= 0.971509754658 ) {
                    return 0.152996465993 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.971509754658
                    return 0.432716345547 < maxgini;
                  }
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.850499987602
        if ( mean_col_support <= 0.985766947269 ) {
          if ( median_col_support <= 0.99950003624 ) {
            if ( median_col_support <= 0.996250033379 ) {
              if ( median_col_coverage <= 0.964265465736 ) {
                if ( median_col_support <= 0.950500011444 ) {
                  if ( median_col_support <= 0.898999989033 ) {
                    return 0.498934911243 < maxgini;
                  }
                  else {  // if median_col_support > 0.898999989033
                    return 0.348002325034 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.950500011444
                  if ( median_col_support <= 0.994500041008 ) {
                    return 0.0820927480115 < maxgini;
                  }
                  else {  // if median_col_support > 0.994500041008
                    return 0.311691106581 < maxgini;
                  }
                }
              }
              else {  // if median_col_coverage > 0.964265465736
                if ( median_col_support <= 0.976750016212 ) {
                  if ( min_col_support <= 0.922999978065 ) {
                    return 0.231111111111 < maxgini;
                  }
                  else {  // if min_col_support > 0.922999978065
                    return 0.497448979592 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.976750016212
                  if ( min_col_coverage <= 0.987975895405 ) {
                    return 0.492670272879 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.987975895405
                    return false;
                  }
                }
              }
            }
            else {  // if median_col_support > 0.996250033379
              if ( min_col_support <= 0.885499954224 ) {
                if ( mean_col_coverage <= 0.589460790157 ) {
                  if ( min_col_support <= 0.855499982834 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.855499982834
                    return 0.473054846939 < maxgini;
                  }
                }
                else {  // if mean_col_coverage > 0.589460790157
                  if ( mean_col_support <= 0.975205898285 ) {
                    return 0.491900826446 < maxgini;
                  }
                  else {  // if mean_col_support > 0.975205898285
                    return false;
                  }
                }
              }
              else {  // if min_col_support > 0.885499954224
                if ( mean_col_coverage <= 0.745734214783 ) {
                  if ( min_col_support <= 0.903499960899 ) {
                    return 0.476625273923 < maxgini;
                  }
                  else {  // if min_col_support > 0.903499960899
                    return 0.21875 < maxgini;
                  }
                }
                else {  // if mean_col_coverage > 0.745734214783
                  if ( mean_col_support <= 0.984323501587 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.984323501587
                    return 0.488165680473 < maxgini;
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.99950003624
            if ( mean_col_support <= 0.971323490143 ) {
              if ( mean_col_support <= 0.9612647295 ) {
                if ( min_col_support <= 0.884500026703 ) {
                  if ( median_col_coverage <= 0.767948746681 ) {
                    return 0.207560433271 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.767948746681
                    return 0.386377358975 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.884500026703
                  if ( mean_col_support <= 0.958382368088 ) {
                    return 0.464444444444 < maxgini;
                  }
                  else {  // if mean_col_support > 0.958382368088
                    return 0.298140574575 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.9612647295
                if ( min_col_support <= 0.894500017166 ) {
                  if ( median_col_coverage <= 0.59221470356 ) {
                    return 0.11783287022 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.59221470356
                    return 0.169817991367 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.894500017166
                  if ( mean_col_support <= 0.962911725044 ) {
                    return 0.481665912177 < maxgini;
                  }
                  else {  // if mean_col_support > 0.962911725044
                    return 0.169814556757 < maxgini;
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.971323490143
              if ( min_col_coverage <= 0.92120051384 ) {
                if ( mean_col_support <= 0.980088233948 ) {
                  if ( mean_col_support <= 0.976251006126 ) {
                    return 0.0935220649293 < maxgini;
                  }
                  else {  // if mean_col_support > 0.976251006126
                    return 0.0698988481779 < maxgini;
                  }
                }
                else {  // if mean_col_support > 0.980088233948
                  if ( median_col_coverage <= 0.850120782852 ) {
                    return 0.047595263425 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.850120782852
                    return 0.0752773403031 < maxgini;
                  }
                }
              }
              else {  // if min_col_coverage > 0.92120051384
                if ( min_col_support <= 0.868499994278 ) {
                  if ( median_col_coverage <= 0.997890293598 ) {
                    return 0.477961600684 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.997890293598
                    return 0.315486068652 < maxgini;
                  }
                }
                else {  // if min_col_support > 0.868499994278
                  if ( min_col_coverage <= 0.963031291962 ) {
                    return 0.117365324873 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.963031291962
                    return 0.275918963437 < maxgini;
                  }
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.985766947269
          if ( median_col_support <= 0.99950003624 ) {
            if ( min_col_support <= 0.911499977112 ) {
              if ( min_col_support <= 0.883499979973 ) {
                if ( median_col_support <= 0.99849998951 ) {
                  if ( min_col_support <= 0.869500041008 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.869500041008
                    return 0.446998377501 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.99849998951
                  if ( min_col_support <= 0.863499999046 ) {
                    return false;
                  }
                  else {  // if min_col_support > 0.863499999046
                    return false;
                  }
                }
              }
              else {  // if min_col_support > 0.883499979973
                if ( median_col_support <= 0.99849998951 ) {
                  if ( median_col_coverage <= 0.871900081635 ) {
                    return 0.294912258553 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.871900081635
                    return 0.49560546875 < maxgini;
                  }
                }
                else {  // if median_col_support > 0.99849998951
                  if ( mean_col_support <= 0.987470567226 ) {
                    return false;
                  }
                  else {  // if mean_col_support > 0.987470567226
                    return 0.492727456377 < maxgini;
                  }
                }
              }
            }
            else {  // if min_col_support > 0.911499977112
              if ( min_col_support <= 0.945500016212 ) {
                if ( mean_col_coverage <= 0.802882909775 ) {
                  if ( median_col_support <= 0.99849998951 ) {
                    return 0.0846664564659 < maxgini;
                  }
                  else {  // if median_col_support > 0.99849998951
                    return 0.291883454735 < maxgini;
                  }
                }
                else {  // if mean_col_coverage > 0.802882909775
                  if ( mean_col_coverage <= 0.992217540741 ) {
                    return 0.380945084726 < maxgini;
                  }
                  else {  // if mean_col_coverage > 0.992217540741
                    return false;
                  }
                }
              }
              else {  // if min_col_support > 0.945500016212
                if ( median_col_coverage <= 0.999225974083 ) {
                  if ( median_col_support <= 0.977499961853 ) {
                    return false;
                  }
                  else {  // if median_col_support > 0.977499961853
                    return 0.0132470244676 < maxgini;
                  }
                }
                else {  // if median_col_coverage > 0.999225974083
                  if ( min_col_support <= 0.960500001907 ) {
                    return 0.391111111111 < maxgini;
                  }
                  else {  // if min_col_support > 0.960500001907
                    return 0.201197122881 < maxgini;
                  }
                }
              }
            }
          }
          else {  // if median_col_support > 0.99950003624
            if ( mean_col_support <= 0.99217915535 ) {
              if ( min_col_support <= 0.879500031471 ) {
                if ( min_col_coverage <= 0.875277757645 ) {
                  if ( median_col_coverage <= 0.667136192322 ) {
                    return 0.0238431993143 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.667136192322
                    return 0.110753849578 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.875277757645
                  if ( median_col_coverage <= 0.998871326447 ) {
                    return 0.345329820575 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.998871326447
                    return 0.161491365972 < maxgini;
                  }
                }
              }
              else {  // if min_col_support > 0.879500031471
                if ( mean_col_support <= 0.988970637321 ) {
                  if ( min_col_coverage <= 0.966158986092 ) {
                    return 0.0274852570054 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.966158986092
                    return 0.122478555943 < maxgini;
                  }
                }
                else {  // if mean_col_support > 0.988970637321
                  if ( min_col_coverage <= 0.962073266506 ) {
                    return 0.0135454868414 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.962073266506
                    return 0.0804671534252 < maxgini;
                  }
                }
              }
            }
            else {  // if mean_col_support > 0.99217915535
              if ( mean_col_support <= 0.993775248528 ) {
                if ( min_col_coverage <= 0.968501985073 ) {
                  if ( min_col_support <= 0.904500007629 ) {
                    return 0.0148915123933 < maxgini;
                  }
                  else {  // if min_col_support > 0.904500007629
                    return 0.00686432075258 < maxgini;
                  }
                }
                else {  // if min_col_coverage > 0.968501985073
                  if ( median_col_coverage <= 0.9987834692 ) {
                    return 0.195529419626 < maxgini;
                  }
                  else {  // if median_col_coverage > 0.9987834692
                    return 0.0419387755102 < maxgini;
                  }
                }
              }
              else {  // if mean_col_support > 0.993775248528
                if ( mean_col_support <= 0.99661552906 ) {
                  if ( min_col_coverage <= 0.978019356728 ) {
                    return 0.00307662850781 < maxgini;
                  }
                  else {  // if min_col_coverage > 0.978019356728
                    return 0.0180062636383 < maxgini;
                  }
                }
                else {  // if mean_col_support > 0.99661552906
                  if ( mean_col_support <= 0.997735261917 ) {
                    return 0.00156424696204 < maxgini;
                  }
                  else {  // if mean_col_support > 0.997735261917
                    return 0.000462411422773 < maxgini;
                  }
                }
              }
            }
          }
        }
      }
    }
  }


#else

//this is with all features, not only those with support > 0.5

//coverage is normalized to number of reads in msa
bool shouldCorrect(double min_col_support, double min_col_coverage,
    double max_col_support, double max_col_coverage,
    double mean_col_support, double mean_col_coverage,
    double median_col_support, double median_col_coverage,
    double maxgini){
  if ( min_col_support <= 0.766499996185 ) {
    if ( min_col_coverage <= 0.454648315907 ) {
      if ( median_col_coverage <= 0.304423034191 ) {
        if ( median_col_support <= 0.74950003624 ) {
          if ( mean_col_support <= 0.851878583431 ) {
            if ( mean_col_coverage <= 0.319407910109 ) {
              if ( min_col_support <= 0.477499991655 ) {
                if ( median_col_support <= 0.529500007629 ) {
                  return 0.479597325537 < maxgini;
                }
                else {  // if median_col_support > 0.529500007629
                  return 0.302866414278 < maxgini;
                }
              }
              else {  // if min_col_support > 0.477499991655
                if ( median_col_support <= 0.559499979019 ) {
                  return 0.488839462866 < maxgini;
                }
                else {  // if median_col_support > 0.559499979019
                  return 0.447969956696 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.319407910109
              if ( median_col_support <= 0.619500041008 ) {
                if ( min_col_support <= 0.491500020027 ) {
                  return 0.475175571033 < maxgini;
                }
                else {  // if min_col_support > 0.491500020027
                  return false;
                }
              }
              else {  // if median_col_support > 0.619500041008
                if ( min_col_support <= 0.490000009537 ) {
                  return 0.315054204713 < maxgini;
                }
                else {  // if min_col_support > 0.490000009537
                  return 0.477757974386 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.851878583431
            if ( median_col_support <= 0.6875 ) {
              if ( mean_col_coverage <= 0.251055836678 ) {
                if ( min_col_coverage <= 0.0506410263479 ) {
                  return 0.410187793228 < maxgini;
                }
                else {  // if min_col_coverage > 0.0506410263479
                  return 0.315511759208 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.251055836678
                if ( median_col_support <= 0.606500029564 ) {
                  return false;
                }
                else {  // if median_col_support > 0.606500029564
                  return 0.452039286125 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.6875
              if ( min_col_coverage <= 0.0519037358463 ) {
                if ( mean_col_coverage <= 0.202869147062 ) {
                  return 0.358675287462 < maxgini;
                }
                else {  // if mean_col_coverage > 0.202869147062
                  return 0.458607057373 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0519037358463
                if ( mean_col_coverage <= 0.268653959036 ) {
                  return 0.245238208617 < maxgini;
                }
                else {  // if mean_col_coverage > 0.268653959036
                  return 0.371277106589 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.74950003624
          if ( mean_col_support <= 0.935433030128 ) {
            if ( median_col_coverage <= 0.0509556420147 ) {
              if ( mean_col_coverage <= 0.104911394417 ) {
                if ( min_col_support <= 0.580500006676 ) {
                  return 0.180793552081 < maxgini;
                }
                else {  // if min_col_support > 0.580500006676
                  return 0.27552091078 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.104911394417
                if ( mean_col_support <= 0.888332724571 ) {
                  return 0.496632514482 < maxgini;
                }
                else {  // if mean_col_support > 0.888332724571
                  return 0.333954204532 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.0509556420147
              if ( mean_col_coverage <= 0.299765050411 ) {
                if ( mean_col_support <= 0.888379812241 ) {
                  return 0.307964104547 < maxgini;
                }
                else {  // if mean_col_support > 0.888379812241
                  return 0.190814316092 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.299765050411
                if ( median_col_support <= 0.949499964714 ) {
                  return 0.280009786122 < maxgini;
                }
                else {  // if median_col_support > 0.949499964714
                  return false;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.935433030128
            if ( min_col_coverage <= 0.200290709734 ) {
              if ( min_col_support <= 0.550500035286 ) {
                if ( min_col_coverage <= 0.150221243501 ) {
                  return 0.189085960677 < maxgini;
                }
                else {  // if min_col_coverage > 0.150221243501
                  return 0.410376117389 < maxgini;
                }
              }
              else {  // if min_col_support > 0.550500035286
                if ( mean_col_support <= 0.957182049751 ) {
                  return 0.135007299242 < maxgini;
                }
                else {  // if mean_col_support > 0.957182049751
                  return 0.071642077646 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.200290709734
              if ( min_col_support <= 0.600499987602 ) {
                if ( median_col_support <= 0.961500048637 ) {
                  return 0.265853182498 < maxgini;
                }
                else {  // if median_col_support > 0.961500048637
                  return false;
                }
              }
              else {  // if min_col_support > 0.600499987602
                if ( min_col_support <= 0.660500049591 ) {
                  return 0.328156137577 < maxgini;
                }
                else {  // if min_col_support > 0.660500049591
                  return 0.14878280298 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_coverage > 0.304423034191
        if ( min_col_support <= 0.620499968529 ) {
          if ( median_col_support <= 0.976500034332 ) {
            if ( median_col_support <= 0.651499986649 ) {
              if ( median_col_support <= 0.588500022888 ) {
                if ( min_col_support <= 0.499500006437 ) {
                  return false;
                }
                else {  // if min_col_support > 0.499500006437
                  return false;
                }
              }
              else {  // if median_col_support > 0.588500022888
                if ( mean_col_coverage <= 0.420652210712 ) {
                  return 0.488142304045 < maxgini;
                }
                else {  // if mean_col_coverage > 0.420652210712
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.651499986649
              if ( median_col_support <= 0.926499962807 ) {
                if ( median_col_support <= 0.706499993801 ) {
                  return 0.476753062618 < maxgini;
                }
                else {  // if median_col_support > 0.706499993801
                  return 0.351813778115 < maxgini;
                }
              }
              else {  // if median_col_support > 0.926499962807
                if ( min_col_support <= 0.543500006199 ) {
                  return false;
                }
                else {  // if min_col_support > 0.543500006199
                  return 0.462983432279 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.976500034332
            if ( min_col_coverage <= 0.350187540054 ) {
              if ( min_col_support <= 0.560500025749 ) {
                if ( median_col_support <= 0.988499999046 ) {
                  return false;
                }
                else {  // if median_col_support > 0.988499999046
                  return false;
                }
              }
              else {  // if min_col_support > 0.560500025749
                if ( mean_col_support <= 0.973499953747 ) {
                  return 0.498835098663 < maxgini;
                }
                else {  // if mean_col_support > 0.973499953747
                  return false;
                }
              }
            }
            else {  // if min_col_coverage > 0.350187540054
              if ( min_col_support <= 0.555500030518 ) {
                if ( mean_col_support <= 0.971029400826 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.971029400826
                  return false;
                }
              }
              else {  // if min_col_support > 0.555500030518
                if ( mean_col_support <= 0.973852932453 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.973852932453
                  return false;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.620499968529
          if ( median_col_support <= 0.764500021935 ) {
            if ( min_col_coverage <= 0.351011097431 ) {
              if ( median_col_support <= 0.743499994278 ) {
                if ( mean_col_coverage <= 0.395481199026 ) {
                  return 0.426284961456 < maxgini;
                }
                else {  // if mean_col_coverage > 0.395481199026
                  return 0.469362457311 < maxgini;
                }
              }
              else {  // if median_col_support > 0.743499994278
                if ( median_col_support <= 0.752499997616 ) {
                  return 0.340822281581 < maxgini;
                }
                else {  // if median_col_support > 0.752499997616
                  return 0.432960075246 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.351011097431
              if ( median_col_support <= 0.712499976158 ) {
                if ( median_col_coverage <= 0.412004798651 ) {
                  return 0.494510198681 < maxgini;
                }
                else {  // if median_col_coverage > 0.412004798651
                  return false;
                }
              }
              else {  // if median_col_support > 0.712499976158
                if ( min_col_coverage <= 0.356653630733 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.356653630733
                  return 0.461432526945 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.764500021935
            if ( median_col_support <= 0.976500034332 ) {
              if ( median_col_support <= 0.816499948502 ) {
                if ( mean_col_coverage <= 0.455973148346 ) {
                  return 0.327875462567 < maxgini;
                }
                else {  // if mean_col_coverage > 0.455973148346
                  return 0.397212657003 < maxgini;
                }
              }
              else {  // if median_col_support > 0.816499948502
                if ( max_col_coverage <= 0.922649562359 ) {
                  return 0.191821166526 < maxgini;
                }
                else {  // if max_col_coverage > 0.922649562359
                  return 0.460190879954 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.976500034332
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_support <= 0.990499973297 ) {
                  return 0.480853739939 < maxgini;
                }
                else {  // if median_col_support > 0.990499973297
                  return false;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_support <= 0.702499985695 ) {
                  return 0.460975260585 < maxgini;
                }
                else {  // if min_col_support > 0.702499985695
                  return 0.228327731192 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if min_col_coverage > 0.454648315907
      if ( median_col_support <= 0.985499978065 ) {
        if ( min_col_support <= 0.631500005722 ) {
          if ( min_col_coverage <= 0.625203609467 ) {
            if ( median_col_support <= 0.658499956131 ) {
              if ( median_col_support <= 0.586500048637 ) {
                if ( max_col_coverage <= 0.591750860214 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.591750860214
                  return false;
                }
              }
              else {  // if median_col_support > 0.586500048637
                if ( min_col_support <= 0.543500006199 ) {
                  return false;
                }
                else {  // if min_col_support > 0.543500006199
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.658499956131
              if ( median_col_support <= 0.87650001049 ) {
                if ( mean_col_coverage <= 0.636569023132 ) {
                  return 0.424442058628 < maxgini;
                }
                else {  // if mean_col_coverage > 0.636569023132
                  return 0.496931342776 < maxgini;
                }
              }
              else {  // if median_col_support > 0.87650001049
                if ( min_col_support <= 0.53149998188 ) {
                  return false;
                }
                else {  // if min_col_support > 0.53149998188
                  return false;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.625203609467
            if ( min_col_coverage <= 0.74082493782 ) {
              if ( max_col_coverage <= 0.731443405151 ) {
                if ( median_col_support <= 0.674000024796 ) {
                  return false;
                }
                else {  // if median_col_support > 0.674000024796
                  return 0.44603735224 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.731443405151
                if ( max_col_coverage <= 0.899852514267 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.899852514267
                  return false;
                }
              }
            }
            else {  // if min_col_coverage > 0.74082493782
              if ( median_col_coverage <= 0.985764920712 ) {
                if ( max_col_coverage <= 0.998983740807 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.998983740807
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.985764920712
                if ( mean_col_support <= 0.94732350111 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.94732350111
                  return 0.498314957427 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.631500005722
          if ( min_col_coverage <= 0.7506762743 ) {
            if ( median_col_support <= 0.747500002384 ) {
              if ( median_col_support <= 0.713500022888 ) {
                if ( mean_col_coverage <= 0.656227707863 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.656227707863
                  return false;
                }
              }
              else {  // if median_col_support > 0.713500022888
                if ( min_col_support <= 0.65750002861 ) {
                  return 0.476702439403 < maxgini;
                }
                else {  // if min_col_support > 0.65750002861
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.747500002384
              if ( min_col_support <= 0.690500020981 ) {
                if ( max_col_coverage <= 0.714619517326 ) {
                  return 0.362826085172 < maxgini;
                }
                else {  // if max_col_coverage > 0.714619517326
                  return 0.479652005859 < maxgini;
                }
              }
              else {  // if min_col_support > 0.690500020981
                if ( median_col_support <= 0.801499962807 ) {
                  return 0.452235898923 < maxgini;
                }
                else {  // if median_col_support > 0.801499962807
                  return 0.308164157973 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.7506762743
            if ( min_col_support <= 0.702499985695 ) {
              if ( max_col_coverage <= 0.998659491539 ) {
                if ( max_col_coverage <= 0.958863794804 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.958863794804
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.998659491539
                if ( min_col_coverage <= 0.875753045082 ) {
                  return 0.48347107438 < maxgini;
                }
                else {  // if min_col_coverage > 0.875753045082
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.702499985695
              if ( min_col_coverage <= 0.863755345345 ) {
                if ( max_col_coverage <= 0.998392283916 ) {
                  return 0.494755423016 < maxgini;
                }
                else {  // if max_col_coverage > 0.998392283916
                  return 0.329773826829 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.863755345345
                if ( max_col_coverage <= 0.9987834692 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.9987834692
                  return false;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.985499978065
        if ( min_col_support <= 0.682500004768 ) {
          if ( min_col_support <= 0.614500045776 ) {
            if ( median_col_coverage <= 0.654041230679 ) {
              if ( mean_col_support <= 0.963617682457 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return false;
                }
              }
              else {  // if mean_col_support > 0.963617682457
                if ( median_col_support <= 0.993499994278 ) {
                  return false;
                }
                else {  // if median_col_support > 0.993499994278
                  return false;
                }
              }
            }
            else {  // if median_col_coverage > 0.654041230679
              if ( median_col_coverage <= 0.993710398674 ) {
                if ( mean_col_support <= 0.966264724731 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.966264724731
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.993710398674
                if ( mean_col_support <= 0.948499977589 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.948499977589
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.614500045776
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_support <= 0.995499968529 ) {
                if ( mean_col_support <= 0.972911775112 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.972911775112
                  return false;
                }
              }
              else {  // if median_col_support > 0.995499968529
                if ( mean_col_support <= 0.975264728069 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.975264728069
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( mean_col_support <= 0.976970613003 ) {
                if ( mean_col_support <= 0.973558783531 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.973558783531
                  return 0.498389805789 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.976970613003
                if ( min_col_support <= 0.639500021935 ) {
                  return false;
                }
                else {  // if min_col_support > 0.639500021935
                  return false;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.682500004768
          if ( median_col_support <= 0.99950003624 ) {
            if ( median_col_support <= 0.991500020027 ) {
              if ( mean_col_support <= 0.977911770344 ) {
                if ( median_col_support <= 0.989500045776 ) {
                  return false;
                }
                else {  // if median_col_support > 0.989500045776
                  return false;
                }
              }
              else {  // if mean_col_support > 0.977911770344
                if ( median_col_support <= 0.989500045776 ) {
                  return 0.415378144528 < maxgini;
                }
                else {  // if median_col_support > 0.989500045776
                  return 0.499873130237 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.991500020027
              if ( median_col_support <= 0.994500041008 ) {
                if ( mean_col_support <= 0.978852987289 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.978852987289
                  return false;
                }
              }
              else {  // if median_col_support > 0.994500041008
                if ( mean_col_support <= 0.981029391289 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.981029391289
                  return false;
                }
              }
            }
          }
          else {  // if median_col_support > 0.99950003624
            if ( median_col_coverage <= 0.667019784451 ) {
              if ( mean_col_support <= 0.980676472187 ) {
                if ( min_col_support <= 0.730499982834 ) {
                  return 0.392772685313 < maxgini;
                }
                else {  // if min_col_support > 0.730499982834
                  return 0.255639023931 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.980676472187
                if ( min_col_support <= 0.710500001907 ) {
                  return false;
                }
                else {  // if min_col_support > 0.710500001907
                  return 0.442729867334 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.667019784451
              if ( mean_col_support <= 0.980852901936 ) {
                if ( mean_col_support <= 0.977205872536 ) {
                  return 0.499529410251 < maxgini;
                }
                else {  // if mean_col_support > 0.977205872536
                  return 0.440457080497 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.980852901936
                if ( min_col_support <= 0.733500003815 ) {
                  return false;
                }
                else {  // if min_col_support > 0.733500003815
                  return 0.492847563867 < maxgini;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if min_col_support > 0.766499996185
    if ( min_col_support <= 0.850499987602 ) {
      if ( median_col_coverage <= 0.667356133461 ) {
        if ( median_col_support <= 0.99950003624 ) {
          if ( median_col_support <= 0.989500045776 ) {
            if ( median_col_support <= 0.861500024796 ) {
              if ( mean_col_coverage <= 0.413277000189 ) {
                if ( median_col_support <= 0.824499964714 ) {
                  return 0.28917484804 < maxgini;
                }
                else {  // if median_col_support > 0.824499964714
                  return 0.187498150094 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.413277000189
                if ( median_col_support <= 0.838500022888 ) {
                  return 0.383993726002 < maxgini;
                }
                else {  // if median_col_support > 0.838500022888
                  return 0.290852050624 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.861500024796
              if ( mean_col_support <= 0.951029419899 ) {
                if ( mean_col_support <= 0.939485371113 ) {
                  return 0.317107586481 < maxgini;
                }
                else {  // if mean_col_support > 0.939485371113
                  return 0.186901121272 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.951029419899
                if ( median_col_support <= 0.984500050545 ) {
                  return 0.0998234910638 < maxgini;
                }
                else {  // if median_col_support > 0.984500050545
                  return 0.307072659117 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.989500045776
            if ( median_col_support <= 0.993499994278 ) {
              if ( mean_col_support <= 0.982382416725 ) {
                if ( median_col_coverage <= 0.34741461277 ) {
                  return 0.355900277008 < maxgini;
                }
                else {  // if median_col_coverage > 0.34741461277
                  return false;
                }
              }
              else {  // if mean_col_support > 0.982382416725
                if ( median_col_support <= 0.992499947548 ) {
                  return 0.356468092589 < maxgini;
                }
                else {  // if median_col_support > 0.992499947548
                  return 0.465901253601 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.993499994278
              if ( mean_col_support <= 0.983088254929 ) {
                if ( min_col_support <= 0.839499950409 ) {
                  return false;
                }
                else {  // if min_col_support > 0.839499950409
                  return false;
                }
              }
              else {  // if mean_col_support > 0.983088254929
                if ( median_col_support <= 0.997500002384 ) {
                  return false;
                }
                else {  // if median_col_support > 0.997500002384
                  return false;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.99950003624
          if ( mean_col_support <= 0.974121332169 ) {
            if ( mean_col_coverage <= 0.35667693615 ) {
              if ( mean_col_support <= 0.961727917194 ) {
                if ( mean_col_coverage <= 0.229935437441 ) {
                  return 0.0890907502839 < maxgini;
                }
                else {  // if mean_col_coverage > 0.229935437441
                  return 0.232302201071 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.961727917194
                if ( min_col_support <= 0.845499992371 ) {
                  return 0.0653583017803 < maxgini;
                }
                else {  // if min_col_support > 0.845499992371
                  return 0.187049351066 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.35667693615
              if ( min_col_support <= 0.825500011444 ) {
                if ( mean_col_support <= 0.967852950096 ) {
                  return 0.239336926607 < maxgini;
                }
                else {  // if mean_col_support > 0.967852950096
                  return 0.124240029184 < maxgini;
                }
              }
              else {  // if min_col_support > 0.825500011444
                if ( min_col_coverage <= 0.540064096451 ) {
                  return 0.267265066455 < maxgini;
                }
                else {  // if min_col_coverage > 0.540064096451
                  return 0.0482795804333 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.974121332169
            if ( mean_col_coverage <= 0.508183300495 ) {
              if ( mean_col_support <= 0.982242703438 ) {
                if ( median_col_coverage <= 0.0230991542339 ) {
                  return 0.200184089415 < maxgini;
                }
                else {  // if median_col_coverage > 0.0230991542339
                  return 0.0360852243514 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.982242703438
                if ( min_col_support <= 0.784500002861 ) {
                  return 0.0562906952321 < maxgini;
                }
                else {  // if min_col_support > 0.784500002861
                  return 0.0162143190685 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.508183300495
              if ( min_col_support <= 0.795500040054 ) {
                if ( mean_col_support <= 0.985735297203 ) {
                  return 0.128502970895 < maxgini;
                }
                else {  // if mean_col_support > 0.985735297203
                  return 0.32963033611 < maxgini;
                }
              }
              else {  // if min_col_support > 0.795500040054
                if ( min_col_support <= 0.825500011444 ) {
                  return 0.0967479784802 < maxgini;
                }
                else {  // if min_col_support > 0.825500011444
                  return 0.0415561046288 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_coverage > 0.667356133461
        if ( median_col_support <= 0.99950003624 ) {
          if ( median_col_support <= 0.992499947548 ) {
            if ( min_col_coverage <= 0.85747051239 ) {
              if ( median_col_support <= 0.989500045776 ) {
                if ( min_col_support <= 0.803499996662 ) {
                  return 0.340346997524 < maxgini;
                }
                else {  // if min_col_support > 0.803499996662
                  return 0.197514914969 < maxgini;
                }
              }
              else {  // if median_col_support > 0.989500045776
                if ( mean_col_support <= 0.983088195324 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.983088195324
                  return 0.296305711915 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.85747051239
              if ( mean_col_support <= 0.983500003815 ) {
                if ( min_col_support <= 0.820500016212 ) {
                  return 0.497503111111 < maxgini;
                }
                else {  // if min_col_support > 0.820500016212
                  return 0.429036134028 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.983500003815
                if ( min_col_support <= 0.799499988556 ) {
                  return 0.453686200378 < maxgini;
                }
                else {  // if min_col_support > 0.799499988556
                  return 0.191238142476 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.992499947548
            if ( median_col_support <= 0.994500041008 ) {
              if ( mean_col_support <= 0.986500024796 ) {
                if ( mean_col_support <= 0.983735322952 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.983735322952
                  return false;
                }
              }
              else {  // if mean_col_support > 0.986500024796
                if ( min_col_support <= 0.828500032425 ) {
                  return 0.493728 < maxgini;
                }
                else {  // if min_col_support > 0.828500032425
                  return 0.327689357308 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.994500041008
              if ( mean_col_support <= 0.986441135406 ) {
                if ( median_col_support <= 0.996500015259 ) {
                  return false;
                }
                else {  // if median_col_support > 0.996500015259
                  return false;
                }
              }
              else {  // if mean_col_support > 0.986441135406
                if ( median_col_support <= 0.996500015259 ) {
                  return false;
                }
                else {  // if median_col_support > 0.996500015259
                  return false;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.99950003624
          if ( min_col_support <= 0.807500004768 ) {
            if ( mean_col_support <= 0.985794067383 ) {
              if ( min_col_coverage <= 0.90969979763 ) {
                if ( median_col_coverage <= 0.670660018921 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.670660018921
                  return 0.222326815507 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.90969979763
                if ( max_col_coverage <= 0.998618781567 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.998618781567
                  return 0.391708579882 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.985794067383
              if ( min_col_support <= 0.783499956131 ) {
                if ( min_col_support <= 0.778499960899 ) {
                  return false;
                }
                else {  // if min_col_support > 0.778499960899
                  return 0.494967585174 < maxgini;
                }
              }
              else {  // if min_col_support > 0.783499956131
                if ( mean_col_support <= 0.986853003502 ) {
                  return 0.23766565475 < maxgini;
                }
                else {  // if mean_col_support > 0.986853003502
                  return 0.452730710794 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.807500004768
            if ( min_col_coverage <= 0.905086398125 ) {
              if ( mean_col_support <= 0.988205850124 ) {
                if ( mean_col_coverage <= 0.991441011429 ) {
                  return 0.0702212767567 < maxgini;
                }
                else {  // if mean_col_coverage > 0.991441011429
                  return false;
                }
              }
              else {  // if mean_col_support > 0.988205850124
                if ( min_col_support <= 0.821500003338 ) {
                  return 0.337663655327 < maxgini;
                }
                else {  // if min_col_support > 0.821500003338
                  return 0.121035570863 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.905086398125
              if ( max_col_coverage <= 0.997727274895 ) {
                if ( mean_col_coverage <= 0.960866689682 ) {
                  return 0.370146079882 < maxgini;
                }
                else {  // if mean_col_coverage > 0.960866689682
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.997727274895
                if ( min_col_coverage <= 0.966849803925 ) {
                  return 0.210458060678 < maxgini;
                }
                else {  // if min_col_coverage > 0.966849803925
                  return 0.373905913312 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if min_col_support > 0.850499987602
      if ( mean_col_support <= 0.985767006874 ) {
        if ( median_col_coverage <= 0.961605429649 ) {
          if ( mean_col_support <= 0.971323490143 ) {
            if ( mean_col_support <= 0.9612647295 ) {
              if ( min_col_support <= 0.880499958992 ) {
                if ( mean_col_support <= 0.944441199303 ) {
                  return 0.348188254894 < maxgini;
                }
                else {  // if mean_col_support > 0.944441199303
                  return 0.202213662132 < maxgini;
                }
              }
              else {  // if min_col_support > 0.880499958992
                if ( mean_col_support <= 0.958368778229 ) {
                  return 0.395328029095 < maxgini;
                }
                else {  // if mean_col_support > 0.958368778229
                  return 0.269539543709 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.9612647295
              if ( median_col_support <= 0.881500005722 ) {
                if ( median_col_coverage <= 0.479130446911 ) {
                  return 0.172634848573 < maxgini;
                }
                else {  // if median_col_coverage > 0.479130446911
                  return 0.323621030697 < maxgini;
                }
              }
              else {  // if median_col_support > 0.881500005722
                if ( min_col_support <= 0.894500017166 ) {
                  return 0.111549335539 < maxgini;
                }
                else {  // if min_col_support > 0.894500017166
                  return 0.176708575338 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.971323490143
            if ( mean_col_support <= 0.980090022087 ) {
              if ( median_col_support <= 0.902500033379 ) {
                if ( min_col_support <= 0.868499994278 ) {
                  return 0.0843856037555 < maxgini;
                }
                else {  // if min_col_support > 0.868499994278
                  return 0.158777076013 < maxgini;
                }
              }
              else {  // if median_col_support > 0.902500033379
                if ( median_col_support <= 0.977499961853 ) {
                  return 0.0725508805644 < maxgini;
                }
                else {  // if median_col_support > 0.977499961853
                  return 0.113448576101 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.980090022087
              if ( median_col_coverage <= 0.800423741341 ) {
                if ( min_col_coverage <= 0.0125791141763 ) {
                  return 0.301065088757 < maxgini;
                }
                else {  // if min_col_coverage > 0.0125791141763
                  return 0.0511282405471 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.800423741341
                if ( min_col_support <= 0.87650001049 ) {
                  return 0.255043940277 < maxgini;
                }
                else {  // if min_col_support > 0.87650001049
                  return 0.0588590043703 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.961605429649
          if ( median_col_support <= 0.99849998951 ) {
            if ( median_col_support <= 0.955500006676 ) {
              if ( max_col_coverage <= 0.998299360275 ) {
                if ( max_col_coverage <= 0.988142132759 ) {
                  return 0.444444444444 < maxgini;
                }
                else {  // if max_col_coverage > 0.988142132759
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.998299360275
                if ( mean_col_support <= 0.982088208199 ) {
                  return 0.263023661019 < maxgini;
                }
                else {  // if mean_col_support > 0.982088208199
                  return 0.0336979591837 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.955500006676
              if ( median_col_support <= 0.992499947548 ) {
                if ( mean_col_support <= 0.979735314846 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.979735314846
                  return 0.434506559494 < maxgini;
                }
              }
              else {  // if median_col_support > 0.992499947548
                if ( min_col_support <= 0.889999985695 ) {
                  return false;
                }
                else {  // if min_col_support > 0.889999985695
                  return 0.375 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.99849998951
            if ( max_col_coverage <= 0.986486494541 ) {
              if ( median_col_coverage <= 0.971780598164 ) {
                if ( min_col_support <= 0.871500015259 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_support > 0.871500015259
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.971780598164
                return false;
              }
            }
            else {  // if max_col_coverage > 0.986486494541
              if ( min_col_support <= 0.856500029564 ) {
                if ( median_col_coverage <= 0.987804889679 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.987804889679
                  return 0.184089414859 < maxgini;
                }
              }
              else {  // if min_col_support > 0.856500029564
                if ( mean_col_support <= 0.985441148281 ) {
                  return 0.0502809573361 < maxgini;
                }
                else {  // if mean_col_support > 0.985441148281
                  return 0.185493460166 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.985767006874
        if ( mean_col_support <= 0.992130279541 ) {
          if ( min_col_support <= 0.879500031471 ) {
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_support <= 0.994500041008 ) {
                if ( median_col_support <= 0.991500020027 ) {
                  return 0.0598172725482 < maxgini;
                }
                else {  // if median_col_support > 0.991500020027
                  return 0.358441321903 < maxgini;
                }
              }
              else {  // if median_col_support > 0.994500041008
                if ( median_col_support <= 0.996500015259 ) {
                  return 0.486457597809 < maxgini;
                }
                else {  // if median_col_support > 0.996500015259
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( median_col_coverage <= 0.961944878101 ) {
                if ( mean_col_coverage <= 0.823789715767 ) {
                  return 0.0131893657998 < maxgini;
                }
                else {  // if mean_col_coverage > 0.823789715767
                  return 0.0372103420548 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.961944878101
                if ( median_col_coverage <= 0.962526082993 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.962526082993
                  return 0.152559707617 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.879500031471
            if ( min_col_coverage <= 0.96485054493 ) {
              if ( mean_col_support <= 0.988970577717 ) {
                if ( max_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if max_col_support > 0.99950003624
                  return 0.0293020682442 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.988970577717
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.022798026018 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0117849644582 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.96485054493
              if ( median_col_support <= 0.999000012875 ) {
                if ( min_col_support <= 0.892500042915 ) {
                  return 0.483194444444 < maxgini;
                }
                else {  // if min_col_support > 0.892500042915
                  return 0.236109145673 < maxgini;
                }
              }
              else {  // if median_col_support > 0.999000012875
                if ( min_col_coverage <= 0.965214729309 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.965214729309
                  return 0.0234405175622 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.992130279541
          if ( mean_col_support <= 0.993775248528 ) {
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_support <= 0.996500015259 ) {
                if ( min_col_support <= 0.903499960899 ) {
                  return 0.274315268853 < maxgini;
                }
                else {  // if min_col_support > 0.903499960899
                  return 0.00752588119705 < maxgini;
                }
              }
              else {  // if median_col_support > 0.996500015259
                if ( min_col_support <= 0.886500000954 ) {
                  return false;
                }
                else {  // if min_col_support > 0.886500000954
                  return 0.431954460552 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_support <= 0.955500006676 ) {
                if ( min_col_coverage <= 0.968501985073 ) {
                  return 0.00593156853136 < maxgini;
                }
                else {  // if min_col_coverage > 0.968501985073
                  return 0.0310602258713 < maxgini;
                }
              }
              else {  // if min_col_support > 0.955500006676
                if ( median_col_coverage <= 0.449137926102 ) {
                  return 0.0549519467703 < maxgini;
                }
                else {  // if median_col_coverage > 0.449137926102
                  return 0.0119526576579 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.993775248528
            if ( mean_col_support <= 0.996615588665 ) {
              if ( min_col_coverage <= 0.978019356728 ) {
                if ( median_col_coverage <= 0.534536957741 ) {
                  return 0.0045731967922 < maxgini;
                }
                else {  // if median_col_coverage > 0.534536957741
                  return 0.00262507914113 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.978019356728
                if ( median_col_support <= 0.99849998951 ) {
                  return 0.159158448389 < maxgini;
                }
                else {  // if median_col_support > 0.99849998951
                  return 0.0121771934096 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.996615588665
              if ( mean_col_support <= 0.997735261917 ) {
                if ( median_col_coverage <= 0.977185964584 ) {
                  return 0.00153593834548 < maxgini;
                }
                else {  // if median_col_coverage > 0.977185964584
                  return 0.00613760076239 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.997735261917
                if ( mean_col_support <= 0.998182058334 ) {
                  return 0.000728583383569 < maxgini;
                }
                else {  // if mean_col_support > 0.998182058334
                  return 0.000330269391951 < maxgini;
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







bool shouldCorrect0(double min_col_support, double min_col_coverage, double max_col_support, double max_col_coverage, double mean_col_support, double mean_col_coverage, double median_col_support, double median_col_coverage, double maxgini) {
  if ( mean_col_support <= 0.981710076332 ) {
    if ( median_col_coverage <= 0.600291550159 ) {
      if ( min_col_support <= 0.703500032425 ) {
        if ( mean_col_coverage <= 0.417863070965 ) {
          if ( min_col_coverage <= 0.200290709734 ) {
            if ( median_col_support <= 0.75150001049 ) {
              if ( median_col_support <= 0.617499947548 ) {
                if ( mean_col_coverage <= 0.166824802756 ) {
                  return 0.368870303511 < maxgini;
                }
                else {  // if mean_col_coverage > 0.166824802756
                  return 0.482705984191 < maxgini;
                }
              }
              else {  // if median_col_support > 0.617499947548
                if ( min_col_coverage <= 0.0507110841572 ) {
                  return 0.406908953897 < maxgini;
                }
                else {  // if min_col_coverage > 0.0507110841572
                  return 0.349756764665 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.75150001049
              if ( min_col_coverage <= 0.0708312988281 ) {
                if ( median_col_support <= 0.816499948502 ) {
                  return 0.304123170128 < maxgini;
                }
                else {  // if median_col_support > 0.816499948502
                  return 0.17032963052 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0708312988281
                if ( median_col_support <= 0.803499996662 ) {
                  return 0.213661976896 < maxgini;
                }
                else {  // if median_col_support > 0.803499996662
                  return 0.134046897141 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.200290709734
            if ( mean_col_coverage <= 0.353037297726 ) {
              if ( mean_col_coverage <= 0.273664116859 ) {
                if ( median_col_support <= 0.667500019073 ) {
                  return 0.428246366782 < maxgini;
                }
                else {  // if median_col_support > 0.667500019073
                  return 0.227110959412 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.273664116859
                if ( min_col_coverage <= 0.203085333109 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.203085333109
                  return 0.395318008076 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.353037297726
              if ( min_col_support <= 0.590499997139 ) {
                if ( max_col_coverage <= 0.545141100883 ) {
                  return 0.496410083641 < maxgini;
                }
                else {  // if max_col_coverage > 0.545141100883
                  return 0.422243461249 < maxgini;
                }
              }
              else {  // if min_col_support > 0.590499997139
                if ( min_col_coverage <= 0.285954773426 ) {
                  return 0.350390302079 < maxgini;
                }
                else {  // if min_col_coverage > 0.285954773426
                  return 0.416377553355 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.417863070965
          if ( median_col_coverage <= 0.458393037319 ) {
            if ( min_col_coverage <= 0.350174427032 ) {
              if ( mean_col_coverage <= 0.650661468506 ) {
                if ( min_col_support <= 0.554499983788 ) {
                  return 0.496616185781 < maxgini;
                }
                else {  // if min_col_support > 0.554499983788
                  return 0.401208115827 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.650661468506
                if ( max_col_coverage <= 0.982975959778 ) {
                  return 0.432853670769 < maxgini;
                }
                else {  // if max_col_coverage > 0.982975959778
                  return false;
                }
              }
            }
            else {  // if min_col_coverage > 0.350174427032
              if ( min_col_support <= 0.617499947548 ) {
                if ( mean_col_coverage <= 0.528965473175 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.528965473175
                  return 0.499344038688 < maxgini;
                }
              }
              else {  // if min_col_support > 0.617499947548
                if ( min_col_coverage <= 0.441763579845 ) {
                  return 0.465633725172 < maxgini;
                }
                else {  // if min_col_coverage > 0.441763579845
                  return 0.351292585501 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.458393037319
            if ( min_col_support <= 0.627499997616 ) {
              if ( mean_col_support <= 0.934499979019 ) {
                if ( mean_col_support <= 0.842764735222 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.842764735222
                  return 0.49889256024 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.934499979019
                if ( mean_col_coverage <= 0.691702306271 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.691702306271
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.627499997616
              if ( median_col_support <= 0.984500050545 ) {
                if ( max_col_coverage <= 0.75746434927 ) {
                  return 0.425455212496 < maxgini;
                }
                else {  // if max_col_coverage > 0.75746434927
                  return 0.3619649526 < maxgini;
                }
              }
              else {  // if median_col_support > 0.984500050545
                if ( min_col_coverage <= 0.425313353539 ) {
                  return 0.485361939597 < maxgini;
                }
                else {  // if min_col_coverage > 0.425313353539
                  return false;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.703500032425
        if ( median_col_support <= 0.838500022888 ) {
          if ( mean_col_coverage <= 0.391240417957 ) {
            if ( median_col_support <= 0.81350004673 ) {
              if ( mean_col_coverage <= 0.300378531218 ) {
                if ( max_col_coverage <= 0.289022743702 ) {
                  return 0.226915082243 < maxgini;
                }
                else {  // if max_col_coverage > 0.289022743702
                  return 0.275988796221 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.300378531218
                if ( mean_col_support <= 0.860205829144 ) {
                  return 0.498866213152 < maxgini;
                }
                else {  // if mean_col_support > 0.860205829144
                  return 0.331880206324 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.81350004673
              if ( mean_col_support <= 0.918558776379 ) {
                if ( min_col_support <= 0.714499950409 ) {
                  return 0.0840478203116 < maxgini;
                }
                else {  // if min_col_support > 0.714499950409
                  return 0.39154579686 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.918558776379
                if ( min_col_coverage <= 0.32071429491 ) {
                  return 0.199560718728 < maxgini;
                }
                else {  // if min_col_coverage > 0.32071429491
                  return 0.0905391143431 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.391240417957
            if ( mean_col_support <= 0.914676427841 ) {
              if ( median_col_support <= 0.757500052452 ) {
                if ( mean_col_coverage <= 0.614209294319 ) {
                  return 0.452336418097 < maxgini;
                }
                else {  // if mean_col_coverage > 0.614209294319
                  return false;
                }
              }
              else {  // if median_col_support > 0.757500052452
                if ( median_col_coverage <= 0.488346874714 ) {
                  return 0.380633760968 < maxgini;
                }
                else {  // if median_col_coverage > 0.488346874714
                  return 0.438147412433 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.914676427841
              if ( mean_col_support <= 0.960676550865 ) {
                if ( min_col_coverage <= 0.350675672293 ) {
                  return 0.355675343072 < maxgini;
                }
                else {  // if min_col_coverage > 0.350675672293
                  return 0.386957744187 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.960676550865
                if ( median_col_coverage <= 0.459415584803 ) {
                  return 0.464268208999 < maxgini;
                }
                else {  // if median_col_coverage > 0.459415584803
                  return false;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.838500022888
          if ( min_col_support <= 0.784500002861 ) {
            if ( mean_col_support <= 0.978343129158 ) {
              if ( median_col_coverage <= 0.450172632933 ) {
                if ( max_col_coverage <= 0.400389105082 ) {
                  return 0.0842843673899 < maxgini;
                }
                else {  // if max_col_coverage > 0.400389105082
                  return 0.142750871146 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.450172632933
                if ( min_col_support <= 0.728500008583 ) {
                  return 0.401789962497 < maxgini;
                }
                else {  // if min_col_support > 0.728500008583
                  return 0.290182027645 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.978343129158
              if ( max_col_coverage <= 0.523936867714 ) {
                if ( median_col_coverage <= 0.322828769684 ) {
                  return 0.0586452742831 < maxgini;
                }
                else {  // if median_col_coverage > 0.322828769684
                  return 0.24714115139 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.523936867714
                if ( mean_col_coverage <= 0.469058036804 ) {
                  return 0.183554947103 < maxgini;
                }
                else {  // if mean_col_coverage > 0.469058036804
                  return 0.425370324719 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.784500002861
            if ( median_col_coverage <= 0.30542576313 ) {
              if ( max_col_coverage <= 0.943303585052 ) {
                if ( mean_col_coverage <= 0.303630948067 ) {
                  return 0.0770039990987 < maxgini;
                }
                else {  // if mean_col_coverage > 0.303630948067
                  return 0.0969493425195 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.943303585052
                if ( min_col_coverage <= 0.0155122661963 ) {
                  return 0.485136741974 < maxgini;
                }
                else {  // if min_col_coverage > 0.0155122661963
                  return 0.252400548697 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.30542576313
              if ( min_col_coverage <= 0.0139866974205 ) {
                if ( min_col_support <= 0.84350001812 ) {
                  return 0.4928 < maxgini;
                }
                else {  // if min_col_support > 0.84350001812
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.0139866974205
                if ( median_col_support <= 0.885499954224 ) {
                  return 0.22008205115 < maxgini;
                }
                else {  // if median_col_support > 0.885499954224
                  return 0.0978694423295 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if median_col_coverage > 0.600291550159
      if ( median_col_support <= 0.984500050545 ) {
        if ( min_col_support <= 0.742499947548 ) {
          if ( min_col_coverage <= 0.727850914001 ) {
            if ( median_col_coverage <= 0.66731774807 ) {
              if ( median_col_coverage <= 0.666103601456 ) {
                if ( mean_col_support <= 0.900558829308 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.900558829308
                  return 0.490777598123 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.666103601456
                if ( mean_col_coverage <= 0.701334297657 ) {
                  return 0.276568407372 < maxgini;
                }
                else {  // if mean_col_coverage > 0.701334297657
                  return 0.446446280992 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.66731774807
              if ( mean_col_support <= 0.970676481724 ) {
                if ( max_col_coverage <= 0.773079454899 ) {
                  return 0.464575363838 < maxgini;
                }
                else {  // if max_col_coverage > 0.773079454899
                  return false;
                }
              }
              else {  // if mean_col_support > 0.970676481724
                if ( max_col_coverage <= 0.83390802145 ) {
                  return 0.283475546306 < maxgini;
                }
                else {  // if max_col_coverage > 0.83390802145
                  return 0.47821425205 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.727850914001
            if ( min_col_support <= 0.627499997616 ) {
              if ( mean_col_support <= 0.969147086143 ) {
                if ( median_col_coverage <= 0.985764920712 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.985764920712
                  return false;
                }
              }
              else {  // if mean_col_support > 0.969147086143
                if ( median_col_support <= 0.981500029564 ) {
                  return 0.4969242599 < maxgini;
                }
                else {  // if median_col_support > 0.981500029564
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.627499997616
              if ( max_col_coverage <= 0.857919216156 ) {
                if ( min_col_support <= 0.719500005245 ) {
                  return 0.489713294562 < maxgini;
                }
                else {  // if min_col_support > 0.719500005245
                  return 0.325624651249 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.857919216156
                if ( median_col_coverage <= 0.750634551048 ) {
                  return 0.411608987603 < maxgini;
                }
                else {  // if median_col_coverage > 0.750634551048
                  return false;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.742499947548
          if ( median_col_support <= 0.863499999046 ) {
            if ( median_col_support <= 0.834499955177 ) {
              if ( mean_col_coverage <= 0.84772002697 ) {
                if ( median_col_support <= 0.791499972343 ) {
                  return 0.497448979592 < maxgini;
                }
                else {  // if median_col_support > 0.791499972343
                  return 0.413566460011 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.84772002697
                if ( mean_col_support <= 0.953176498413 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.953176498413
                  return 0.408163265306 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.834499955177
              if ( median_col_coverage <= 0.92078435421 ) {
                if ( median_col_support <= 0.862499952316 ) {
                  return 0.309454623649 < maxgini;
                }
                else {  // if median_col_support > 0.862499952316
                  return 0.458897745611 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.92078435421
                if ( mean_col_support <= 0.959735274315 ) {
                  return 0.4955500178 < maxgini;
                }
                else {  // if mean_col_support > 0.959735274315
                  return 0.0 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.863499999046
            if ( min_col_support <= 0.816499948502 ) {
              if ( median_col_coverage <= 0.852545261383 ) {
                if ( mean_col_coverage <= 0.771969139576 ) {
                  return 0.210116118723 < maxgini;
                }
                else {  // if mean_col_coverage > 0.771969139576
                  return 0.329110472069 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.852545261383
                if ( min_col_coverage <= 0.920083999634 ) {
                  return 0.473736412426 < maxgini;
                }
                else {  // if min_col_coverage > 0.920083999634
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.816499948502
              if ( min_col_coverage <= 0.909249305725 ) {
                if ( median_col_support <= 0.895500004292 ) {
                  return 0.23915475357 < maxgini;
                }
                else {  // if median_col_support > 0.895500004292
                  return 0.0997017822453 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.909249305725
                if ( min_col_coverage <= 0.96336555481 ) {
                  return 0.314152004225 < maxgini;
                }
                else {  // if min_col_coverage > 0.96336555481
                  return 0.439317689755 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.984500050545
        if ( mean_col_support <= 0.976852893829 ) {
          if ( min_col_coverage <= 0.668171644211 ) {
            if ( min_col_support <= 0.745499968529 ) {
              if ( min_col_support <= 0.612499952316 ) {
                if ( median_col_support <= 0.990499973297 ) {
                  return false;
                }
                else {  // if median_col_support > 0.990499973297
                  return false;
                }
              }
              else {  // if min_col_support > 0.612499952316
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.498509705114 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.745499968529
              if ( median_col_support <= 0.99950003624 ) {
                if ( min_col_support <= 0.841500043869 ) {
                  return false;
                }
                else {  // if min_col_support > 0.841500043869
                  return 0.294543063774 < maxgini;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( median_col_coverage <= 0.604957163334 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.604957163334
                  return 0.20911726335 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.668171644211
            if ( min_col_support <= 0.734500050545 ) {
              if ( min_col_coverage <= 0.995491027832 ) {
                if ( min_col_coverage <= 0.876126527786 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.876126527786
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.995491027832
                if ( median_col_coverage <= 0.997624576092 ) {
                  return 0.375 < maxgini;
                }
                else {  // if median_col_coverage > 0.997624576092
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.734500050545
              if ( max_col_coverage <= 0.818447589874 ) {
                if ( min_col_coverage <= 0.676178455353 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.676178455353
                  return 0.302636989118 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.818447589874
                if ( max_col_coverage <= 0.997584939003 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.997584939003
                  return 0.496108637237 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.976852893829
          if ( median_col_coverage <= 0.833540081978 ) {
            if ( min_col_support <= 0.794499993324 ) {
              if ( min_col_support <= 0.703500032425 ) {
                if ( min_col_support <= 0.647500038147 ) {
                  return false;
                }
                else {  // if min_col_support > 0.647500038147
                  return false;
                }
              }
              else {  // if min_col_support > 0.703500032425
                if ( min_col_coverage <= 0.811752736568 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.811752736568
                  return 0.299711584682 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.794499993324
              if ( min_col_coverage <= 0.572675466537 ) {
                if ( median_col_coverage <= 0.605751395226 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.605751395226
                  return 0.109172853524 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.572675466537
                if ( min_col_support <= 0.846500039101 ) {
                  return 0.429431266695 < maxgini;
                }
                else {  // if min_col_support > 0.846500039101
                  return 0.101280312149 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.833540081978
            if ( mean_col_coverage <= 0.999871015549 ) {
              if ( max_col_coverage <= 0.995337963104 ) {
                if ( mean_col_support <= 0.979558825493 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.979558825493
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.995337963104
                if ( min_col_support <= 0.800500035286 ) {
                  return false;
                }
                else {  // if min_col_support > 0.800500035286
                  return 0.399017679989 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.999871015549
              if ( min_col_support <= 0.682500004768 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return false;
                }
              }
              else {  // if min_col_support > 0.682500004768
                if ( median_col_support <= 0.999000012875 ) {
                  return false;
                }
                else {  // if median_col_support > 0.999000012875
                  return 0.425078043704 < maxgini;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if mean_col_support > 0.981710076332
    if ( median_col_coverage <= 0.950073301792 ) {
      if ( mean_col_support <= 0.987303316593 ) {
        if ( min_col_support <= 0.797500014305 ) {
          if ( median_col_support <= 0.99950003624 ) {
            if ( median_col_support <= 0.992499947548 ) {
              if ( max_col_coverage <= 0.539347946644 ) {
                if ( median_col_support <= 0.991500020027 ) {
                  return 0.143569149127 < maxgini;
                }
                else {  // if median_col_support > 0.991500020027
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.539347946644
                if ( median_col_support <= 0.990499973297 ) {
                  return 0.26437366563 < maxgini;
                }
                else {  // if median_col_support > 0.990499973297
                  return 0.483395685992 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.992499947548
              if ( mean_col_coverage <= 0.691917836666 ) {
                if ( max_col_coverage <= 0.655805528164 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.655805528164
                  return false;
                }
              }
              else {  // if mean_col_coverage > 0.691917836666
                if ( median_col_coverage <= 0.713206768036 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.713206768036
                  return false;
                }
              }
            }
          }
          else {  // if median_col_support > 0.99950003624
            if ( median_col_coverage <= 0.502204179764 ) {
              if ( min_col_coverage <= 0.302478581667 ) {
                if ( mean_col_coverage <= 0.606218755245 ) {
                  return 0.0422596927417 < maxgini;
                }
                else {  // if mean_col_coverage > 0.606218755245
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.302478581667
                if ( max_col_coverage <= 0.458758503199 ) {
                  return 0.107962077777 < maxgini;
                }
                else {  // if max_col_coverage > 0.458758503199
                  return 0.277061953701 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.502204179764
              if ( median_col_coverage <= 0.652358174324 ) {
                if ( min_col_support <= 0.756500005722 ) {
                  return 0.497690053742 < maxgini;
                }
                else {  // if min_col_support > 0.756500005722
                  return 0.267951062753 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.652358174324
                if ( median_col_coverage <= 0.916790306568 ) {
                  return 0.489537597405 < maxgini;
                }
                else {  // if median_col_coverage > 0.916790306568
                  return false;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.797500014305
          if ( max_col_coverage <= 0.800574183464 ) {
            if ( min_col_support <= 0.833500027657 ) {
              if ( median_col_coverage <= 0.45867484808 ) {
                if ( median_col_coverage <= 0.350213676691 ) {
                  return 0.02760548023 < maxgini;
                }
                else {  // if median_col_coverage > 0.350213676691
                  return 0.0765466784778 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.45867484808
                if ( max_col_coverage <= 0.700162887573 ) {
                  return 0.154724359632 < maxgini;
                }
                else {  // if max_col_coverage > 0.700162887573
                  return 0.27061446113 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.833500027657
              if ( min_col_support <= 0.9375 ) {
                if ( median_col_coverage <= 0.0125525211915 ) {
                  return 0.248845320011 < maxgini;
                }
                else {  // if median_col_coverage > 0.0125525211915
                  return 0.0417802109877 < maxgini;
                }
              }
              else {  // if min_col_support > 0.9375
                if ( min_col_coverage <= 0.320256412029 ) {
                  return 0.183529496643 < maxgini;
                }
                else {  // if min_col_coverage > 0.320256412029
                  return 0.0563254018642 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.800574183464
            if ( min_col_support <= 0.857499957085 ) {
              if ( median_col_coverage <= 0.91310441494 ) {
                if ( mean_col_coverage <= 0.709185957909 ) {
                  return 0.138477318484 < maxgini;
                }
                else {  // if mean_col_coverage > 0.709185957909
                  return 0.324172507643 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.91310441494
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.240041551247 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.857499957085
              if ( min_col_support <= 0.881500005722 ) {
                if ( max_col_coverage <= 0.804805874825 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.804805874825
                  return 0.127597059257 < maxgini;
                }
              }
              else {  // if min_col_support > 0.881500005722
                if ( max_col_coverage <= 0.802041649818 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.802041649818
                  return 0.0335875911001 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.987303316593
        if ( mean_col_support <= 0.990710139275 ) {
          if ( min_col_support <= 0.857499957085 ) {
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_support <= 0.994500041008 ) {
                if ( min_col_support <= 0.820500016212 ) {
                  return false;
                }
                else {  // if min_col_support > 0.820500016212
                  return 0.245164488658 < maxgini;
                }
              }
              else {  // if median_col_support > 0.994500041008
                if ( mean_col_coverage <= 0.760064959526 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.760064959526
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( mean_col_support <= 0.988902747631 ) {
                if ( mean_col_coverage <= 0.624673008919 ) {
                  return 0.0359426742964 < maxgini;
                }
                else {  // if mean_col_coverage > 0.624673008919
                  return 0.194163808678 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.988902747631
                if ( mean_col_support <= 0.989852905273 ) {
                  return 0.0571810522763 < maxgini;
                }
                else {  // if mean_col_support > 0.989852905273
                  return 0.0400899264921 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.857499957085
            if ( median_col_coverage <= 0.913108587265 ) {
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_support <= 0.993499994278 ) {
                  return 0.0214625693542 < maxgini;
                }
                else {  // if median_col_support > 0.993499994278
                  return 0.473268956887 < maxgini;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( mean_col_coverage <= 0.6565721035 ) {
                  return 0.0206702358752 < maxgini;
                }
                else {  // if mean_col_coverage > 0.6565721035
                  return 0.00903936167883 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.913108587265
              if ( min_col_support <= 0.879500031471 ) {
                if ( median_col_coverage <= 0.913639545441 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.913639545441
                  return 0.268105380168 < maxgini;
                }
              }
              else {  // if min_col_support > 0.879500031471
                if ( mean_col_support <= 0.989617645741 ) {
                  return 0.0513875107598 < maxgini;
                }
                else {  // if mean_col_support > 0.989617645741
                  return 0.0235607152614 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.990710139275
          if ( median_col_coverage <= 0.513416409492 ) {
            if ( min_col_coverage <= 0.024742372334 ) {
              if ( min_col_support <= 0.913499951363 ) {
                if ( min_col_coverage <= 0.0137990098447 ) {
                  return 0.060546875 < maxgini;
                }
                else {  // if min_col_coverage > 0.0137990098447
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_support > 0.913499951363
                if ( median_col_support <= 0.984500050545 ) {
                  return 0.387811634349 < maxgini;
                }
                else {  // if median_col_support > 0.984500050545
                  return 0.0598679208718 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.024742372334
              if ( min_col_coverage <= 0.0528464019299 ) {
                if ( min_col_support <= 0.863499999046 ) {
                  return 0.0502809573361 < maxgini;
                }
                else {  // if min_col_support > 0.863499999046
                  return 0.0106948780206 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0528464019299
                if ( mean_col_support <= 0.993794083595 ) {
                  return 0.0116270729433 < maxgini;
                }
                else {  // if mean_col_support > 0.993794083595
                  return 0.00286051610287 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.513416409492
            if ( min_col_support <= 0.887500047684 ) {
              if ( median_col_coverage <= 0.711204111576 ) {
                if ( min_col_coverage <= 0.594673395157 ) {
                  return 0.0204693621839 < maxgini;
                }
                else {  // if min_col_coverage > 0.594673395157
                  return 0.0597517792499 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.711204111576
                if ( min_col_coverage <= 0.913188457489 ) {
                  return 0.119009252615 < maxgini;
                }
                else {  // if min_col_coverage > 0.913188457489
                  return 0.30066054405 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.887500047684
              if ( min_col_support <= 0.914499998093 ) {
                if ( min_col_coverage <= 0.900199174881 ) {
                  return 0.010224487188 < maxgini;
                }
                else {  // if min_col_coverage > 0.900199174881
                  return 0.0467022275131 < maxgini;
                }
              }
              else {  // if min_col_support > 0.914499998093
                if ( mean_col_coverage <= 0.491232097149 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.491232097149
                  return 0.00198864781003 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if median_col_coverage > 0.950073301792
      if ( mean_col_support <= 0.98761767149 ) {
        if ( median_col_coverage <= 0.998951792717 ) {
          if ( median_col_support <= 0.991500020027 ) {
            if ( mean_col_coverage <= 0.994262814522 ) {
              if ( median_col_coverage <= 0.971724748611 ) {
                if ( max_col_coverage <= 0.997663557529 ) {
                  return 0.27555 < maxgini;
                }
                else {  // if max_col_coverage > 0.997663557529
                  return 0.0976834906416 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.971724748611
                if ( mean_col_support <= 0.987558841705 ) {
                  return 0.466027268031 < maxgini;
                }
                else {  // if mean_col_support > 0.987558841705
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.994262814522
              if ( median_col_support <= 0.988499999046 ) {
                if ( median_col_coverage <= 0.99358767271 ) {
                  return 0.486111111111 < maxgini;
                }
                else {  // if median_col_coverage > 0.99358767271
                  return false;
                }
              }
              else {  // if median_col_support > 0.988499999046
                return false;
              }
            }
          }
          else {  // if median_col_support > 0.991500020027
            if ( median_col_support <= 0.99950003624 ) {
              if ( min_col_support <= 0.890499949455 ) {
                if ( median_col_support <= 0.993499994278 ) {
                  return false;
                }
                else {  // if median_col_support > 0.993499994278
                  return false;
                }
              }
              else {  // if min_col_support > 0.890499949455
                if ( mean_col_support <= 0.987352967262 ) {
                  return 0.48347107438 < maxgini;
                }
                else {  // if mean_col_support > 0.987352967262
                  return 0.0 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_coverage <= 0.92017698288 ) {
                if ( median_col_coverage <= 0.974103212357 ) {
                  return 0.179619839653 < maxgini;
                }
                else {  // if median_col_coverage > 0.974103212357
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.92017698288
                if ( min_col_support <= 0.792500019073 ) {
                  return false;
                }
                else {  // if min_col_support > 0.792500019073
                  return 0.169416898213 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.998951792717
          if ( min_col_support <= 0.807500004768 ) {
            if ( min_col_support <= 0.760499954224 ) {
              if ( min_col_support <= 0.732499957085 ) {
                if ( min_col_support <= 0.708999991417 ) {
                  return false;
                }
                else {  // if min_col_support > 0.708999991417
                  return false;
                }
              }
              else {  // if min_col_support > 0.732499957085
                if ( min_col_coverage <= 0.948684215546 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.948684215546
                  return 0.499881656805 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.760499954224
              if ( min_col_support <= 0.797500014305 ) {
                if ( mean_col_support <= 0.987441122532 ) {
                  return 0.41947281856 < maxgini;
                }
                else {  // if mean_col_support > 0.987441122532
                  return false;
                }
              }
              else {  // if min_col_support > 0.797500014305
                if ( min_col_support <= 0.803499996662 ) {
                  return false;
                }
                else {  // if min_col_support > 0.803499996662
                  return 0.432132963989 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.807500004768
            if ( median_col_support <= 0.99950003624 ) {
              if ( max_col_support <= 0.99950003624 ) {
                if ( min_col_support <= 0.953999996185 ) {
                  return false;
                }
                else {  // if min_col_support > 0.953999996185
                  return 0.0 < maxgini;
                }
              }
              else {  // if max_col_support > 0.99950003624
                if ( median_col_support <= 0.959499955177 ) {
                  return 0.0838281103397 < maxgini;
                }
                else {  // if median_col_support > 0.959499955177
                  return 0.389985652035 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( mean_col_coverage <= 0.966360330582 ) {
                if ( mean_col_support <= 0.987147092819 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_support > 0.987147092819
                  return false;
                }
              }
              else {  // if mean_col_coverage > 0.966360330582
                if ( min_col_support <= 0.879500031471 ) {
                  return 0.128068055936 < maxgini;
                }
                else {  // if min_col_support > 0.879500031471
                  return 0.0107331401066 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.98761767149
        if ( min_col_coverage <= 0.937779188156 ) {
          if ( mean_col_coverage <= 0.996264219284 ) {
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_coverage <= 0.95016682148 ) {
                if ( mean_col_support <= 0.993646979332 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.993646979332
                  return 0.0 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.95016682148
                if ( min_col_coverage <= 0.713578522205 ) {
                  return 0.491493383743 < maxgini;
                }
                else {  // if min_col_coverage > 0.713578522205
                  return 0.0466743631307 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_coverage <= 0.753787875175 ) {
                if ( min_col_support <= 0.868999958038 ) {
                  return 0.32 < maxgini;
                }
                else {  // if min_col_support > 0.868999958038
                  return 0.0229853975122 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.753787875175
                if ( mean_col_coverage <= 0.98992061615 ) {
                  return 0.00230690433683 < maxgini;
                }
                else {  // if mean_col_coverage > 0.98992061615
                  return 0.00854685111164 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.996264219284
            if ( median_col_support <= 0.995000004768 ) {
              return 0.0 < maxgini;
            }
            else {  // if median_col_support > 0.995000004768
              if ( min_col_support <= 0.886999964714 ) {
                if ( mean_col_support <= 0.990411758423 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.990411758423
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_support > 0.886999964714
                return 0.0 < maxgini;
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.937779188156
          if ( min_col_support <= 0.858500003815 ) {
            if ( median_col_support <= 0.99950003624 ) {
              if ( max_col_coverage <= 0.979662716389 ) {
                if ( median_col_coverage <= 0.952713906765 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.952713906765
                  return 0.0997229916898 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.979662716389
                if ( mean_col_support <= 0.990382373333 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.990382373333
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_support <= 0.833500027657 ) {
                if ( max_col_coverage <= 0.991250514984 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.991250514984
                  return 0.428641975309 < maxgini;
                }
              }
              else {  // if min_col_support > 0.833500027657
                if ( median_col_coverage <= 0.996913552284 ) {
                  return 0.339576507409 < maxgini;
                }
                else {  // if median_col_coverage > 0.996913552284
                  return 0.133280736358 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.858500003815
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_coverage <= 0.980732083321 ) {
                if ( mean_col_coverage <= 0.962841272354 ) {
                  return 0.155036127704 < maxgini;
                }
                else {  // if mean_col_coverage > 0.962841272354
                  return 0.0583968367095 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.980732083321
                if ( mean_col_coverage <= 0.998081445694 ) {
                  return 0.138942282099 < maxgini;
                }
                else {  // if mean_col_coverage > 0.998081445694
                  return 0.211322691306 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_support <= 0.886500000954 ) {
                if ( min_col_coverage <= 0.961873888969 ) {
                  return 0.0217838498709 < maxgini;
                }
                else {  // if min_col_coverage > 0.961873888969
                  return 0.118048236096 < maxgini;
                }
              }
              else {  // if min_col_support > 0.886500000954
                if ( mean_col_support <= 0.99532353878 ) {
                  return 0.0122781362743 < maxgini;
                }
                else {  // if mean_col_support > 0.99532353878
                  return 0.00253186406403 < maxgini;
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
  if ( max_col_coverage <= 0.459404528141 ) {
    if ( median_col_coverage <= 0.0500783696771 ) {
      if ( mean_col_support <= 0.926939368248 ) {
        if ( min_col_support <= 0.544499993324 ) {
          if ( min_col_coverage <= 0.0254787411541 ) {
            if ( mean_col_support <= 0.922852933407 ) {
              if ( min_col_coverage <= 0.00604235101491 ) {
                if ( min_col_support <= 0.50049996376 ) {
                  return 0.355029585799 < maxgini;
                }
                else {  // if min_col_support > 0.50049996376
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.00604235101491
                if ( min_col_support <= 0.543500006199 ) {
                  return 0.285405228026 < maxgini;
                }
                else {  // if min_col_support > 0.543500006199
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.922852933407
              if ( min_col_coverage <= 0.00917740166187 ) {
                if ( median_col_support <= 0.93900001049 ) {
                  return 0.0 < maxgini;
                }
                else {  // if median_col_support > 0.93900001049
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.00917740166187
                if ( median_col_coverage <= 0.00972340255976 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.00972340255976
                  return 0.490306040699 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.0254787411541
            if ( median_col_support <= 0.74950003624 ) {
              if ( mean_col_support <= 0.918147087097 ) {
                if ( median_col_coverage <= 0.0386482477188 ) {
                  return 0.425512398151 < maxgini;
                }
                else {  // if median_col_coverage > 0.0386482477188
                  return 0.459989049829 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.918147087097
                if ( min_col_support <= 0.503499984741 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_support > 0.503499984741
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.74950003624
              if ( median_col_coverage <= 0.045804195106 ) {
                if ( median_col_support <= 0.925500035286 ) {
                  return 0.328180737218 < maxgini;
                }
                else {  // if median_col_support > 0.925500035286
                  return 0.448513031856 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.045804195106
                if ( mean_col_support <= 0.86726474762 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.86726474762
                  return 0.339222145329 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.544499993324
          if ( max_col_coverage <= 0.317627489567 ) {
            if ( min_col_support <= 0.580500006676 ) {
              if ( max_col_coverage <= 0.203011780977 ) {
                if ( median_col_support <= 0.679499983788 ) {
                  return 0.340834225098 < maxgini;
                }
                else {  // if median_col_support > 0.679499983788
                  return 0.196113069757 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.203011780977
                if ( median_col_support <= 0.727499961853 ) {
                  return 0.393414869474 < maxgini;
                }
                else {  // if median_col_support > 0.727499961853
                  return 0.282994454928 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.580500006676
              if ( min_col_coverage <= 0.0225995928049 ) {
                if ( min_col_coverage <= 0.00416017975658 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.00416017975658
                  return 0.199893895271 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0225995928049
                if ( min_col_support <= 0.629500031471 ) {
                  return 0.416309842062 < maxgini;
                }
                else {  // if min_col_support > 0.629500031471
                  return 0.33718761429 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.317627489567
            if ( mean_col_coverage <= 0.180268883705 ) {
              if ( min_col_support <= 0.565500020981 ) {
                if ( max_col_coverage <= 0.377155184746 ) {
                  return 0.337249861496 < maxgini;
                }
                else {  // if max_col_coverage > 0.377155184746
                  return 0.161680541103 < maxgini;
                }
              }
              else {  // if min_col_support > 0.565500020981
                if ( min_col_coverage <= 0.0202040821314 ) {
                  return 0.0886965927528 < maxgini;
                }
                else {  // if min_col_coverage > 0.0202040821314
                  return 0.401669700947 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.180268883705
              if ( mean_col_coverage <= 0.226571023464 ) {
                if ( min_col_coverage <= 0.0298573970795 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.0298573970795
                  return 0.453312923005 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.226571023464
                if ( median_col_support <= 0.712000012398 ) {
                  return false;
                }
                else {  // if median_col_support > 0.712000012398
                  return 0.490815655651 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.926939368248
        if ( mean_col_support <= 0.947656869888 ) {
          if ( min_col_support <= 0.601500034332 ) {
            if ( mean_col_support <= 0.936667263508 ) {
              if ( max_col_coverage <= 0.230063512921 ) {
                if ( median_col_support <= 0.830500006676 ) {
                  return 0.0997229916898 < maxgini;
                }
                else {  // if median_col_support > 0.830500006676
                  return 0.171864541909 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.230063512921
                if ( min_col_coverage <= 0.0413251370192 ) {
                  return 0.352149598247 < maxgini;
                }
                else {  // if min_col_coverage > 0.0413251370192
                  return 0.280896647491 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.936667263508
              if ( median_col_support <= 0.844500005245 ) {
                if ( mean_col_support <= 0.943264722824 ) {
                  return 0.331247165533 < maxgini;
                }
                else {  // if mean_col_support > 0.943264722824
                  return false;
                }
              }
              else {  // if median_col_support > 0.844500005245
                if ( mean_col_coverage <= 0.118572987616 ) {
                  return 0.0975516091622 < maxgini;
                }
                else {  // if mean_col_coverage > 0.118572987616
                  return 0.242039027973 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.601500034332
            if ( mean_col_support <= 0.941217660904 ) {
              if ( mean_col_coverage <= 0.198393240571 ) {
                if ( max_col_coverage <= 0.256696432829 ) {
                  return 0.269081051486 < maxgini;
                }
                else {  // if max_col_coverage > 0.256696432829
                  return 0.333761499265 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.198393240571
                if ( median_col_support <= 0.844500005245 ) {
                  return 0.495176977041 < maxgini;
                }
                else {  // if median_col_support > 0.844500005245
                  return 0.228832401899 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.941217660904
              if ( median_col_support <= 0.807000041008 ) {
                if ( min_col_support <= 0.641999959946 ) {
                  return false;
                }
                else {  // if min_col_support > 0.641999959946
                  return 0.323228634039 < maxgini;
                }
              }
              else {  // if median_col_support > 0.807000041008
                if ( mean_col_coverage <= 0.225748181343 ) {
                  return 0.201568895716 < maxgini;
                }
                else {  // if mean_col_coverage > 0.225748181343
                  return 0.448558205767 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.947656869888
          if ( mean_col_coverage <= 0.194718852639 ) {
            if ( median_col_coverage <= 0.018561584875 ) {
              if ( median_col_support <= 0.916499972343 ) {
                if ( min_col_support <= 0.870999991894 ) {
                  return 0.302205114555 < maxgini;
                }
                else {  // if min_col_support > 0.870999991894
                  return false;
                }
              }
              else {  // if median_col_support > 0.916499972343
                if ( min_col_coverage <= 0.00682601798326 ) {
                  return 0.0647919852689 < maxgini;
                }
                else {  // if min_col_coverage > 0.00682601798326
                  return 0.166030176267 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.018561584875
              if ( mean_col_support <= 0.962601780891 ) {
                if ( min_col_support <= 0.735499978065 ) {
                  return 0.126891187877 < maxgini;
                }
                else {  // if min_col_support > 0.735499978065
                  return 0.241452697586 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.962601780891
                if ( min_col_coverage <= 0.00420234818012 ) {
                  return 0.359861591696 < maxgini;
                }
                else {  // if min_col_coverage > 0.00420234818012
                  return 0.0690636822628 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.194718852639
            if ( mean_col_support <= 0.961615562439 ) {
              if ( min_col_coverage <= 0.0392307713628 ) {
                if ( max_col_coverage <= 0.445906430483 ) {
                  return 0.499653739612 < maxgini;
                }
                else {  // if max_col_coverage > 0.445906430483
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0392307713628
                if ( min_col_support <= 0.539000034332 ) {
                  return 0.403816558941 < maxgini;
                }
                else {  // if min_col_support > 0.539000034332
                  return 0.261666353913 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.961615562439
              if ( mean_col_support <= 0.968088269234 ) {
                if ( min_col_support <= 0.801499962807 ) {
                  return 0.153129306201 < maxgini;
                }
                else {  // if min_col_support > 0.801499962807
                  return 0.309256055363 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.968088269234
                if ( median_col_support <= 0.886500000954 ) {
                  return false;
                }
                else {  // if median_col_support > 0.886500000954
                  return 0.0636157203312 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if median_col_coverage > 0.0500783696771
      if ( min_col_coverage <= 0.0507430285215 ) {
        if ( mean_col_support <= 0.919445991516 ) {
          if ( min_col_coverage <= 0.0347658209503 ) {
            if ( median_col_coverage <= 0.12701612711 ) {
              if ( median_col_support <= 0.746500015259 ) {
                if ( median_col_support <= 0.609500050545 ) {
                  return 0.433358528597 < maxgini;
                }
                else {  // if median_col_support > 0.609500050545
                  return 0.332346152792 < maxgini;
                }
              }
              else {  // if median_col_support > 0.746500015259
                if ( min_col_support <= 0.504500031471 ) {
                  return 0.0285654274312 < maxgini;
                }
                else {  // if min_col_support > 0.504500031471
                  return 0.200889856848 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.12701612711
              if ( median_col_support <= 0.615000009537 ) {
                if ( max_col_coverage <= 0.233592867851 ) {
                  return 0.375 < maxgini;
                }
                else {  // if max_col_coverage > 0.233592867851
                  return false;
                }
              }
              else {  // if median_col_support > 0.615000009537
                if ( mean_col_support <= 0.909992158413 ) {
                  return 0.436224489796 < maxgini;
                }
                else {  // if mean_col_support > 0.909992158413
                  return 0.225651577503 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.0347658209503
            if ( max_col_coverage <= 0.273804396391 ) {
              if ( median_col_support <= 0.691499948502 ) {
                if ( min_col_support <= 0.473500013351 ) {
                  return 0.125219584703 < maxgini;
                }
                else {  // if min_col_support > 0.473500013351
                  return 0.387131127165 < maxgini;
                }
              }
              else {  // if median_col_support > 0.691499948502
                if ( median_col_support <= 0.74950003624 ) {
                  return 0.288902873535 < maxgini;
                }
                else {  // if median_col_support > 0.74950003624
                  return 0.206640521297 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.273804396391
              if ( median_col_support <= 0.743499994278 ) {
                if ( max_col_coverage <= 0.392080724239 ) {
                  return 0.435878233448 < maxgini;
                }
                else {  // if max_col_coverage > 0.392080724239
                  return 0.472115333578 < maxgini;
                }
              }
              else {  // if median_col_support > 0.743499994278
                if ( mean_col_support <= 0.888441205025 ) {
                  return 0.428386648759 < maxgini;
                }
                else {  // if mean_col_support > 0.888441205025
                  return 0.278643056086 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.919445991516
          if ( min_col_support <= 0.712499976158 ) {
            if ( mean_col_coverage <= 0.185304403305 ) {
              if ( mean_col_coverage <= 0.131704986095 ) {
                if ( median_col_support <= 0.8125 ) {
                  return 0.158979678504 < maxgini;
                }
                else {  // if median_col_support > 0.8125
                  return 0.0564877580372 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.131704986095
                if ( mean_col_support <= 0.938782334328 ) {
                  return 0.172074473733 < maxgini;
                }
                else {  // if mean_col_support > 0.938782334328
                  return 0.0579351715929 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.185304403305
              if ( min_col_support <= 0.709499955177 ) {
                if ( min_col_coverage <= 0.0388386137784 ) {
                  return 0.283855362438 < maxgini;
                }
                else {  // if min_col_coverage > 0.0388386137784
                  return 0.158362856674 < maxgini;
                }
              }
              else {  // if min_col_support > 0.709499955177
                if ( mean_col_support <= 0.948513746262 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.948513746262
                  return 0.0624349635796 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.712499976158
            if ( max_col_coverage <= 0.31851118803 ) {
              if ( median_col_support <= 0.924499988556 ) {
                if ( mean_col_coverage <= 0.249542117119 ) {
                  return 0.138043763597 < maxgini;
                }
                else {  // if mean_col_coverage > 0.249542117119
                  return false;
                }
              }
              else {  // if median_col_support > 0.924499988556
                if ( max_col_coverage <= 0.136809274554 ) {
                  return 0.0612882550873 < maxgini;
                }
                else {  // if max_col_coverage > 0.136809274554
                  return 0.0269497475915 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.31851118803
              if ( max_col_coverage <= 0.318994760513 ) {
                return false;
              }
              else {  // if max_col_coverage > 0.318994760513
                if ( min_col_support <= 0.810500025749 ) {
                  return 0.112259312337 < maxgini;
                }
                else {  // if min_col_support > 0.810500025749
                  return 0.0381468790872 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_coverage > 0.0507430285215
        if ( min_col_support <= 0.711500048637 ) {
          if ( mean_col_support <= 0.871970593929 ) {
            if ( mean_col_coverage <= 0.23021556437 ) {
              if ( mean_col_coverage <= 0.164317935705 ) {
                if ( max_col_coverage <= 0.132704406977 ) {
                  return 0.0277722674073 < maxgini;
                }
                else {  // if max_col_coverage > 0.132704406977
                  return 0.282247084256 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.164317935705
                if ( median_col_support <= 0.609500050545 ) {
                  return 0.444881028651 < maxgini;
                }
                else {  // if median_col_support > 0.609500050545
                  return 0.293582976435 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.23021556437
              if ( mean_col_coverage <= 0.273593366146 ) {
                if ( median_col_coverage <= 0.0944940447807 ) {
                  return 0.494649227111 < maxgini;
                }
                else {  // if median_col_coverage > 0.0944940447807
                  return 0.44239230626 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.273593366146
                if ( median_col_support <= 0.620499968529 ) {
                  return 0.499848087085 < maxgini;
                }
                else {  // if median_col_support > 0.620499968529
                  return 0.429877405383 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.871970593929
            if ( max_col_coverage <= 0.364146143198 ) {
              if ( median_col_support <= 0.728500008583 ) {
                if ( median_col_support <= 0.684499979019 ) {
                  return 0.352675104987 < maxgini;
                }
                else {  // if median_col_support > 0.684499979019
                  return 0.274678878465 < maxgini;
                }
              }
              else {  // if median_col_support > 0.728500008583
                if ( min_col_coverage <= 0.150187969208 ) {
                  return 0.129901275 < maxgini;
                }
                else {  // if min_col_coverage > 0.150187969208
                  return 0.209744842476 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.364146143198
              if ( mean_col_coverage <= 0.306136101484 ) {
                if ( median_col_support <= 0.727499961853 ) {
                  return 0.35712416193 < maxgini;
                }
                else {  // if median_col_support > 0.727499961853
                  return 0.19151114128 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.306136101484
                if ( mean_col_coverage <= 0.341748267412 ) {
                  return 0.349978548982 < maxgini;
                }
                else {  // if mean_col_coverage > 0.341748267412
                  return 0.397813338913 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.711500048637
          if ( min_col_support <= 0.824499964714 ) {
            if ( min_col_coverage <= 0.200378790498 ) {
              if ( median_col_coverage <= 0.208406955004 ) {
                if ( mean_col_coverage <= 0.232986420393 ) {
                  return 0.0714475747974 < maxgini;
                }
                else {  // if mean_col_coverage > 0.232986420393
                  return 0.116040865907 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.208406955004
                if ( min_col_support <= 0.794499993324 ) {
                  return 0.151431956472 < maxgini;
                }
                else {  // if min_col_support > 0.794499993324
                  return 0.103689490199 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.200378790498
              if ( min_col_support <= 0.747500002384 ) {
                if ( mean_col_support <= 0.931499958038 ) {
                  return 0.353860179582 < maxgini;
                }
                else {  // if mean_col_support > 0.931499958038
                  return 0.166179443953 < maxgini;
                }
              }
              else {  // if min_col_support > 0.747500002384
                if ( max_col_coverage <= 0.409269869328 ) {
                  return 0.117283784173 < maxgini;
                }
                else {  // if max_col_coverage > 0.409269869328
                  return 0.155644181704 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.824499964714
            if ( min_col_support <= 0.888499975204 ) {
              if ( min_col_support <= 0.838500022888 ) {
                if ( mean_col_support <= 0.964735269547 ) {
                  return 0.171608509789 < maxgini;
                }
                else {  // if mean_col_support > 0.964735269547
                  return 0.0433466046792 < maxgini;
                }
              }
              else {  // if min_col_support > 0.838500022888
                if ( median_col_support <= 0.905499994755 ) {
                  return 0.149957239191 < maxgini;
                }
                else {  // if median_col_support > 0.905499994755
                  return 0.0330483219654 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.888499975204
              if ( max_col_coverage <= 0.425637245178 ) {
                if ( mean_col_coverage <= 0.373826503754 ) {
                  return 0.0218809071558 < maxgini;
                }
                else {  // if mean_col_coverage > 0.373826503754
                  return 0.0111968888194 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.425637245178
                if ( min_col_support <= 0.898499965668 ) {
                  return 0.0359247895754 < maxgini;
                }
                else {  // if min_col_support > 0.898499965668
                  return 0.0136231109825 < maxgini;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if max_col_coverage > 0.459404528141
    if ( mean_col_support <= 0.981280386448 ) {
      if ( median_col_support <= 0.982499957085 ) {
        if ( min_col_support <= 0.703500032425 ) {
          if ( median_col_support <= 0.667500019073 ) {
            if ( median_col_coverage <= 0.350396156311 ) {
              if ( median_col_support <= 0.617499947548 ) {
                if ( median_col_coverage <= 0.258661866188 ) {
                  return 0.490898211136 < maxgini;
                }
                else {  // if median_col_coverage > 0.258661866188
                  return false;
                }
              }
              else {  // if median_col_support > 0.617499947548
                if ( median_col_support <= 0.662500023842 ) {
                  return 0.470134725642 < maxgini;
                }
                else {  // if median_col_support > 0.662500023842
                  return false;
                }
              }
            }
            else {  // if median_col_coverage > 0.350396156311
              if ( mean_col_support <= 0.841558814049 ) {
                if ( min_col_support <= 0.503499984741 ) {
                  return false;
                }
                else {  // if min_col_support > 0.503499984741
                  return false;
                }
              }
              else {  // if mean_col_support > 0.841558814049
                if ( median_col_support <= 0.602499961853 ) {
                  return false;
                }
                else {  // if median_col_support > 0.602499961853
                  return false;
                }
              }
            }
          }
          else {  // if median_col_support > 0.667500019073
            if ( median_col_support <= 0.923500001431 ) {
              if ( median_col_support <= 0.74950003624 ) {
                if ( max_col_coverage <= 0.770271539688 ) {
                  return 0.451021399212 < maxgini;
                }
                else {  // if max_col_coverage > 0.770271539688
                  return false;
                }
              }
              else {  // if median_col_support > 0.74950003624
                if ( mean_col_coverage <= 0.651040256023 ) {
                  return 0.292527591285 < maxgini;
                }
                else {  // if mean_col_coverage > 0.651040256023
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.923500001431
              if ( min_col_support <= 0.628499984741 ) {
                if ( median_col_support <= 0.96749997139 ) {
                  return false;
                }
                else {  // if median_col_support > 0.96749997139
                  return false;
                }
              }
              else {  // if min_col_support > 0.628499984741
                if ( median_col_coverage <= 0.667445480824 ) {
                  return 0.377175763485 < maxgini;
                }
                else {  // if median_col_coverage > 0.667445480824
                  return false;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.703500032425
          if ( median_col_coverage <= 0.86967843771 ) {
            if ( min_col_support <= 0.81350004673 ) {
              if ( mean_col_support <= 0.935485363007 ) {
                if ( mean_col_coverage <= 0.409457027912 ) {
                  return 0.307581234574 < maxgini;
                }
                else {  // if mean_col_coverage > 0.409457027912
                  return 0.399706373288 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.935485363007
                if ( median_col_coverage <= 0.667184233665 ) {
                  return 0.181648012262 < maxgini;
                }
                else {  // if median_col_coverage > 0.667184233665
                  return 0.36521181694 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.81350004673
              if ( min_col_support <= 0.848500013351 ) {
                if ( mean_col_support <= 0.95667642355 ) {
                  return 0.262271527241 < maxgini;
                }
                else {  // if mean_col_support > 0.95667642355
                  return 0.116524543065 < maxgini;
                }
              }
              else {  // if min_col_support > 0.848500013351
                if ( min_col_support <= 0.865499973297 ) {
                  return 0.119436602552 < maxgini;
                }
                else {  // if min_col_support > 0.865499973297
                  return 0.0911205855787 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.86967843771
            if ( min_col_support <= 0.823500037193 ) {
              if ( min_col_coverage <= 0.85737401247 ) {
                if ( max_col_coverage <= 0.997641503811 ) {
                  return 0.487723313934 < maxgini;
                }
                else {  // if max_col_coverage > 0.997641503811
                  return 0.394697210051 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.85737401247
                if ( mean_col_support <= 0.979676485062 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.979676485062
                  return 0.457856399584 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.823500037193
              if ( min_col_coverage <= 0.962382078171 ) {
                if ( min_col_support <= 0.859500050545 ) {
                  return 0.344708442488 < maxgini;
                }
                else {  // if min_col_support > 0.859500050545
                  return 0.168181418618 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.962382078171
                if ( median_col_support <= 0.949499964714 ) {
                  return 0.304269495839 < maxgini;
                }
                else {  // if median_col_support > 0.949499964714
                  return 0.493827160494 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.982499957085
        if ( min_col_support <= 0.715499997139 ) {
          if ( mean_col_coverage <= 0.444343626499 ) {
            if ( min_col_support <= 0.601500034332 ) {
              if ( median_col_coverage <= 0.269966602325 ) {
                if ( mean_col_coverage <= 0.31274703145 ) {
                  return 0.226249662177 < maxgini;
                }
                else {  // if mean_col_coverage > 0.31274703145
                  return 0.369329071701 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.269966602325
                if ( median_col_support <= 0.999000012875 ) {
                  return false;
                }
                else {  // if median_col_support > 0.999000012875
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.601500034332
              if ( min_col_support <= 0.666499972343 ) {
                if ( median_col_support <= 0.999000012875 ) {
                  return false;
                }
                else {  // if median_col_support > 0.999000012875
                  return 0.333235682952 < maxgini;
                }
              }
              else {  // if min_col_support > 0.666499972343
                if ( max_col_coverage <= 0.476076006889 ) {
                  return 0.471580193851 < maxgini;
                }
                else {  // if max_col_coverage > 0.476076006889
                  return 0.278604958233 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.444343626499
            if ( median_col_support <= 0.99950003624 ) {
              if ( mean_col_support <= 0.966205835342 ) {
                if ( mean_col_coverage <= 0.992613911629 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.992613911629
                  return false;
                }
              }
              else {  // if mean_col_support > 0.966205835342
                if ( min_col_coverage <= 0.605307161808 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.605307161808
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_coverage <= 0.550361633301 ) {
                if ( min_col_coverage <= 0.35018247366 ) {
                  return 0.464308606725 < maxgini;
                }
                else {  // if min_col_coverage > 0.35018247366
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.550361633301
                if ( min_col_support <= 0.639500021935 ) {
                  return false;
                }
                else {  // if min_col_support > 0.639500021935
                  return false;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.715499997139
          if ( median_col_support <= 0.99950003624 ) {
            if ( min_col_support <= 0.823500037193 ) {
              if ( min_col_coverage <= 0.527848243713 ) {
                if ( median_col_coverage <= 0.300927937031 ) {
                  return 0.27173119065 < maxgini;
                }
                else {  // if median_col_coverage > 0.300927937031
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.527848243713
                if ( min_col_support <= 0.786499977112 ) {
                  return false;
                }
                else {  // if min_col_support > 0.786499977112
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.823500037193
              if ( min_col_support <= 0.851500034332 ) {
                if ( median_col_coverage <= 0.916805148125 ) {
                  return 0.476532958426 < maxgini;
                }
                else {  // if median_col_coverage > 0.916805148125
                  return false;
                }
              }
              else {  // if min_col_support > 0.851500034332
                if ( mean_col_coverage <= 0.969400525093 ) {
                  return 0.270664038756 < maxgini;
                }
                else {  // if mean_col_coverage > 0.969400525093
                  return 0.498866213152 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.99950003624
            if ( min_col_coverage <= 0.750609755516 ) {
              if ( mean_col_coverage <= 0.455363273621 ) {
                if ( max_col_coverage <= 0.476108938456 ) {
                  return 0.15088457005 < maxgini;
                }
                else {  // if max_col_coverage > 0.476108938456
                  return 0.0729676865817 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.455363273621
                if ( mean_col_support <= 0.973264753819 ) {
                  return 0.264689659691 < maxgini;
                }
                else {  // if mean_col_support > 0.973264753819
                  return 0.128748850917 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.750609755516
              if ( min_col_coverage <= 0.920827746391 ) {
                if ( min_col_coverage <= 0.754311621189 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.754311621189
                  return 0.323096111132 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.920827746391
                if ( mean_col_coverage <= 0.94991350174 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.94991350174
                  return 0.448252809326 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if mean_col_support > 0.981280386448
      if ( min_col_coverage <= 0.933539509773 ) {
        if ( min_col_coverage <= 0.909396409988 ) {
          if ( max_col_coverage <= 0.800531983376 ) {
            if ( min_col_support <= 0.785500049591 ) {
              if ( median_col_support <= 0.99950003624 ) {
                if ( mean_col_coverage <= 0.511362075806 ) {
                  return 0.496166833018 < maxgini;
                }
                else {  // if mean_col_coverage > 0.511362075806
                  return false;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_support <= 0.737499952316 ) {
                  return 0.492860704336 < maxgini;
                }
                else {  // if min_col_support > 0.737499952316
                  return 0.263685091902 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.785500049591
              if ( min_col_coverage <= 0.432387828827 ) {
                if ( median_col_coverage <= 0.0348884388804 ) {
                  return 0.129799891833 < maxgini;
                }
                else {  // if median_col_coverage > 0.0348884388804
                  return 0.0137230460771 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.432387828827
                if ( mean_col_support <= 0.989441156387 ) {
                  return 0.0587546219831 < maxgini;
                }
                else {  // if mean_col_support > 0.989441156387
                  return 0.00402359422518 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.800531983376
            if ( min_col_support <= 0.822499990463 ) {
              if ( mean_col_coverage <= 0.703437685966 ) {
                if ( min_col_support <= 0.743999958038 ) {
                  return false;
                }
                else {  // if min_col_support > 0.743999958038
                  return 0.226907112586 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.703437685966
                if ( mean_col_support <= 0.98449999094 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.98449999094
                  return 0.496576947167 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.822499990463
              if ( mean_col_support <= 0.991088271141 ) {
                if ( median_col_coverage <= 0.667033791542 ) {
                  return 0.0336819376566 < maxgini;
                }
                else {  // if median_col_coverage > 0.667033791542
                  return 0.0796073802473 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.991088271141
                if ( min_col_coverage <= 0.0180311892182 ) {
                  return 0.262716049383 < maxgini;
                }
                else {  // if min_col_coverage > 0.0180311892182
                  return 0.00255034470604 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.909396409988
          if ( min_col_coverage <= 0.910210371017 ) {
            if ( mean_col_coverage <= 0.95579969883 ) {
              if ( mean_col_support <= 0.990617632866 ) {
                if ( median_col_support <= 0.992499947548 ) {
                  return 0.0 < maxgini;
                }
                else {  // if median_col_support > 0.992499947548
                  return false;
                }
              }
              else {  // if mean_col_support > 0.990617632866
                return 0.0 < maxgini;
              }
            }
            else {  // if mean_col_coverage > 0.95579969883
              if ( min_col_support <= 0.832499980927 ) {
                if ( min_col_coverage <= 0.909442663193 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_coverage > 0.909442663193
                  return false;
                }
              }
              else {  // if min_col_support > 0.832499980927
                if ( median_col_coverage <= 0.946026861668 ) {
                  return 0.1171875 < maxgini;
                }
                else {  // if median_col_coverage > 0.946026861668
                  return 0.444444444444 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.910210371017
            if ( median_col_support <= 0.99950003624 ) {
              if ( min_col_support <= 0.878499984741 ) {
                if ( min_col_support <= 0.796499967575 ) {
                  return false;
                }
                else {  // if min_col_support > 0.796499967575
                  return false;
                }
              }
              else {  // if min_col_support > 0.878499984741
                if ( mean_col_coverage <= 0.982176959515 ) {
                  return 0.0151694975646 < maxgini;
                }
                else {  // if mean_col_coverage > 0.982176959515
                  return 0.0586618829978 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_coverage <= 0.910603702068 ) {
                if ( min_col_support <= 0.786000013351 ) {
                  return false;
                }
                else {  // if min_col_support > 0.786000013351
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.910603702068
                if ( min_col_coverage <= 0.923240602016 ) {
                  return 0.00792251719252 < maxgini;
                }
                else {  // if min_col_coverage > 0.923240602016
                  return 0.0133614487489 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_coverage > 0.933539509773
        if ( min_col_coverage <= 0.934750318527 ) {
          if ( median_col_coverage <= 0.94390141964 ) {
            if ( median_col_coverage <= 0.93703353405 ) {
              if ( min_col_support <= 0.960500001907 ) {
                if ( mean_col_coverage <= 0.935531616211 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.935531616211
                  return 0.484429065744 < maxgini;
                }
              }
              else {  // if min_col_support > 0.960500001907
                return 0.0 < maxgini;
              }
            }
            else {  // if median_col_coverage > 0.93703353405
              if ( min_col_support <= 0.922500014305 ) {
                if ( mean_col_coverage <= 0.967356741428 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.967356741428
                  return false;
                }
              }
              else {  // if min_col_support > 0.922500014305
                return 0.0 < maxgini;
              }
            }
          }
          else {  // if median_col_coverage > 0.94390141964
            if ( max_col_coverage <= 0.990941166878 ) {
              if ( min_col_coverage <= 0.934178471565 ) {
                if ( min_col_support <= 0.774999976158 ) {
                  return false;
                }
                else {  // if min_col_support > 0.774999976158
                  return 0.0831758034026 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.934178471565
                return 0.0 < maxgini;
              }
            }
            else {  // if max_col_coverage > 0.990941166878
              if ( max_col_coverage <= 0.996931433678 ) {
                if ( median_col_coverage <= 0.964881360531 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.964881360531
                  return 0.0 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.996931433678
                if ( min_col_support <= 0.901499986649 ) {
                  return false;
                }
                else {  // if min_col_support > 0.901499986649
                  return 0.0 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.934750318527
          if ( max_col_coverage <= 0.99903845787 ) {
            if ( max_col_coverage <= 0.981524169445 ) {
              if ( median_col_support <= 0.99849998951 ) {
                if ( median_col_support <= 0.994500041008 ) {
                  return 0.160992628232 < maxgini;
                }
                else {  // if median_col_support > 0.994500041008
                  return 0.449252094917 < maxgini;
                }
              }
              else {  // if median_col_support > 0.99849998951
                if ( max_col_coverage <= 0.979660749435 ) {
                  return 0.0332422028655 < maxgini;
                }
                else {  // if max_col_coverage > 0.979660749435
                  return 0.170303578495 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.981524169445
              if ( median_col_coverage <= 0.944529891014 ) {
                if ( median_col_support <= 0.987499952316 ) {
                  return false;
                }
                else {  // if median_col_support > 0.987499952316
                  return 0.209002019343 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.944529891014
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.459726077098 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.373666676098 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.99903845787
            if ( min_col_coverage <= 0.973753094673 ) {
              if ( median_col_coverage <= 0.950073301792 ) {
                if ( mean_col_support <= 0.986088275909 ) {
                  return 0.443137299606 < maxgini;
                }
                else {  // if mean_col_support > 0.986088275909
                  return 0.0052103625941 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.950073301792
                if ( min_col_support <= 0.869500041008 ) {
                  return false;
                }
                else {  // if min_col_support > 0.869500041008
                  return 0.00970395294964 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.973753094673
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_coverage <= 0.977382361889 ) {
                  return 0.0777940102264 < maxgini;
                }
                else {  // if median_col_coverage > 0.977382361889
                  return 0.352180890283 < maxgini;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_coverage <= 0.974248409271 ) {
                  return 0.497777777778 < maxgini;
                }
                else {  // if min_col_coverage > 0.974248409271
                  return 0.0407378755032 < maxgini;
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
  if ( min_col_support <= 0.766499996185 ) {
    if ( median_col_coverage <= 0.500617265701 ) {
      if ( mean_col_support <= 0.888242721558 ) {
        if ( median_col_coverage <= 0.261415779591 ) {
          if ( max_col_coverage <= 0.276455760002 ) {
            if ( median_col_coverage <= 0.051787994802 ) {
              if ( min_col_coverage <= 0.0260304920375 ) {
                if ( min_col_support <= 0.50150001049 ) {
                  return 0.44091796875 < maxgini;
                }
                else {  // if min_col_support > 0.50150001049
                  return 0.210083326319 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0260304920375
                if ( mean_col_coverage <= 0.146966904402 ) {
                  return 0.407620515057 < maxgini;
                }
                else {  // if mean_col_coverage > 0.146966904402
                  return 0.483860959494 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.051787994802
              if ( mean_col_support <= 0.829147100449 ) {
                if ( median_col_support <= 0.567499995232 ) {
                  return 0.45726135611 < maxgini;
                }
                else {  // if median_col_support > 0.567499995232
                  return 0.393762795813 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.829147100449
                if ( min_col_support <= 0.479499995708 ) {
                  return 0.193622346144 < maxgini;
                }
                else {  // if min_col_support > 0.479499995708
                  return 0.335194216754 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.276455760002
            if ( mean_col_support <= 0.850606620312 ) {
              if ( min_col_support <= 0.489499986172 ) {
                if ( min_col_coverage <= 0.0659420341253 ) {
                  return 0.482421875 < maxgini;
                }
                else {  // if min_col_coverage > 0.0659420341253
                  return 0.356439795455 < maxgini;
                }
              }
              else {  // if min_col_support > 0.489499986172
                if ( mean_col_coverage <= 0.327548384666 ) {
                  return 0.469478590922 < maxgini;
                }
                else {  // if mean_col_coverage > 0.327548384666
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.850606620312
              if ( mean_col_coverage <= 0.252211391926 ) {
                if ( min_col_support <= 0.499500006437 ) {
                  return 0.24848992793 < maxgini;
                }
                else {  // if min_col_support > 0.499500006437
                  return 0.38464771713 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.252211391926
                if ( min_col_support <= 0.491500020027 ) {
                  return 0.26084260824 < maxgini;
                }
                else {  // if min_col_support > 0.491500020027
                  return 0.425357336053 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.261415779591
          if ( median_col_support <= 0.619500041008 ) {
            if ( mean_col_coverage <= 0.364859521389 ) {
              if ( mean_col_coverage <= 0.322055131197 ) {
                if ( max_col_coverage <= 0.32071429491 ) {
                  return 0.430006016012 < maxgini;
                }
                else {  // if max_col_coverage > 0.32071429491
                  return 0.494043024425 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.322055131197
                if ( median_col_support <= 0.551499962807 ) {
                  return false;
                }
                else {  // if median_col_support > 0.551499962807
                  return 0.499741735537 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.364859521389
              if ( median_col_support <= 0.586500048637 ) {
                if ( max_col_coverage <= 0.837719321251 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.837719321251
                  return 0.491626297578 < maxgini;
                }
              }
              else {  // if median_col_support > 0.586500048637
                if ( mean_col_coverage <= 0.455562651157 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.455562651157
                  return false;
                }
              }
            }
          }
          else {  // if median_col_support > 0.619500041008
            if ( median_col_support <= 0.712499976158 ) {
              if ( mean_col_coverage <= 0.411958217621 ) {
                if ( median_col_support <= 0.665500044823 ) {
                  return 0.464000952551 < maxgini;
                }
                else {  // if median_col_support > 0.665500044823
                  return 0.42035706271 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.411958217621
                if ( mean_col_coverage <= 0.460402160883 ) {
                  return 0.484985591169 < maxgini;
                }
                else {  // if mean_col_coverage > 0.460402160883
                  return 0.499915830766 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.712499976158
              if ( median_col_coverage <= 0.350675672293 ) {
                if ( mean_col_coverage <= 0.361770510674 ) {
                  return 0.314834099056 < maxgini;
                }
                else {  // if mean_col_coverage > 0.361770510674
                  return 0.393928512388 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.350675672293
                if ( min_col_support <= 0.709499955177 ) {
                  return 0.445199779717 < maxgini;
                }
                else {  // if min_col_support > 0.709499955177
                  return 0.38285140335 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.888242721558
        if ( min_col_coverage <= 0.304437279701 ) {
          if ( mean_col_support <= 0.92622256279 ) {
            if ( mean_col_support <= 0.90814357996 ) {
              if ( min_col_support <= 0.522500038147 ) {
                if ( min_col_support <= 0.499500006437 ) {
                  return 0.21360554251 < maxgini;
                }
                else {  // if min_col_support > 0.499500006437
                  return 0.398885380339 < maxgini;
                }
              }
              else {  // if min_col_support > 0.522500038147
                if ( mean_col_coverage <= 0.300779283047 ) {
                  return 0.309402004943 < maxgini;
                }
                else {  // if mean_col_coverage > 0.300779283047
                  return 0.374558174129 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.90814357996
              if ( median_col_coverage <= 0.051787994802 ) {
                if ( median_col_support <= 0.761500000954 ) {
                  return 0.407129336434 < maxgini;
                }
                else {  // if median_col_support > 0.761500000954
                  return 0.309352683798 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.051787994802
                if ( median_col_coverage <= 0.25076687336 ) {
                  return 0.23734232513 < maxgini;
                }
                else {  // if median_col_coverage > 0.25076687336
                  return 0.338517808083 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.92622256279
            if ( median_col_coverage <= 0.250475287437 ) {
              if ( max_col_coverage <= 0.251582354307 ) {
                if ( mean_col_support <= 0.941083908081 ) {
                  return 0.16393664252 < maxgini;
                }
                else {  // if mean_col_support > 0.941083908081
                  return 0.0810466304082 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.251582354307
                if ( min_col_support <= 0.550500035286 ) {
                  return 0.258259750729 < maxgini;
                }
                else {  // if min_col_support > 0.550500035286
                  return 0.136291568999 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.250475287437
              if ( median_col_support <= 0.962499976158 ) {
                if ( median_col_support <= 0.809499979019 ) {
                  return 0.346287773045 < maxgini;
                }
                else {  // if median_col_support > 0.809499979019
                  return 0.16255581925 < maxgini;
                }
              }
              else {  // if median_col_support > 0.962499976158
                if ( max_col_coverage <= 0.350396156311 ) {
                  return 0.180286161399 < maxgini;
                }
                else {  // if max_col_coverage > 0.350396156311
                  return 0.404977755869 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.304437279701
          if ( mean_col_coverage <= 0.442613691092 ) {
            if ( mean_col_support <= 0.96744120121 ) {
              if ( min_col_support <= 0.604499995708 ) {
                if ( mean_col_support <= 0.939911723137 ) {
                  return 0.452595484861 < maxgini;
                }
                else {  // if mean_col_support > 0.939911723137
                  return false;
                }
              }
              else {  // if min_col_support > 0.604499995708
                if ( median_col_support <= 0.770500004292 ) {
                  return 0.451456093265 < maxgini;
                }
                else {  // if median_col_support > 0.770500004292
                  return 0.251394581285 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.96744120121
              if ( median_col_support <= 0.982499957085 ) {
                if ( mean_col_coverage <= 0.429302811623 ) {
                  return 0.155420525097 < maxgini;
                }
                else {  // if mean_col_coverage > 0.429302811623
                  return 0.3318 < maxgini;
                }
              }
              else {  // if median_col_support > 0.982499957085
                if ( median_col_coverage <= 0.335169315338 ) {
                  return 0.388555340391 < maxgini;
                }
                else {  // if median_col_coverage > 0.335169315338
                  return 0.491790960462 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.442613691092
            if ( min_col_coverage <= 0.36373308301 ) {
              if ( median_col_coverage <= 0.363780200481 ) {
                if ( median_col_support <= 0.973500013351 ) {
                  return 0.263690371732 < maxgini;
                }
                else {  // if median_col_support > 0.973500013351
                  return 0.432473220162 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.363780200481
                if ( median_col_coverage <= 0.399745553732 ) {
                  return 0.457339305028 < maxgini;
                }
                else {  // if median_col_coverage > 0.399745553732
                  return 0.375148411979 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.36373308301
              if ( median_col_support <= 0.977499961853 ) {
                if ( median_col_support <= 0.756500005722 ) {
                  return 0.48821514466 < maxgini;
                }
                else {  // if median_col_support > 0.756500005722
                  return 0.345792799226 < maxgini;
                }
              }
              else {  // if median_col_support > 0.977499961853
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.499898128804 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if median_col_coverage > 0.500617265701
      if ( median_col_support <= 0.984500050545 ) {
        if ( max_col_coverage <= 0.863706707954 ) {
          if ( min_col_coverage <= 0.600300312042 ) {
            if ( median_col_coverage <= 0.513473153114 ) {
              if ( median_col_coverage <= 0.506153941154 ) {
                if ( mean_col_support <= 0.950323462486 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.950323462486
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.506153941154
                if ( min_col_support <= 0.674499988556 ) {
                  return false;
                }
                else {  // if min_col_support > 0.674499988556
                  return 0.475308641975 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.513473153114
              if ( max_col_coverage <= 0.66705429554 ) {
                if ( median_col_support <= 0.734500050545 ) {
                  return false;
                }
                else {  // if median_col_support > 0.734500050545
                  return 0.330441834534 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.66705429554
                if ( median_col_support <= 0.724500000477 ) {
                  return false;
                }
                else {  // if median_col_support > 0.724500000477
                  return 0.451340263849 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.600300312042
            if ( median_col_coverage <= 0.619147241116 ) {
              if ( mean_col_support <= 0.870676457882 ) {
                if ( min_col_coverage <= 0.608471512794 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.608471512794
                  return false;
                }
              }
              else {  // if mean_col_support > 0.870676457882
                if ( mean_col_support <= 0.955529332161 ) {
                  return 0.367570595753 < maxgini;
                }
                else {  // if mean_col_support > 0.955529332161
                  return 0.160367105422 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.619147241116
              if ( median_col_support <= 0.710500001907 ) {
                if ( mean_col_support <= 0.864970684052 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.864970684052
                  return false;
                }
              }
              else {  // if median_col_support > 0.710500001907
                if ( min_col_support <= 0.658499956131 ) {
                  return false;
                }
                else {  // if min_col_support > 0.658499956131
                  return 0.423218058607 < maxgini;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.863706707954
          if ( min_col_support <= 0.666499972343 ) {
            if ( median_col_coverage <= 0.70876455307 ) {
              if ( min_col_coverage <= 0.574271082878 ) {
                if ( median_col_support <= 0.70300000906 ) {
                  return false;
                }
                else {  // if median_col_support > 0.70300000906
                  return 0.477330316273 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.574271082878
                if ( min_col_support <= 0.48299998045 ) {
                  return 0.290657439446 < maxgini;
                }
                else {  // if min_col_support > 0.48299998045
                  return false;
                }
              }
            }
            else {  // if median_col_coverage > 0.70876455307
              if ( max_col_coverage <= 0.997355937958 ) {
                if ( median_col_coverage <= 0.846344232559 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.846344232559
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.997355937958
                if ( min_col_coverage <= 0.313657402992 ) {
                  return 0.296608593217 < maxgini;
                }
                else {  // if min_col_coverage > 0.313657402992
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.666499972343
            if ( mean_col_coverage <= 0.842482924461 ) {
              if ( median_col_coverage <= 0.714462995529 ) {
                if ( median_col_support <= 0.74849998951 ) {
                  return false;
                }
                else {  // if median_col_support > 0.74849998951
                  return 0.323364411823 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.714462995529
                if ( mean_col_support <= 0.972794175148 ) {
                  return 0.486716218052 < maxgini;
                }
                else {  // if mean_col_support > 0.972794175148
                  return 0.399973553719 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.842482924461
              if ( mean_col_coverage <= 0.945960044861 ) {
                if ( min_col_coverage <= 0.800462961197 ) {
                  return 0.44987055213 < maxgini;
                }
                else {  // if min_col_coverage > 0.800462961197
                  return false;
                }
              }
              else {  // if mean_col_coverage > 0.945960044861
                if ( min_col_coverage <= 0.925116837025 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.925116837025
                  return false;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.984500050545
        if ( mean_col_support <= 0.976852893829 ) {
          if ( median_col_coverage <= 0.65415096283 ) {
            if ( min_col_support <= 0.611500024796 ) {
              if ( median_col_support <= 0.990499973297 ) {
                if ( min_col_support <= 0.53149998188 ) {
                  return false;
                }
                else {  // if min_col_support > 0.53149998188
                  return false;
                }
              }
              else {  // if median_col_support > 0.990499973297
                if ( mean_col_support <= 0.962970554829 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.962970554829
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.611500024796
              if ( min_col_support <= 0.705500006676 ) {
                if ( min_col_coverage <= 0.446185410023 ) {
                  return 0.45544664736 < maxgini;
                }
                else {  // if min_col_coverage > 0.446185410023
                  return false;
                }
              }
              else {  // if min_col_support > 0.705500006676
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.374846484509 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.65415096283
            if ( median_col_support <= 0.990499973297 ) {
              if ( min_col_support <= 0.619500041008 ) {
                if ( max_col_coverage <= 0.748786449432 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.748786449432
                  return false;
                }
              }
              else {  // if min_col_support > 0.619500041008
                if ( max_col_coverage <= 0.771541059017 ) {
                  return 0.473372781065 < maxgini;
                }
                else {  // if max_col_coverage > 0.771541059017
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.990499973297
              if ( median_col_coverage <= 0.994922697544 ) {
                if ( min_col_coverage <= 0.886127412319 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.886127412319
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.994922697544
                if ( mean_col_support <= 0.948382377625 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.948382377625
                  return false;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.976852893829
          if ( mean_col_coverage <= 0.71208691597 ) {
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_support <= 0.991500020027 ) {
                if ( min_col_support <= 0.670500040054 ) {
                  return false;
                }
                else {  // if min_col_support > 0.670500040054
                  return 0.474427591848 < maxgini;
                }
              }
              else {  // if median_col_support > 0.991500020027
                if ( max_col_coverage <= 0.630552768707 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.630552768707
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( mean_col_coverage <= 0.541647791862 ) {
                if ( mean_col_support <= 0.977088272572 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.977088272572
                  return 0.309051887388 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.541647791862
                if ( max_col_coverage <= 0.807515859604 ) {
                  return 0.499481309738 < maxgini;
                }
                else {  // if max_col_coverage > 0.807515859604
                  return 0.433008187967 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.71208691597
            if ( median_col_support <= 0.99950003624 ) {
              if ( mean_col_support <= 0.984088242054 ) {
                if ( min_col_support <= 0.713500022888 ) {
                  return false;
                }
                else {  // if min_col_support > 0.713500022888
                  return false;
                }
              }
              else {  // if mean_col_support > 0.984088242054
                if ( max_col_coverage <= 0.825099289417 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.825099289417
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_support <= 0.702499985695 ) {
                if ( min_col_support <= 0.639500021935 ) {
                  return false;
                }
                else {  // if min_col_support > 0.639500021935
                  return false;
                }
              }
              else {  // if min_col_support > 0.702499985695
                if ( min_col_coverage <= 0.882555782795 ) {
                  return 0.491418840692 < maxgini;
                }
                else {  // if min_col_coverage > 0.882555782795
                  return false;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if min_col_support > 0.766499996185
    if ( min_col_support <= 0.856500029564 ) {
      if ( median_col_coverage <= 0.667447686195 ) {
        if ( mean_col_support <= 0.952485322952 ) {
          if ( mean_col_support <= 0.938531398773 ) {
            if ( median_col_coverage <= 0.386391460896 ) {
              if ( min_col_support <= 0.787500023842 ) {
                if ( max_col_coverage <= 0.397368431091 ) {
                  return 0.224887037334 < maxgini;
                }
                else {  // if max_col_coverage > 0.397368431091
                  return 0.273272300612 < maxgini;
                }
              }
              else {  // if min_col_support > 0.787500023842
                if ( mean_col_support <= 0.91991174221 ) {
                  return 0.466827364555 < maxgini;
                }
                else {  // if mean_col_support > 0.91991174221
                  return 0.31182221172 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.386391460896
              if ( median_col_support <= 0.830500006676 ) {
                if ( max_col_coverage <= 0.453079164028 ) {
                  return 0.497346607605 < maxgini;
                }
                else {  // if max_col_coverage > 0.453079164028
                  return 0.406261290401 < maxgini;
                }
              }
              else {  // if median_col_support > 0.830500006676
                if ( median_col_support <= 0.845499992371 ) {
                  return 0.342406716225 < maxgini;
                }
                else {  // if median_col_support > 0.845499992371
                  return 0.293157833489 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.938531398773
            if ( mean_col_coverage <= 0.429057717323 ) {
              if ( median_col_support <= 0.821500003338 ) {
                if ( mean_col_support <= 0.947088301182 ) {
                  return 0.318492431757 < maxgini;
                }
                else {  // if mean_col_support > 0.947088301182
                  return 0.251044770525 < maxgini;
                }
              }
              else {  // if median_col_support > 0.821500003338
                if ( min_col_coverage <= 0.271246254444 ) {
                  return 0.180872102334 < maxgini;
                }
                else {  // if min_col_coverage > 0.271246254444
                  return 0.135519165884 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.429057717323
              if ( max_col_coverage <= 0.993589758873 ) {
                if ( median_col_support <= 0.836500048637 ) {
                  return 0.377599067601 < maxgini;
                }
                else {  // if median_col_support > 0.836500048637
                  return 0.23249095423 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.993589758873
                if ( min_col_coverage <= 0.387499988079 ) {
                  return 0.484756865709 < maxgini;
                }
                else {  // if min_col_coverage > 0.387499988079
                  return 0.244897959184 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.952485322952
          if ( median_col_support <= 0.99950003624 ) {
            if ( min_col_coverage <= 0.500582754612 ) {
              if ( median_col_support <= 0.989500045776 ) {
                if ( max_col_coverage <= 0.963463485241 ) {
                  return 0.11086734495 < maxgini;
                }
                else {  // if max_col_coverage > 0.963463485241
                  return 0.398368284976 < maxgini;
                }
              }
              else {  // if median_col_support > 0.989500045776
                if ( median_col_coverage <= 0.280151516199 ) {
                  return 0.383554862503 < maxgini;
                }
                else {  // if median_col_coverage > 0.280151516199
                  return false;
                }
              }
            }
            else {  // if min_col_coverage > 0.500582754612
              if ( median_col_support <= 0.990499973297 ) {
                if ( median_col_coverage <= 0.666247904301 ) {
                  return 0.168601322027 < maxgini;
                }
                else {  // if median_col_coverage > 0.666247904301
                  return 0.0879566537359 < maxgini;
                }
              }
              else {  // if median_col_support > 0.990499973297
                if ( mean_col_support <= 0.98344117403 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.98344117403
                  return false;
                }
              }
            }
          }
          else {  // if median_col_support > 0.99950003624
            if ( min_col_support <= 0.787500023842 ) {
              if ( median_col_coverage <= 0.503624141216 ) {
                if ( median_col_coverage <= 0.305216789246 ) {
                  return 0.0545944017758 < maxgini;
                }
                else {  // if median_col_coverage > 0.305216789246
                  return 0.118684321636 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.503624141216
                if ( min_col_support <= 0.779500007629 ) {
                  return 0.263342076517 < maxgini;
                }
                else {  // if min_col_support > 0.779500007629
                  return 0.20801720136 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.787500023842
              if ( mean_col_coverage <= 0.385344862938 ) {
                if ( mean_col_support <= 0.974902749062 ) {
                  return 0.0734179191087 < maxgini;
                }
                else {  // if mean_col_support > 0.974902749062
                  return 0.0189634236796 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.385344862938
                if ( min_col_coverage <= 0.500778794289 ) {
                  return 0.0545306627056 < maxgini;
                }
                else {  // if min_col_coverage > 0.500778794289
                  return 0.0726472264452 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_coverage > 0.667447686195
        if ( mean_col_coverage <= 0.929460406303 ) {
          if ( max_col_coverage <= 0.998863637447 ) {
            if ( max_col_coverage <= 0.800276994705 ) {
              if ( min_col_support <= 0.799499988556 ) {
                if ( median_col_support <= 0.990499973297 ) {
                  return 0.215765936004 < maxgini;
                }
                else {  // if median_col_support > 0.990499973297
                  return 0.448197089994 < maxgini;
                }
              }
              else {  // if min_col_support > 0.799499988556
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.23716857345 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0660989749743 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.800276994705
              if ( min_col_support <= 0.809499979019 ) {
                if ( median_col_support <= 0.988499999046 ) {
                  return 0.36168499358 < maxgini;
                }
                else {  // if median_col_support > 0.988499999046
                  return false;
                }
              }
              else {  // if min_col_support > 0.809499979019
                if ( mean_col_coverage <= 0.877247095108 ) {
                  return 0.308377376874 < maxgini;
                }
                else {  // if mean_col_coverage > 0.877247095108
                  return 0.417942147656 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.998863637447
            if ( median_col_coverage <= 0.675337851048 ) {
              if ( mean_col_coverage <= 0.849785208702 ) {
                if ( mean_col_coverage <= 0.841296195984 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_coverage > 0.841296195984
                  return 0.444444444444 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.849785208702
                return false;
              }
            }
            else {  // if median_col_coverage > 0.675337851048
              if ( min_col_coverage <= 0.34787389636 ) {
                if ( mean_col_coverage <= 0.844498336315 ) {
                  return 0.14201183432 < maxgini;
                }
                else {  // if mean_col_coverage > 0.844498336315
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.34787389636
                if ( min_col_support <= 0.806499958038 ) {
                  return 0.321313864222 < maxgini;
                }
                else {  // if min_col_support > 0.806499958038
                  return 0.111059195973 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.929460406303
          if ( min_col_coverage <= 0.909228205681 ) {
            if ( max_col_coverage <= 0.998721241951 ) {
              if ( mean_col_coverage <= 0.929772853851 ) {
                if ( mean_col_support <= 0.985352993011 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.985352993011
                  return false;
                }
              }
              else {  // if mean_col_coverage > 0.929772853851
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.401094623984 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.998721241951
              if ( min_col_support <= 0.830500006676 ) {
                if ( median_col_coverage <= 0.948163866997 ) {
                  return 0.465421095518 < maxgini;
                }
                else {  // if median_col_coverage > 0.948163866997
                  return 0.32 < maxgini;
                }
              }
              else {  // if min_col_support > 0.830500006676
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.393889660396 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0771111478619 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.909228205681
            if ( mean_col_coverage <= 0.990174472332 ) {
              if ( median_col_support <= 0.99950003624 ) {
                if ( max_col_coverage <= 0.997402429581 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.997402429581
                  return false;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_coverage <= 0.968378961086 ) {
                  return 0.410369686399 < maxgini;
                }
                else {  // if min_col_coverage > 0.968378961086
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.990174472332
              if ( mean_col_coverage <= 0.999830961227 ) {
                if ( min_col_coverage <= 0.969107866287 ) {
                  return 0.293817281815 < maxgini;
                }
                else {  // if min_col_coverage > 0.969107866287
                  return false;
                }
              }
              else {  // if mean_col_coverage > 0.999830961227
                if ( mean_col_support <= 0.984617590904 ) {
                  return 0.470058753954 < maxgini;
                }
                else {  // if mean_col_support > 0.984617590904
                  return 0.345679012346 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if min_col_support > 0.856500029564
      if ( median_col_support <= 0.959499955177 ) {
        if ( max_col_coverage <= 0.58160674572 ) {
          if ( median_col_coverage <= 0.275151818991 ) {
            if ( mean_col_support <= 0.966794073582 ) {
              if ( mean_col_coverage <= 0.30758947134 ) {
                if ( min_col_support <= 0.886500000954 ) {
                  return 0.247023062382 < maxgini;
                }
                else {  // if min_col_support > 0.886500000954
                  return 0.482853223594 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.30758947134
                if ( max_col_coverage <= 0.453463196754 ) {
                  return 0.157902758726 < maxgini;
                }
                else {  // if max_col_coverage > 0.453463196754
                  return 0.268398843337 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.966794073582
              if ( median_col_support <= 0.90649998188 ) {
                if ( min_col_support <= 0.870499968529 ) {
                  return 0.108828050749 < maxgini;
                }
                else {  // if min_col_support > 0.870499968529
                  return 0.200361872231 < maxgini;
                }
              }
              else {  // if median_col_support > 0.90649998188
                if ( median_col_support <= 0.944499969482 ) {
                  return 0.112448016898 < maxgini;
                }
                else {  // if median_col_support > 0.944499969482
                  return 0.0784973502795 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.275151818991
            if ( min_col_support <= 0.872500002384 ) {
              if ( median_col_support <= 0.883499979973 ) {
                if ( mean_col_coverage <= 0.487226873636 ) {
                  return 0.183764529332 < maxgini;
                }
                else {  // if mean_col_coverage > 0.487226873636
                  return 0.372006094104 < maxgini;
                }
              }
              else {  // if median_col_support > 0.883499979973
                if ( min_col_support <= 0.871500015259 ) {
                  return 0.0949612893976 < maxgini;
                }
                else {  // if min_col_support > 0.871500015259
                  return 0.139183543315 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.872500002384
              if ( median_col_support <= 0.901499986649 ) {
                if ( mean_col_support <= 0.95714700222 ) {
                  return 0.345940462511 < maxgini;
                }
                else {  // if mean_col_support > 0.95714700222
                  return 0.146796977991 < maxgini;
                }
              }
              else {  // if median_col_support > 0.901499986649
                if ( min_col_support <= 0.942499995232 ) {
                  return 0.0736250773226 < maxgini;
                }
                else {  // if min_col_support > 0.942499995232
                  return 0.141693636775 < maxgini;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.58160674572
          if ( mean_col_support <= 0.969970583916 ) {
            if ( median_col_support <= 0.883499979973 ) {
              if ( mean_col_coverage <= 0.504061579704 ) {
                if ( mean_col_support <= 0.944500029087 ) {
                  return 0.444444444444 < maxgini;
                }
                else {  // if mean_col_support > 0.944500029087
                  return 0.124960140306 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.504061579704
                if ( mean_col_coverage <= 0.900662660599 ) {
                  return 0.309606204798 < maxgini;
                }
                else {  // if mean_col_coverage > 0.900662660599
                  return 0.0333237575409 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.883499979973
              if ( median_col_support <= 0.892500042915 ) {
                if ( min_col_coverage <= 0.854497373104 ) {
                  return 0.182691891304 < maxgini;
                }
                else {  // if min_col_coverage > 0.854497373104
                  return 0.498866213152 < maxgini;
                }
              }
              else {  // if median_col_support > 0.892500042915
                if ( mean_col_coverage <= 0.956026554108 ) {
                  return 0.134613294662 < maxgini;
                }
                else {  // if mean_col_coverage > 0.956026554108
                  return 0.356698296092 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.969970583916
            if ( min_col_coverage <= 0.972095608711 ) {
              if ( mean_col_support <= 0.976852893829 ) {
                if ( median_col_support <= 0.894500017166 ) {
                  return 0.177761093358 < maxgini;
                }
                else {  // if median_col_support > 0.894500017166
                  return 0.0867155589866 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.976852893829
                if ( min_col_coverage <= 0.213203459978 ) {
                  return 0.145819179402 < maxgini;
                }
                else {  // if min_col_coverage > 0.213203459978
                  return 0.0407670947412 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.972095608711
              if ( min_col_coverage <= 0.997237563133 ) {
                if ( median_col_support <= 0.923500001431 ) {
                  return false;
                }
                else {  // if median_col_support > 0.923500001431
                  return 0.447292966773 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.997237563133
                if ( mean_col_support <= 0.982029438019 ) {
                  return 0.340264650284 < maxgini;
                }
                else {  // if mean_col_support > 0.982029438019
                  return 0.0715265306122 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.959499955177
        if ( mean_col_coverage <= 0.998351275921 ) {
          if ( mean_col_support <= 0.991719603539 ) {
            if ( mean_col_coverage <= 0.956607937813 ) {
              if ( mean_col_support <= 0.985558867455 ) {
                if ( max_col_support <= 0.997500002384 ) {
                  return 0.495 < maxgini;
                }
                else {  // if max_col_support > 0.997500002384
                  return 0.0614584412199 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.985558867455
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.0423197777874 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0162384867263 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.956607937813
              if ( min_col_coverage <= 0.968847036362 ) {
                if ( max_col_coverage <= 0.998806715012 ) {
                  return 0.444119558847 < maxgini;
                }
                else {  // if max_col_coverage > 0.998806715012
                  return 0.0807133135047 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.968847036362
                if ( median_col_support <= 0.989500045776 ) {
                  return 0.343971462545 < maxgini;
                }
                else {  // if median_col_support > 0.989500045776
                  return 0.49721541546 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.991719603539
            if ( median_col_support <= 0.99950003624 ) {
              if ( min_col_coverage <= 0.982287526131 ) {
                if ( mean_col_support <= 0.992852926254 ) {
                  return 0.0324853274607 < maxgini;
                }
                else {  // if mean_col_support > 0.992852926254
                  return 0.00615874513621 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.982287526131
                if ( min_col_coverage <= 0.982404530048 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.982404530048
                  return 0.124444444444 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_support <= 0.965499997139 ) {
                if ( mean_col_coverage <= 0.613117694855 ) {
                  return 0.00431650754623 < maxgini;
                }
                else {  // if mean_col_coverage > 0.613117694855
                  return 0.00250342187482 < maxgini;
                }
              }
              else {  // if min_col_support > 0.965499997139
                if ( min_col_coverage <= 0.579099059105 ) {
                  return 0.00205501558633 < maxgini;
                }
                else {  // if min_col_coverage > 0.579099059105
                  return 0.000707025710303 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.998351275921
          if ( mean_col_support <= 0.985617697239 ) {
            if ( median_col_support <= 0.999000012875 ) {
              if ( mean_col_support <= 0.97979414463 ) {
                if ( min_col_support <= 0.872500002384 ) {
                  return 0.499773653237 < maxgini;
                }
                else {  // if min_col_support > 0.872500002384
                  return false;
                }
              }
              else {  // if mean_col_support > 0.97979414463
                if ( mean_col_coverage <= 0.998682141304 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.998682141304
                  return 0.473935312084 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.999000012875
              if ( min_col_support <= 0.868999958038 ) {
                if ( min_col_support <= 0.866500020027 ) {
                  return 0.0654984199943 < maxgini;
                }
                else {  // if min_col_support > 0.866500020027
                  return 0.4296875 < maxgini;
                }
              }
              else {  // if min_col_support > 0.868999958038
                if ( mean_col_support <= 0.985411763191 ) {
                  return 0.0173146965161 < maxgini;
                }
                else {  // if mean_col_support > 0.985411763191
                  return 0.444444444444 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.985617697239
            if ( max_col_support <= 0.99950003624 ) {
              return 0.0 < maxgini;
            }
            else {  // if max_col_support > 0.99950003624
              if ( mean_col_support <= 0.992323517799 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.308312592772 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0334779983197 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.992323517799
                if ( min_col_support <= 0.93649995327 ) {
                  return 0.0276831465535 < maxgini;
                }
                else {  // if min_col_support > 0.93649995327
                  return 0.0102894192883 < maxgini;
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
  if ( min_col_support <= 0.770500004292 ) {
    if ( max_col_coverage <= 0.667507052422 ) {
      if ( max_col_coverage <= 0.500605344772 ) {
        if ( min_col_support <= 0.551499962807 ) {
          if ( median_col_support <= 0.619500041008 ) {
            if ( mean_col_coverage <= 0.282950997353 ) {
              if ( mean_col_coverage <= 0.188565641642 ) {
                if ( median_col_coverage <= 0.0254787411541 ) {
                  return 0.130501692829 < maxgini;
                }
                else {  // if median_col_coverage > 0.0254787411541
                  return 0.431044031564 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.188565641642
                if ( mean_col_coverage <= 0.224057808518 ) {
                  return 0.462514172336 < maxgini;
                }
                else {  // if mean_col_coverage > 0.224057808518
                  return 0.48256678679 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.282950997353
              if ( median_col_support <= 0.574499964714 ) {
                if ( min_col_coverage <= 0.261204004288 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.261204004288
                  return false;
                }
              }
              else {  // if median_col_support > 0.574499964714
                if ( mean_col_coverage <= 0.301688790321 ) {
                  return 0.477713083824 < maxgini;
                }
                else {  // if mean_col_coverage > 0.301688790321
                  return false;
                }
              }
            }
          }
          else {  // if median_col_support > 0.619500041008
            if ( min_col_coverage <= 0.250333338976 ) {
              if ( mean_col_support <= 0.918207168579 ) {
                if ( min_col_coverage <= 0.0501046031713 ) {
                  return 0.40349713006 < maxgini;
                }
                else {  // if min_col_coverage > 0.0501046031713
                  return 0.319613823606 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.918207168579
                if ( median_col_support <= 0.940500020981 ) {
                  return 0.196548143334 < maxgini;
                }
                else {  // if median_col_support > 0.940500020981
                  return 0.302263789988 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.250333338976
              if ( median_col_coverage <= 0.399421989918 ) {
                if ( median_col_support <= 0.911000013351 ) {
                  return 0.356382517952 < maxgini;
                }
                else {  // if median_col_support > 0.911000013351
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.399421989918
                if ( median_col_support <= 0.954499959946 ) {
                  return 0.267109329213 < maxgini;
                }
                else {  // if median_col_support > 0.954499959946
                  return false;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.551499962807
          if ( median_col_coverage <= 0.250609755516 ) {
            if ( median_col_support <= 0.779500007629 ) {
              if ( max_col_coverage <= 0.275181174278 ) {
                if ( max_col_coverage <= 0.202828317881 ) {
                  return 0.259805946577 < maxgini;
                }
                else {  // if max_col_coverage > 0.202828317881
                  return 0.306596370935 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.275181174278
                if ( median_col_coverage <= 0.0607908368111 ) {
                  return 0.436571389769 < maxgini;
                }
                else {  // if median_col_coverage > 0.0607908368111
                  return 0.354900365537 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.779500007629
              if ( min_col_coverage <= 0.0477821268141 ) {
                if ( max_col_coverage <= 0.335702091455 ) {
                  return 0.137426521329 < maxgini;
                }
                else {  // if max_col_coverage > 0.335702091455
                  return 0.197462689848 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0477821268141
                if ( mean_col_support <= 0.947594165802 ) {
                  return 0.159953123717 < maxgini;
                }
                else {  // if mean_col_support > 0.947594165802
                  return 0.0796256992963 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.250609755516
            if ( min_col_support <= 0.679499983788 ) {
              if ( min_col_support <= 0.621500015259 ) {
                if ( max_col_coverage <= 0.499295771122 ) {
                  return 0.456351144549 < maxgini;
                }
                else {  // if max_col_coverage > 0.499295771122
                  return 0.397860796364 < maxgini;
                }
              }
              else {  // if min_col_support > 0.621500015259
                if ( mean_col_coverage <= 0.386644005775 ) {
                  return 0.352708990979 < maxgini;
                }
                else {  // if mean_col_coverage > 0.386644005775
                  return 0.412915114679 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.679499983788
              if ( mean_col_coverage <= 0.370683342218 ) {
                if ( mean_col_coverage <= 0.30008649826 ) {
                  return 0.14180021831 < maxgini;
                }
                else {  // if mean_col_coverage > 0.30008649826
                  return 0.225605279833 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.370683342218
                if ( mean_col_coverage <= 0.370720118284 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.370720118284
                  return 0.295214135593 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if max_col_coverage > 0.500605344772
        if ( max_col_coverage <= 0.600257754326 ) {
          if ( median_col_coverage <= 0.350126922131 ) {
            if ( median_col_support <= 0.694499969482 ) {
              if ( min_col_support <= 0.582499980927 ) {
                if ( median_col_support <= 0.613499999046 ) {
                  return false;
                }
                else {  // if median_col_support > 0.613499999046
                  return 0.482638130374 < maxgini;
                }
              }
              else {  // if min_col_support > 0.582499980927
                if ( median_col_support <= 0.611500024796 ) {
                  return false;
                }
                else {  // if median_col_support > 0.611500024796
                  return 0.441751228083 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.694499969482
              if ( min_col_support <= 0.703500032425 ) {
                if ( median_col_support <= 0.971500039101 ) {
                  return 0.281059271905 < maxgini;
                }
                else {  // if median_col_support > 0.971500039101
                  return 0.414868865644 < maxgini;
                }
              }
              else {  // if min_col_support > 0.703500032425
                if ( median_col_support <= 0.824499964714 ) {
                  return 0.328520300895 < maxgini;
                }
                else {  // if median_col_support > 0.824499964714
                  return 0.116231949898 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.350126922131
            if ( median_col_support <= 0.982499957085 ) {
              if ( mean_col_support <= 0.892499983311 ) {
                if ( median_col_support <= 0.662500023842 ) {
                  return false;
                }
                else {  // if median_col_support > 0.662500023842
                  return 0.465878800098 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.892499983311
                if ( min_col_support <= 0.587499976158 ) {
                  return 0.483502474734 < maxgini;
                }
                else {  // if min_col_support > 0.587499976158
                  return 0.31863994733 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.982499957085
              if ( mean_col_support <= 0.974617660046 ) {
                if ( median_col_support <= 0.999000012875 ) {
                  return false;
                }
                else {  // if median_col_support > 0.999000012875
                  return false;
                }
              }
              else {  // if mean_col_support > 0.974617660046
                if ( min_col_support <= 0.695500016212 ) {
                  return false;
                }
                else {  // if min_col_support > 0.695500016212
                  return 0.404215071635 < maxgini;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.600257754326
          if ( min_col_coverage <= 0.40082681179 ) {
            if ( mean_col_coverage <= 0.447641372681 ) {
              if ( max_col_coverage <= 0.602574706078 ) {
                if ( min_col_coverage <= 0.275461375713 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.275461375713
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.602574706078
                if ( median_col_support <= 0.759500026703 ) {
                  return 0.469719489735 < maxgini;
                }
                else {  // if median_col_support > 0.759500026703
                  return 0.215377883472 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.447641372681
              if ( min_col_coverage <= 0.363810539246 ) {
                if ( median_col_support <= 0.681499958038 ) {
                  return false;
                }
                else {  // if median_col_support > 0.681499958038
                  return 0.348590299935 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.363810539246
                if ( median_col_support <= 0.981500029564 ) {
                  return 0.385933130522 < maxgini;
                }
                else {  // if median_col_support > 0.981500029564
                  return false;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.40082681179
            if ( max_col_coverage <= 0.666228055954 ) {
              if ( max_col_coverage <= 0.657193303108 ) {
                if ( min_col_coverage <= 0.498645961285 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.498645961285
                  return 0.485125494445 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.657193303108
                if ( median_col_coverage <= 0.500649333 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.500649333
                  return false;
                }
              }
            }
            else {  // if max_col_coverage > 0.666228055954
              if ( median_col_coverage <= 0.569035947323 ) {
                if ( max_col_coverage <= 0.667461752892 ) {
                  return 0.442068002327 < maxgini;
                }
                else {  // if max_col_coverage > 0.667461752892
                  return 0.0 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.569035947323
                if ( median_col_support <= 0.986000001431 ) {
                  return 0.234178766789 < maxgini;
                }
                else {  // if median_col_support > 0.986000001431
                  return 0.476680111971 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if max_col_coverage > 0.667507052422
      if ( mean_col_support <= 0.939088225365 ) {
        if ( mean_col_coverage <= 0.649894118309 ) {
          if ( min_col_coverage <= 0.502958476543 ) {
            if ( median_col_support <= 0.71749997139 ) {
              if ( median_col_support <= 0.620499968529 ) {
                if ( min_col_support <= 0.427999973297 ) {
                  return 0.39239026464 < maxgini;
                }
                else {  // if min_col_support > 0.427999973297
                  return false;
                }
              }
              else {  // if median_col_support > 0.620499968529
                if ( max_col_coverage <= 0.848076939583 ) {
                  return 0.489464099896 < maxgini;
                }
                else {  // if max_col_coverage > 0.848076939583
                  return 0.415224913495 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.71749997139
              if ( mean_col_support <= 0.872823536396 ) {
                if ( min_col_support <= 0.731500029564 ) {
                  return 0.419033241399 < maxgini;
                }
                else {  // if min_col_support > 0.731500029564
                  return false;
                }
              }
              else {  // if mean_col_support > 0.872823536396
                if ( min_col_coverage <= 0.458448439837 ) {
                  return 0.335021062517 < maxgini;
                }
                else {  // if min_col_coverage > 0.458448439837
                  return 0.455324657533 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.502958476543
            if ( min_col_support <= 0.637500047684 ) {
              if ( median_col_support <= 0.976500034332 ) {
                if ( max_col_coverage <= 0.758555650711 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.758555650711
                  return 0.457776316751 < maxgini;
                }
              }
              else {  // if median_col_support > 0.976500034332
                if ( mean_col_coverage <= 0.622398853302 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.622398853302
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.637500047684
              if ( max_col_coverage <= 0.756331145763 ) {
                if ( min_col_support <= 0.756500005722 ) {
                  return 0.48246036021 < maxgini;
                }
                else {  // if min_col_support > 0.756500005722
                  return 0.290657439446 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.756331145763
                if ( mean_col_coverage <= 0.647544980049 ) {
                  return 0.167396006237 < maxgini;
                }
                else {  // if mean_col_coverage > 0.647544980049
                  return 0.461419753086 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.649894118309
          if ( min_col_support <= 0.622500002384 ) {
            if ( median_col_coverage <= 0.655414700508 ) {
              if ( median_col_support <= 0.972000002861 ) {
                if ( mean_col_support <= 0.85591173172 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.85591173172
                  return 0.499603840878 < maxgini;
                }
              }
              else {  // if median_col_support > 0.972000002861
                if ( min_col_support <= 0.56149995327 ) {
                  return false;
                }
                else {  // if min_col_support > 0.56149995327
                  return false;
                }
              }
            }
            else {  // if median_col_coverage > 0.655414700508
              if ( mean_col_support <= 0.91167652607 ) {
                if ( max_col_coverage <= 0.996774196625 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.996774196625
                  return false;
                }
              }
              else {  // if mean_col_support > 0.91167652607
                if ( min_col_support <= 0.550500035286 ) {
                  return false;
                }
                else {  // if min_col_support > 0.550500035286
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.622500002384
            if ( min_col_coverage <= 0.809854507446 ) {
              if ( max_col_coverage <= 0.949820160866 ) {
                if ( min_col_support <= 0.692499995232 ) {
                  return 0.498738164473 < maxgini;
                }
                else {  // if min_col_support > 0.692499995232
                  return 0.439424925979 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.949820160866
                if ( median_col_support <= 0.71899998188 ) {
                  return false;
                }
                else {  // if median_col_support > 0.71899998188
                  return 0.418743601217 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.809854507446
              if ( median_col_coverage <= 0.943126678467 ) {
                if ( median_col_support <= 0.975000023842 ) {
                  return false;
                }
                else {  // if median_col_support > 0.975000023842
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.943126678467
                if ( mean_col_support <= 0.860235333443 ) {
                  return 0.152777777778 < maxgini;
                }
                else {  // if mean_col_support > 0.860235333443
                  return false;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.939088225365
        if ( median_col_support <= 0.985499978065 ) {
          if ( median_col_coverage <= 0.762000799179 ) {
            if ( min_col_support <= 0.65649998188 ) {
              if ( min_col_coverage <= 0.50287437439 ) {
                if ( mean_col_coverage <= 0.529539704323 ) {
                  return 0.272006012256 < maxgini;
                }
                else {  // if mean_col_coverage > 0.529539704323
                  return 0.476311317488 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.50287437439
                if ( median_col_coverage <= 0.560172438622 ) {
                  return 0.498966942149 < maxgini;
                }
                else {  // if median_col_coverage > 0.560172438622
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.65649998188
              if ( mean_col_support <= 0.964970588684 ) {
                if ( median_col_support <= 0.970499992371 ) {
                  return 0.299439046528 < maxgini;
                }
                else {  // if median_col_support > 0.970499992371
                  return 0.450307382936 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.964970588684
                if ( median_col_coverage <= 0.524662137032 ) {
                  return 0.242057596482 < maxgini;
                }
                else {  // if median_col_coverage > 0.524662137032
                  return 0.399898178597 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.762000799179
            if ( max_col_coverage <= 0.996996879578 ) {
              if ( mean_col_support <= 0.973441183567 ) {
                if ( min_col_support <= 0.681499958038 ) {
                  return false;
                }
                else {  // if min_col_support > 0.681499958038
                  return false;
                }
              }
              else {  // if mean_col_support > 0.973441183567
                if ( min_col_coverage <= 0.834313750267 ) {
                  return 0.42044464944 < maxgini;
                }
                else {  // if min_col_coverage > 0.834313750267
                  return false;
                }
              }
            }
            else {  // if max_col_coverage > 0.996996879578
              if ( mean_col_support <= 0.973911762238 ) {
                if ( mean_col_support <= 0.953205823898 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.953205823898
                  return false;
                }
              }
              else {  // if mean_col_support > 0.973911762238
                if ( mean_col_support <= 0.976794064045 ) {
                  return 0.494613042782 < maxgini;
                }
                else {  // if mean_col_support > 0.976794064045
                  return 0.431100780533 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.985499978065
          if ( mean_col_coverage <= 0.56177341938 ) {
            if ( median_col_support <= 0.999000012875 ) {
              if ( max_col_coverage <= 0.689788818359 ) {
                if ( max_col_coverage <= 0.671312451363 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.671312451363
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.689788818359
                if ( min_col_coverage <= 0.181292891502 ) {
                  return 0.274119860253 < maxgini;
                }
                else {  // if min_col_coverage > 0.181292891502
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.999000012875
              if ( min_col_support <= 0.659500002861 ) {
                if ( median_col_coverage <= 0.325401067734 ) {
                  return 0.337585697412 < maxgini;
                }
                else {  // if median_col_coverage > 0.325401067734
                  return false;
                }
              }
              else {  // if min_col_support > 0.659500002861
                if ( max_col_coverage <= 0.678268790245 ) {
                  return 0.411678218862 < maxgini;
                }
                else {  // if max_col_coverage > 0.678268790245
                  return 0.17322629543 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.56177341938
            if ( median_col_support <= 0.99950003624 ) {
              if ( min_col_support <= 0.646499991417 ) {
                if ( mean_col_support <= 0.966617643833 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.966617643833
                  return false;
                }
              }
              else {  // if min_col_support > 0.646499991417
                if ( median_col_support <= 0.993499994278 ) {
                  return false;
                }
                else {  // if median_col_support > 0.993499994278
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( median_col_coverage <= 0.667019784451 ) {
                if ( median_col_coverage <= 0.501322746277 ) {
                  return 0.484031102637 < maxgini;
                }
                else {  // if median_col_coverage > 0.501322746277
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.667019784451
                if ( min_col_support <= 0.680500030518 ) {
                  return false;
                }
                else {  // if min_col_support > 0.680500030518
                  return false;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if min_col_support > 0.770500004292
    if ( mean_col_support <= 0.988485336304 ) {
      if ( min_col_support <= 0.827499985695 ) {
        if ( median_col_coverage <= 0.667065382004 ) {
          if ( median_col_coverage <= 0.50069642067 ) {
            if ( mean_col_support <= 0.952970564365 ) {
              if ( max_col_coverage <= 0.51087474823 ) {
                if ( mean_col_coverage <= 0.377143651247 ) {
                  return 0.214671998959 < maxgini;
                }
                else {  // if mean_col_coverage > 0.377143651247
                  return 0.27507815319 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.51087474823
                if ( mean_col_support <= 0.938617646694 ) {
                  return 0.355843067012 < maxgini;
                }
                else {  // if mean_col_support > 0.938617646694
                  return 0.245141314258 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.952970564365
              if ( mean_col_coverage <= 0.431978285313 ) {
                if ( median_col_support <= 0.865499973297 ) {
                  return 0.195562285219 < maxgini;
                }
                else {  // if median_col_support > 0.865499973297
                  return 0.0655894419935 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.431978285313
                if ( median_col_support <= 0.854499995708 ) {
                  return 0.394143550755 < maxgini;
                }
                else {  // if median_col_support > 0.854499995708
                  return 0.129948088509 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.50069642067
            if ( median_col_support <= 0.99950003624 ) {
              if ( min_col_support <= 0.787500023842 ) {
                if ( mean_col_support <= 0.973147034645 ) {
                  return 0.304526605519 < maxgini;
                }
                else {  // if mean_col_support > 0.973147034645
                  return 0.496798600634 < maxgini;
                }
              }
              else {  // if min_col_support > 0.787500023842
                if ( median_col_support <= 0.989500045776 ) {
                  return 0.211772562376 < maxgini;
                }
                else {  // if median_col_support > 0.989500045776
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( mean_col_support <= 0.98602938652 ) {
                if ( max_col_coverage <= 0.817821025848 ) {
                  return 0.106553558705 < maxgini;
                }
                else {  // if max_col_coverage > 0.817821025848
                  return 0.0544127835078 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.98602938652
                if ( max_col_coverage <= 0.63172352314 ) {
                  return 0.117951978092 < maxgini;
                }
                else {  // if max_col_coverage > 0.63172352314
                  return 0.268244564765 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.667065382004
          if ( min_col_support <= 0.799499988556 ) {
            if ( median_col_support <= 0.99950003624 ) {
              if ( mean_col_support <= 0.973382353783 ) {
                if ( min_col_coverage <= 0.864517807961 ) {
                  return 0.393389590543 < maxgini;
                }
                else {  // if min_col_coverage > 0.864517807961
                  return false;
                }
              }
              else {  // if mean_col_support > 0.973382353783
                if ( median_col_support <= 0.992499947548 ) {
                  return 0.422821841266 < maxgini;
                }
                else {  // if median_col_support > 0.992499947548
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_support <= 0.78149998188 ) {
                if ( min_col_coverage <= 0.909319341183 ) {
                  return 0.405566467616 < maxgini;
                }
                else {  // if min_col_coverage > 0.909319341183
                  return 0.49951171875 < maxgini;
                }
              }
              else {  // if min_col_support > 0.78149998188
                if ( min_col_coverage <= 0.908627092838 ) {
                  return 0.310240980636 < maxgini;
                }
                else {  // if min_col_coverage > 0.908627092838
                  return 0.479506191426 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.799499988556
            if ( mean_col_coverage <= 0.931037902832 ) {
              if ( max_col_coverage <= 0.800311565399 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.268110879667 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0516598874065 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.800311565399
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.464080799538 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.184296855391 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.931037902832
              if ( max_col_coverage <= 0.99876844883 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.438836176563 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.99876844883
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.242037229738 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.827499985695
        if ( median_col_support <= 0.892500042915 ) {
          if ( max_col_coverage <= 0.604957163334 ) {
            if ( max_col_coverage <= 0.417288541794 ) {
              if ( median_col_support <= 0.865499973297 ) {
                if ( min_col_coverage <= 0.0317540317774 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.0317540317774
                  return 0.184089414859 < maxgini;
                }
              }
              else {  // if median_col_support > 0.865499973297
                if ( median_col_support <= 0.889500021935 ) {
                  return 0.124993066589 < maxgini;
                }
                else {  // if median_col_support > 0.889500021935
                  return 0.0812782753696 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.417288541794
              if ( median_col_coverage <= 0.479130446911 ) {
                if ( mean_col_support <= 0.938088178635 ) {
                  return 0.401727777778 < maxgini;
                }
                else {  // if mean_col_support > 0.938088178635
                  return 0.181351803173 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.479130446911
                if ( median_col_coverage <= 0.493243247271 ) {
                  return 0.377115617652 < maxgini;
                }
                else {  // if median_col_coverage > 0.493243247271
                  return 0.23720327435 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.604957163334
            if ( min_col_coverage <= 0.736068129539 ) {
              if ( min_col_coverage <= 0.467708349228 ) {
                if ( max_col_coverage <= 0.606601715088 ) {
                  return 0.428061831153 < maxgini;
                }
                else {  // if max_col_coverage > 0.606601715088
                  return 0.226376886961 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.467708349228
                if ( median_col_support <= 0.87549996376 ) {
                  return 0.338128292375 < maxgini;
                }
                else {  // if median_col_support > 0.87549996376
                  return 0.213556513247 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.736068129539
              if ( mean_col_coverage <= 0.817035913467 ) {
                if ( median_col_coverage <= 0.760952353477 ) {
                  return 0.0 < maxgini;
                }
                else {  // if median_col_coverage > 0.760952353477
                  return 0.152777777778 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.817035913467
                if ( median_col_support <= 0.837499976158 ) {
                  return false;
                }
                else {  // if median_col_support > 0.837499976158
                  return 0.147778242066 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.892500042915
          if ( min_col_coverage <= 0.913170576096 ) {
            if ( mean_col_support <= 0.981029391289 ) {
              if ( median_col_support <= 0.911499977112 ) {
                if ( min_col_support <= 0.894500017166 ) {
                  return 0.108538596809 < maxgini;
                }
                else {  // if min_col_support > 0.894500017166
                  return 0.158790170132 < maxgini;
                }
              }
              else {  // if median_col_support > 0.911499977112
                if ( mean_col_support <= 0.966355085373 ) {
                  return 0.169147861606 < maxgini;
                }
                else {  // if mean_col_support > 0.966355085373
                  return 0.0833250312441 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.981029391289
              if ( median_col_coverage <= 0.800273239613 ) {
                if ( max_col_coverage <= 0.774264931679 ) {
                  return 0.0420982254758 < maxgini;
                }
                else {  // if max_col_coverage > 0.774264931679
                  return 0.0618771589513 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.800273239613
                if ( mean_col_support <= 0.985794126987 ) {
                  return 0.153931431908 < maxgini;
                }
                else {  // if mean_col_support > 0.985794126987
                  return 0.0774614554722 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.913170576096
            if ( median_col_coverage <= 0.925985455513 ) {
              if ( mean_col_support <= 0.976911723614 ) {
                if ( median_col_support <= 0.96850001812 ) {
                  return 0.457465277778 < maxgini;
                }
                else {  // if median_col_support > 0.96850001812
                  return 0.0 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.976911723614
                if ( median_col_coverage <= 0.917237460613 ) {
                  return 0.0353867380894 < maxgini;
                }
                else {  // if median_col_coverage > 0.917237460613
                  return 0.0954108546527 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.925985455513
              if ( min_col_support <= 0.880499958992 ) {
                if ( median_col_coverage <= 0.998953938484 ) {
                  return 0.477208571115 < maxgini;
                }
                else {  // if median_col_coverage > 0.998953938484
                  return 0.367346382088 < maxgini;
                }
              }
              else {  // if min_col_support > 0.880499958992
                if ( median_col_coverage <= 0.964412331581 ) {
                  return 0.105327673081 < maxgini;
                }
                else {  // if median_col_coverage > 0.964412331581
                  return 0.281602782651 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if mean_col_support > 0.988485336304
      if ( mean_col_support <= 0.991719603539 ) {
        if ( min_col_support <= 0.870499968529 ) {
          if ( max_col_coverage <= 0.809620201588 ) {
            if ( max_col_coverage <= 0.669594466686 ) {
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_support <= 0.997500002384 ) {
                  return 0.381988343381 < maxgini;
                }
                else {  // if median_col_support > 0.997500002384
                  return false;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( median_col_coverage <= 0.455880820751 ) {
                  return 0.00905474596203 < maxgini;
                }
                else {  // if median_col_coverage > 0.455880820751
                  return 0.0307565780605 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.669594466686
              if ( median_col_coverage <= 0.571750342846 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.471099564156 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.015975000057 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.571750342846
                if ( median_col_coverage <= 0.576843261719 ) {
                  return 0.46106003245 < maxgini;
                }
                else {  // if median_col_coverage > 0.576843261719
                  return 0.171003120799 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.809620201588
            if ( mean_col_support <= 0.990382373333 ) {
              if ( max_col_coverage <= 0.96726667881 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.499754766099 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.102641240648 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.96726667881
                if ( mean_col_support <= 0.988794088364 ) {
                  return 0.453278779427 < maxgini;
                }
                else {  // if mean_col_support > 0.988794088364
                  return 0.405041911992 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.990382373333
              if ( median_col_coverage <= 0.913125514984 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0684141031109 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.913125514984
                if ( mean_col_coverage <= 0.996187806129 ) {
                  return 0.465410020616 < maxgini;
                }
                else {  // if mean_col_coverage > 0.996187806129
                  return 0.18836565097 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.870499968529
          if ( mean_col_coverage <= 0.974213242531 ) {
            if ( mean_col_support <= 0.990029394627 ) {
              if ( mean_col_support <= 0.988970577717 ) {
                if ( median_col_coverage <= 0.318665385246 ) {
                  return 0.0494535007126 < maxgini;
                }
                else {  // if median_col_coverage > 0.318665385246
                  return 0.0239861828157 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.988970577717
                if ( min_col_support <= 0.888499975204 ) {
                  return 0.0332545177424 < maxgini;
                }
                else {  // if min_col_support > 0.888499975204
                  return 0.0183547036011 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.990029394627
              if ( min_col_support <= 0.879500031471 ) {
                if ( max_col_coverage <= 0.808451414108 ) {
                  return 0.0316551331269 < maxgini;
                }
                else {  // if max_col_coverage > 0.808451414108
                  return 0.130123489775 < maxgini;
                }
              }
              else {  // if min_col_support > 0.879500031471
                if ( median_col_coverage <= 0.458114027977 ) {
                  return 0.0169275292837 < maxgini;
                }
                else {  // if median_col_coverage > 0.458114027977
                  return 0.0122615917554 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.974213242531
            if ( mean_col_coverage <= 0.97424274683 ) {
              return false;
            }
            else {  // if mean_col_coverage > 0.97424274683
              if ( max_col_coverage <= 0.99874997139 ) {
                if ( median_col_support <= 0.996500015259 ) {
                  return 0.168038408779 < maxgini;
                }
                else {  // if median_col_support > 0.996500015259
                  return 0.477040816327 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.99874997139
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.17918504403 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0219294186388 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.991719603539
        if ( min_col_coverage <= 0.980327665806 ) {
          if ( min_col_coverage <= 0.499490857124 ) {
            if ( min_col_support <= 0.972499966621 ) {
              if ( max_col_coverage <= 0.975154340267 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.0178836476802 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.00449916613522 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.975154340267
                if ( max_col_coverage <= 0.975606143475 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.975606143475
                  return 0.0336601613811 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.972499966621
              if ( median_col_support <= 0.975499987602 ) {
                if ( median_col_coverage <= 0.478813558817 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.478813558817
                  return 0.0 < maxgini;
                }
              }
              else {  // if median_col_support > 0.975499987602
                if ( mean_col_support <= 0.997205913067 ) {
                  return 0.00723901795571 < maxgini;
                }
                else {  // if mean_col_support > 0.997205913067
                  return 0.000603750746804 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.499490857124
            if ( min_col_support <= 0.896499991417 ) {
              if ( median_col_coverage <= 0.92709851265 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.486265174134 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.00834933887057 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.92709851265
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.499414477584 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0359680736058 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.896499991417
              if ( mean_col_support <= 0.99532353878 ) {
                if ( min_col_support <= 0.908499956131 ) {
                  return 0.0117404582816 < maxgini;
                }
                else {  // if min_col_support > 0.908499956131
                  return 0.00412112405219 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.99532353878
                if ( median_col_coverage <= 0.659772396088 ) {
                  return 0.00166432321174 < maxgini;
                }
                else {  // if median_col_coverage > 0.659772396088
                  return 0.00119051843072 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.980327665806
          if ( median_col_coverage <= 0.990361392498 ) {
            if ( mean_col_coverage <= 0.99260365963 ) {
              if ( mean_col_support <= 0.993147134781 ) {
                if ( min_col_coverage <= 0.988080024719 ) {
                  return 0.18 < maxgini;
                }
                else {  // if min_col_coverage > 0.988080024719
                  return false;
                }
              }
              else {  // if mean_col_support > 0.993147134781
                if ( max_col_coverage <= 0.988922894001 ) {
                  return 0.108727810651 < maxgini;
                }
                else {  // if max_col_coverage > 0.988922894001
                  return 0.0 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.99260365963
              if ( min_col_support <= 0.942000031471 ) {
                if ( mean_col_coverage <= 0.992943048477 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.992943048477
                  return 0.355029585799 < maxgini;
                }
              }
              else {  // if min_col_support > 0.942000031471
                if ( median_col_support <= 0.986500024796 ) {
                  return 0.375 < maxgini;
                }
                else {  // if median_col_support > 0.986500024796
                  return 0.0412188365651 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.990361392498
            if ( min_col_support <= 0.915500044823 ) {
              if ( median_col_support <= 0.99849998951 ) {
                if ( mean_col_support <= 0.991970658302 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_support > 0.991970658302
                  return false;
                }
              }
              else {  // if median_col_support > 0.99849998951
                if ( min_col_support <= 0.904500007629 ) {
                  return 0.0680471266733 < maxgini;
                }
                else {  // if min_col_support > 0.904500007629
                  return 0.0275808107976 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.915500044823
              if ( min_col_support <= 0.942499995232 ) {
                if ( median_col_support <= 0.999000012875 ) {
                  return 0.362402579952 < maxgini;
                }
                else {  // if median_col_support > 0.999000012875
                  return 0.0114102086073 < maxgini;
                }
              }
              else {  // if min_col_support > 0.942499995232
                if ( min_col_coverage <= 0.999115049839 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_coverage > 0.999115049839
                  return 0.00811894494329 < maxgini;
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
  if ( median_col_coverage <= 0.264539599419 ) {
    if ( mean_col_support <= 0.920193791389 ) {
      if ( mean_col_support <= 0.855710148811 ) {
        if ( mean_col_coverage <= 0.238272994757 ) {
          if ( max_col_coverage <= 0.21492728591 ) {
            if ( mean_col_support <= 0.792558789253 ) {
              if ( mean_col_support <= 0.79169857502 ) {
                if ( median_col_support <= 0.558500051498 ) {
                  return 0.34875 < maxgini;
                }
                else {  // if median_col_support > 0.558500051498
                  return 0.482770984976 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.79169857502
                if ( min_col_support <= 0.522500038147 ) {
                  return false;
                }
                else {  // if min_col_support > 0.522500038147
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.792558789253
              if ( min_col_coverage <= 0.124228395522 ) {
                if ( mean_col_support <= 0.855464100838 ) {
                  return 0.3619649526 < maxgini;
                }
                else {  // if mean_col_support > 0.855464100838
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.124228395522
                if ( min_col_support <= 0.534500002861 ) {
                  return 0.118570741868 < maxgini;
                }
                else {  // if min_col_support > 0.534500002861
                  return 0.255 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.21492728591
            if ( median_col_coverage <= 0.0504237264395 ) {
              if ( max_col_coverage <= 0.403703689575 ) {
                if ( mean_col_support <= 0.821537852287 ) {
                  return 0.4712 < maxgini;
                }
                else {  // if mean_col_support > 0.821537852287
                  return 0.497862024933 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.403703689575
                if ( median_col_support <= 0.534000039101 ) {
                  return 0.0 < maxgini;
                }
                else {  // if median_col_support > 0.534000039101
                  return 0.447862413194 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.0504237264395
              if ( median_col_coverage <= 0.0754985809326 ) {
                if ( min_col_support <= 0.533499956131 ) {
                  return 0.186441244278 < maxgini;
                }
                else {  // if min_col_support > 0.533499956131
                  return 0.422091412742 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.0754985809326
                if ( min_col_support <= 0.50049996376 ) {
                  return 0.2956364853 < maxgini;
                }
                else {  // if min_col_support > 0.50049996376
                  return 0.44355363318 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.238272994757
          if ( median_col_support <= 0.619500041008 ) {
            if ( min_col_support <= 0.452499985695 ) {
              if ( min_col_coverage <= 0.0976190492511 ) {
                if ( median_col_coverage <= 0.113171353936 ) {
                  return 0.132653061224 < maxgini;
                }
                else {  // if median_col_coverage > 0.113171353936
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.0976190492511
                if ( median_col_support <= 0.46899998188 ) {
                  return false;
                }
                else {  // if median_col_support > 0.46899998188
                  return 0.292679145407 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.452499985695
              if ( max_col_coverage <= 0.508064508438 ) {
                if ( median_col_support <= 0.557500004768 ) {
                  return false;
                }
                else {  // if median_col_support > 0.557500004768
                  return 0.487056007561 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.508064508438
                if ( median_col_support <= 0.607499957085 ) {
                  return false;
                }
                else {  // if median_col_support > 0.607499957085
                  return false;
                }
              }
            }
          }
          else {  // if median_col_support > 0.619500041008
            if ( max_col_coverage <= 0.522774338722 ) {
              if ( median_col_support <= 0.667500019073 ) {
                if ( median_col_coverage <= 0.261168718338 ) {
                  return 0.454088232768 < maxgini;
                }
                else {  // if median_col_coverage > 0.261168718338
                  return false;
                }
              }
              else {  // if median_col_support > 0.667500019073
                if ( min_col_support <= 0.517500042915 ) {
                  return 0.276742984196 < maxgini;
                }
                else {  // if min_col_support > 0.517500042915
                  return 0.445325311374 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.522774338722
              if ( mean_col_support <= 0.844970583916 ) {
                if ( mean_col_support <= 0.769147157669 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.769147157669
                  return 0.492097130595 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.844970583916
                if ( max_col_coverage <= 0.697826087475 ) {
                  return 0.422025234289 < maxgini;
                }
                else {  // if max_col_coverage > 0.697826087475
                  return 0.172335600907 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.855710148811
        if ( median_col_support <= 0.726500034332 ) {
          if ( mean_col_coverage <= 0.269974827766 ) {
            if ( min_col_coverage <= 0.0523821413517 ) {
              if ( max_col_coverage <= 0.238883405924 ) {
                if ( median_col_support <= 0.516499996185 ) {
                  return 0.4902 < maxgini;
                }
                else {  // if median_col_support > 0.516499996185
                  return 0.370323510711 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.238883405924
                if ( mean_col_coverage <= 0.21655087173 ) {
                  return 0.41977420976 < maxgini;
                }
                else {  // if mean_col_coverage > 0.21655087173
                  return 0.471422146928 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.0523821413517
              if ( mean_col_coverage <= 0.187335163355 ) {
                if ( mean_col_coverage <= 0.1644359231 ) {
                  return 0.211974003002 < maxgini;
                }
                else {  // if mean_col_coverage > 0.1644359231
                  return 0.272303180708 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.187335163355
                if ( median_col_coverage <= 0.0881176441908 ) {
                  return 0.406747691027 < maxgini;
                }
                else {  // if median_col_coverage > 0.0881176441908
                  return 0.334235522515 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.269974827766
            if ( mean_col_coverage <= 0.342885136604 ) {
              if ( min_col_support <= 0.604499995708 ) {
                if ( mean_col_coverage <= 0.342324376106 ) {
                  return 0.452843363889 < maxgini;
                }
                else {  // if mean_col_coverage > 0.342324376106
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_support > 0.604499995708
                if ( median_col_support <= 0.685500025749 ) {
                  return 0.430773391022 < maxgini;
                }
                else {  // if median_col_support > 0.685500025749
                  return 0.363124577895 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.342885136604
              if ( min_col_support <= 0.435499995947 ) {
                return 0.0 < maxgini;
              }
              else {  // if min_col_support > 0.435499995947
                if ( median_col_coverage <= 0.261387169361 ) {
                  return 0.475213531591 < maxgini;
                }
                else {  // if median_col_coverage > 0.261387169361
                  return false;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.726500034332
          if ( max_col_coverage <= 0.358732461929 ) {
            if ( min_col_support <= 0.789000034332 ) {
              if ( mean_col_support <= 0.875899553299 ) {
                if ( min_col_coverage <= 0.0510647520423 ) {
                  return 0.412228674943 < maxgini;
                }
                else {  // if min_col_coverage > 0.0510647520423
                  return 0.245890090634 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.875899553299
                if ( median_col_support <= 0.908499956131 ) {
                  return 0.234361395012 < maxgini;
                }
                else {  // if median_col_support > 0.908499956131
                  return 0.321408405935 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.789000034332
              if ( mean_col_support <= 0.916970551014 ) {
                if ( min_col_support <= 0.793500006199 ) {
                  return 0.475308641975 < maxgini;
                }
                else {  // if min_col_support > 0.793500006199
                  return false;
                }
              }
              else {  // if mean_col_support > 0.916970551014
                if ( min_col_support <= 0.79699999094 ) {
                  return 0.456747404844 < maxgini;
                }
                else {  // if min_col_support > 0.79699999094
                  return 0.0 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.358732461929
            if ( median_col_support <= 0.765499949455 ) {
              if ( mean_col_support <= 0.86414706707 ) {
                if ( median_col_support <= 0.757500052452 ) {
                  return 0.444444444444 < maxgini;
                }
                else {  // if median_col_support > 0.757500052452
                  return false;
                }
              }
              else {  // if mean_col_support > 0.86414706707
                if ( min_col_coverage <= 0.0845238119364 ) {
                  return 0.429110202553 < maxgini;
                }
                else {  // if min_col_coverage > 0.0845238119364
                  return 0.30832197634 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.765499949455
              if ( min_col_coverage <= 0.0504237264395 ) {
                if ( max_col_coverage <= 0.429802954197 ) {
                  return 0.308876157143 < maxgini;
                }
                else {  // if max_col_coverage > 0.429802954197
                  return 0.402957924446 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0504237264395
                if ( median_col_support <= 0.909500002861 ) {
                  return 0.237785671519 < maxgini;
                }
                else {  // if median_col_support > 0.909500002861
                  return 0.481859410431 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if mean_col_support > 0.920193791389
      if ( min_col_coverage <= 0.0509879142046 ) {
        if ( max_col_coverage <= 0.243168592453 ) {
          if ( median_col_support <= 0.759500026703 ) {
            if ( min_col_coverage <= 0.0421099290252 ) {
              if ( max_col_coverage <= 0.165300548077 ) {
                if ( median_col_support <= 0.745499968529 ) {
                  return 0.113064616428 < maxgini;
                }
                else {  // if median_col_support > 0.745499968529
                  return 0.424382716049 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.165300548077
                if ( min_col_support <= 0.608000040054 ) {
                  return 0.279739995301 < maxgini;
                }
                else {  // if min_col_support > 0.608000040054
                  return 0.483404087921 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.0421099290252
              if ( min_col_coverage <= 0.0504237264395 ) {
                if ( mean_col_coverage <= 0.135421991348 ) {
                  return 0.294617768595 < maxgini;
                }
                else {  // if mean_col_coverage > 0.135421991348
                  return 0.107300369321 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0504237264395
                return false;
              }
            }
          }
          else {  // if median_col_support > 0.759500026703
            if ( mean_col_support <= 0.937466025352 ) {
              if ( min_col_coverage <= 0.00754727749154 ) {
                if ( min_col_coverage <= 0.00749074202031 ) {
                  return 0.325570260935 < maxgini;
                }
                else {  // if min_col_coverage > 0.00749074202031
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.00754727749154
                if ( min_col_coverage <= 0.0320020467043 ) {
                  return 0.163592373017 < maxgini;
                }
                else {  // if min_col_coverage > 0.0320020467043
                  return 0.201217522985 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.937466025352
              if ( mean_col_support <= 0.961727976799 ) {
                if ( min_col_support <= 0.868499994278 ) {
                  return 0.114391205848 < maxgini;
                }
                else {  // if min_col_support > 0.868499994278
                  return false;
                }
              }
              else {  // if mean_col_support > 0.961727976799
                if ( median_col_coverage <= 0.0646321624517 ) {
                  return 0.0649195726662 < maxgini;
                }
                else {  // if median_col_coverage > 0.0646321624517
                  return 0.0301852612986 < maxgini;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.243168592453
          if ( max_col_coverage <= 0.381101191044 ) {
            if ( mean_col_support <= 0.947766959667 ) {
              if ( median_col_coverage <= 0.0515030957758 ) {
                if ( mean_col_coverage <= 0.127840429544 ) {
                  return 0.222435537491 < maxgini;
                }
                else {  // if mean_col_coverage > 0.127840429544
                  return 0.314870003549 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.0515030957758
                if ( min_col_coverage <= 0.0347852408886 ) {
                  return 0.279516307626 < maxgini;
                }
                else {  // if min_col_coverage > 0.0347852408886
                  return 0.167261397639 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.947766959667
              if ( mean_col_support <= 0.965828478336 ) {
                if ( median_col_coverage <= 0.0478095263243 ) {
                  return 0.184352345006 < maxgini;
                }
                else {  // if median_col_coverage > 0.0478095263243
                  return 0.0865806778321 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.965828478336
                if ( mean_col_coverage <= 0.0732105523348 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.0732105523348
                  return 0.0509722714598 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.381101191044
            if ( max_col_coverage <= 0.984090924263 ) {
              if ( median_col_support <= 0.81149995327 ) {
                if ( median_col_support <= 0.742499947548 ) {
                  return false;
                }
                else {  // if median_col_support > 0.742499947548
                  return 0.415817226696 < maxgini;
                }
              }
              else {  // if median_col_support > 0.81149995327
                if ( median_col_support <= 0.866500020027 ) {
                  return 0.242825981486 < maxgini;
                }
                else {  // if median_col_support > 0.866500020027
                  return 0.139815848008 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.984090924263
              if ( mean_col_support <= 0.979088187218 ) {
                if ( min_col_coverage <= 0.00790526159108 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.00790526159108
                  return 0.346849769523 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.979088187218
                if ( median_col_support <= 0.974500000477 ) {
                  return 0.387811634349 < maxgini;
                }
                else {  // if median_col_support > 0.974500000477
                  return 0.0327777777778 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_coverage > 0.0509879142046
        if ( min_col_support <= 0.783499956131 ) {
          if ( min_col_coverage <= 0.150117367506 ) {
            if ( min_col_support <= 0.550500035286 ) {
              if ( median_col_support <= 0.914499998093 ) {
                if ( mean_col_coverage <= 0.0951720327139 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.0951720327139
                  return 0.144277225095 < maxgini;
                }
              }
              else {  // if median_col_support > 0.914499998093
                if ( median_col_coverage <= 0.150342464447 ) {
                  return 0.18961585793 < maxgini;
                }
                else {  // if median_col_coverage > 0.150342464447
                  return 0.333965756934 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.550500035286
              if ( max_col_coverage <= 0.849242448807 ) {
                if ( mean_col_coverage <= 0.194523960352 ) {
                  return 0.0705173372443 < maxgini;
                }
                else {  // if mean_col_coverage > 0.194523960352
                  return 0.118765326943 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.849242448807
                if ( max_col_coverage <= 0.850531935692 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.850531935692
                  return 0.298465353518 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.150117367506
            if ( median_col_coverage <= 0.263240396976 ) {
              if ( median_col_support <= 0.807500004768 ) {
                if ( min_col_coverage <= 0.161726236343 ) {
                  return 0.38246097337 < maxgini;
                }
                else {  // if min_col_coverage > 0.161726236343
                  return 0.270577167587 < maxgini;
                }
              }
              else {  // if median_col_support > 0.807500004768
                if ( median_col_coverage <= 0.250621497631 ) {
                  return 0.154211647296 < maxgini;
                }
                else {  // if median_col_coverage > 0.250621497631
                  return 0.242113460975 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.263240396976
              if ( median_col_support <= 0.993499994278 ) {
                if ( median_col_coverage <= 0.263812601566 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.263812601566
                  return 0.340264650284 < maxgini;
                }
              }
              else {  // if median_col_support > 0.993499994278
                if ( min_col_support <= 0.694000005722 ) {
                  return false;
                }
                else {  // if min_col_support > 0.694000005722
                  return 0.0 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.783499956131
          if ( min_col_support <= 0.888499975204 ) {
            if ( median_col_support <= 0.90649998188 ) {
              if ( min_col_support <= 0.886500000954 ) {
                if ( min_col_coverage <= 0.118929751217 ) {
                  return 0.135116879995 < maxgini;
                }
                else {  // if min_col_coverage > 0.118929751217
                  return 0.168765710649 < maxgini;
                }
              }
              else {  // if min_col_support > 0.886500000954
                if ( median_col_coverage <= 0.092857144773 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.092857144773
                  return 0.300826390755 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.90649998188
              if ( median_col_coverage <= 0.261452913284 ) {
                if ( median_col_support <= 0.940500020981 ) {
                  return 0.0815595276029 < maxgini;
                }
                else {  // if median_col_support > 0.940500020981
                  return 0.028946003504 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.261452913284
                if ( median_col_support <= 0.995499968529 ) {
                  return 0.249012440787 < maxgini;
                }
                else {  // if median_col_support > 0.995499968529
                  return 0.0323145099064 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.888499975204
            if ( min_col_support <= 0.932500004768 ) {
              if ( median_col_support <= 0.952499985695 ) {
                if ( mean_col_support <= 0.964764714241 ) {
                  return 0.499540863177 < maxgini;
                }
                else {  // if mean_col_support > 0.964764714241
                  return 0.119801042521 < maxgini;
                }
              }
              else {  // if median_col_support > 0.952499985695
                if ( mean_col_support <= 0.988970577717 ) {
                  return 0.046556975745 < maxgini;
                }
                else {  // if mean_col_support > 0.988970577717
                  return 0.00679059139041 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.932500004768
              if ( median_col_support <= 0.956499993801 ) {
                if ( mean_col_coverage <= 0.309033602476 ) {
                  return 0.160558420948 < maxgini;
                }
                else {  // if mean_col_coverage > 0.309033602476
                  return 0.316044074174 < maxgini;
                }
              }
              else {  // if median_col_support > 0.956499993801
                if ( min_col_coverage <= 0.0528309419751 ) {
                  return 0.165289256198 < maxgini;
                }
                else {  // if min_col_coverage > 0.0528309419751
                  return 0.00729844569945 < maxgini;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if median_col_coverage > 0.264539599419
    if ( mean_col_coverage <= 0.960810780525 ) {
      if ( median_col_support <= 0.99950003624 ) {
        if ( min_col_support <= 0.762500047684 ) {
          if ( mean_col_coverage <= 0.615216553211 ) {
            if ( median_col_coverage <= 0.409152507782 ) {
              if ( min_col_support <= 0.62650001049 ) {
                if ( min_col_support <= 0.550500035286 ) {
                  return 0.490830240335 < maxgini;
                }
                else {  // if min_col_support > 0.550500035286
                  return 0.443335513992 < maxgini;
                }
              }
              else {  // if min_col_support > 0.62650001049
                if ( median_col_support <= 0.986500024796 ) {
                  return 0.304389007531 < maxgini;
                }
                else {  // if median_col_support > 0.986500024796
                  return false;
                }
              }
            }
            else {  // if median_col_coverage > 0.409152507782
              if ( median_col_support <= 0.984500050545 ) {
                if ( min_col_support <= 0.630499958992 ) {
                  return 0.498493838461 < maxgini;
                }
                else {  // if min_col_support > 0.630499958992
                  return 0.358982397857 < maxgini;
                }
              }
              else {  // if median_col_support > 0.984500050545
                if ( min_col_support <= 0.715499997139 ) {
                  return false;
                }
                else {  // if min_col_support > 0.715499997139
                  return false;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.615216553211
            if ( mean_col_coverage <= 0.715727984905 ) {
              if ( min_col_support <= 0.68850004673 ) {
                if ( max_col_coverage <= 0.837786793709 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.837786793709
                  return 0.495616177973 < maxgini;
                }
              }
              else {  // if min_col_support > 0.68850004673
                if ( median_col_support <= 0.988499999046 ) {
                  return 0.357018270688 < maxgini;
                }
                else {  // if median_col_support > 0.988499999046
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.715727984905
              if ( median_col_coverage <= 0.653950095177 ) {
                if ( min_col_support <= 0.684499979019 ) {
                  return false;
                }
                else {  // if min_col_support > 0.684499979019
                  return 0.429838142699 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.653950095177
                if ( median_col_support <= 0.988499999046 ) {
                  return false;
                }
                else {  // if median_col_support > 0.988499999046
                  return false;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.762500047684
          if ( median_col_support <= 0.994500041008 ) {
            if ( median_col_coverage <= 0.498688459396 ) {
              if ( min_col_support <= 0.847499966621 ) {
                if ( mean_col_coverage <= 0.412189543247 ) {
                  return 0.143922349501 < maxgini;
                }
                else {  // if mean_col_coverage > 0.412189543247
                  return 0.202654525069 < maxgini;
                }
              }
              else {  // if min_col_support > 0.847499966621
                if ( min_col_support <= 0.904500007629 ) {
                  return 0.0915183427953 < maxgini;
                }
                else {  // if min_col_support > 0.904500007629
                  return 0.0420632294862 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.498688459396
              if ( mean_col_support <= 0.981264710426 ) {
                if ( mean_col_support <= 0.950500011444 ) {
                  return 0.344309667873 < maxgini;
                }
                else {  // if mean_col_support > 0.950500011444
                  return 0.163125575508 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.981264710426
                if ( mean_col_support <= 0.986558794975 ) {
                  return 0.0832349989505 < maxgini;
                }
                else {  // if mean_col_support > 0.986558794975
                  return 0.0100630383832 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.994500041008
            if ( median_col_coverage <= 0.621126055717 ) {
              if ( median_col_coverage <= 0.463206797838 ) {
                if ( min_col_support <= 0.850499987602 ) {
                  return false;
                }
                else {  // if min_col_support > 0.850499987602
                  return 0.0713588261505 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.463206797838
                if ( min_col_support <= 0.896499991417 ) {
                  return false;
                }
                else {  // if min_col_support > 0.896499991417
                  return 0.0328310841899 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.621126055717
              if ( max_col_coverage <= 0.983812212944 ) {
                if ( max_col_coverage <= 0.842166304588 ) {
                  return 0.406546045704 < maxgini;
                }
                else {  // if max_col_coverage > 0.842166304588
                  return 0.43919813059 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.983812212944
                if ( mean_col_support <= 0.992558836937 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.992558836937
                  return 0.0144451581612 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.99950003624
        if ( min_col_support <= 0.706499993801 ) {
          if ( min_col_support <= 0.619500041008 ) {
            if ( mean_col_support <= 0.970147073269 ) {
              if ( min_col_coverage <= 0.458681106567 ) {
                if ( mean_col_coverage <= 0.389071822166 ) {
                  return 0.482231515201 < maxgini;
                }
                else {  // if mean_col_coverage > 0.389071822166
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.458681106567
                if ( median_col_coverage <= 0.636617541313 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.636617541313
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.970147073269
              if ( median_col_coverage <= 0.580741763115 ) {
                if ( min_col_support <= 0.574499964714 ) {
                  return false;
                }
                else {  // if min_col_support > 0.574499964714
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.580741763115
                if ( min_col_coverage <= 0.725095391273 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.725095391273
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.619500041008
            if ( min_col_coverage <= 0.500929355621 ) {
              if ( min_col_coverage <= 0.334067553282 ) {
                if ( median_col_coverage <= 0.320087730885 ) {
                  return 0.266370301155 < maxgini;
                }
                else {  // if median_col_coverage > 0.320087730885
                  return 0.385005062654 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.334067553282
                if ( max_col_coverage <= 0.459614783525 ) {
                  return 0.336734693878 < maxgini;
                }
                else {  // if max_col_coverage > 0.459614783525
                  return 0.490585949331 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.500929355621
              if ( max_col_coverage <= 0.778093457222 ) {
                if ( min_col_support <= 0.676499962807 ) {
                  return false;
                }
                else {  // if min_col_support > 0.676499962807
                  return 0.495106828519 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.778093457222
                if ( mean_col_support <= 0.977147042751 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.977147042751
                  return false;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.706499993801
          if ( min_col_support <= 0.779500007629 ) {
            if ( min_col_support <= 0.756500005722 ) {
              if ( median_col_coverage <= 0.636634230614 ) {
                if ( median_col_coverage <= 0.461803734303 ) {
                  return 0.18319860121 < maxgini;
                }
                else {  // if median_col_coverage > 0.461803734303
                  return 0.373707276623 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.636634230614
                if ( mean_col_support <= 0.982558846474 ) {
                  return 0.446412195311 < maxgini;
                }
                else {  // if mean_col_support > 0.982558846474
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.756500005722
              if ( min_col_coverage <= 0.501552820206 ) {
                if ( mean_col_coverage <= 0.494875192642 ) {
                  return 0.0974423498771 < maxgini;
                }
                else {  // if mean_col_coverage > 0.494875192642
                  return 0.189917032999 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.501552820206
                if ( min_col_coverage <= 0.660232663155 ) {
                  return 0.30542827599 < maxgini;
                }
                else {  // if min_col_coverage > 0.660232663155
                  return 0.440357274747 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.779500007629
            if ( max_col_coverage <= 0.638805747032 ) {
              if ( min_col_coverage <= 0.35753184557 ) {
                if ( mean_col_coverage <= 0.257762521505 ) {
                  return 0.21875 < maxgini;
                }
                else {  // if mean_col_coverage > 0.257762521505
                  return 0.016433764285 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.35753184557
                if ( max_col_coverage <= 0.638716101646 ) {
                  return 0.00917622773536 < maxgini;
                }
                else {  // if max_col_coverage > 0.638716101646
                  return false;
                }
              }
            }
            else {  // if max_col_coverage > 0.638805747032
              if ( mean_col_coverage <= 0.615268707275 ) {
                if ( min_col_coverage <= 0.0248456783593 ) {
                  return 0.48 < maxgini;
                }
                else {  // if min_col_coverage > 0.0248456783593
                  return 0.00730294823112 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.615268707275
                if ( mean_col_support <= 0.988911747932 ) {
                  return 0.0584407838138 < maxgini;
                }
                else {  // if mean_col_support > 0.988911747932
                  return 0.0023229268481 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if mean_col_coverage > 0.960810780525
      if ( min_col_coverage <= 0.937574982643 ) {
        if ( median_col_coverage <= 0.933568000793 ) {
          if ( mean_col_support <= 0.982088267803 ) {
            if ( median_col_support <= 0.986500024796 ) {
              if ( min_col_support <= 0.743499994278 ) {
                if ( mean_col_coverage <= 0.966047406197 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.966047406197
                  return 0.477737564178 < maxgini;
                }
              }
              else {  // if min_col_support > 0.743499994278
                if ( min_col_support <= 0.805000007153 ) {
                  return 0.277777777778 < maxgini;
                }
                else {  // if min_col_support > 0.805000007153
                  return 0.111284541565 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.986500024796
              if ( mean_col_support <= 0.977500021458 ) {
                if ( min_col_coverage <= 0.841686725616 ) {
                  return 0.367309458219 < maxgini;
                }
                else {  // if min_col_coverage > 0.841686725616
                  return false;
                }
              }
              else {  // if mean_col_support > 0.977500021458
                if ( max_col_coverage <= 0.997191011906 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.997191011906
                  return false;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.982088267803
            if ( min_col_support <= 0.847499966621 ) {
              if ( max_col_coverage <= 0.998533725739 ) {
                if ( min_col_coverage <= 0.925000011921 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.925000011921
                  return 0.0 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.998533725739
                if ( mean_col_coverage <= 0.967628479004 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.967628479004
                  return 0.350056689342 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.847499966621
              if ( mean_col_support <= 0.992205917835 ) {
                if ( min_col_coverage <= 0.918761789799 ) {
                  return 0.0586682032902 < maxgini;
                }
                else {  // if min_col_coverage > 0.918761789799
                  return 0.00618232077399 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.992205917835
                if ( median_col_coverage <= 0.9311825037 ) {
                  return 0.00167843786759 < maxgini;
                }
                else {  // if median_col_coverage > 0.9311825037
                  return 0.00335194586031 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.933568000793
          if ( min_col_coverage <= 0.923194885254 ) {
            if ( mean_col_support <= 0.985088229179 ) {
              if ( median_col_coverage <= 0.949921607971 ) {
                if ( min_col_coverage <= 0.889125287533 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.889125287533
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.949921607971
                if ( min_col_coverage <= 0.905086398125 ) {
                  return 0.496640687987 < maxgini;
                }
                else {  // if min_col_coverage > 0.905086398125
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.985088229179
              if ( mean_col_support <= 0.987911701202 ) {
                if ( min_col_coverage <= 0.913118958473 ) {
                  return 0.225424796424 < maxgini;
                }
                else {  // if min_col_coverage > 0.913118958473
                  return 0.478057773126 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.987911701202
                if ( mean_col_support <= 0.990088284016 ) {
                  return 0.130790970741 < maxgini;
                }
                else {  // if mean_col_support > 0.990088284016
                  return 0.00347251826129 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.923194885254
            if ( mean_col_coverage <= 0.972402095795 ) {
              if ( mean_col_coverage <= 0.972198367119 ) {
                if ( min_col_coverage <= 0.92492890358 ) {
                  return 0.490335541471 < maxgini;
                }
                else {  // if min_col_coverage > 0.92492890358
                  return 0.218568869752 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.972198367119
                if ( min_col_coverage <= 0.924849152565 ) {
                  return 0.345679012346 < maxgini;
                }
                else {  // if min_col_coverage > 0.924849152565
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.972402095795
              if ( median_col_support <= 0.99950003624 ) {
                if ( min_col_coverage <= 0.937427520752 ) {
                  return 0.450799630093 < maxgini;
                }
                else {  // if min_col_coverage > 0.937427520752
                  return 0.250045625868 < maxgini;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( max_col_coverage <= 0.99364554882 ) {
                  return 0.4608 < maxgini;
                }
                else {  // if max_col_coverage > 0.99364554882
                  return 0.0570552599316 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_coverage > 0.937574982643
        if ( max_col_coverage <= 0.997965335846 ) {
          if ( mean_col_support <= 0.989382386208 ) {
            if ( max_col_coverage <= 0.982583403587 ) {
              if ( mean_col_support <= 0.9860881567 ) {
                if ( min_col_support <= 0.882500052452 ) {
                  return false;
                }
                else {  // if min_col_support > 0.882500052452
                  return 0.418801514814 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.9860881567
                if ( min_col_coverage <= 0.954638957977 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.954638957977
                  return 0.341796875 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.982583403587
              if ( median_col_coverage <= 0.978201508522 ) {
                if ( min_col_support <= 0.889500021935 ) {
                  return false;
                }
                else {  // if min_col_support > 0.889500021935
                  return 0.413194444444 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.978201508522
                if ( median_col_support <= 0.949499964714 ) {
                  return 0.499334008205 < maxgini;
                }
                else {  // if median_col_support > 0.949499964714
                  return false;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.989382386208
            if ( min_col_support <= 0.894500017166 ) {
              if ( min_col_coverage <= 0.959910690784 ) {
                if ( max_col_coverage <= 0.981529891491 ) {
                  return 0.287334593573 < maxgini;
                }
                else {  // if max_col_coverage > 0.981529891491
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.959910690784
                if ( mean_col_support <= 0.992735266685 ) {
                  return 0.394448502557 < maxgini;
                }
                else {  // if mean_col_support > 0.992735266685
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.894500017166
              if ( mean_col_coverage <= 0.987626850605 ) {
                if ( mean_col_coverage <= 0.980467140675 ) {
                  return 0.00423726903503 < maxgini;
                }
                else {  // if mean_col_coverage > 0.980467140675
                  return 0.060546875 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.987626850605
                if ( max_col_coverage <= 0.997888445854 ) {
                  return 0.152777777778 < maxgini;
                }
                else {  // if max_col_coverage > 0.997888445854
                  return false;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.997965335846
          if ( min_col_support <= 0.845499992371 ) {
            if ( mean_col_coverage <= 0.99113291502 ) {
              if ( mean_col_support <= 0.979970574379 ) {
                if ( min_col_coverage <= 0.949895381927 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.949895381927
                  return false;
                }
              }
              else {  // if mean_col_support > 0.979970574379
                if ( min_col_coverage <= 0.965576469898 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.965576469898
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.99113291502
              if ( min_col_support <= 0.772500038147 ) {
                if ( median_col_support <= 0.980499982834 ) {
                  return false;
                }
                else {  // if median_col_support > 0.980499982834
                  return false;
                }
              }
              else {  // if min_col_support > 0.772500038147
                if ( min_col_coverage <= 0.9603921175 ) {
                  return 0.190131281123 < maxgini;
                }
                else {  // if min_col_coverage > 0.9603921175
                  return 0.473063323589 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.845499992371
            if ( mean_col_coverage <= 0.998230814934 ) {
              if ( median_col_support <= 0.99950003624 ) {
                if ( min_col_coverage <= 0.983239710331 ) {
                  return 0.0944519652129 < maxgini;
                }
                else {  // if min_col_coverage > 0.983239710331
                  return 0.387989007231 < maxgini;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( mean_col_coverage <= 0.989868879318 ) {
                  return 0.00309861591108 < maxgini;
                }
                else {  // if mean_col_coverage > 0.989868879318
                  return 0.0086716345028 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.998230814934
              if ( median_col_coverage <= 0.998069047928 ) {
                if ( min_col_support <= 0.962999999523 ) {
                  return false;
                }
                else {  // if min_col_support > 0.962999999523
                  return 0.0 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.998069047928
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.348514622004 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0144604056385 < maxgini;
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
  if ( min_col_coverage <= 0.1599458009 ) {
    if ( mean_col_support <= 0.918062806129 ) {
      if ( mean_col_support <= 0.855355143547 ) {
        if ( mean_col_coverage <= 0.260385841131 ) {
          if ( min_col_support <= 0.503499984741 ) {
            if ( median_col_support <= 0.495500028133 ) {
              if ( mean_col_support <= 0.662264764309 ) {
                return 0.0 < maxgini;
              }
              else {  // if mean_col_support > 0.662264764309
                if ( mean_col_coverage <= 0.140607923269 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_coverage > 0.140607923269
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.495500028133
              if ( max_col_support <= 0.988499999046 ) {
                return false;
              }
              else {  // if max_col_support > 0.988499999046
                if ( max_col_coverage <= 0.315037608147 ) {
                  return 0.290023342915 < maxgini;
                }
                else {  // if max_col_coverage > 0.315037608147
                  return 0.388964997367 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.503499984741
            if ( max_col_coverage <= 0.27704679966 ) {
              if ( mean_col_support <= 0.834676504135 ) {
                if ( median_col_support <= 0.575500011444 ) {
                  return 0.476478284439 < maxgini;
                }
                else {  // if median_col_support > 0.575500011444
                  return 0.393511680859 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.834676504135
                if ( mean_col_support <= 0.855188310146 ) {
                  return 0.362208634455 < maxgini;
                }
                else {  // if mean_col_support > 0.855188310146
                  return false;
                }
              }
            }
            else {  // if max_col_coverage > 0.27704679966
              if ( min_col_coverage <= 0.0606617629528 ) {
                if ( min_col_support <= 0.639500021935 ) {
                  return 0.476090811052 < maxgini;
                }
                else {  // if min_col_support > 0.639500021935
                  return 0.28624260355 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0606617629528
                if ( max_col_coverage <= 0.430952370167 ) {
                  return 0.448375635649 < maxgini;
                }
                else {  // if max_col_coverage > 0.430952370167
                  return 0.36294994667 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.260385841131
          if ( mean_col_coverage <= 0.328089654446 ) {
            if ( median_col_support <= 0.619500041008 ) {
              if ( median_col_support <= 0.557500004768 ) {
                if ( min_col_support <= 0.452000021935 ) {
                  return 0.399524375743 < maxgini;
                }
                else {  // if min_col_support > 0.452000021935
                  return false;
                }
              }
              else {  // if median_col_support > 0.557500004768
                if ( min_col_support <= 0.441500008106 ) {
                  return 0.161512578112 < maxgini;
                }
                else {  // if min_col_support > 0.441500008106
                  return 0.494433770823 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.619500041008
              if ( median_col_coverage <= 0.139610394835 ) {
                if ( min_col_support <= 0.515499949455 ) {
                  return 0.38677764566 < maxgini;
                }
                else {  // if min_col_support > 0.515499949455
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.139610394835
                if ( min_col_support <= 0.513499975204 ) {
                  return 0.324083042982 < maxgini;
                }
                else {  // if min_col_support > 0.513499975204
                  return 0.459183673469 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.328089654446
            if ( min_col_support <= 0.474500000477 ) {
              if ( median_col_support <= 0.509999990463 ) {
                if ( median_col_support <= 0.5 ) {
                  return 0.444444444444 < maxgini;
                }
                else {  // if median_col_support > 0.5
                  return false;
                }
              }
              else {  // if median_col_support > 0.509999990463
                if ( mean_col_coverage <= 0.753340005875 ) {
                  return 0.268132033573 < maxgini;
                }
                else {  // if mean_col_coverage > 0.753340005875
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.474500000477
              if ( median_col_support <= 0.667500019073 ) {
                if ( min_col_coverage <= 0.155870437622 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.155870437622
                  return false;
                }
              }
              else {  // if median_col_support > 0.667500019073
                if ( min_col_support <= 0.644500017166 ) {
                  return 0.479694660903 < maxgini;
                }
                else {  // if min_col_support > 0.644500017166
                  return 0.204142011834 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.855355143547
        if ( median_col_coverage <= 0.0517584830523 ) {
          if ( mean_col_support <= 0.89424264431 ) {
            if ( mean_col_coverage <= 0.192157924175 ) {
              if ( median_col_coverage <= 0.0305361300707 ) {
                if ( mean_col_support <= 0.893999993801 ) {
                  return 0.283846875 < maxgini;
                }
                else {  // if mean_col_support > 0.893999993801
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.0305361300707
                if ( min_col_support <= 0.673500001431 ) {
                  return 0.411090143707 < maxgini;
                }
                else {  // if min_col_support > 0.673500001431
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.192157924175
              if ( max_col_coverage <= 0.404545456171 ) {
                if ( median_col_support <= 0.636000037193 ) {
                  return 0.383769132653 < maxgini;
                }
                else {  // if median_col_support > 0.636000037193
                  return 0.492075761252 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.404545456171
                if ( min_col_coverage <= 0.0180194806308 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_coverage > 0.0180194806308
                  return false;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.89424264431
            if ( median_col_support <= 0.726500034332 ) {
              if ( median_col_support <= 0.573000013828 ) {
                if ( min_col_coverage <= 0.0110480561852 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.0110480561852
                  return 0.116761798059 < maxgini;
                }
              }
              else {  // if median_col_support > 0.573000013828
                if ( mean_col_support <= 0.894757330418 ) {
                  return 0.203341855369 < maxgini;
                }
                else {  // if mean_col_support > 0.894757330418
                  return 0.436911660611 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.726500034332
              if ( mean_col_coverage <= 0.200569033623 ) {
                if ( median_col_coverage <= 0.00419448595494 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.00419448595494
                  return 0.311753079559 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.200569033623
                if ( min_col_coverage <= 0.0392307713628 ) {
                  return 0.246997555532 < maxgini;
                }
                else {  // if min_col_coverage > 0.0392307713628
                  return 0.474664472937 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.0517584830523
          if ( mean_col_support <= 0.894072890282 ) {
            if ( median_col_support <= 0.711500048637 ) {
              if ( max_col_coverage <= 0.394338130951 ) {
                if ( median_col_support <= 0.551499962807 ) {
                  return 0.452007038278 < maxgini;
                }
                else {  // if median_col_support > 0.551499962807
                  return 0.351390027295 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.394338130951
                if ( median_col_support <= 0.643499970436 ) {
                  return 0.486540948448 < maxgini;
                }
                else {  // if median_col_support > 0.643499970436
                  return 0.438977318373 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.711500048637
              if ( mean_col_support <= 0.878677070141 ) {
                if ( max_col_coverage <= 0.377976179123 ) {
                  return 0.303117379989 < maxgini;
                }
                else {  // if max_col_coverage > 0.377976179123
                  return 0.41022694628 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.878677070141
                if ( max_col_coverage <= 0.442222237587 ) {
                  return 0.265404967309 < maxgini;
                }
                else {  // if max_col_coverage > 0.442222237587
                  return 0.379384295473 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.894072890282
            if ( median_col_coverage <= 0.303689062595 ) {
              if ( median_col_support <= 0.741500020027 ) {
                if ( mean_col_support <= 0.913441181183 ) {
                  return 0.341208647469 < maxgini;
                }
                else {  // if mean_col_support > 0.913441181183
                  return 0.29359849257 < maxgini;
                }
              }
              else {  // if median_col_support > 0.741500020027
                if ( min_col_support <= 0.787500023842 ) {
                  return 0.227237474438 < maxgini;
                }
                else {  // if min_col_support > 0.787500023842
                  return false;
                }
              }
            }
            else {  // if median_col_coverage > 0.303689062595
              if ( min_col_support <= 0.721500039101 ) {
                if ( median_col_support <= 0.731999993324 ) {
                  return false;
                }
                else {  // if median_col_support > 0.731999993324
                  return 0.393932611353 < maxgini;
                }
              }
              else {  // if min_col_support > 0.721500039101
                if ( median_col_coverage <= 0.346989989281 ) {
                  return 0.290657439446 < maxgini;
                }
                else {  // if median_col_coverage > 0.346989989281
                  return false;
                }
              }
            }
          }
        }
      }
    }
    else {  // if mean_col_support > 0.918062806129
      if ( min_col_coverage <= 0.0509879142046 ) {
        if ( mean_col_coverage <= 0.567511558533 ) {
          if ( median_col_support <= 0.863499999046 ) {
            if ( median_col_support <= 0.761500000954 ) {
              if ( mean_col_coverage <= 0.181085973978 ) {
                if ( mean_col_coverage <= 0.126568630338 ) {
                  return 0.250169047359 < maxgini;
                }
                else {  // if mean_col_coverage > 0.126568630338
                  return 0.335006790123 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.181085973978
                if ( mean_col_coverage <= 0.18202906847 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.18202906847
                  return 0.454747546077 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.761500000954
              if ( mean_col_coverage <= 0.247321546078 ) {
                if ( min_col_coverage <= 0.0349857211113 ) {
                  return 0.295984741942 < maxgini;
                }
                else {  // if min_col_coverage > 0.0349857211113
                  return 0.20491010545 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.247321546078
                if ( min_col_support <= 0.517500042915 ) {
                  return 0.447406866326 < maxgini;
                }
                else {  // if min_col_support > 0.517500042915
                  return 0.294179262276 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.863499999046
            if ( min_col_coverage <= 0.0183767657727 ) {
              if ( min_col_support <= 0.522500038147 ) {
                if ( min_col_support <= 0.514500021935 ) {
                  return 0.229385394632 < maxgini;
                }
                else {  // if min_col_support > 0.514500021935
                  return 0.417823228634 < maxgini;
                }
              }
              else {  // if min_col_support > 0.522500038147
                if ( median_col_support <= 0.929499983788 ) {
                  return 0.218931832871 < maxgini;
                }
                else {  // if median_col_support > 0.929499983788
                  return 0.114152993886 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.0183767657727
              if ( median_col_coverage <= 0.051787994802 ) {
                if ( mean_col_support <= 0.953531384468 ) {
                  return 0.223471396486 < maxgini;
                }
                else {  // if mean_col_support > 0.953531384468
                  return 0.0953430023136 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.051787994802
                if ( min_col_coverage <= 0.0501798540354 ) {
                  return 0.0704811774969 < maxgini;
                }
                else {  // if min_col_coverage > 0.0501798540354
                  return 0.265927977839 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.567511558533
          if ( min_col_support <= 0.863499999046 ) {
            if ( mean_col_coverage <= 0.641284346581 ) {
              if ( max_col_coverage <= 0.866852283478 ) {
                if ( mean_col_coverage <= 0.576269626617 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_coverage > 0.576269626617
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.866852283478
                if ( max_col_coverage <= 0.918333351612 ) {
                  return 0.124444444444 < maxgini;
                }
                else {  // if max_col_coverage > 0.918333351612
                  return 0.406388683432 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.641284346581
              if ( min_col_support <= 0.787999987602 ) {
                if ( median_col_coverage <= 0.413103580475 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.413103580475
                  return 0.419077817413 < maxgini;
                }
              }
              else {  // if min_col_support > 0.787999987602
                if ( median_col_support <= 0.930000007153 ) {
                  return 0.28875 < maxgini;
                }
                else {  // if median_col_support > 0.930000007153
                  return 0.476625273923 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.863499999046
            if ( median_col_coverage <= 0.516026139259 ) {
              if ( min_col_coverage <= 0.0122023811564 ) {
                if ( median_col_support <= 0.981999993324 ) {
                  return 0.478298611111 < maxgini;
                }
                else {  // if median_col_support > 0.981999993324
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0122023811564
                if ( median_col_coverage <= 0.418855547905 ) {
                  return 0.0860319779239 < maxgini;
                }
                else {  // if median_col_coverage > 0.418855547905
                  return 0.308390022676 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.516026139259
              if ( median_col_support <= 0.976500034332 ) {
                if ( max_col_coverage <= 0.99295771122 ) {
                  return 0.21875 < maxgini;
                }
                else {  // if max_col_coverage > 0.99295771122
                  return false;
                }
              }
              else {  // if median_col_support > 0.976500034332
                return 0.0 < maxgini;
              }
            }
          }
        }
      }
      else {  // if min_col_coverage > 0.0509879142046
        if ( min_col_support <= 0.793500006199 ) {
          if ( max_col_coverage <= 0.858591675758 ) {
            if ( max_col_coverage <= 0.253572165966 ) {
              if ( mean_col_support <= 0.945382356644 ) {
                if ( median_col_support <= 0.821500003338 ) {
                  return 0.167710302457 < maxgini;
                }
                else {  // if median_col_support > 0.821500003338
                  return 0.0793079589216 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.945382356644
                if ( median_col_coverage <= 0.0776083767414 ) {
                  return 0.0234935693677 < maxgini;
                }
                else {  // if median_col_coverage > 0.0776083767414
                  return 0.0541991629973 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.253572165966
              if ( min_col_support <= 0.551499962807 ) {
                if ( min_col_support <= 0.528499960899 ) {
                  return 0.196397668303 < maxgini;
                }
                else {  // if min_col_support > 0.528499960899
                  return 0.247182656692 < maxgini;
                }
              }
              else {  // if min_col_support > 0.551499962807
                if ( min_col_support <= 0.710500001907 ) {
                  return 0.133330414395 < maxgini;
                }
                else {  // if min_col_support > 0.710500001907
                  return 0.10336210256 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.858591675758
            if ( mean_col_support <= 0.960470616817 ) {
              if ( min_col_support <= 0.716500043869 ) {
                if ( median_col_support <= 0.948500037193 ) {
                  return 0.336262487888 < maxgini;
                }
                else {  // if median_col_support > 0.948500037193
                  return 0.484329571209 < maxgini;
                }
              }
              else {  // if min_col_support > 0.716500043869
                if ( mean_col_support <= 0.934058785439 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.934058785439
                  return 0.499540863177 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.960470616817
              if ( mean_col_coverage <= 0.644516706467 ) {
                if ( max_col_coverage <= 0.876096487045 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.876096487045
                  return 0.240129799892 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.644516706467
                if ( min_col_support <= 0.675999999046 ) {
                  return false;
                }
                else {  // if min_col_support > 0.675999999046
                  return 0.356505102041 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.793500006199
          if ( max_col_support <= 0.99849998951 ) {
            if ( max_col_support <= 0.995999991894 ) {
              return 0.0 < maxgini;
            }
            else {  // if max_col_support > 0.995999991894
              return false;
            }
          }
          else {  // if max_col_support > 0.99849998951
            if ( min_col_support <= 0.840499997139 ) {
              if ( mean_col_support <= 0.962617635727 ) {
                if ( min_col_support <= 0.814499974251 ) {
                  return 0.152193474604 < maxgini;
                }
                else {  // if min_col_support > 0.814499974251
                  return 0.217888459687 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.962617635727
                if ( min_col_coverage <= 0.159497380257 ) {
                  return 0.041689296395 < maxgini;
                }
                else {  // if min_col_coverage > 0.159497380257
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.840499997139
              if ( mean_col_support <= 0.975323557854 ) {
                if ( min_col_support <= 0.881500005722 ) {
                  return 0.11122612819 < maxgini;
                }
                else {  // if min_col_support > 0.881500005722
                  return 0.228957855331 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.975323557854
                if ( mean_col_coverage <= 0.786180257797 ) {
                  return 0.0246222295534 < maxgini;
                }
                else {  // if mean_col_coverage > 0.786180257797
                  return 0.449380165289 < maxgini;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if min_col_coverage > 0.1599458009
    if ( min_col_support <= 0.745499968529 ) {
      if ( min_col_coverage <= 0.458550065756 ) {
        if ( min_col_coverage <= 0.350174427032 ) {
          if ( mean_col_coverage <= 0.341754436493 ) {
            if ( median_col_coverage <= 0.200267374516 ) {
              if ( median_col_support <= 0.68649995327 ) {
                if ( max_col_coverage <= 0.246621623635 ) {
                  return 0.277777777778 < maxgini;
                }
                else {  // if max_col_coverage > 0.246621623635
                  return 0.421583797005 < maxgini;
                }
              }
              else {  // if median_col_support > 0.68649995327
                if ( max_col_coverage <= 0.286421507597 ) {
                  return 0.132906325334 < maxgini;
                }
                else {  // if max_col_coverage > 0.286421507597
                  return 0.181256603351 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.200267374516
              if ( mean_col_coverage <= 0.289468884468 ) {
                if ( mean_col_coverage <= 0.238890588284 ) {
                  return 0.210500726648 < maxgini;
                }
                else {  // if mean_col_coverage > 0.238890588284
                  return 0.310506076986 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.289468884468
                if ( median_col_support <= 0.685500025749 ) {
                  return 0.474602146604 < maxgini;
                }
                else {  // if median_col_support > 0.685500025749
                  return 0.296141076027 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.341754436493
            if ( mean_col_support <= 0.851147055626 ) {
              if ( median_col_coverage <= 0.350654184818 ) {
                if ( mean_col_support <= 0.797970533371 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.797970533371
                  return 0.482853223594 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.350654184818
                if ( median_col_coverage <= 0.562608718872 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.562608718872
                  return 0.38983756768 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.851147055626
              if ( median_col_support <= 0.970499992371 ) {
                if ( min_col_support <= 0.587499976158 ) {
                  return 0.402015291191 < maxgini;
                }
                else {  // if min_col_support > 0.587499976158
                  return 0.312746327288 < maxgini;
                }
              }
              else {  // if median_col_support > 0.970499992371
                if ( median_col_coverage <= 0.300190120935 ) {
                  return 0.362444079727 < maxgini;
                }
                else {  // if median_col_coverage > 0.300190120935
                  return 0.492110114445 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.350174427032
          if ( max_col_coverage <= 0.500666677952 ) {
            if ( mean_col_support <= 0.845382332802 ) {
              if ( mean_col_coverage <= 0.370039403439 ) {
                return 0.0 < maxgini;
              }
              else {  // if mean_col_coverage > 0.370039403439
                if ( min_col_support <= 0.477999985218 ) {
                  return 0.359861591696 < maxgini;
                }
                else {  // if min_col_support > 0.477999985218
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.845382332802
              if ( median_col_coverage <= 0.428268760443 ) {
                if ( min_col_coverage <= 0.359894186258 ) {
                  return 0.494255010871 < maxgini;
                }
                else {  // if min_col_coverage > 0.359894186258
                  return 0.399451904297 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.428268760443
                if ( median_col_coverage <= 0.458578437567 ) {
                  return 0.297181712752 < maxgini;
                }
                else {  // if median_col_coverage > 0.458578437567
                  return 0.408630033215 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.500666677952
            if ( min_col_support <= 0.616500020027 ) {
              if ( median_col_coverage <= 0.499122798443 ) {
                if ( median_col_support <= 0.981500029564 ) {
                  return 0.499838056333 < maxgini;
                }
                else {  // if median_col_support > 0.981500029564
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.499122798443
                if ( min_col_support <= 0.540500044823 ) {
                  return false;
                }
                else {  // if min_col_support > 0.540500044823
                  return 0.493682309454 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.616500020027
              if ( median_col_coverage <= 0.499347269535 ) {
                if ( max_col_coverage <= 0.695482373238 ) {
                  return 0.45613106213 < maxgini;
                }
                else {  // if max_col_coverage > 0.695482373238
                  return 0.37438456221 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.499347269535
                if ( mean_col_support <= 0.909676492214 ) {
                  return 0.486719107401 < maxgini;
                }
                else {  // if mean_col_support > 0.909676492214
                  return 0.332514877246 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_coverage > 0.458550065756
        if ( mean_col_coverage <= 0.714763641357 ) {
          if ( min_col_support <= 0.630499958992 ) {
            if ( min_col_coverage <= 0.501361608505 ) {
              if ( median_col_support <= 0.984500050545 ) {
                if ( max_col_coverage <= 0.571877837181 ) {
                  return 0.405730609418 < maxgini;
                }
                else {  // if max_col_coverage > 0.571877837181
                  return false;
                }
              }
              else {  // if median_col_support > 0.984500050545
                if ( min_col_support <= 0.609500050545 ) {
                  return false;
                }
                else {  // if min_col_support > 0.609500050545
                  return false;
                }
              }
            }
            else {  // if min_col_coverage > 0.501361608505
              if ( max_col_coverage <= 0.667106389999 ) {
                if ( min_col_support <= 0.580500006676 ) {
                  return false;
                }
                else {  // if min_col_support > 0.580500006676
                  return 0.499444444444 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.667106389999
                if ( median_col_support <= 0.984500050545 ) {
                  return false;
                }
                else {  // if median_col_support > 0.984500050545
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.630499958992
            if ( min_col_coverage <= 0.50046902895 ) {
              if ( max_col_coverage <= 0.600373148918 ) {
                if ( median_col_support <= 0.775499999523 ) {
                  return 0.498671484749 < maxgini;
                }
                else {  // if median_col_support > 0.775499999523
                  return 0.341478462833 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.600373148918
                if ( mean_col_coverage <= 0.599974572659 ) {
                  return 0.491462152503 < maxgini;
                }
                else {  // if mean_col_coverage > 0.599974572659
                  return 0.406175245851 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.50046902895
              if ( max_col_coverage <= 0.66705429554 ) {
                if ( median_col_support <= 0.986500024796 ) {
                  return 0.302344784961 < maxgini;
                }
                else {  // if median_col_support > 0.986500024796
                  return 0.499794105378 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.66705429554
                if ( median_col_coverage <= 0.665338456631 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.665338456631
                  return 0.478634531224 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.714763641357
          if ( mean_col_support <= 0.976558864117 ) {
            if ( min_col_support <= 0.614500045776 ) {
              if ( mean_col_coverage <= 0.987109959126 ) {
                if ( mean_col_support <= 0.92714703083 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.92714703083
                  return false;
                }
              }
              else {  // if mean_col_coverage > 0.987109959126
                if ( median_col_support <= 0.976500034332 ) {
                  return false;
                }
                else {  // if median_col_support > 0.976500034332
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.614500045776
              if ( median_col_support <= 0.986500024796 ) {
                if ( median_col_coverage <= 0.833728253841 ) {
                  return 0.486982120435 < maxgini;
                }
                else {  // if median_col_coverage > 0.833728253841
                  return false;
                }
              }
              else {  // if median_col_support > 0.986500024796
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return false;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.976558864117
            if ( median_col_support <= 0.991500020027 ) {
              if ( median_col_support <= 0.986500024796 ) {
                if ( min_col_coverage <= 0.926956295967 ) {
                  return 0.370544019778 < maxgini;
                }
                else {  // if min_col_coverage > 0.926956295967
                  return false;
                }
              }
              else {  // if median_col_support > 0.986500024796
                if ( max_col_coverage <= 0.782981038094 ) {
                  return 0.0 < maxgini;
                }
                else {  // if max_col_coverage > 0.782981038094
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.991500020027
              if ( max_col_coverage <= 0.800959348679 ) {
                if ( min_col_coverage <= 0.666169166565 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.666169166565
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.800959348679
                if ( mean_col_coverage <= 0.933394610882 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.933394610882
                  return false;
                }
              }
            }
          }
        }
      }
    }
    else {  // if min_col_support > 0.745499968529
      if ( mean_col_support <= 0.987500011921 ) {
        if ( max_col_coverage <= 0.857350468636 ) {
          if ( mean_col_coverage <= 0.65364831686 ) {
            if ( mean_col_support <= 0.95085299015 ) {
              if ( median_col_support <= 0.829499959946 ) {
                if ( mean_col_coverage <= 0.371806740761 ) {
                  return 0.247179589057 < maxgini;
                }
                else {  // if mean_col_coverage > 0.371806740761
                  return 0.389254155783 < maxgini;
                }
              }
              else {  // if median_col_support > 0.829499959946
                if ( min_col_coverage <= 0.404508620501 ) {
                  return 0.215776996664 < maxgini;
                }
                else {  // if min_col_coverage > 0.404508620501
                  return 0.274238949944 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.95085299015
              if ( median_col_coverage <= 0.501225590706 ) {
                if ( min_col_support <= 0.836500048637 ) {
                  return 0.116760333762 < maxgini;
                }
                else {  // if min_col_support > 0.836500048637
                  return 0.070023059885 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.501225590706
                if ( min_col_support <= 0.797500014305 ) {
                  return 0.354196322755 < maxgini;
                }
                else {  // if min_col_support > 0.797500014305
                  return 0.0869837743896 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.65364831686
            if ( median_col_support <= 0.988499999046 ) {
              if ( min_col_support <= 0.846500039101 ) {
                if ( mean_col_support <= 0.951205849648 ) {
                  return 0.354743377672 < maxgini;
                }
                else {  // if mean_col_support > 0.951205849648
                  return 0.184740830607 < maxgini;
                }
              }
              else {  // if min_col_support > 0.846500039101
                if ( max_col_coverage <= 0.856499373913 ) {
                  return 0.0579883460292 < maxgini;
                }
                else {  // if max_col_coverage > 0.856499373913
                  return 0.0291971273885 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.988499999046
              if ( median_col_support <= 0.99950003624 ) {
                if ( mean_col_support <= 0.981676459312 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.981676459312
                  return false;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( median_col_coverage <= 0.600613474846 ) {
                  return 0.0610160002668 < maxgini;
                }
                else {  // if median_col_coverage > 0.600613474846
                  return 0.113772296361 < maxgini;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.857350468636
          if ( median_col_support <= 0.988499999046 ) {
            if ( min_col_coverage <= 0.920321941376 ) {
              if ( median_col_support <= 0.894500017166 ) {
                if ( mean_col_support <= 0.901176452637 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.901176452637
                  return 0.290207288925 < maxgini;
                }
              }
              else {  // if median_col_support > 0.894500017166
                if ( min_col_coverage <= 0.85747051239 ) {
                  return 0.129743461252 < maxgini;
                }
                else {  // if min_col_coverage > 0.85747051239
                  return 0.255762034459 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.920321941376
              if ( max_col_coverage <= 0.997581601143 ) {
                if ( min_col_coverage <= 0.989236116409 ) {
                  return 0.490991396784 < maxgini;
                }
                else {  // if min_col_coverage > 0.989236116409
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.997581601143
                if ( min_col_support <= 0.829499959946 ) {
                  return 0.495810940932 < maxgini;
                }
                else {  // if min_col_support > 0.829499959946
                  return 0.33108288424 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.988499999046
            if ( median_col_coverage <= 0.714808106422 ) {
              if ( min_col_coverage <= 0.608969688416 ) {
                if ( mean_col_support <= 0.985441207886 ) {
                  return 0.142180802929 < maxgini;
                }
                else {  // if mean_col_support > 0.985441207886
                  return 0.0332131117604 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.608969688416
                if ( max_col_coverage <= 0.899245142937 ) {
                  return 0.3801800538 < maxgini;
                }
                else {  // if max_col_coverage > 0.899245142937
                  return 0.0969126301494 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.714808106422
              if ( mean_col_support <= 0.981147050858 ) {
                if ( max_col_coverage <= 0.997584939003 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.997584939003
                  return 0.483850664551 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.981147050858
                if ( mean_col_support <= 0.986382365227 ) {
                  return 0.453980085875 < maxgini;
                }
                else {  // if mean_col_support > 0.986382365227
                  return 0.343021218569 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.987500011921
        if ( min_col_coverage <= 0.937588512897 ) {
          if ( median_col_coverage <= 0.498859524727 ) {
            if ( mean_col_support <= 0.992735266685 ) {
              if ( min_col_support <= 0.945500016212 ) {
                if ( median_col_coverage <= 0.49337720871 ) {
                  return 0.0177526360653 < maxgini;
                }
                else {  // if median_col_coverage > 0.49337720871
                  return 0.494341330919 < maxgini;
                }
              }
              else {  // if min_col_support > 0.945500016212
                if ( min_col_coverage <= 0.408280134201 ) {
                  return 0.057770575149 < maxgini;
                }
                else {  // if min_col_coverage > 0.408280134201
                  return 0.0241459094195 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.992735266685
              if ( median_col_coverage <= 0.498797744513 ) {
                if ( max_col_coverage <= 0.638971090317 ) {
                  return 0.00358723803278 < maxgini;
                }
                else {  // if max_col_coverage > 0.638971090317
                  return 0.00262844807666 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.498797744513
                return false;
              }
            }
          }
          else {  // if median_col_coverage > 0.498859524727
            if ( median_col_support <= 0.99950003624 ) {
              if ( min_col_coverage <= 0.91676568985 ) {
                if ( mean_col_support <= 0.990970551968 ) {
                  return 0.09486608583 < maxgini;
                }
                else {  // if mean_col_support > 0.990970551968
                  return 0.0128473657807 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.91676568985
                if ( median_col_coverage <= 0.926869750023 ) {
                  return 0.0114108072067 < maxgini;
                }
                else {  // if median_col_coverage > 0.926869750023
                  return 0.0888849305592 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( mean_col_support <= 0.990382373333 ) {
                if ( median_col_coverage <= 0.904950141907 ) {
                  return 0.0280032481714 < maxgini;
                }
                else {  // if median_col_coverage > 0.904950141907
                  return 0.090555890611 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.990382373333
                if ( min_col_coverage <= 0.226539582014 ) {
                  return 0.345679012346 < maxgini;
                }
                else {  // if min_col_coverage > 0.226539582014
                  return 0.00223365247689 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.937588512897
          if ( max_col_coverage <= 0.99903845787 ) {
            if ( mean_col_support <= 0.992029428482 ) {
              if ( mean_col_coverage <= 0.960822939873 ) {
                if ( min_col_coverage <= 0.938686013222 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.938686013222
                  return 0.119173553719 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.960822939873
                if ( mean_col_coverage <= 0.981934547424 ) {
                  return 0.48400789295 < maxgini;
                }
                else {  // if mean_col_coverage > 0.981934547424
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.992029428482
              if ( mean_col_coverage <= 0.982404589653 ) {
                if ( min_col_support <= 0.897500038147 ) {
                  return 0.269471117208 < maxgini;
                }
                else {  // if min_col_support > 0.897500038147
                  return 0.00224813541351 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.982404589653
                if ( median_col_support <= 0.988499999046 ) {
                  return false;
                }
                else {  // if median_col_support > 0.988499999046
                  return 0.16350532468 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.99903845787
            if ( min_col_coverage <= 0.938433647156 ) {
              if ( median_col_support <= 0.99849998951 ) {
                if ( mean_col_coverage <= 0.958508372307 ) {
                  return 0.444444444444 < maxgini;
                }
                else {  // if mean_col_coverage > 0.958508372307
                  return 0.0327777777778 < maxgini;
                }
              }
              else {  // if median_col_support > 0.99849998951
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.107755102041 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.938433647156
              if ( mean_col_support <= 0.990852952003 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.358278119421 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0771505079244 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.990852952003
                if ( min_col_support <= 0.884500026703 ) {
                  return 0.265412022032 < maxgini;
                }
                else {  // if min_col_support > 0.884500026703
                  return 0.00650585886854 < maxgini;
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
  if ( min_col_support <= 0.766499996185 ) {
    if ( median_col_support <= 0.96749997139 ) {
      if ( min_col_coverage <= 0.500554323196 ) {
        if ( median_col_support <= 0.71850001812 ) {
          if ( median_col_coverage <= 0.261162549257 ) {
            if ( min_col_coverage <= 0.150471702218 ) {
              if ( mean_col_coverage <= 0.245911329985 ) {
                if ( min_col_coverage <= 0.0518360957503 ) {
                  return 0.419374559783 < maxgini;
                }
                else {  // if min_col_coverage > 0.0518360957503
                  return 0.343267504478 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.245911329985
                if ( median_col_support <= 0.619500041008 ) {
                  return 0.493434508379 < maxgini;
                }
                else {  // if median_col_support > 0.619500041008
                  return 0.451219438126 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.150471702218
              if ( median_col_coverage <= 0.202243030071 ) {
                if ( mean_col_coverage <= 0.210372954607 ) {
                  return 0.245277152057 < maxgini;
                }
                else {  // if mean_col_coverage > 0.210372954607
                  return 0.419247879746 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.202243030071
                if ( mean_col_support <= 0.82747066021 ) {
                  return 0.493433279386 < maxgini;
                }
                else {  // if mean_col_support > 0.82747066021
                  return 0.438349553097 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.261162549257
            if ( mean_col_coverage <= 0.432439446449 ) {
              if ( median_col_support <= 0.619500041008 ) {
                if ( mean_col_coverage <= 0.342357993126 ) {
                  return 0.499360511476 < maxgini;
                }
                else {  // if mean_col_coverage > 0.342357993126
                  return false;
                }
              }
              else {  // if median_col_support > 0.619500041008
                if ( median_col_support <= 0.675500035286 ) {
                  return 0.471469545193 < maxgini;
                }
                else {  // if median_col_support > 0.675500035286
                  return 0.424174599522 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.432439446449
              if ( mean_col_coverage <= 0.47358673811 ) {
                if ( min_col_support <= 0.620499968529 ) {
                  return false;
                }
                else {  // if min_col_support > 0.620499968529
                  return 0.492892327875 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.47358673811
                if ( max_col_coverage <= 0.681334614754 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.681334614754
                  return false;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.71850001812
          if ( mean_col_coverage <= 0.42357686162 ) {
            if ( min_col_coverage <= 0.0510251522064 ) {
              if ( mean_col_coverage <= 0.215571120381 ) {
                if ( min_col_support <= 0.522500038147 ) {
                  return 0.316091128436 < maxgini;
                }
                else {  // if min_col_support > 0.522500038147
                  return 0.224887704361 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.215571120381
                if ( min_col_support <= 0.712499976158 ) {
                  return 0.354307442885 < maxgini;
                }
                else {  // if min_col_support > 0.712499976158
                  return 0.203040690527 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.0510251522064
              if ( min_col_support <= 0.551499962807 ) {
                if ( median_col_support <= 0.931499958038 ) {
                  return 0.221840893199 < maxgini;
                }
                else {  // if median_col_support > 0.931499958038
                  return 0.404912657584 < maxgini;
                }
              }
              else {  // if min_col_support > 0.551499962807
                if ( mean_col_coverage <= 0.31127756834 ) {
                  return 0.16094739157 < maxgini;
                }
                else {  // if mean_col_coverage > 0.31127756834
                  return 0.249867115998 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.42357686162
            if ( median_col_coverage <= 0.400280117989 ) {
              if ( min_col_support <= 0.552500009537 ) {
                if ( min_col_coverage <= 0.390096604824 ) {
                  return 0.415281606596 < maxgini;
                }
                else {  // if min_col_coverage > 0.390096604824
                  return 0.190973458328 < maxgini;
                }
              }
              else {  // if min_col_support > 0.552500009537
                if ( min_col_coverage <= 0.0231663696468 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.0231663696468
                  return 0.295649362132 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.400280117989
              if ( min_col_support <= 0.583500027657 ) {
                if ( median_col_support <= 0.919499993324 ) {
                  return 0.430827151804 < maxgini;
                }
                else {  // if median_col_support > 0.919499993324
                  return false;
                }
              }
              else {  // if min_col_support > 0.583500027657
                if ( mean_col_support <= 0.928852915764 ) {
                  return 0.409871107088 < maxgini;
                }
                else {  // if mean_col_support > 0.928852915764
                  return 0.266313809263 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_coverage > 0.500554323196
        if ( min_col_coverage <= 0.714572548866 ) {
          if ( median_col_support <= 0.738499999046 ) {
            if ( mean_col_coverage <= 0.749855816364 ) {
              if ( min_col_support <= 0.466499984264 ) {
                if ( median_col_coverage <= 0.611616134644 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.611616134644
                  return false;
                }
              }
              else {  // if min_col_support > 0.466499984264
                if ( mean_col_coverage <= 0.742592394352 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.742592394352
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.749855816364
              if ( min_col_coverage <= 0.520869553089 ) {
                if ( min_col_support <= 0.630999982357 ) {
                  return false;
                }
                else {  // if min_col_support > 0.630999982357
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.520869553089
                if ( mean_col_support <= 0.905411720276 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.905411720276
                  return 0.379853902345 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.738499999046
            if ( min_col_support <= 0.630499958992 ) {
              if ( max_col_coverage <= 0.700140058994 ) {
                if ( mean_col_coverage <= 0.646885752678 ) {
                  return 0.466869428441 < maxgini;
                }
                else {  // if mean_col_coverage > 0.646885752678
                  return 0.176085663296 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.700140058994
                if ( mean_col_coverage <= 0.796561837196 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.796561837196
                  return 0.494271268746 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.630499958992
              if ( max_col_coverage <= 0.714543581009 ) {
                if ( min_col_coverage <= 0.511428952217 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.511428952217
                  return 0.307013855671 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.714543581009
                if ( min_col_support <= 0.689499974251 ) {
                  return 0.47534664914 < maxgini;
                }
                else {  // if min_col_support > 0.689499974251
                  return 0.367604656685 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.714572548866
          if ( min_col_support <= 0.695500016212 ) {
            if ( min_col_support <= 0.639500021935 ) {
              if ( max_col_coverage <= 0.994389891624 ) {
                if ( median_col_coverage <= 0.974021553993 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.974021553993
                  return 0.4872 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.994389891624
                if ( min_col_coverage <= 0.99069583416 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.99069583416
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.639500021935
              if ( mean_col_support <= 0.970500051975 ) {
                if ( mean_col_coverage <= 0.879833936691 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.879833936691
                  return false;
                }
              }
              else {  // if mean_col_support > 0.970500051975
                if ( mean_col_coverage <= 0.901083171368 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.901083171368
                  return 0.402042461704 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.695500016212
            if ( min_col_support <= 0.753499984741 ) {
              if ( min_col_coverage <= 0.852155447006 ) {
                if ( min_col_coverage <= 0.75035816431 ) {
                  return 0.432132963989 < maxgini;
                }
                else {  // if min_col_coverage > 0.75035816431
                  return 0.49109485343 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.852155447006
                if ( median_col_support <= 0.923500001431 ) {
                  return false;
                }
                else {  // if median_col_support > 0.923500001431
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.753499984741
              if ( min_col_coverage <= 0.833793759346 ) {
                if ( mean_col_coverage <= 0.898150324821 ) {
                  return 0.367115466224 < maxgini;
                }
                else {  // if mean_col_coverage > 0.898150324821
                  return 0.150497178178 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.833793759346
                if ( min_col_coverage <= 0.83854675293 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.83854675293
                  return 0.495497895466 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if median_col_support > 0.96749997139
      if ( median_col_coverage <= 0.350126922131 ) {
        if ( median_col_support <= 0.999000012875 ) {
          if ( median_col_support <= 0.987499952316 ) {
            if ( median_col_support <= 0.977499961853 ) {
              if ( mean_col_coverage <= 0.197215691209 ) {
                if ( mean_col_support <= 0.928441166878 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.928441166878
                  return 0.0974287801315 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.197215691209
                if ( min_col_support <= 0.640499949455 ) {
                  return 0.467126890204 < maxgini;
                }
                else {  // if min_col_support > 0.640499949455
                  return 0.183768367537 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.977499961853
              if ( mean_col_support <= 0.956470608711 ) {
                if ( mean_col_coverage <= 0.227455914021 ) {
                  return 0.3046875 < maxgini;
                }
                else {  // if mean_col_coverage > 0.227455914021
                  return false;
                }
              }
              else {  // if mean_col_support > 0.956470608711
                if ( median_col_support <= 0.985499978065 ) {
                  return 0.372461993674 < maxgini;
                }
                else {  // if median_col_support > 0.985499978065
                  return 0.468421918259 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.987499952316
            if ( min_col_coverage <= 0.144707202911 ) {
              if ( max_col_coverage <= 0.294666051865 ) {
                if ( min_col_support <= 0.72000002861 ) {
                  return false;
                }
                else {  // if min_col_support > 0.72000002861
                  return 0.0 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.294666051865
                if ( mean_col_coverage <= 0.654210627079 ) {
                  return 0.415844838922 < maxgini;
                }
                else {  // if mean_col_coverage > 0.654210627079
                  return false;
                }
              }
            }
            else {  // if min_col_coverage > 0.144707202911
              if ( min_col_support <= 0.684499979019 ) {
                if ( median_col_support <= 0.991500020027 ) {
                  return false;
                }
                else {  // if median_col_support > 0.991500020027
                  return false;
                }
              }
              else {  // if min_col_support > 0.684499979019
                if ( mean_col_support <= 0.965117573738 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.965117573738
                  return false;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.999000012875
          if ( median_col_coverage <= 0.250356137753 ) {
            if ( min_col_support <= 0.550500035286 ) {
              if ( median_col_coverage <= 0.150342464447 ) {
                if ( median_col_coverage <= 0.0506410263479 ) {
                  return 0.313074757584 < maxgini;
                }
                else {  // if median_col_coverage > 0.0506410263479
                  return 0.17798731722 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.150342464447
                if ( min_col_support <= 0.458499997854 ) {
                  return 0.127066115702 < maxgini;
                }
                else {  // if min_col_support > 0.458499997854
                  return 0.414253287197 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.550500035286
              if ( median_col_coverage <= 0.0502100847661 ) {
                if ( max_col_coverage <= 0.277171224356 ) {
                  return 0.119782654066 < maxgini;
                }
                else {  // if max_col_coverage > 0.277171224356
                  return 0.167049806332 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.0502100847661
                if ( mean_col_support <= 0.957242667675 ) {
                  return 0.11532113042 < maxgini;
                }
                else {  // if mean_col_support > 0.957242667675
                  return 0.0578133228936 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.250356137753
            if ( mean_col_support <= 0.947088241577 ) {
              if ( min_col_support <= 0.557500004768 ) {
                if ( median_col_coverage <= 0.347597241402 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.347597241402
                  return 0.462809917355 < maxgini;
                }
              }
              else {  // if min_col_support > 0.557500004768
                if ( mean_col_support <= 0.934088230133 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.934088230133
                  return 0.278662243456 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.947088241577
              if ( min_col_support <= 0.611500024796 ) {
                if ( min_col_coverage <= 0.183501690626 ) {
                  return 0.297520661157 < maxgini;
                }
                else {  // if min_col_coverage > 0.183501690626
                  return false;
                }
              }
              else {  // if min_col_support > 0.611500024796
                if ( mean_col_coverage <= 0.355892956257 ) {
                  return 0.144783829091 < maxgini;
                }
                else {  // if mean_col_coverage > 0.355892956257
                  return 0.230297829554 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_coverage > 0.350126922131
        if ( median_col_support <= 0.986500024796 ) {
          if ( min_col_support <= 0.678499996662 ) {
            if ( median_col_coverage <= 0.655956983566 ) {
              if ( min_col_support <= 0.612499952316 ) {
                if ( max_col_coverage <= 0.654814600945 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.654814600945
                  return false;
                }
              }
              else {  // if min_col_support > 0.612499952316
                if ( median_col_coverage <= 0.649561405182 ) {
                  return 0.4903921522 < maxgini;
                }
                else {  // if median_col_coverage > 0.649561405182
                  return 0.24169921875 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.655956983566
              if ( median_col_coverage <= 0.985257863998 ) {
                if ( min_col_support <= 0.616500020027 ) {
                  return false;
                }
                else {  // if min_col_support > 0.616500020027
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.985257863998
                if ( max_col_coverage <= 0.992129206657 ) {
                  return 0.0 < maxgini;
                }
                else {  // if max_col_coverage > 0.992129206657
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.678499996662
            if ( min_col_coverage <= 0.843080282211 ) {
              if ( mean_col_support <= 0.9685587883 ) {
                if ( mean_col_support <= 0.967617630959 ) {
                  return 0.426597968576 < maxgini;
                }
                else {  // if mean_col_support > 0.967617630959
                  return 0.472089723463 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.9685587883
                if ( max_col_coverage <= 0.781353473663 ) {
                  return 0.281326179413 < maxgini;
                }
                else {  // if max_col_coverage > 0.781353473663
                  return 0.393687182895 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.843080282211
              if ( min_col_coverage <= 0.914811134338 ) {
                if ( min_col_coverage <= 0.848713755608 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.848713755608
                  return 0.494244989255 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.914811134338
                if ( min_col_support <= 0.743499994278 ) {
                  return false;
                }
                else {  // if min_col_support > 0.743499994278
                  return 0.498810232005 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.986500024796
          if ( mean_col_coverage <= 0.660181283951 ) {
            if ( min_col_support <= 0.674499988556 ) {
              if ( min_col_support <= 0.613499999046 ) {
                if ( median_col_coverage <= 0.500666677952 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.500666677952
                  return false;
                }
              }
              else {  // if min_col_support > 0.613499999046
                if ( mean_col_support <= 0.976970613003 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.976970613003
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.674499988556
              if ( min_col_support <= 0.729499995708 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.430546926609 < maxgini;
                }
              }
              else {  // if min_col_support > 0.729499995708
                if ( median_col_coverage <= 0.502186059952 ) {
                  return 0.354130155198 < maxgini;
                }
                else {  // if median_col_coverage > 0.502186059952
                  return 0.488836041749 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.660181283951
            if ( mean_col_support <= 0.977090001106 ) {
              if ( median_col_support <= 0.99950003624 ) {
                if ( min_col_support <= 0.706499993801 ) {
                  return false;
                }
                else {  // if min_col_support > 0.706499993801
                  return false;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( median_col_coverage <= 0.668693184853 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.668693184853
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.977090001106
              if ( min_col_support <= 0.698500037193 ) {
                if ( mean_col_support <= 0.980264723301 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.980264723301
                  return false;
                }
              }
              else {  // if min_col_support > 0.698500037193
                if ( median_col_coverage <= 0.880223929882 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.880223929882
                  return false;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if min_col_support > 0.766499996185
    if ( median_col_support <= 0.99950003624 ) {
      if ( min_col_support <= 0.861500024796 ) {
        if ( median_col_support <= 0.991500020027 ) {
          if ( min_col_support <= 0.819499969482 ) {
            if ( median_col_support <= 0.835500001907 ) {
              if ( median_col_support <= 0.824499964714 ) {
                if ( min_col_coverage <= 0.311422407627 ) {
                  return 0.300152092326 < maxgini;
                }
                else {  // if min_col_coverage > 0.311422407627
                  return 0.399475271423 < maxgini;
                }
              }
              else {  // if median_col_support > 0.824499964714
                if ( max_col_support <= 0.989499986172 ) {
                  return false;
                }
                else {  // if max_col_support > 0.989499986172
                  return 0.300471982284 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.835500001907
              if ( median_col_support <= 0.984500050545 ) {
                if ( min_col_support <= 0.785500049591 ) {
                  return 0.182332212089 < maxgini;
                }
                else {  // if min_col_support > 0.785500049591
                  return 0.154682526476 < maxgini;
                }
              }
              else {  // if median_col_support > 0.984500050545
                if ( min_col_coverage <= 0.921240568161 ) {
                  return 0.418983231116 < maxgini;
                }
                else {  // if min_col_coverage > 0.921240568161
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.819499969482
            if ( median_col_support <= 0.872500002384 ) {
              if ( median_col_support <= 0.860499978065 ) {
                if ( mean_col_support <= 0.938029408455 ) {
                  return 0.366376662905 < maxgini;
                }
                else {  // if mean_col_support > 0.938029408455
                  return 0.260378099545 < maxgini;
                }
              }
              else {  // if median_col_support > 0.860499978065
                if ( min_col_support <= 0.849500000477 ) {
                  return 0.182495117188 < maxgini;
                }
                else {  // if min_col_support > 0.849500000477
                  return 0.279320447341 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.872500002384
              if ( median_col_support <= 0.986500024796 ) {
                if ( median_col_coverage <= 0.863909959793 ) {
                  return 0.09876532858 < maxgini;
                }
                else {  // if median_col_coverage > 0.863909959793
                  return 0.36148618275 < maxgini;
                }
              }
              else {  // if median_col_support > 0.986500024796
                if ( min_col_support <= 0.840499997139 ) {
                  return 0.32218085395 < maxgini;
                }
                else {  // if min_col_support > 0.840499997139
                  return 0.238293648742 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.991500020027
          if ( median_col_support <= 0.994500041008 ) {
            if ( min_col_support <= 0.826499998569 ) {
              if ( min_col_coverage <= 0.515877008438 ) {
                if ( min_col_coverage <= 0.292499989271 ) {
                  return 0.356698296092 < maxgini;
                }
                else {  // if min_col_coverage > 0.292499989271
                  return 0.498951952388 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.515877008438
                if ( median_col_support <= 0.992499947548 ) {
                  return 0.498161268294 < maxgini;
                }
                else {  // if median_col_support > 0.992499947548
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.826499998569
              if ( min_col_coverage <= 0.943126678467 ) {
                if ( min_col_coverage <= 0.77073776722 ) {
                  return 0.48149945231 < maxgini;
                }
                else {  // if min_col_coverage > 0.77073776722
                  return 0.414530465554 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.943126678467
                if ( mean_col_support <= 0.985205829144 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.985205829144
                  return 0.483740929858 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.994500041008
            if ( median_col_support <= 0.997500002384 ) {
              if ( median_col_coverage <= 0.619517087936 ) {
                if ( min_col_support <= 0.833500027657 ) {
                  return false;
                }
                else {  // if min_col_support > 0.833500027657
                  return 0.499847637763 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.619517087936
                if ( median_col_support <= 0.996500015259 ) {
                  return false;
                }
                else {  // if median_col_support > 0.996500015259
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.997500002384
              if ( mean_col_support <= 0.979088306427 ) {
                if ( mean_col_coverage <= 0.704805016518 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.704805016518
                  return false;
                }
              }
              else {  // if mean_col_support > 0.979088306427
                if ( min_col_support <= 0.815500020981 ) {
                  return false;
                }
                else {  // if min_col_support > 0.815500020981
                  return false;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.861500024796
        if ( mean_col_support <= 0.976970553398 ) {
          if ( mean_col_support <= 0.959794104099 ) {
            if ( max_col_coverage <= 0.626059293747 ) {
              if ( mean_col_support <= 0.94061768055 ) {
                if ( mean_col_support <= 0.937117636204 ) {
                  return 0.255 < maxgini;
                }
                else {  // if mean_col_support > 0.937117636204
                  return false;
                }
              }
              else {  // if mean_col_support > 0.94061768055
                if ( mean_col_support <= 0.959676504135 ) {
                  return 0.271308764075 < maxgini;
                }
                else {  // if mean_col_support > 0.959676504135
                  return 0.48150887574 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.626059293747
              if ( median_col_support <= 0.928499996662 ) {
                if ( min_col_coverage <= 0.98647660017 ) {
                  return 0.218326590199 < maxgini;
                }
                else {  // if min_col_coverage > 0.98647660017
                  return false;
                }
              }
              else {  // if median_col_support > 0.928499996662
                if ( min_col_support <= 0.880499958992 ) {
                  return false;
                }
                else {  // if min_col_support > 0.880499958992
                  return 0.172335600907 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.959794104099
            if ( median_col_support <= 0.892500042915 ) {
              if ( median_col_coverage <= 0.479130446911 ) {
                if ( min_col_coverage <= 0.290994614363 ) {
                  return 0.186474627682 < maxgini;
                }
                else {  // if min_col_coverage > 0.290994614363
                  return 0.123372481937 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.479130446911
                if ( mean_col_support <= 0.96538233757 ) {
                  return 0.158878195886 < maxgini;
                }
                else {  // if mean_col_support > 0.96538233757
                  return 0.28470743034 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.892500042915
              if ( min_col_coverage <= 0.962696909904 ) {
                if ( min_col_coverage <= 0.0735503435135 ) {
                  return 0.244897959184 < maxgini;
                }
                else {  // if min_col_coverage > 0.0735503435135
                  return 0.0966398501699 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.962696909904
                if ( median_col_coverage <= 0.966210007668 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.966210007668
                  return 0.439864412928 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.976970553398
          if ( mean_col_coverage <= 0.998326361179 ) {
            if ( median_col_coverage <= 0.977381229401 ) {
              if ( mean_col_support <= 0.992485284805 ) {
                if ( mean_col_coverage <= 0.954629004002 ) {
                  return 0.0455818671359 < maxgini;
                }
                else {  // if mean_col_coverage > 0.954629004002
                  return 0.166059688679 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.992485284805
                if ( mean_col_support <= 0.993852972984 ) {
                  return 0.0177112857725 < maxgini;
                }
                else {  // if mean_col_support > 0.993852972984
                  return 0.00395331439246 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.977381229401
              if ( max_col_support <= 0.99950003624 ) {
                if ( min_col_support <= 0.93850004673 ) {
                  return false;
                }
                else {  // if min_col_support > 0.93850004673
                  return 0.0 < maxgini;
                }
              }
              else {  // if max_col_support > 0.99950003624
                if ( mean_col_support <= 0.987500011921 ) {
                  return 0.356307790413 < maxgini;
                }
                else {  // if mean_col_support > 0.987500011921
                  return 0.114763363576 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.998326361179
            if ( min_col_coverage <= 0.972401976585 ) {
              return false;
            }
            else {  // if min_col_coverage > 0.972401976585
              if ( min_col_support <= 0.943500041962 ) {
                if ( median_col_support <= 0.947499990463 ) {
                  return 0.0713305898491 < maxgini;
                }
                else {  // if median_col_support > 0.947499990463
                  return 0.423668521469 < maxgini;
                }
              }
              else {  // if min_col_support > 0.943500041962
                if ( mean_col_support <= 0.986558794975 ) {
                  return 0.470692520776 < maxgini;
                }
                else {  // if mean_col_support > 0.986558794975
                  return 0.12766019587 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if median_col_support > 0.99950003624
      if ( median_col_coverage <= 0.444054365158 ) {
        if ( mean_col_support <= 0.982067883015 ) {
          if ( mean_col_coverage <= 0.368384182453 ) {
            if ( min_col_support <= 0.775499999523 ) {
              if ( min_col_coverage <= 0.0513157919049 ) {
                if ( min_col_support <= 0.772500038147 ) {
                  return 0.101119143428 < maxgini;
                }
                else {  // if min_col_support > 0.772500038147
                  return 0.2654912764 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0513157919049
                if ( min_col_support <= 0.773499965668 ) {
                  return 0.038080874729 < maxgini;
                }
                else {  // if min_col_support > 0.773499965668
                  return 0.122281421958 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.775499999523
              if ( min_col_support <= 0.844500005245 ) {
                if ( mean_col_support <= 0.974639594555 ) {
                  return 0.0679034467578 < maxgini;
                }
                else {  // if mean_col_support > 0.974639594555
                  return 0.026856838201 < maxgini;
                }
              }
              else {  // if min_col_support > 0.844500005245
                if ( mean_col_support <= 0.974088311195 ) {
                  return 0.175761635569 < maxgini;
                }
                else {  // if mean_col_support > 0.974088311195
                  return 0.0521224380277 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.368384182453
            if ( mean_col_coverage <= 0.447613686323 ) {
              if ( max_col_coverage <= 0.536149859428 ) {
                if ( mean_col_support <= 0.973794102669 ) {
                  return 0.216055404607 < maxgini;
                }
                else {  // if mean_col_support > 0.973794102669
                  return 0.078290010373 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.536149859428
                if ( mean_col_coverage <= 0.392994642258 ) {
                  return 0.0269008264463 < maxgini;
                }
                else {  // if mean_col_coverage > 0.392994642258
                  return 0.0664623039298 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.447613686323
              if ( min_col_support <= 0.885499954224 ) {
                if ( max_col_coverage <= 0.647853732109 ) {
                  return 0.137727426485 < maxgini;
                }
                else {  // if max_col_coverage > 0.647853732109
                  return 0.0673616554988 < maxgini;
                }
              }
              else {  // if min_col_support > 0.885499954224
                if ( mean_col_support <= 0.980088233948 ) {
                  return 0.460223537147 < maxgini;
                }
                else {  // if mean_col_support > 0.980088233948
                  return 0.223002628504 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.982067883015
          if ( max_col_coverage <= 0.927051663399 ) {
            if ( mean_col_support <= 0.990499973297 ) {
              if ( median_col_coverage <= 0.0106954928488 ) {
                if ( median_col_coverage <= 0.00995049439371 ) {
                  return 0.150247933884 < maxgini;
                }
                else {  // if median_col_coverage > 0.00995049439371
                  return 0.498614958449 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.0106954928488
                if ( max_col_coverage <= 0.505222558975 ) {
                  return 0.0219667995215 < maxgini;
                }
                else {  // if max_col_coverage > 0.505222558975
                  return 0.0274298386252 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.990499973297
              if ( mean_col_support <= 0.993911743164 ) {
                if ( median_col_coverage <= 0.440390765667 ) {
                  return 0.0103938506558 < maxgini;
                }
                else {  // if median_col_coverage > 0.440390765667
                  return 0.0294384991412 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.993911743164
                if ( median_col_coverage <= 0.244855612516 ) {
                  return 0.00125027530692 < maxgini;
                }
                else {  // if median_col_coverage > 0.244855612516
                  return 0.00281643194622 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.927051663399
            if ( median_col_coverage <= 0.0317540317774 ) {
              if ( median_col_coverage <= 0.0298573970795 ) {
                if ( mean_col_support <= 0.994794130325 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_support > 0.994794130325
                  return 0.244897959184 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.0298573970795
                if ( min_col_coverage <= 0.0307765156031 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.0307765156031
                  return 0.408163265306 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.0317540317774
              if ( min_col_coverage <= 0.337121218443 ) {
                if ( min_col_coverage <= 0.331223636866 ) {
                  return 0.0673336938886 < maxgini;
                }
                else {  // if min_col_coverage > 0.331223636866
                  return 0.444444444444 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.337121218443
                return 0.0 < maxgini;
              }
            }
          }
        }
      }
      else {  // if median_col_coverage > 0.444054365158
        if ( max_col_coverage <= 0.710717797279 ) {
          if ( mean_col_support <= 0.987735271454 ) {
            if ( mean_col_coverage <= 0.630017995834 ) {
              if ( min_col_coverage <= 0.563696861267 ) {
                if ( min_col_coverage <= 0.562895536423 ) {
                  return 0.0719583909754 < maxgini;
                }
                else {  // if min_col_coverage > 0.562895536423
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.563696861267
                if ( mean_col_support <= 0.979264736176 ) {
                  return 0.0824426035503 < maxgini;
                }
                else {  // if mean_col_support > 0.979264736176
                  return 0.015872 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.630017995834
              if ( min_col_coverage <= 0.606601715088 ) {
                if ( max_col_coverage <= 0.708914399147 ) {
                  return 0.0395965020065 < maxgini;
                }
                else {  // if max_col_coverage > 0.708914399147
                  return 0.158790170132 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.606601715088
                if ( min_col_coverage <= 0.635642170906 ) {
                  return 0.0225252363403 < maxgini;
                }
                else {  // if min_col_coverage > 0.635642170906
                  return 0.00531204731906 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.987735271454
            if ( min_col_support <= 0.825500011444 ) {
              if ( min_col_coverage <= 0.458505511284 ) {
                if ( max_col_coverage <= 0.581531405449 ) {
                  return 0.0 < maxgini;
                }
                else {  // if max_col_coverage > 0.581531405449
                  return 0.0776757369615 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.458505511284
                if ( median_col_coverage <= 0.608111262321 ) {
                  return 0.212619389913 < maxgini;
                }
                else {  // if median_col_coverage > 0.608111262321
                  return 0.0 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.825500011444
              if ( max_col_coverage <= 0.710609078407 ) {
                if ( mean_col_coverage <= 0.569523096085 ) {
                  return 0.00471387527285 < maxgini;
                }
                else {  // if mean_col_coverage > 0.569523096085
                  return 0.00355342275086 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.710609078407
                if ( median_col_coverage <= 0.586477994919 ) {
                  return 0.0 < maxgini;
                }
                else {  // if median_col_coverage > 0.586477994919
                  return false;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.710717797279
          if ( mean_col_coverage <= 0.995555818081 ) {
            if ( min_col_support <= 0.807500004768 ) {
              if ( min_col_coverage <= 0.600653648376 ) {
                if ( mean_col_coverage <= 0.618874669075 ) {
                  return 0.0689687793858 < maxgini;
                }
                else {  // if mean_col_coverage > 0.618874669075
                  return 0.176738496862 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.600653648376
                if ( mean_col_support <= 0.985970616341 ) {
                  return 0.260983253187 < maxgini;
                }
                else {  // if mean_col_support > 0.985970616341
                  return 0.465868154986 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.807500004768
              if ( mean_col_support <= 0.990382373333 ) {
                if ( median_col_coverage <= 0.913188457489 ) {
                  return 0.0292455759583 < maxgini;
                }
                else {  // if median_col_coverage > 0.913188457489
                  return 0.0846715828064 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.990382373333
                if ( median_col_coverage <= 0.609143137932 ) {
                  return 0.00299159212179 < maxgini;
                }
                else {  // if median_col_coverage > 0.609143137932
                  return 0.00186856817543 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.995555818081
            if ( min_col_support <= 0.851500034332 ) {
              if ( max_col_coverage <= 0.998618781567 ) {
                if ( mean_col_coverage <= 0.996517896652 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_coverage > 0.996517896652
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.998618781567
                if ( min_col_coverage <= 0.980002641678 ) {
                  return 0.18325697626 < maxgini;
                }
                else {  // if min_col_coverage > 0.980002641678
                  return 0.419565808124 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.851500034332
              if ( median_col_coverage <= 0.981306791306 ) {
                if ( mean_col_support <= 0.99426472187 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.99426472187
                  return 0.0 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.981306791306
                if ( median_col_coverage <= 0.99756193161 ) {
                  return 0.0624349635796 < maxgini;
                }
                else {  // if median_col_coverage > 0.99756193161
                  return 0.0158485860777 < maxgini;
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
  if ( min_col_support <= 0.770500004292 ) {
    if ( median_col_coverage <= 0.501316070557 ) {
      if ( max_col_coverage <= 0.454697489738 ) {
        if ( median_col_coverage <= 0.20843206346 ) {
          if ( mean_col_support <= 0.918029427528 ) {
            if ( min_col_support <= 0.556499958038 ) {
              if ( min_col_support <= 0.490500003099 ) {
                if ( max_col_coverage <= 0.210101872683 ) {
                  return 0.183890541344 < maxgini;
                }
                else {  // if max_col_coverage > 0.210101872683
                  return 0.278869225107 < maxgini;
                }
              }
              else {  // if min_col_support > 0.490500003099
                if ( median_col_coverage <= 0.178791880608 ) {
                  return 0.400455565986 < maxgini;
                }
                else {  // if median_col_coverage > 0.178791880608
                  return 0.439634301214 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.556499958038
              if ( mean_col_support <= 0.875710129738 ) {
                if ( mean_col_support <= 0.857250988483 ) {
                  return 0.427239191378 < maxgini;
                }
                else {  // if mean_col_support > 0.857250988483
                  return 0.369693709472 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.875710129738
                if ( min_col_support <= 0.574499964714 ) {
                  return 0.239941629758 < maxgini;
                }
                else {  // if min_col_support > 0.574499964714
                  return 0.307493995946 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.918029427528
            if ( median_col_support <= 0.814499974251 ) {
              if ( mean_col_coverage <= 0.215777903795 ) {
                if ( min_col_coverage <= 0.0470653399825 ) {
                  return 0.312444459128 < maxgini;
                }
                else {  // if min_col_coverage > 0.0470653399825
                  return 0.189688089968 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.215777903795
                if ( median_col_coverage <= 0.0960061401129 ) {
                  return 0.431452881027 < maxgini;
                }
                else {  // if median_col_coverage > 0.0960061401129
                  return 0.270241095627 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.814499974251
              if ( min_col_coverage <= 0.0598507449031 ) {
                if ( mean_col_support <= 0.94106066227 ) {
                  return 0.22029330543 < maxgini;
                }
                else {  // if mean_col_support > 0.94106066227
                  return 0.103520190408 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0598507449031
                if ( mean_col_coverage <= 0.218908622861 ) {
                  return 0.0766475157952 < maxgini;
                }
                else {  // if mean_col_coverage > 0.218908622861
                  return 0.11720706804 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.20843206346
          if ( mean_col_coverage <= 0.297182261944 ) {
            if ( mean_col_support <= 0.881558775902 ) {
              if ( median_col_support <= 0.65649998188 ) {
                if ( median_col_support <= 0.574499964714 ) {
                  return 0.498793579437 < maxgini;
                }
                else {  // if median_col_support > 0.574499964714
                  return 0.451524872643 < maxgini;
                }
              }
              else {  // if median_col_support > 0.65649998188
                if ( min_col_coverage <= 0.226539582014 ) {
                  return 0.361083411163 < maxgini;
                }
                else {  // if min_col_coverage > 0.226539582014
                  return 0.24795 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.881558775902
              if ( mean_col_support <= 0.93132352829 ) {
                if ( mean_col_coverage <= 0.266045570374 ) {
                  return 0.219067169414 < maxgini;
                }
                else {  // if mean_col_coverage > 0.266045570374
                  return 0.285563424486 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.93132352829
                if ( min_col_support <= 0.599500000477 ) {
                  return 0.379353098521 < maxgini;
                }
                else {  // if min_col_support > 0.599500000477
                  return 0.134159063519 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.297182261944
            if ( median_col_coverage <= 0.250356137753 ) {
              if ( mean_col_support <= 0.88691174984 ) {
                if ( median_col_coverage <= 0.213203459978 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.213203459978
                  return 0.4661324338 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.88691174984
                if ( mean_col_support <= 0.925499975681 ) {
                  return 0.32008041997 < maxgini;
                }
                else {  // if mean_col_support > 0.925499975681
                  return 0.184684830476 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.250356137753
              if ( mean_col_support <= 0.876147031784 ) {
                if ( max_col_coverage <= 0.454378962517 ) {
                  return 0.477203690658 < maxgini;
                }
                else {  // if max_col_coverage > 0.454378962517
                  return 0.433385385267 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.876147031784
                if ( median_col_coverage <= 0.256912589073 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.256912589073
                  return 0.3108377975 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if max_col_coverage > 0.454697489738
        if ( median_col_support <= 0.976500034332 ) {
          if ( max_col_coverage <= 0.550167798996 ) {
            if ( mean_col_coverage <= 0.373829722404 ) {
              if ( min_col_coverage <= 0.051003344357 ) {
                if ( median_col_support <= 0.794499993324 ) {
                  return 0.467760065295 < maxgini;
                }
                else {  // if median_col_support > 0.794499993324
                  return 0.233169623275 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.051003344357
                if ( max_col_support <= 0.981000006199 ) {
                  return false;
                }
                else {  // if max_col_support > 0.981000006199
                  return 0.302401111719 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.373829722404
              if ( median_col_support <= 0.711500048637 ) {
                if ( median_col_support <= 0.620499968529 ) {
                  return false;
                }
                else {  // if median_col_support > 0.620499968529
                  return 0.489044104917 < maxgini;
                }
              }
              else {  // if median_col_support > 0.711500048637
                if ( max_col_coverage <= 0.549827337265 ) {
                  return 0.320172378134 < maxgini;
                }
                else {  // if max_col_coverage > 0.549827337265
                  return 0.188507302945 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.550167798996
            if ( mean_col_support <= 0.877205848694 ) {
              if ( median_col_support <= 0.617499947548 ) {
                if ( min_col_coverage <= 0.321825385094 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.321825385094
                  return false;
                }
              }
              else {  // if median_col_support > 0.617499947548
                if ( mean_col_support <= 0.837323606014 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.837323606014
                  return 0.471198483153 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.877205848694
              if ( mean_col_coverage <= 0.457613110542 ) {
                if ( median_col_support <= 0.795500040054 ) {
                  return 0.384041330371 < maxgini;
                }
                else {  // if median_col_support > 0.795500040054
                  return 0.187428832074 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.457613110542
                if ( mean_col_support <= 0.929615557194 ) {
                  return 0.422385708415 < maxgini;
                }
                else {  // if mean_col_support > 0.929615557194
                  return 0.303802412192 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.976500034332
          if ( median_col_coverage <= 0.304458171129 ) {
            if ( mean_col_support <= 0.946441173553 ) {
              if ( mean_col_coverage <= 0.305078983307 ) {
                if ( min_col_support <= 0.611500024796 ) {
                  return 0.270344331888 < maxgini;
                }
                else {  // if min_col_support > 0.611500024796
                  return 0.47896120973 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.305078983307
                if ( median_col_support <= 0.99849998951 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99849998951
                  return 0.468913826717 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.946441173553
              if ( mean_col_coverage <= 0.331867218018 ) {
                if ( median_col_coverage <= 0.0621141977608 ) {
                  return 0.20879501385 < maxgini;
                }
                else {  // if median_col_coverage > 0.0621141977608
                  return 0.0829391548313 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.331867218018
                if ( min_col_support <= 0.652500033379 ) {
                  return 0.42498541332 < maxgini;
                }
                else {  // if min_col_support > 0.652500033379
                  return 0.147516909932 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.304458171129
            if ( mean_col_coverage <= 0.448203980923 ) {
              if ( min_col_support <= 0.679499983788 ) {
                if ( max_col_coverage <= 0.545101642609 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.545101642609
                  return 0.499707336108 < maxgini;
                }
              }
              else {  // if min_col_support > 0.679499983788
                if ( min_col_support <= 0.727499961853 ) {
                  return 0.436769491918 < maxgini;
                }
                else {  // if min_col_support > 0.727499961853
                  return 0.261601387614 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.448203980923
              if ( min_col_coverage <= 0.35141825676 ) {
                if ( mean_col_support <= 0.947441160679 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.947441160679
                  return 0.441815870085 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.35141825676
                if ( median_col_support <= 0.99849998951 ) {
                  return false;
                }
                else {  // if median_col_support > 0.99849998951
                  return 0.498740406983 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if median_col_coverage > 0.501316070557
      if ( median_col_coverage <= 0.667019784451 ) {
        if ( median_col_coverage <= 0.600200772285 ) {
          if ( min_col_support <= 0.672500014305 ) {
            if ( median_col_support <= 0.984500050545 ) {
              if ( median_col_support <= 0.669499993324 ) {
                if ( max_col_coverage <= 0.768874645233 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.768874645233
                  return false;
                }
              }
              else {  // if median_col_support > 0.669499993324
                if ( min_col_coverage <= 0.455193459988 ) {
                  return 0.399732453605 < maxgini;
                }
                else {  // if min_col_coverage > 0.455193459988
                  return 0.49336436998 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.984500050545
              if ( min_col_support <= 0.611500024796 ) {
                if ( max_col_coverage <= 0.630820870399 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.630820870399
                  return false;
                }
              }
              else {  // if min_col_support > 0.611500024796
                if ( mean_col_coverage <= 0.66897046566 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.66897046566
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.672500014305
            if ( mean_col_support <= 0.964617609978 ) {
              if ( mean_col_coverage <= 0.664608955383 ) {
                if ( mean_col_coverage <= 0.612260580063 ) {
                  return 0.379525267752 < maxgini;
                }
                else {  // if mean_col_coverage > 0.612260580063
                  return 0.416049371988 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.664608955383
                if ( mean_col_support <= 0.894441127777 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.894441127777
                  return 0.278839770152 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.964617609978
              if ( median_col_support <= 0.986500024796 ) {
                if ( median_col_support <= 0.980499982834 ) {
                  return 0.306649842276 < maxgini;
                }
                else {  // if median_col_support > 0.980499982834
                  return 0.441406061158 < maxgini;
                }
              }
              else {  // if median_col_support > 0.986500024796
                if ( mean_col_coverage <= 0.652631521225 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.652631521225
                  return 0.456974563986 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.600200772285
          if ( min_col_support <= 0.6875 ) {
            if ( mean_col_coverage <= 0.642436146736 ) {
              if ( min_col_coverage <= 0.525431036949 ) {
                if ( mean_col_support <= 0.846029400826 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.846029400826
                  return 0.329534705644 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.525431036949
                if ( max_col_coverage <= 0.668675363064 ) {
                  return 0.490464852608 < maxgini;
                }
                else {  // if max_col_coverage > 0.668675363064
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.642436146736
              if ( mean_col_support <= 0.933676481247 ) {
                if ( median_col_support <= 0.934499979019 ) {
                  return false;
                }
                else {  // if median_col_support > 0.934499979019
                  return false;
                }
              }
              else {  // if mean_col_support > 0.933676481247
                if ( min_col_coverage <= 0.665703296661 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.665703296661
                  return 0.468391836735 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.6875
            if ( max_col_coverage <= 0.709733843803 ) {
              if ( median_col_coverage <= 0.602524280548 ) {
                if ( mean_col_coverage <= 0.63265645504 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.63265645504
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.602524280548
                if ( median_col_support <= 0.984500050545 ) {
                  return 0.224377022088 < maxgini;
                }
                else {  // if median_col_support > 0.984500050545
                  return 0.414982121009 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.709733843803
              if ( median_col_coverage <= 0.666266024113 ) {
                if ( mean_col_support <= 0.963911771774 ) {
                  return 0.421969474187 < maxgini;
                }
                else {  // if mean_col_support > 0.963911771774
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.666266024113
                if ( median_col_support <= 0.986500024796 ) {
                  return 0.223249927421 < maxgini;
                }
                else {  // if median_col_support > 0.986500024796
                  return 0.47096821019 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_coverage > 0.667019784451
        if ( max_col_coverage <= 0.785839557648 ) {
          if ( max_col_coverage <= 0.750432491302 ) {
            if ( mean_col_coverage <= 0.728945910931 ) {
              if ( min_col_support <= 0.680500030518 ) {
                if ( median_col_support <= 0.989500045776 ) {
                  return 0.489308917764 < maxgini;
                }
                else {  // if median_col_support > 0.989500045776
                  return false;
                }
              }
              else {  // if min_col_support > 0.680500030518
                if ( median_col_coverage <= 0.670808911324 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.670808911324
                  return 0.285374554102 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.728945910931
              if ( max_col_coverage <= 0.748983740807 ) {
                if ( min_col_support <= 0.652999997139 ) {
                  return false;
                }
                else {  // if min_col_support > 0.652999997139
                  return 0.165289256198 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.748983740807
                if ( mean_col_coverage <= 0.748284339905 ) {
                  return 0.110726643599 < maxgini;
                }
                else {  // if mean_col_coverage > 0.748284339905
                  return 0.438276113952 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.750432491302
            if ( min_col_support <= 0.660500049591 ) {
              if ( mean_col_support <= 0.912529408932 ) {
                if ( max_col_coverage <= 0.782329976559 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.782329976559
                  return 0.45848893996 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.912529408932
                if ( mean_col_coverage <= 0.733882308006 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.733882308006
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.660500049591
              if ( min_col_coverage <= 0.679249763489 ) {
                if ( min_col_support <= 0.74849998951 ) {
                  return false;
                }
                else {  // if min_col_support > 0.74849998951
                  return 0.437716262976 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.679249763489
                if ( mean_col_support <= 0.892470598221 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.892470598221
                  return 0.380097498248 < maxgini;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.785839557648
          if ( mean_col_coverage <= 0.991819739342 ) {
            if ( median_col_support <= 0.988499999046 ) {
              if ( median_col_coverage <= 0.809620201588 ) {
                if ( min_col_support <= 0.663499951363 ) {
                  return false;
                }
                else {  // if min_col_support > 0.663499951363
                  return 0.461183946349 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.809620201588
                if ( mean_col_support <= 0.970911800861 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.970911800861
                  return 0.499922987502 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.988499999046
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_support <= 0.994500041008 ) {
                  return false;
                }
                else {  // if median_col_support > 0.994500041008
                  return false;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_coverage <= 0.863972783089 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.863972783089
                  return false;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.991819739342
            if ( median_col_support <= 0.976500034332 ) {
              if ( mean_col_support <= 0.94679415226 ) {
                if ( min_col_support <= 0.633499979973 ) {
                  return false;
                }
                else {  // if min_col_support > 0.633499979973
                  return false;
                }
              }
              else {  // if mean_col_support > 0.94679415226
                if ( mean_col_coverage <= 0.992103695869 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_coverage > 0.992103695869
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.976500034332
              if ( mean_col_coverage <= 0.999897122383 ) {
                if ( median_col_coverage <= 0.977525234222 ) {
                  return 0.491217883583 < maxgini;
                }
                else {  // if median_col_coverage > 0.977525234222
                  return false;
                }
              }
              else {  // if mean_col_coverage > 0.999897122383
                if ( mean_col_support <= 0.951147079468 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.951147079468
                  return false;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if min_col_support > 0.770500004292
    if ( median_col_coverage <= 0.410083293915 ) {
      if ( median_col_support <= 0.904500007629 ) {
        if ( mean_col_support <= 0.939382314682 ) {
          if ( min_col_support <= 0.848500013351 ) {
            if ( mean_col_coverage <= 0.387066364288 ) {
              if ( min_col_coverage <= 0.225148111582 ) {
                if ( min_col_support <= 0.780499994755 ) {
                  return 0.232336652703 < maxgini;
                }
                else {  // if min_col_support > 0.780499994755
                  return 0.376601363118 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.225148111582
                if ( min_col_coverage <= 0.228553920984 ) {
                  return 0.138352694558 < maxgini;
                }
                else {  // if min_col_coverage > 0.228553920984
                  return 0.249578310745 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.387066364288
              if ( min_col_support <= 0.772500038147 ) {
                if ( max_col_coverage <= 0.465476185083 ) {
                  return 0.486992715921 < maxgini;
                }
                else {  // if max_col_coverage > 0.465476185083
                  return 0.175608133702 < maxgini;
                }
              }
              else {  // if min_col_support > 0.772500038147
                if ( min_col_coverage <= 0.209429830313 ) {
                  return 0.40650109569 < maxgini;
                }
                else {  // if min_col_coverage > 0.209429830313
                  return 0.342297748146 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.848500013351
            if ( mean_col_support <= 0.918411850929 ) {
              return false;
            }
            else {  // if mean_col_support > 0.918411850929
              if ( max_col_coverage <= 0.492424249649 ) {
                if ( median_col_support <= 0.852499961853 ) {
                  return 0.455 < maxgini;
                }
                else {  // if median_col_support > 0.852499961853
                  return 0.221118410511 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.492424249649
                if ( mean_col_coverage <= 0.424843072891 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.424843072891
                  return 0.42 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.939382314682
          if ( median_col_coverage <= 0.0437171533704 ) {
            if ( mean_col_coverage <= 0.365888744593 ) {
              if ( mean_col_coverage <= 0.0990196093917 ) {
                if ( median_col_support <= 0.865999996662 ) {
                  return 0.301783264746 < maxgini;
                }
                else {  // if median_col_support > 0.865999996662
                  return 0.0907029478458 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.0990196093917
                if ( max_col_coverage <= 0.158947363496 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.158947363496
                  return 0.311019091886 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.365888744593
              return false;
            }
          }
          else {  // if median_col_coverage > 0.0437171533704
            if ( min_col_support <= 0.783499956131 ) {
              if ( max_col_coverage <= 0.292408883572 ) {
                if ( mean_col_support <= 0.949617743492 ) {
                  return 0.0208310249307 < maxgini;
                }
                else {  // if mean_col_support > 0.949617743492
                  return 0.149333866074 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.292408883572
                if ( max_col_coverage <= 0.299253731966 ) {
                  return 0.408163265306 < maxgini;
                }
                else {  // if max_col_coverage > 0.299253731966
                  return 0.19750709919 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.783499956131
              if ( median_col_support <= 0.849500000477 ) {
                if ( max_col_coverage <= 0.413418292999 ) {
                  return 0.20055140696 < maxgini;
                }
                else {  // if max_col_coverage > 0.413418292999
                  return 0.255986245225 < maxgini;
                }
              }
              else {  // if median_col_support > 0.849500000477
                if ( min_col_coverage <= 0.00674725323915 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.00674725323915
                  return 0.138332474823 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.904500007629
        if ( min_col_coverage <= 0.051787994802 ) {
          if ( mean_col_coverage <= 0.537477374077 ) {
            if ( mean_col_support <= 0.97471010685 ) {
              if ( mean_col_support <= 0.949072897434 ) {
                if ( mean_col_support <= 0.941916584969 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.941916584969
                  return 0.489795918367 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.949072897434
                if ( max_col_coverage <= 0.0358272120357 ) {
                  return 0.493827160494 < maxgini;
                }
                else {  // if max_col_coverage > 0.0358272120357
                  return 0.110204514549 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.97471010685
              if ( mean_col_coverage <= 0.428701817989 ) {
                if ( mean_col_support <= 0.983121335506 ) {
                  return 0.0489382234329 < maxgini;
                }
                else {  // if mean_col_support > 0.983121335506
                  return 0.0235107150237 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.428701817989
                if ( median_col_coverage <= 0.0493902415037 ) {
                  return 0.0235260770975 < maxgini;
                }
                else {  // if median_col_coverage > 0.0493902415037
                  return 0.202448979592 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.537477374077
            if ( mean_col_coverage <= 0.540011048317 ) {
              return false;
            }
            else {  // if mean_col_coverage > 0.540011048317
              if ( min_col_support <= 0.787500023842 ) {
                if ( mean_col_support <= 0.978519558907 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.978519558907
                  return 0.0 < maxgini;
                }
              }
              else {  // if min_col_support > 0.787500023842
                if ( mean_col_support <= 0.985443115234 ) {
                  return 0.349226502723 < maxgini;
                }
                else {  // if mean_col_support > 0.985443115234
                  return 0.0831758034026 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.051787994802
          if ( min_col_support <= 0.887500047684 ) {
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_coverage <= 0.409156382084 ) {
                if ( median_col_support <= 0.990499973297 ) {
                  return 0.0680139942352 < maxgini;
                }
                else {  // if median_col_support > 0.990499973297
                  return 0.480695598646 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.409156382084
                if ( median_col_coverage <= 0.409684538841 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.409684538841
                  return 0.226843100189 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( median_col_coverage <= 0.305409371853 ) {
                if ( min_col_coverage <= 0.240155041218 ) {
                  return 0.0259601356151 < maxgini;
                }
                else {  // if min_col_coverage > 0.240155041218
                  return 0.0331613663121 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.305409371853
                if ( min_col_support <= 0.839499950409 ) {
                  return 0.0639255171531 < maxgini;
                }
                else {  // if min_col_support > 0.839499950409
                  return 0.0299176571665 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.887500047684
            if ( min_col_coverage <= 0.280056029558 ) {
              if ( mean_col_support <= 0.985970616341 ) {
                if ( median_col_support <= 0.946500003338 ) {
                  return 0.116476430033 < maxgini;
                }
                else {  // if median_col_support > 0.946500003338
                  return 0.0495433671462 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.985970616341
                if ( max_col_coverage <= 0.996763765812 ) {
                  return 0.0116066881205 < maxgini;
                }
                else {  // if max_col_coverage > 0.996763765812
                  return 0.102264426589 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.280056029558
              if ( max_col_coverage <= 0.609823107719 ) {
                if ( median_col_support <= 0.959499955177 ) {
                  return 0.0805729091856 < maxgini;
                }
                else {  // if median_col_support > 0.959499955177
                  return 0.0104385751803 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.609823107719
                if ( min_col_support <= 0.960500001907 ) {
                  return 0.0137761991182 < maxgini;
                }
                else {  // if min_col_support > 0.960500001907
                  return 0.00230069866247 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if median_col_coverage > 0.410083293915
      if ( min_col_support <= 0.845499992371 ) {
        if ( median_col_coverage <= 0.714519917965 ) {
          if ( median_col_coverage <= 0.600314497948 ) {
            if ( median_col_support <= 0.857499957085 ) {
              if ( min_col_coverage <= 0.196153849363 ) {
                if ( min_col_support <= 0.791000008583 ) {
                  return false;
                }
                else {  // if min_col_support > 0.791000008583
                  return 0.48 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.196153849363
                if ( max_col_coverage <= 0.534523844719 ) {
                  return 0.392534820409 < maxgini;
                }
                else {  // if max_col_coverage > 0.534523844719
                  return 0.352680695036 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.857499957085
              if ( max_col_coverage <= 0.608817100525 ) {
                if ( mean_col_support <= 0.939029455185 ) {
                  return 0.351913554537 < maxgini;
                }
                else {  // if mean_col_support > 0.939029455185
                  return 0.115698510586 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.608817100525
                if ( median_col_coverage <= 0.50126850605 ) {
                  return 0.139915658042 < maxgini;
                }
                else {  // if median_col_coverage > 0.50126850605
                  return 0.216001918678 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.600314497948
            if ( mean_col_coverage <= 0.657627761364 ) {
              if ( max_col_coverage <= 0.692559123039 ) {
                if ( median_col_coverage <= 0.617914438248 ) {
                  return 0.153673257024 < maxgini;
                }
                else {  // if median_col_coverage > 0.617914438248
                  return 0.0674281573578 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.692559123039
                if ( median_col_coverage <= 0.605047821999 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.605047821999
                  return 0.224276716042 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.657627761364
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_coverage <= 0.714065909386 ) {
                  return 0.395583764586 < maxgini;
                }
                else {  // if median_col_coverage > 0.714065909386
                  return 0.175976288204 < maxgini;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_coverage <= 0.255411267281 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.255411267281
                  return 0.15090709342 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.714519917965
          if ( mean_col_support <= 0.975852966309 ) {
            if ( median_col_support <= 0.986500024796 ) {
              if ( mean_col_support <= 0.908088326454 ) {
                if ( min_col_coverage <= 0.852089285851 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.852089285851
                  return false;
                }
              }
              else {  // if mean_col_support > 0.908088326454
                if ( median_col_coverage <= 0.864635825157 ) {
                  return 0.288008517986 < maxgini;
                }
                else {  // if median_col_coverage > 0.864635825157
                  return 0.468863409451 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.986500024796
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_support <= 0.992499947548 ) {
                  return 0.497225463569 < maxgini;
                }
                else {  // if median_col_support > 0.992499947548
                  return false;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( median_col_coverage <= 0.920761942863 ) {
                  return 0.217346909136 < maxgini;
                }
                else {  // if median_col_coverage > 0.920761942863
                  return 0.467911111111 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.975852966309
            if ( median_col_support <= 0.99950003624 ) {
              if ( median_col_coverage <= 0.90924346447 ) {
                if ( mean_col_support <= 0.985676407814 ) {
                  return 0.488707020096 < maxgini;
                }
                else {  // if mean_col_support > 0.985676407814
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.90924346447
                if ( mean_col_support <= 0.985617637634 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.985617637634
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.99950003624
              if ( min_col_coverage <= 0.90969979763 ) {
                if ( median_col_coverage <= 0.718682348728 ) {
                  return 0.497256515775 < maxgini;
                }
                else {  // if median_col_coverage > 0.718682348728
                  return 0.215375282034 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.90969979763
                if ( median_col_coverage <= 0.998618781567 ) {
                  return 0.436639054298 < maxgini;
                }
                else {  // if median_col_coverage > 0.998618781567
                  return 0.326312381012 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.845499992371
        if ( median_col_support <= 0.99950003624 ) {
          if ( min_col_coverage <= 0.980066239834 ) {
            if ( median_col_support <= 0.995499968529 ) {
              if ( mean_col_support <= 0.985757350922 ) {
                if ( median_col_support <= 0.990499973297 ) {
                  return 0.0770070147476 < maxgini;
                }
                else {  // if median_col_support > 0.990499973297
                  return 0.494906689415 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.985757350922
                if ( max_col_support <= 0.99950003624 ) {
                  return 0.48 < maxgini;
                }
                else {  // if max_col_support > 0.99950003624
                  return 0.0134299533573 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.995499968529
              if ( max_col_coverage <= 0.797441065311 ) {
                if ( median_col_support <= 0.99849998951 ) {
                  return 0.184046520394 < maxgini;
                }
                else {  // if median_col_support > 0.99849998951
                  return 0.39756232687 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.797441065311
                if ( median_col_support <= 0.997500002384 ) {
                  return 0.200893991456 < maxgini;
                }
                else {  // if median_col_support > 0.997500002384
                  return 0.334784178686 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.980066239834
            if ( min_col_coverage <= 0.9809705019 ) {
              if ( mean_col_support <= 0.991588234901 ) {
                if ( median_col_coverage <= 0.980468451977 ) {
                  return 0.489795918367 < maxgini;
                }
                else {  // if median_col_coverage > 0.980468451977
                  return false;
                }
              }
              else {  // if mean_col_support > 0.991588234901
                if ( median_col_support <= 0.982500016689 ) {
                  return false;
                }
                else {  // if median_col_support > 0.982500016689
                  return 0.0 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.9809705019
              if ( median_col_support <= 0.949499964714 ) {
                if ( mean_col_coverage <= 0.999901294708 ) {
                  return 0.498866213152 < maxgini;
                }
                else {  // if mean_col_coverage > 0.999901294708
                  return 0.116974263686 < maxgini;
                }
              }
              else {  // if median_col_support > 0.949499964714
                if ( median_col_support <= 0.981500029564 ) {
                  return 0.381210197381 < maxgini;
                }
                else {  // if median_col_support > 0.981500029564
                  return 0.312565103885 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.99950003624
          if ( mean_col_support <= 0.984088242054 ) {
            if ( max_col_coverage <= 0.789315164089 ) {
              if ( mean_col_coverage <= 0.463141024113 ) {
                if ( mean_col_support <= 0.974323511124 ) {
                  return 0.336734693878 < maxgini;
                }
                else {  // if mean_col_support > 0.974323511124
                  return 0.0414127218935 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.463141024113
                if ( min_col_support <= 0.871500015259 ) {
                  return 0.0766283319078 < maxgini;
                }
                else {  // if min_col_support > 0.871500015259
                  return 0.120463961316 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.789315164089
              if ( min_col_coverage <= 0.939209163189 ) {
                if ( max_col_coverage <= 0.949576258659 ) {
                  return 0.0467083398671 < maxgini;
                }
                else {  // if max_col_coverage > 0.949576258659
                  return 0.0221437309753 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.939209163189
                if ( min_col_coverage <= 0.943376064301 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.943376064301
                  return 0.0907029478458 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.984088242054
            if ( min_col_support <= 0.920500040054 ) {
              if ( mean_col_support <= 0.991147041321 ) {
                if ( min_col_coverage <= 0.961032390594 ) {
                  return 0.0184383400555 < maxgini;
                }
                else {  // if min_col_coverage > 0.961032390594
                  return 0.0891271701353 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.991147041321
                if ( min_col_support <= 0.87650001049 ) {
                  return 0.01728756166 < maxgini;
                }
                else {  // if min_col_support > 0.87650001049
                  return 0.0042208674395 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.920500040054
              if ( mean_col_support <= 0.993323564529 ) {
                if ( median_col_coverage <= 0.56840467453 ) {
                  return 0.0140677461687 < maxgini;
                }
                else {  // if median_col_coverage > 0.56840467453
                  return 0.00595225316701 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.993323564529
                if ( median_col_coverage <= 0.607215166092 ) {
                  return 0.00238523163839 < maxgini;
                }
                else {  // if median_col_coverage > 0.607215166092
                  return 0.00140437642464 < maxgini;
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
  if ( median_col_coverage <= 0.255219221115 ) {
    if ( mean_col_support <= 0.923710107803 ) {
      if ( mean_col_support <= 0.855710148811 ) {
        if ( max_col_coverage <= 0.276969730854 ) {
          if ( mean_col_coverage <= 0.155946299434 ) {
            if ( median_col_support <= 0.574499964714 ) {
              if ( max_col_coverage <= 0.21942153573 ) {
                if ( min_col_support <= 0.503499984741 ) {
                  return 0.1996875 < maxgini;
                }
                else {  // if min_col_support > 0.503499984741
                  return 0.361382757185 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.21942153573
                if ( median_col_coverage <= 0.0493902415037 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.0493902415037
                  return 0.412253469686 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.574499964714
              if ( min_col_coverage <= 0.0602418743074 ) {
                if ( min_col_support <= 0.515499949455 ) {
                  return 0.265381000904 < maxgini;
                }
                else {  // if min_col_support > 0.515499949455
                  return 0.398648164452 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0602418743074
                if ( max_col_coverage <= 0.137147337198 ) {
                  return 0.32 < maxgini;
                }
                else {  // if max_col_coverage > 0.137147337198
                  return 0.143020251187 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.155946299434
            if ( median_col_support <= 0.59350001812 ) {
              if ( min_col_support <= 0.511500000954 ) {
                if ( min_col_coverage <= 0.0444664061069 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.0444664061069
                  return 0.385879949634 < maxgini;
                }
              }
              else {  // if min_col_support > 0.511500000954
                if ( min_col_support <= 0.516499996185 ) {
                  return false;
                }
                else {  // if min_col_support > 0.516499996185
                  return 0.471106112532 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.59350001812
              if ( min_col_support <= 0.512500047684 ) {
                if ( median_col_support <= 0.759999990463 ) {
                  return 0.190405804891 < maxgini;
                }
                else {  // if median_col_support > 0.759999990463
                  return false;
                }
              }
              else {  // if min_col_support > 0.512500047684
                if ( min_col_support <= 0.634000003338 ) {
                  return 0.415541651819 < maxgini;
                }
                else {  // if min_col_support > 0.634000003338
                  return false;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.276969730854
          if ( mean_col_coverage <= 0.324408888817 ) {
            if ( mean_col_support <= 0.833615601063 ) {
              if ( min_col_support <= 0.490500003099 ) {
                if ( mean_col_support <= 0.763088226318 ) {
                  return 0.45580715851 < maxgini;
                }
                else {  // if mean_col_support > 0.763088226318
                  return 0.329771904722 < maxgini;
                }
              }
              else {  // if min_col_support > 0.490500003099
                if ( median_col_coverage <= 0.201307192445 ) {
                  return 0.472472156327 < maxgini;
                }
                else {  // if median_col_coverage > 0.201307192445
                  return 0.486931872917 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.833615601063
              if ( min_col_support <= 0.492500007153 ) {
                if ( min_col_support <= 0.426499992609 ) {
                  return 0.364556615711 < maxgini;
                }
                else {  // if min_col_support > 0.426499992609
                  return 0.235685079251 < maxgini;
                }
              }
              else {  // if min_col_support > 0.492500007153
                if ( min_col_support <= 0.62650001049 ) {
                  return 0.458746426115 < maxgini;
                }
                else {  // if min_col_support > 0.62650001049
                  return 0.30796193975 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.324408888817
            if ( min_col_support <= 0.459500014782 ) {
              if ( mean_col_coverage <= 0.340664952993 ) {
                if ( median_col_support <= 0.519999980927 ) {
                  return false;
                }
                else {  // if median_col_support > 0.519999980927
                  return 0.0285654274312 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.340664952993
                if ( mean_col_support <= 0.849588215351 ) {
                  return 0.389955799421 < maxgini;
                }
                else {  // if mean_col_support > 0.849588215351
                  return 0.0 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.459500014782
              if ( min_col_support <= 0.581499993801 ) {
                if ( median_col_support <= 0.667500019073 ) {
                  return false;
                }
                else {  // if median_col_support > 0.667500019073
                  return 0.432282327434 < maxgini;
                }
              }
              else {  // if min_col_support > 0.581499993801
                if ( max_col_coverage <= 0.638181805611 ) {
                  return 0.44148199446 < maxgini;
                }
                else {  // if max_col_coverage > 0.638181805611
                  return false;
                }
              }
            }
          }
        }
      }
      else {  // if mean_col_support > 0.855710148811
        if ( mean_col_support <= 0.897355079651 ) {
          if ( min_col_coverage <= 0.0510647520423 ) {
            if ( median_col_support <= 0.761500000954 ) {
              if ( max_col_coverage <= 0.392080724239 ) {
                if ( max_col_coverage <= 0.24880951643 ) {
                  return 0.375276579183 < maxgini;
                }
                else {  // if max_col_coverage > 0.24880951643
                  return 0.414012511275 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.392080724239
                if ( mean_col_coverage <= 0.258027136326 ) {
                  return 0.451270170144 < maxgini;
                }
                else {  // if mean_col_coverage > 0.258027136326
                  return 0.495276443357 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.761500000954
              if ( max_col_coverage <= 0.274294674397 ) {
                if ( max_col_coverage <= 0.260064423084 ) {
                  return 0.238660975685 < maxgini;
                }
                else {  // if max_col_coverage > 0.260064423084
                  return 0.0246875 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.274294674397
                if ( min_col_coverage <= 0.0452186241746 ) {
                  return 0.298106072745 < maxgini;
                }
                else {  // if min_col_coverage > 0.0452186241746
                  return 0.426291989968 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.0510647520423
            if ( mean_col_coverage <= 0.260011970997 ) {
              if ( median_col_support <= 0.614500045776 ) {
                if ( mean_col_coverage <= 0.185407236218 ) {
                  return 0.265448294878 < maxgini;
                }
                else {  // if mean_col_coverage > 0.185407236218
                  return 0.473348948246 < maxgini;
                }
              }
              else {  // if median_col_support > 0.614500045776
                if ( min_col_support <= 0.621500015259 ) {
                  return 0.251938472955 < maxgini;
                }
                else {  // if min_col_support > 0.621500015259
                  return 0.319413203323 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.260011970997
              if ( mean_col_coverage <= 0.360788196325 ) {
                if ( min_col_support <= 0.482500016689 ) {
                  return 0.183890541344 < maxgini;
                }
                else {  // if min_col_support > 0.482500016689
                  return 0.391831618281 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.360788196325
                if ( mean_col_coverage <= 0.539705872536 ) {
                  return 0.44975277263 < maxgini;
                }
                else {  // if mean_col_coverage > 0.539705872536
                  return false;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.897355079651
          if ( mean_col_support <= 0.913730323315 ) {
            if ( median_col_coverage <= 0.0510647520423 ) {
              if ( median_col_support <= 0.725499987602 ) {
                if ( min_col_support <= 0.518499970436 ) {
                  return 0.480223461934 < maxgini;
                }
                else {  // if min_col_support > 0.518499970436
                  return 0.409042323332 < maxgini;
                }
              }
              else {  // if median_col_support > 0.725499987602
                if ( mean_col_support <= 0.912405848503 ) {
                  return 0.319678140431 < maxgini;
                }
                else {  // if mean_col_support > 0.912405848503
                  return 0.401064209275 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.0510647520423
              if ( median_col_support <= 0.726500034332 ) {
                if ( min_col_coverage <= 0.0521778613329 ) {
                  return 0.391833423472 < maxgini;
                }
                else {  // if min_col_coverage > 0.0521778613329
                  return 0.325980324207 < maxgini;
                }
              }
              else {  // if median_col_support > 0.726500034332
                if ( min_col_support <= 0.787500023842 ) {
                  return 0.240057838605 < maxgini;
                }
                else {  // if min_col_support > 0.787500023842
                  return false;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.913730323315
            if ( min_col_support <= 0.792500019073 ) {
              if ( mean_col_coverage <= 0.298733025789 ) {
                if ( median_col_support <= 0.756500005722 ) {
                  return 0.333112769675 < maxgini;
                }
                else {  // if median_col_support > 0.756500005722
                  return 0.21821120702 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.298733025789
                if ( min_col_coverage <= 0.00694179395214 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.00694179395214
                  return 0.313915737925 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.792500019073
              if ( min_col_support <= 0.805500030518 ) {
                if ( min_col_coverage <= 0.103571429849 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_coverage > 0.103571429849
                  return 0.494121062564 < maxgini;
                }
              }
              else {  // if min_col_support > 0.805500030518
                if ( min_col_support <= 0.837999999523 ) {
                  return false;
                }
                else {  // if min_col_support > 0.837999999523
                  return 0.444444444444 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if mean_col_support > 0.923710107803
      if ( min_col_support <= 0.713500022888 ) {
        if ( min_col_support <= 0.550500035286 ) {
          if ( min_col_coverage <= 0.150187969208 ) {
            if ( median_col_coverage <= 0.0478479862213 ) {
              if ( median_col_support <= 0.78149998188 ) {
                if ( mean_col_coverage <= 0.0959986895323 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_coverage > 0.0959986895323
                  return false;
                }
              }
              else {  // if median_col_support > 0.78149998188
                if ( max_col_coverage <= 0.233018875122 ) {
                  return 0.253149997987 < maxgini;
                }
                else {  // if max_col_coverage > 0.233018875122
                  return 0.332159464071 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.0478479862213
              if ( mean_col_support <= 0.928264737129 ) {
                if ( median_col_support <= 0.968000054359 ) {
                  return 0.203043375639 < maxgini;
                }
                else {  // if median_col_support > 0.968000054359
                  return 0.341563966493 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.928264737129
                if ( min_col_coverage <= 0.103807471693 ) {
                  return 0.139219966286 < maxgini;
                }
                else {  // if min_col_coverage > 0.103807471693
                  return 0.227841975309 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.150187969208
            if ( median_col_support <= 0.943500041962 ) {
              if ( mean_col_support <= 0.956941127777 ) {
                if ( median_col_coverage <= 0.250899285078 ) {
                  return 0.154838204082 < maxgini;
                }
                else {  // if median_col_coverage > 0.250899285078
                  return false;
                }
              }
              else {  // if mean_col_support > 0.956941127777
                if ( mean_col_coverage <= 0.289542794228 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.289542794228
                  return 0.0 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.943500041962
              if ( mean_col_support <= 0.969058811665 ) {
                if ( median_col_coverage <= 0.175199478865 ) {
                  return 0.228531855956 < maxgini;
                }
                else {  // if median_col_coverage > 0.175199478865
                  return 0.458425726303 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.969058811665
                if ( mean_col_coverage <= 0.281825065613 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.281825065613
                  return false;
                }
              }
            }
          }
        }
        else {  // if min_col_support > 0.550500035286
          if ( min_col_support <= 0.574499964714 ) {
            if ( mean_col_support <= 0.928970575333 ) {
              if ( median_col_coverage <= 0.0479059070349 ) {
                if ( mean_col_coverage <= 0.14635565877 ) {
                  return 0.227156915802 < maxgini;
                }
                else {  // if mean_col_coverage > 0.14635565877
                  return 0.37384527172 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.0479059070349
                if ( mean_col_support <= 0.928251028061 ) {
                  return 0.166677333655 < maxgini;
                }
                else {  // if mean_col_support > 0.928251028061
                  return 0.062826783606 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.928970575333
              if ( max_col_coverage <= 0.304491788149 ) {
                if ( mean_col_support <= 0.955420136452 ) {
                  return 0.0825507224767 < maxgini;
                }
                else {  // if mean_col_support > 0.955420136452
                  return 0.0463397985831 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.304491788149
                if ( median_col_support <= 0.947499990463 ) {
                  return 0.105459506403 < maxgini;
                }
                else {  // if median_col_support > 0.947499990463
                  return 0.181593157329 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.574499964714
            if ( median_col_coverage <= 0.250830829144 ) {
              if ( min_col_coverage <= 0.0452221557498 ) {
                if ( median_col_coverage <= 0.0436246544123 ) {
                  return 0.227902377188 < maxgini;
                }
                else {  // if median_col_coverage > 0.0436246544123
                  return 0.153023121153 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0452221557498
                if ( min_col_coverage <= 0.150187969208 ) {
                  return 0.132065980131 < maxgini;
                }
                else {  // if min_col_coverage > 0.150187969208
                  return 0.184868919996 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.250830829144
              if ( mean_col_coverage <= 0.523440897465 ) {
                if ( median_col_support <= 0.975000023842 ) {
                  return 0.478298611111 < maxgini;
                }
                else {  // if median_col_support > 0.975000023842
                  return false;
                }
              }
              else {  // if mean_col_coverage > 0.523440897465
                return 0.0 < maxgini;
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.713500022888
        if ( mean_col_support <= 0.962899565697 ) {
          if ( mean_col_support <= 0.947594165802 ) {
            if ( min_col_support <= 0.733500003815 ) {
              if ( median_col_coverage <= 0.045804195106 ) {
                if ( median_col_coverage <= 0.0084770116955 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.0084770116955
                  return 0.315720803179 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.045804195106
                if ( max_col_coverage <= 0.962962985039 ) {
                  return 0.149942877379 < maxgini;
                }
                else {  // if max_col_coverage > 0.962962985039
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.733500003815
              if ( min_col_coverage <= 0.226539582014 ) {
                if ( median_col_coverage <= 0.0635080635548 ) {
                  return 0.313802460093 < maxgini;
                }
                else {  // if median_col_coverage > 0.0635080635548
                  return 0.234974077768 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.226539582014
                if ( min_col_support <= 0.806499958038 ) {
                  return 0.148535216567 < maxgini;
                }
                else {  // if min_col_support > 0.806499958038
                  return 0.347299168975 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.947594165802
            if ( median_col_support <= 0.905499994755 ) {
              if ( max_col_coverage <= 0.990566015244 ) {
                if ( median_col_support <= 0.845499992371 ) {
                  return 0.185941033529 < maxgini;
                }
                else {  // if median_col_support > 0.845499992371
                  return 0.135901389871 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.990566015244
                if ( median_col_coverage <= 0.0239361710846 ) {
                  return 0.0 < maxgini;
                }
                else {  // if median_col_coverage > 0.0239361710846
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.905499994755
              if ( min_col_support <= 0.898999989033 ) {
                if ( mean_col_support <= 0.957303404808 ) {
                  return 0.101828785539 < maxgini;
                }
                else {  // if mean_col_support > 0.957303404808
                  return 0.070288925141 < maxgini;
                }
              }
              else {  // if min_col_support > 0.898999989033
                return false;
              }
            }
          }
        }
        else {  // if mean_col_support > 0.962899565697
          if ( mean_col_support <= 0.98465692997 ) {
            if ( max_col_coverage <= 0.86100178957 ) {
              if ( median_col_support <= 0.929499983788 ) {
                if ( min_col_support <= 0.879500031471 ) {
                  return 0.0933984197531 < maxgini;
                }
                else {  // if min_col_support > 0.879500031471
                  return 0.184516999942 < maxgini;
                }
              }
              else {  // if median_col_support > 0.929499983788
                if ( min_col_support <= 0.904500007629 ) {
                  return 0.0450585388595 < maxgini;
                }
                else {  // if min_col_support > 0.904500007629
                  return 0.115094175469 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.86100178957
              if ( min_col_coverage <= 0.0481997653842 ) {
                if ( mean_col_support <= 0.979638576508 ) {
                  return 0.408163265306 < maxgini;
                }
                else {  // if mean_col_support > 0.979638576508
                  return 0.137174211248 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0481997653842
                if ( min_col_support <= 0.786000013351 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_support > 0.786000013351
                  return 0.187573696145 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.98465692997
            if ( median_col_support <= 0.957499980927 ) {
              if ( min_col_coverage <= 0.189832687378 ) {
                if ( min_col_support <= 0.917500019073 ) {
                  return 0.119511090991 < maxgini;
                }
                else {  // if min_col_support > 0.917500019073
                  return 0.0205457310138 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.189832687378
                if ( min_col_coverage <= 0.190982788801 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.190982788801
                  return 0.29809022301 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.957499980927
              if ( min_col_coverage <= 0.0176214873791 ) {
                if ( min_col_coverage <= 0.0171921178699 ) {
                  return 0.0773490755179 < maxgini;
                }
                else {  // if min_col_coverage > 0.0171921178699
                  return 0.357777777778 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0176214873791
                if ( min_col_support <= 0.932500004768 ) {
                  return 0.0142565953946 < maxgini;
                }
                else {  // if min_col_support > 0.932500004768
                  return 0.00638467419815 < maxgini;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if median_col_coverage > 0.255219221115
    if ( min_col_coverage <= 0.928648591042 ) {
      if ( median_col_support <= 0.99950003624 ) {
        if ( max_col_coverage <= 0.800528407097 ) {
          if ( max_col_coverage <= 0.714624106884 ) {
            if ( median_col_coverage <= 0.599358797073 ) {
              if ( min_col_support <= 0.721500039101 ) {
                if ( min_col_coverage <= 0.350187540054 ) {
                  return 0.413655804708 < maxgini;
                }
                else {  // if min_col_coverage > 0.350187540054
                  return 0.499856641215 < maxgini;
                }
              }
              else {  // if min_col_support > 0.721500039101
                if ( median_col_support <= 0.853500008583 ) {
                  return 0.33378934298 < maxgini;
                }
                else {  // if median_col_support > 0.853500008583
                  return 0.0959864405141 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.599358797073
              if ( median_col_support <= 0.990499973297 ) {
                if ( min_col_coverage <= 0.459935903549 ) {
                  return 0.219193087761 < maxgini;
                }
                else {  // if min_col_coverage > 0.459935903549
                  return 0.102041056994 < maxgini;
                }
              }
              else {  // if median_col_support > 0.990499973297
                if ( median_col_support <= 0.994500041008 ) {
                  return 0.414054480674 < maxgini;
                }
                else {  // if median_col_support > 0.994500041008
                  return false;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.714624106884
            if ( min_col_coverage <= 0.500746250153 ) {
              if ( mean_col_support <= 0.959441184998 ) {
                if ( median_col_support <= 0.972499966621 ) {
                  return 0.368028343021 < maxgini;
                }
                else {  // if median_col_support > 0.972499966621
                  return false;
                }
              }
              else {  // if mean_col_support > 0.959441184998
                if ( median_col_support <= 0.992499947548 ) {
                  return 0.0700396926232 < maxgini;
                }
                else {  // if median_col_support > 0.992499947548
                  return 0.491025159422 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.500746250153
              if ( mean_col_support <= 0.980088233948 ) {
                if ( median_col_support <= 0.985499978065 ) {
                  return 0.356240497677 < maxgini;
                }
                else {  // if median_col_support > 0.985499978065
                  return false;
                }
              }
              else {  // if mean_col_support > 0.980088233948
                if ( min_col_support <= 0.853500008583 ) {
                  return 0.497830239859 < maxgini;
                }
                else {  // if min_col_support > 0.853500008583
                  return 0.0281719856259 < maxgini;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.800528407097
          if ( median_col_coverage <= 0.667033791542 ) {
            if ( mean_col_support <= 0.977125525475 ) {
              if ( max_col_coverage <= 0.806294500828 ) {
                if ( max_col_coverage <= 0.804829478264 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.804829478264
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.806294500828
                if ( median_col_support <= 0.981500029564 ) {
                  return 0.346462647225 < maxgini;
                }
                else {  // if median_col_support > 0.981500029564
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.977125525475
              if ( median_col_support <= 0.994500041008 ) {
                if ( mean_col_coverage <= 0.884135484695 ) {
                  return 0.0472392647379 < maxgini;
                }
                else {  // if mean_col_coverage > 0.884135484695
                  return false;
                }
              }
              else {  // if median_col_support > 0.994500041008
                if ( median_col_support <= 0.996500015259 ) {
                  return 0.4487817907 < maxgini;
                }
                else {  // if median_col_support > 0.996500015259
                  return 0.498214314059 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.667033791542
            if ( mean_col_support <= 0.982617616653 ) {
              if ( median_col_coverage <= 0.857312560081 ) {
                if ( min_col_support <= 0.769500017166 ) {
                  return false;
                }
                else {  // if min_col_support > 0.769500017166
                  return 0.26767388521 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.857312560081
                if ( min_col_coverage <= 0.80031645298 ) {
                  return 0.471092033834 < maxgini;
                }
                else {  // if min_col_coverage > 0.80031645298
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.982617616653
              if ( median_col_coverage <= 0.671583175659 ) {
                if ( median_col_support <= 0.997500002384 ) {
                  return 0.34875 < maxgini;
                }
                else {  // if median_col_support > 0.997500002384
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.671583175659
                if ( max_col_coverage <= 0.997819900513 ) {
                  return 0.116636399061 < maxgini;
                }
                else {  // if max_col_coverage > 0.997819900513
                  return 0.0604881858871 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.99950003624
        if ( mean_col_coverage <= 0.763345777988 ) {
          if ( min_col_support <= 0.701499998569 ) {
            if ( median_col_coverage <= 0.40040487051 ) {
              if ( mean_col_support <= 0.969911754131 ) {
                if ( mean_col_support <= 0.950147032738 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.950147032738
                  return 0.400749579754 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.969911754131
                if ( mean_col_support <= 0.975441217422 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.975441217422
                  return 0.489230037847 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.40040487051
              if ( min_col_support <= 0.614500045776 ) {
                if ( mean_col_support <= 0.975558757782 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.975558757782
                  return false;
                }
              }
              else {  // if min_col_support > 0.614500045776
                if ( mean_col_support <= 0.976970613003 ) {
                  return 0.496787774605 < maxgini;
                }
                else {  // if mean_col_support > 0.976970613003
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.701499998569
            if ( mean_col_support <= 0.985852956772 ) {
              if ( min_col_coverage <= 0.500692486763 ) {
                if ( mean_col_support <= 0.974147081375 ) {
                  return 0.167702722381 < maxgini;
                }
                else {  // if mean_col_support > 0.974147081375
                  return 0.0694019794666 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.500692486763
                if ( max_col_coverage <= 0.667953133583 ) {
                  return 0.102938935637 < maxgini;
                }
                else {  // if max_col_coverage > 0.667953133583
                  return 0.190745980193 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.985852956772
              if ( min_col_support <= 0.807500004768 ) {
                if ( min_col_coverage <= 0.458726406097 ) {
                  return 0.0735683560578 < maxgini;
                }
                else {  // if min_col_coverage > 0.458726406097
                  return 0.335463327053 < maxgini;
                }
              }
              else {  // if min_col_support > 0.807500004768
                if ( median_col_coverage <= 0.48429864645 ) {
                  return 0.00714856051472 < maxgini;
                }
                else {  // if median_col_coverage > 0.48429864645
                  return 0.00348728207924 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.763345777988
          if ( min_col_support <= 0.758499979973 ) {
            if ( min_col_support <= 0.678499996662 ) {
              if ( mean_col_support <= 0.974558830261 ) {
                if ( mean_col_coverage <= 0.97361010313 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.97361010313
                  return false;
                }
              }
              else {  // if mean_col_support > 0.974558830261
                if ( max_col_coverage <= 0.820071935654 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.820071935654
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.678499996662
              if ( min_col_support <= 0.712499976158 ) {
                if ( min_col_coverage <= 0.863857030869 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.863857030869
                  return false;
                }
              }
              else {  // if min_col_support > 0.712499976158
                if ( mean_col_support <= 0.982735276222 ) {
                  return 0.441649627715 < maxgini;
                }
                else {  // if mean_col_support > 0.982735276222
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.758499979973
            if ( max_col_coverage <= 0.998466253281 ) {
              if ( min_col_support <= 0.810500025749 ) {
                if ( min_col_coverage <= 0.876779794693 ) {
                  return 0.363841987052 < maxgini;
                }
                else {  // if min_col_coverage > 0.876779794693
                  return 0.496806907715 < maxgini;
                }
              }
              else {  // if min_col_support > 0.810500025749
                if ( min_col_support <= 0.84249997139 ) {
                  return 0.156157049375 < maxgini;
                }
                else {  // if min_col_support > 0.84249997139
                  return 0.00209647936908 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.998466253281
              if ( mean_col_support <= 0.98861759901 ) {
                if ( min_col_support <= 0.808500051498 ) {
                  return 0.322884673771 < maxgini;
                }
                else {  // if min_col_support > 0.808500051498
                  return 0.0242349440679 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.98861759901
                if ( mean_col_support <= 0.990088224411 ) {
                  return 0.0320171393653 < maxgini;
                }
                else {  // if mean_col_support > 0.990088224411
                  return 0.00166236603026 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if min_col_coverage > 0.928648591042
      if ( max_col_coverage <= 0.997682332993 ) {
        if ( median_col_coverage <= 0.939447879791 ) {
          if ( min_col_support <= 0.885499954224 ) {
            if ( median_col_support <= 0.994500041008 ) {
              if ( max_col_coverage <= 0.968501985073 ) {
                if ( median_col_coverage <= 0.930795073509 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.930795073509
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.968501985073
                if ( mean_col_support <= 0.982617616653 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.982617616653
                  return 0.0 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.994500041008
              if ( median_col_coverage <= 0.933444082737 ) {
                if ( min_col_coverage <= 0.92949461937 ) {
                  return 0.444444444444 < maxgini;
                }
                else {  // if min_col_coverage > 0.92949461937
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.933444082737
                if ( min_col_support <= 0.78149998188 ) {
                  return false;
                }
                else {  // if min_col_support > 0.78149998188
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.885499954224
            if ( mean_col_support <= 0.987970590591 ) {
              if ( mean_col_coverage <= 0.932575762272 ) {
                return false;
              }
              else {  // if mean_col_coverage > 0.932575762272
                if ( min_col_coverage <= 0.93582701683 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_coverage > 0.93582701683
                  return 0.110726643599 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.987970590591
              if ( max_col_coverage <= 0.988700211048 ) {
                if ( min_col_coverage <= 0.935948729515 ) {
                  return 0.0 < maxgini;
                }
                else {  // if min_col_coverage > 0.935948729515
                  return 0.00408161558374 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.988700211048
                if ( mean_col_coverage <= 0.957837402821 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_coverage > 0.957837402821
                  return 0.132653061224 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.939447879791
          if ( min_col_support <= 0.87549996376 ) {
            if ( min_col_coverage <= 0.947030186653 ) {
              if ( median_col_support <= 0.992499947548 ) {
                if ( mean_col_coverage <= 0.980407118797 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.980407118797
                  return 0.0 < maxgini;
                }
              }
              else {  // if median_col_support > 0.992499947548
                if ( max_col_coverage <= 0.950728178024 ) {
                  return 0.488909008601 < maxgini;
                }
                else {  // if max_col_coverage > 0.950728178024
                  return false;
                }
              }
            }
            else {  // if min_col_coverage > 0.947030186653
              if ( min_col_coverage <= 0.972157597542 ) {
                if ( mean_col_coverage <= 0.962058186531 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.962058186531
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.972157597542
                if ( mean_col_support <= 0.959264755249 ) {
                  return 0.499149750239 < maxgini;
                }
                else {  // if mean_col_support > 0.959264755249
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.87549996376
            if ( mean_col_coverage <= 0.980367660522 ) {
              if ( mean_col_coverage <= 0.971454024315 ) {
                if ( max_col_coverage <= 0.978999972343 ) {
                  return 0.0121892249837 < maxgini;
                }
                else {  // if max_col_coverage > 0.978999972343
                  return 0.0609072547771 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.971454024315
                if ( min_col_coverage <= 0.931623756886 ) {
                  return 0.462809917355 < maxgini;
                }
                else {  // if min_col_coverage > 0.931623756886
                  return 0.103475893701 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.980367660522
              if ( min_col_support <= 0.93700003624 ) {
                if ( min_col_support <= 0.887500047684 ) {
                  return false;
                }
                else {  // if min_col_support > 0.887500047684
                  return 0.483628117914 < maxgini;
                }
              }
              else {  // if min_col_support > 0.93700003624
                if ( max_col_coverage <= 0.988537847996 ) {
                  return 0.232118184717 < maxgini;
                }
                else {  // if max_col_coverage > 0.988537847996
                  return 0.0256367009614 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if max_col_coverage > 0.997682332993
        if ( min_col_coverage <= 0.973013401031 ) {
          if ( mean_col_coverage <= 0.964772999287 ) {
            if ( min_col_coverage <= 0.930189847946 ) {
              if ( min_col_coverage <= 0.92954647541 ) {
                if ( median_col_support <= 0.991500020027 ) {
                  return 0.48 < maxgini;
                }
                else {  // if median_col_support > 0.991500020027
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.92954647541
                if ( mean_col_support <= 0.982676446438 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.982676446438
                  return 0.0447526367927 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.930189847946
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_coverage <= 0.941299557686 ) {
                  return 0.209731951316 < maxgini;
                }
                else {  // if median_col_coverage > 0.941299557686
                  return 0.423659565063 < maxgini;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_coverage <= 0.931104302406 ) {
                  return 0.021824151706 < maxgini;
                }
                else {  // if min_col_coverage > 0.931104302406
                  return 0.0349864268142 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.964772999287
            if ( median_col_coverage <= 0.944682240486 ) {
              if ( min_col_coverage <= 0.930158674717 ) {
                if ( mean_col_coverage <= 0.966766953468 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.966766953468
                  return 0.476009070295 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.930158674717
                if ( mean_col_support <= 0.985647022724 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.985647022724
                  return 0.00532248142868 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.944682240486
              if ( mean_col_support <= 0.986323535442 ) {
                if ( median_col_coverage <= 0.99627995491 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.99627995491
                  return 0.448220928261 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.986323535442
                if ( median_col_coverage <= 0.944742918015 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.944742918015
                  return 0.0231393664479 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.973013401031
          if ( min_col_support <= 0.835500001907 ) {
            if ( median_col_support <= 0.986500024796 ) {
              if ( mean_col_coverage <= 0.999107480049 ) {
                if ( median_col_coverage <= 0.976284205914 ) {
                  return 0.49994685939 < maxgini;
                }
                else {  // if median_col_coverage > 0.976284205914
                  return false;
                }
              }
              else {  // if mean_col_coverage > 0.999107480049
                if ( median_col_coverage <= 0.998387098312 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.998387098312
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.986500024796
              if ( median_col_coverage <= 0.996268630028 ) {
                if ( min_col_coverage <= 0.987915277481 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.987915277481
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.996268630028
                if ( min_col_support <= 0.728500008583 ) {
                  return false;
                }
                else {  // if min_col_support > 0.728500008583
                  return 0.486012081914 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_support > 0.835500001907
            if ( mean_col_support <= 0.987205862999 ) {
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_support <= 0.950500011444 ) {
                  return 0.213039485767 < maxgini;
                }
                else {  // if median_col_support > 0.950500011444
                  return 0.475055284294 < maxgini;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( min_col_support <= 0.871500015259 ) {
                  return 0.151683303367 < maxgini;
                }
                else {  // if min_col_support > 0.871500015259
                  return 0.0345572081513 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.987205862999
              if ( mean_col_support <= 0.992735266685 ) {
                if ( median_col_coverage <= 0.998871326447 ) {
                  return 0.469391008609 < maxgini;
                }
                else {  // if median_col_coverage > 0.998871326447
                  return 0.113370600438 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.992735266685
                if ( median_col_coverage <= 0.982650518417 ) {
                  return 0.0 < maxgini;
                }
                else {  // if median_col_coverage > 0.982650518417
                  return 0.0170801804161 < maxgini;
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
  if ( min_col_support <= 0.767500042915 ) {
    if ( median_col_coverage <= 0.500617265701 ) {
      if ( median_col_support <= 0.712499976158 ) {
        if ( median_col_coverage <= 0.261162549257 ) {
          if ( mean_col_support <= 0.849635958672 ) {
            if ( min_col_support <= 0.458499997854 ) {
              if ( max_col_coverage <= 0.568323016167 ) {
                if ( median_col_support <= 0.469500005245 ) {
                  return false;
                }
                else {  // if median_col_support > 0.469500005245
                  return 0.299035463859 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.568323016167
                if ( mean_col_support <= 0.84861767292 ) {
                  return 0.43496371943 < maxgini;
                }
                else {  // if mean_col_support > 0.84861767292
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.458499997854
              if ( max_col_coverage <= 0.27704679966 ) {
                if ( max_col_coverage <= 0.194026142359 ) {
                  return 0.350067298226 < maxgini;
                }
                else {  // if max_col_coverage > 0.194026142359
                  return 0.430904519379 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.27704679966
                if ( mean_col_coverage <= 0.32652130723 ) {
                  return 0.472302946861 < maxgini;
                }
                else {  // if mean_col_coverage > 0.32652130723
                  return false;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.849635958672
            if ( min_col_support <= 0.488499999046 ) {
              if ( max_col_coverage <= 0.462912082672 ) {
                if ( min_col_coverage <= 0.0439613536 ) {
                  return 0.33710723296 < maxgini;
                }
                else {  // if min_col_coverage > 0.0439613536
                  return 0.18874673849 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.462912082672
                if ( min_col_support <= 0.379999995232 ) {
                  return false;
                }
                else {  // if min_col_support > 0.379999995232
                  return 0.328419873254 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.488499999046
              if ( min_col_support <= 0.554499983788 ) {
                if ( mean_col_coverage <= 0.25958108902 ) {
                  return 0.399926519123 < maxgini;
                }
                else {  // if mean_col_coverage > 0.25958108902
                  return 0.47018855137 < maxgini;
                }
              }
              else {  // if min_col_support > 0.554499983788
                if ( max_col_coverage <= 0.256776571274 ) {
                  return 0.315770182064 < maxgini;
                }
                else {  // if max_col_coverage > 0.256776571274
                  return 0.391999090189 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.261162549257
          if ( median_col_coverage <= 0.36381816864 ) {
            if ( mean_col_coverage <= 0.396317929029 ) {
              if ( min_col_coverage <= 0.106203004718 ) {
                if ( min_col_coverage <= 0.10263158381 ) {
                  return false;
                }
                else {  // if min_col_coverage > 0.10263158381
                  return false;
                }
              }
              else {  // if min_col_coverage > 0.106203004718
                if ( median_col_support <= 0.606500029564 ) {
                  return false;
                }
                else {  // if median_col_support > 0.606500029564
                  return 0.442266628423 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.396317929029
              if ( max_col_coverage <= 0.533789992332 ) {
                if ( max_col_coverage <= 0.524236619473 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.524236619473
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.533789992332
                if ( median_col_coverage <= 0.3451410532 ) {
                  return 0.499197381344 < maxgini;
                }
                else {  // if median_col_coverage > 0.3451410532
                  return 0.481216439826 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_coverage > 0.36381816864
            if ( max_col_coverage <= 0.460277199745 ) {
              if ( min_col_coverage <= 0.293981492519 ) {
                if ( median_col_coverage <= 0.380131363869 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.380131363869
                  return 0.477625739645 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.293981492519
                if ( median_col_support <= 0.644999980927 ) {
                  return false;
                }
                else {  // if median_col_support > 0.644999980927
                  return 0.375978469819 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.460277199745
              if ( min_col_support <= 0.68649995327 ) {
                if ( mean_col_coverage <= 0.45207965374 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.45207965374
                  return false;
                }
              }
              else {  // if min_col_support > 0.68649995327
                if ( mean_col_support <= 0.844764590263 ) {
                  return 0.267069790879 < maxgini;
                }
                else {  // if mean_col_support > 0.844764590263
                  return 0.486166722222 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.712499976158
        if ( max_col_coverage <= 0.500605344772 ) {
          if ( mean_col_coverage <= 0.312243878841 ) {
            if ( median_col_support <= 0.814499974251 ) {
              if ( min_col_coverage <= 0.0692049860954 ) {
                if ( median_col_support <= 0.757500052452 ) {
                  return 0.363441661488 < maxgini;
                }
                else {  // if median_col_support > 0.757500052452
                  return 0.290003084311 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.0692049860954
                if ( mean_col_support <= 0.875165104866 ) {
                  return 0.326347770993 < maxgini;
                }
                else {  // if mean_col_support > 0.875165104866
                  return 0.215255845049 < maxgini;
                }
              }
            }
            else {  // if median_col_support > 0.814499974251
              if ( min_col_support <= 0.550500035286 ) {
                if ( min_col_coverage <= 0.160480648279 ) {
                  return 0.22505048565 < maxgini;
                }
                else {  // if min_col_coverage > 0.160480648279
                  return 0.414807356993 < maxgini;
                }
              }
              else {  // if min_col_support > 0.550500035286
                if ( max_col_coverage <= 0.304491788149 ) {
                  return 0.101602362215 < maxgini;
                }
                else {  // if max_col_coverage > 0.304491788149
                  return 0.12920704118 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_coverage > 0.312243878841
            if ( mean_col_coverage <= 0.355892956257 ) {
              if ( min_col_coverage <= 0.190657943487 ) {
                if ( min_col_coverage <= 0.114835165441 ) {
                  return 0.283924705736 < maxgini;
                }
                else {  // if min_col_coverage > 0.114835165441
                  return 0.203564602909 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.190657943487
                if ( min_col_support <= 0.599500000477 ) {
                  return 0.425937248585 < maxgini;
                }
                else {  // if min_col_support > 0.599500000477
                  return 0.24951272252 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.355892956257
              if ( median_col_support <= 0.971500039101 ) {
                if ( min_col_support <= 0.557500004768 ) {
                  return 0.368696048416 < maxgini;
                }
                else {  // if min_col_support > 0.557500004768
                  return 0.285419782995 < maxgini;
                }
              }
              else {  // if median_col_support > 0.971500039101
                if ( max_col_coverage <= 0.499340355396 ) {
                  return 0.483519204159 < maxgini;
                }
                else {  // if max_col_coverage > 0.499340355396
                  return 0.332514308501 < maxgini;
                }
              }
            }
          }
        }
        else {  // if max_col_coverage > 0.500605344772
          if ( mean_col_support <= 0.96909725666 ) {
            if ( min_col_support <= 0.607499957085 ) {
              if ( min_col_support <= 0.555500030518 ) {
                if ( mean_col_support <= 0.939147114754 ) {
                  return 0.429533322873 < maxgini;
                }
                else {  // if mean_col_support > 0.939147114754
                  return false;
                }
              }
              else {  // if min_col_support > 0.555500030518
                if ( mean_col_support <= 0.946676492691 ) {
                  return 0.367998534843 < maxgini;
                }
                else {  // if mean_col_support > 0.946676492691
                  return 0.497144956033 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.607499957085
              if ( min_col_coverage <= 0.350416123867 ) {
                if ( min_col_coverage <= 0.300406515598 ) {
                  return 0.24861750173 < maxgini;
                }
                else {  // if min_col_coverage > 0.300406515598
                  return 0.314460122033 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.350416123867
                if ( median_col_support <= 0.978500008583 ) {
                  return 0.3387163692 < maxgini;
                }
                else {  // if median_col_support > 0.978500008583
                  return false;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.96909725666
            if ( min_col_support <= 0.675500035286 ) {
              if ( min_col_support <= 0.59350001812 ) {
                if ( min_col_coverage <= 0.213371276855 ) {
                  return 0.371644444444 < maxgini;
                }
                else {  // if min_col_coverage > 0.213371276855
                  return false;
                }
              }
              else {  // if min_col_support > 0.59350001812
                if ( mean_col_coverage <= 0.374323934317 ) {
                  return 0.0916598079561 < maxgini;
                }
                else {  // if mean_col_coverage > 0.374323934317
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.675500035286
              if ( median_col_support <= 0.999000012875 ) {
                if ( mean_col_support <= 0.974911749363 ) {
                  return 0.354960080627 < maxgini;
                }
                else {  // if mean_col_support > 0.974911749363
                  return 0.494490806012 < maxgini;
                }
              }
              else {  // if median_col_support > 0.999000012875
                if ( min_col_coverage <= 0.334192454815 ) {
                  return 0.108979458377 < maxgini;
                }
                else {  // if min_col_coverage > 0.334192454815
                  return 0.332053918696 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if median_col_coverage > 0.500617265701
      if ( min_col_support <= 0.679499983788 ) {
        if ( median_col_support <= 0.986500024796 ) {
          if ( min_col_coverage <= 0.667673945427 ) {
            if ( min_col_support <= 0.630499958992 ) {
              if ( mean_col_support <= 0.843205869198 ) {
                if ( min_col_coverage <= 0.310096144676 ) {
                  return 0.457856399584 < maxgini;
                }
                else {  // if min_col_coverage > 0.310096144676
                  return false;
                }
              }
              else {  // if mean_col_support > 0.843205869198
                if ( median_col_support <= 0.929499983788 ) {
                  return 0.497484010838 < maxgini;
                }
                else {  // if median_col_support > 0.929499983788
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.630499958992
              if ( min_col_coverage <= 0.666140913963 ) {
                if ( median_col_coverage <= 0.513841032982 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.513841032982
                  return 0.470892103727 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.666140913963
                if ( median_col_coverage <= 0.715476155281 ) {
                  return 0.156423130194 < maxgini;
                }
                else {  // if median_col_coverage > 0.715476155281
                  return 0.439156300703 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.667673945427
            if ( mean_col_coverage <= 0.988531529903 ) {
              if ( min_col_support <= 0.62450003624 ) {
                if ( median_col_support <= 0.860499978065 ) {
                  return false;
                }
                else {  // if median_col_support > 0.860499978065
                  return false;
                }
              }
              else {  // if min_col_support > 0.62450003624
                if ( max_col_coverage <= 0.842264771461 ) {
                  return 0.492228890656 < maxgini;
                }
                else {  // if max_col_coverage > 0.842264771461
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.988531529903
              if ( min_col_support <= 0.608500003815 ) {
                if ( max_col_coverage <= 0.995099425316 ) {
                  return 0.0 < maxgini;
                }
                else {  // if max_col_coverage > 0.995099425316
                  return false;
                }
              }
              else {  // if min_col_support > 0.608500003815
                if ( max_col_coverage <= 0.996200203896 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.996200203896
                  return false;
                }
              }
            }
          }
        }
        else {  // if median_col_support > 0.986500024796
          if ( min_col_support <= 0.614500045776 ) {
            if ( mean_col_coverage <= 0.728417992592 ) {
              if ( mean_col_support <= 0.964735269547 ) {
                if ( max_col_coverage <= 0.974342107773 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.974342107773
                  return 0.197530864198 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.964735269547
                if ( median_col_support <= 0.993499994278 ) {
                  return false;
                }
                else {  // if median_col_support > 0.993499994278
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.728417992592
              if ( mean_col_support <= 0.966264665127 ) {
                if ( max_col_coverage <= 0.971369981766 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.971369981766
                  return false;
                }
              }
              else {  // if mean_col_support > 0.966264665127
                if ( median_col_support <= 0.994500041008 ) {
                  return false;
                }
                else {  // if median_col_support > 0.994500041008
                  return false;
                }
              }
            }
          }
          else {  // if min_col_support > 0.614500045776
            if ( max_col_coverage <= 0.801120638847 ) {
              if ( mean_col_coverage <= 0.734295368195 ) {
                if ( mean_col_support <= 0.972911775112 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.972911775112
                  return false;
                }
              }
              else {  // if mean_col_coverage > 0.734295368195
                if ( mean_col_coverage <= 0.735589921474 ) {
                  return 0.345679012346 < maxgini;
                }
                else {  // if mean_col_coverage > 0.735589921474
                  return false;
                }
              }
            }
            else {  // if max_col_coverage > 0.801120638847
              if ( median_col_coverage <= 0.996095657349 ) {
                if ( mean_col_support <= 0.972264647484 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.972264647484
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.996095657349
                if ( median_col_coverage <= 0.998161792755 ) {
                  return 0.0 < maxgini;
                }
                else {  // if median_col_coverage > 0.998161792755
                  return false;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.679499983788
        if ( median_col_coverage <= 0.667071223259 ) {
          if ( median_col_support <= 0.984500050545 ) {
            if ( median_col_support <= 0.758499979973 ) {
              if ( min_col_coverage <= 0.524159669876 ) {
                if ( min_col_support <= 0.735499978065 ) {
                  return false;
                }
                else {  // if min_col_support > 0.735499978065
                  return 0.413848344607 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.524159669876
                if ( mean_col_support <= 0.900588274002 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.900588274002
                  return false;
                }
              }
            }
            else {  // if median_col_support > 0.758499979973
              if ( median_col_coverage <= 0.513825178146 ) {
                if ( min_col_coverage <= 0.464157372713 ) {
                  return 0.389509825407 < maxgini;
                }
                else {  // if min_col_coverage > 0.464157372713
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.513825178146
                if ( min_col_coverage <= 0.64826798439 ) {
                  return 0.314735475499 < maxgini;
                }
                else {  // if min_col_coverage > 0.64826798439
                  return 0.113141672165 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.984500050545
            if ( median_col_coverage <= 0.600200772285 ) {
              if ( median_col_coverage <= 0.599689483643 ) {
                if ( min_col_support <= 0.731500029564 ) {
                  return false;
                }
                else {  // if min_col_support > 0.731500029564
                  return 0.483920704552 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.599689483643
                if ( min_col_coverage <= 0.597058832645 ) {
                  return 0.420315893078 < maxgini;
                }
                else {  // if min_col_coverage > 0.597058832645
                  return 0.0 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.600200772285
              if ( median_col_support <= 0.99950003624 ) {
                if ( median_col_support <= 0.991500020027 ) {
                  return 0.499348355902 < maxgini;
                }
                else {  // if median_col_support > 0.991500020027
                  return false;
                }
              }
              else {  // if median_col_support > 0.99950003624
                if ( mean_col_support <= 0.980205893517 ) {
                  return 0.393466553322 < maxgini;
                }
                else {  // if mean_col_support > 0.980205893517
                  return 0.499278487338 < maxgini;
                }
              }
            }
          }
        }
        else {  // if median_col_coverage > 0.667071223259
          if ( min_col_coverage <= 0.850197196007 ) {
            if ( mean_col_support <= 0.962617635727 ) {
              if ( max_col_coverage <= 0.800576388836 ) {
                if ( median_col_support <= 0.990000009537 ) {
                  return 0.330822345848 < maxgini;
                }
                else {  // if median_col_support > 0.990000009537
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.800576388836
                if ( max_col_coverage <= 0.998392283916 ) {
                  return 0.49494100346 < maxgini;
                }
                else {  // if max_col_coverage > 0.998392283916
                  return 0.441868293364 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.962617635727
              if ( max_col_coverage <= 0.769381046295 ) {
                if ( mean_col_support <= 0.982088208199 ) {
                  return 0.352315689981 < maxgini;
                }
                else {  // if mean_col_support > 0.982088208199
                  return 0.499405469679 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.769381046295
                if ( mean_col_support <= 0.979970574379 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.979970574379
                  return false;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.850197196007
            if ( mean_col_coverage <= 0.991711735725 ) {
              if ( median_col_support <= 0.989500045776 ) {
                if ( mean_col_support <= 0.977147042751 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.977147042751
                  return 0.477727835156 < maxgini;
                }
              }
              else {  // if median_col_support > 0.989500045776
                if ( min_col_support <= 0.712499976158 ) {
                  return false;
                }
                else {  // if min_col_support > 0.712499976158
                  return false;
                }
              }
            }
            else {  // if mean_col_coverage > 0.991711735725
              if ( min_col_coverage <= 0.960769176483 ) {
                if ( min_col_coverage <= 0.946457803249 ) {
                  return 0.489795918367 < maxgini;
                }
                else {  // if min_col_coverage > 0.946457803249
                  return 0.36815193572 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.960769176483
                if ( median_col_coverage <= 0.99686729908 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.99686729908
                  return false;
                }
              }
            }
          }
        }
      }
    }
  }
  else {  // if min_col_support > 0.767500042915
    if ( mean_col_support <= 0.987970590591 ) {
      if ( min_col_support <= 0.824499964714 ) {
        if ( mean_col_coverage <= 0.745140314102 ) {
          if ( min_col_coverage <= 0.454673886299 ) {
            if ( mean_col_support <= 0.953205883503 ) {
              if ( min_col_support <= 0.769500017166 ) {
                if ( mean_col_coverage <= 0.408957242966 ) {
                  return 0.183698285177 < maxgini;
                }
                else {  // if mean_col_coverage > 0.408957242966
                  return 0.277432041195 < maxgini;
                }
              }
              else {  // if min_col_support > 0.769500017166
                if ( mean_col_support <= 0.939323544502 ) {
                  return 0.326186414294 < maxgini;
                }
                else {  // if mean_col_support > 0.939323544502
                  return 0.218401935882 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.953205883503
              if ( median_col_support <= 0.856500029564 ) {
                if ( mean_col_coverage <= 0.369346410036 ) {
                  return 0.196064349325 < maxgini;
                }
                else {  // if mean_col_coverage > 0.369346410036
                  return 0.322099107418 < maxgini;
                }
              }
              else {  // if median_col_support > 0.856500029564
                if ( mean_col_coverage <= 0.402972459793 ) {
                  return 0.0631826440594 < maxgini;
                }
                else {  // if mean_col_coverage > 0.402972459793
                  return 0.12161703373 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.454673886299
            if ( mean_col_support <= 0.985558867455 ) {
              if ( median_col_coverage <= 0.600335597992 ) {
                if ( min_col_coverage <= 0.582874655724 ) {
                  return 0.258263626359 < maxgini;
                }
                else {  // if min_col_coverage > 0.582874655724
                  return 0.0971025272733 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.600335597992
                if ( mean_col_support <= 0.972088217735 ) {
                  return 0.272701406527 < maxgini;
                }
                else {  // if mean_col_support > 0.972088217735
                  return 0.382877188827 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.985558867455
              if ( mean_col_coverage <= 0.631321966648 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.498793474915 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.220332275342 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.631321966648
                if ( min_col_support <= 0.807500004768 ) {
                  return 0.486987156659 < maxgini;
                }
                else {  // if min_col_support > 0.807500004768
                  return 0.326278583776 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.745140314102
          if ( median_col_support <= 0.99950003624 ) {
            if ( min_col_coverage <= 0.857695937157 ) {
              if ( mean_col_support <= 0.975852906704 ) {
                if ( median_col_coverage <= 0.800423741341 ) {
                  return 0.346981644167 < maxgini;
                }
                else {  // if median_col_coverage > 0.800423741341
                  return 0.446714214608 < maxgini;
                }
              }
              else {  // if mean_col_support > 0.975852906704
                if ( min_col_coverage <= 0.658141493797 ) {
                  return 0.439248979592 < maxgini;
                }
                else {  // if min_col_coverage > 0.658141493797
                  return false;
                }
              }
            }
            else {  // if min_col_coverage > 0.857695937157
              if ( median_col_coverage <= 0.998618781567 ) {
                if ( median_col_coverage <= 0.920321941376 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.920321941376
                  return false;
                }
              }
              else {  // if median_col_coverage > 0.998618781567
                if ( min_col_coverage <= 0.962433874607 ) {
                  return 0.425078043704 < maxgini;
                }
                else {  // if min_col_coverage > 0.962433874607
                  return 0.498201468489 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.99950003624
            if ( mean_col_support <= 0.985794067383 ) {
              if ( mean_col_coverage <= 0.94128382206 ) {
                if ( min_col_coverage <= 0.905532896519 ) {
                  return 0.18500812669 < maxgini;
                }
                else {  // if min_col_coverage > 0.905532896519
                  return 0.48150887574 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.94128382206
                if ( max_col_coverage <= 0.998618781567 ) {
                  return false;
                }
                else {  // if max_col_coverage > 0.998618781567
                  return 0.336484957308 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.985794067383
              if ( min_col_coverage <= 0.905294299126 ) {
                if ( min_col_support <= 0.786499977112 ) {
                  return false;
                }
                else {  // if min_col_support > 0.786499977112
                  return 0.256471468144 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.905294299126
                if ( min_col_support <= 0.799499988556 ) {
                  return false;
                }
                else {  // if min_col_support > 0.799499988556
                  return 0.324864639733 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if min_col_support > 0.824499964714
        if ( mean_col_coverage <= 0.950606048107 ) {
          if ( median_col_support <= 0.892500042915 ) {
            if ( max_col_coverage <= 0.604957163334 ) {
              if ( median_col_support <= 0.867499947548 ) {
                if ( mean_col_coverage <= 0.4328571558 ) {
                  return 0.201995227947 < maxgini;
                }
                else {  // if mean_col_coverage > 0.4328571558
                  return 0.321819088885 < maxgini;
                }
              }
              else {  // if median_col_support > 0.867499947548
                if ( mean_col_support <= 0.946205914021 ) {
                  return 0.347360267636 < maxgini;
                }
                else {  // if mean_col_support > 0.946205914021
                  return 0.138753880205 < maxgini;
                }
              }
            }
            else {  // if max_col_coverage > 0.604957163334
              if ( median_col_support <= 0.853500008583 ) {
                if ( max_col_coverage <= 0.614709854126 ) {
                  return 0.487638938333 < maxgini;
                }
                else {  // if max_col_coverage > 0.614709854126
                  return 0.362952677853 < maxgini;
                }
              }
              else {  // if median_col_support > 0.853500008583
                if ( mean_col_support <= 0.94473528862 ) {
                  return 0.359057816984 < maxgini;
                }
                else {  // if mean_col_support > 0.94473528862
                  return 0.229703581474 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.892500042915
            if ( min_col_coverage <= 0.818337440491 ) {
              if ( median_col_coverage <= 0.66750305891 ) {
                if ( mean_col_coverage <= 0.824114561081 ) {
                  return 0.063250127102 < maxgini;
                }
                else {  // if mean_col_coverage > 0.824114561081
                  return 0.298034567901 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.66750305891
                if ( median_col_support <= 0.986500024796 ) {
                  return 0.060453955815 < maxgini;
                }
                else {  // if median_col_support > 0.986500024796
                  return 0.146603831674 < maxgini;
                }
              }
            }
            else {  // if min_col_coverage > 0.818337440491
              if ( min_col_support <= 0.859500050545 ) {
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.442763160528 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0757036691704 < maxgini;
                }
              }
              else {  // if min_col_support > 0.859500050545
                if ( median_col_support <= 0.99950003624 ) {
                  return 0.0927237731366 < maxgini;
                }
                else {  // if median_col_support > 0.99950003624
                  return 0.0127597156635 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_coverage > 0.950606048107
          if ( median_col_support <= 0.99950003624 ) {
            if ( min_col_support <= 0.868499994278 ) {
              if ( min_col_support <= 0.839499950409 ) {
                if ( min_col_coverage <= 0.885763764381 ) {
                  return 0.368618035449 < maxgini;
                }
                else {  // if min_col_coverage > 0.885763764381
                  return 0.497053520637 < maxgini;
                }
              }
              else {  // if min_col_support > 0.839499950409
                if ( median_col_support <= 0.994500041008 ) {
                  return 0.374484004676 < maxgini;
                }
                else {  // if median_col_support > 0.994500041008
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.868499994278
              if ( median_col_support <= 0.994500041008 ) {
                if ( median_col_support <= 0.956499993801 ) {
                  return 0.143822596038 < maxgini;
                }
                else {  // if median_col_support > 0.956499993801
                  return 0.309382774473 < maxgini;
                }
              }
              else {  // if median_col_support > 0.994500041008
                if ( mean_col_support <= 0.987117648125 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.987117648125
                  return 0.460223537147 < maxgini;
                }
              }
            }
          }
          else {  // if median_col_support > 0.99950003624
            if ( max_col_coverage <= 0.989998996258 ) {
              if ( mean_col_coverage <= 0.967133641243 ) {
                if ( min_col_coverage <= 0.95087403059 ) {
                  return 0.285374554102 < maxgini;
                }
                else {  // if min_col_coverage > 0.95087403059
                  return 0.0 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.967133641243
                if ( max_col_coverage <= 0.978250622749 ) {
                  return 0.444444444444 < maxgini;
                }
                else {  // if max_col_coverage > 0.978250622749
                  return false;
                }
              }
            }
            else {  // if max_col_coverage > 0.989998996258
              if ( min_col_support <= 0.867499947548 ) {
                if ( min_col_coverage <= 0.966849803925 ) {
                  return 0.0810004702854 < maxgini;
                }
                else {  // if min_col_coverage > 0.966849803925
                  return 0.317070241406 < maxgini;
                }
              }
              else {  // if min_col_support > 0.867499947548
                if ( median_col_coverage <= 0.940285205841 ) {
                  return 0.00766272189349 < maxgini;
                }
                else {  // if median_col_coverage > 0.940285205841
                  return 0.0355755093468 < maxgini;
                }
              }
            }
          }
        }
      }
    }
    else {  // if mean_col_support > 0.987970590591
      if ( median_col_support <= 0.99950003624 ) {
        if ( min_col_coverage <= 0.93341255188 ) {
          if ( max_col_coverage <= 0.800627946854 ) {
            if ( median_col_coverage <= 0.688063263893 ) {
              if ( min_col_support <= 0.871500015259 ) {
                if ( median_col_coverage <= 0.474537551403 ) {
                  return 0.401526557193 < maxgini;
                }
                else {  // if median_col_coverage > 0.474537551403
                  return 0.494984419641 < maxgini;
                }
              }
              else {  // if min_col_support > 0.871500015259
                if ( median_col_support <= 0.995499968529 ) {
                  return 0.0135574851903 < maxgini;
                }
                else {  // if median_col_support > 0.995499968529
                  return 0.0912225283077 < maxgini;
                }
              }
            }
            else {  // if median_col_coverage > 0.688063263893
              if ( min_col_coverage <= 0.688272058964 ) {
                if ( min_col_coverage <= 0.688153803349 ) {
                  return 0.0119961355529 < maxgini;
                }
                else {  // if min_col_coverage > 0.688153803349
                  return 0.375 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.688272058964
                if ( median_col_coverage <= 0.738278388977 ) {
                  return 0.0 < maxgini;
                }
                else {  // if median_col_coverage > 0.738278388977
                  return 0.00507096114874 < maxgini;
                }
              }
            }
          }
          else {  // if max_col_coverage > 0.800627946854
            if ( mean_col_support <= 0.991676449776 ) {
              if ( mean_col_coverage <= 0.916030347347 ) {
                if ( max_col_coverage <= 0.803394794464 ) {
                  return 0.462809917355 < maxgini;
                }
                else {  // if max_col_coverage > 0.803394794464
                  return 0.101349751312 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.916030347347
                if ( mean_col_support <= 0.98914706707 ) {
                  return 0.279087661292 < maxgini;
                }
                else {  // if mean_col_support > 0.98914706707
                  return 0.147962042928 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.991676449776
              if ( max_col_coverage <= 0.80064868927 ) {
                if ( mean_col_coverage <= 0.717237472534 ) {
                  return 0.0 < maxgini;
                }
                else {  // if mean_col_coverage > 0.717237472534
                  return false;
                }
              }
              else {  // if max_col_coverage > 0.80064868927
                if ( min_col_support <= 0.897500038147 ) {
                  return 0.499116512648 < maxgini;
                }
                else {  // if min_col_support > 0.897500038147
                  return 0.00536547235453 < maxgini;
                }
              }
            }
          }
        }
        else {  // if min_col_coverage > 0.93341255188
          if ( min_col_coverage <= 0.978300094604 ) {
            if ( mean_col_support <= 0.992205917835 ) {
              if ( median_col_support <= 0.995499968529 ) {
                if ( min_col_support <= 0.883499979973 ) {
                  return 0.393674997987 < maxgini;
                }
                else {  // if min_col_support > 0.883499979973
                  return 0.0348474196292 < maxgini;
                }
              }
              else {  // if median_col_support > 0.995499968529
                if ( mean_col_support <= 0.988852977753 ) {
                  return false;
                }
                else {  // if mean_col_support > 0.988852977753
                  return false;
                }
              }
            }
            else {  // if mean_col_support > 0.992205917835
              if ( min_col_support <= 0.921499967575 ) {
                if ( min_col_support <= 0.889500021935 ) {
                  return false;
                }
                else {  // if min_col_support > 0.889500021935
                  return 0.308806452821 < maxgini;
                }
              }
              else {  // if min_col_support > 0.921499967575
                if ( median_col_support <= 0.994500041008 ) {
                  return 0.00111575976393 < maxgini;
                }
                else {  // if median_col_support > 0.994500041008
                  return 0.00773433395982 < maxgini;
                }
              }
            }
          }
          else {  // if min_col_coverage > 0.978300094604
            if ( min_col_support <= 0.899500012398 ) {
              if ( median_col_support <= 0.989500045776 ) {
                if ( median_col_coverage <= 0.98958337307 ) {
                  return 0.0 < maxgini;
                }
                else {  // if median_col_coverage > 0.98958337307
                  return 0.253954110047 < maxgini;
                }
              }
              else {  // if median_col_support > 0.989500045776
                if ( median_col_coverage <= 0.984769999981 ) {
                  return false;
                }
                else {  // if median_col_coverage > 0.984769999981
                  return false;
                }
              }
            }
            else {  // if min_col_support > 0.899500012398
              if ( min_col_support <= 0.946500003338 ) {
                if ( median_col_support <= 0.988499999046 ) {
                  return 0.229814481821 < maxgini;
                }
                else {  // if median_col_support > 0.988499999046
                  return 0.443494794024 < maxgini;
                }
              }
              else {  // if min_col_support > 0.946500003338
                if ( min_col_support <= 0.985499978065 ) {
                  return 0.108493936066 < maxgini;
                }
                else {  // if min_col_support > 0.985499978065
                  return 0.0 < maxgini;
                }
              }
            }
          }
        }
      }
      else {  // if median_col_support > 0.99950003624
        if ( mean_col_support <= 0.991188287735 ) {
          if ( mean_col_support <= 0.988970577717 ) {
            if ( mean_col_coverage <= 0.900204479694 ) {
              if ( mean_col_coverage <= 0.636459887028 ) {
                if ( min_col_support <= 0.935500025749 ) {
                  return 0.0257854119586 < maxgini;
                }
                else {  // if min_col_support > 0.935500025749
                  return 0.124444444444 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.636459887028
                if ( mean_col_coverage <= 0.636496245861 ) {
                  return false;
                }
                else {  // if mean_col_coverage > 0.636496245861
                  return 0.0404552911649 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.900204479694
              if ( max_col_coverage <= 0.99527156353 ) {
                if ( min_col_support <= 0.822499990463 ) {
                  return false;
                }
                else {  // if min_col_support > 0.822499990463
                  return 0.0290318095025 < maxgini;
                }
              }
              else {  // if max_col_coverage > 0.99527156353
                if ( min_col_support <= 0.821500003338 ) {
                  return 0.390576359852 < maxgini;
                }
                else {  // if min_col_support > 0.821500003338
                  return 0.0202986171744 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.988970577717
            if ( min_col_support <= 0.857499957085 ) {
              if ( median_col_coverage <= 0.672429203987 ) {
                if ( max_col_coverage <= 0.602418422699 ) {
                  return 0.00817552215055 < maxgini;
                }
                else {  // if max_col_coverage > 0.602418422699
                  return 0.0424159907374 < maxgini;
                }
              }
              else {  // if median_col_coverage > 0.672429203987
                if ( min_col_support <= 0.833500027657 ) {
                  return 0.267487298592 < maxgini;
                }
                else {  // if min_col_support > 0.833500027657
                  return 0.113539811304 < maxgini;
                }
              }
            }
            else {  // if min_col_support > 0.857499957085
              if ( mean_col_coverage <= 0.603956580162 ) {
                if ( max_col_coverage <= 0.885703921318 ) {
                  return 0.0156433483213 < maxgini;
                }
                else {  // if max_col_coverage > 0.885703921318
                  return 0.154731204126 < maxgini;
                }
              }
              else {  // if mean_col_coverage > 0.603956580162
                if ( min_col_coverage <= 0.923402905464 ) {
                  return 0.00872045129005 < maxgini;
                }
                else {  // if min_col_coverage > 0.923402905464
                  return 0.0357734104804 < maxgini;
                }
              }
            }
          }
        }
        else {  // if mean_col_support > 0.991188287735
          if ( mean_col_support <= 0.994323551655 ) {
            if ( mean_col_support <= 0.992970585823 ) {
              if ( min_col_coverage <= 0.961921811104 ) {
                if ( min_col_support <= 0.952499985695 ) {
                  return 0.00815629402324 < maxgini;
                }
                else {  // if min_col_support > 0.952499985695
                  return 0.0242183433186 < maxgini;
                }
              }
              else {  // if min_col_coverage > 0.961921811104
                if ( max_col_coverage <= 0.998106062412 ) {
                  return 0.284081632653 < maxgini;
                }
                else {  // if max_col_coverage > 0.998106062412
                  return 0.0288399762046 < maxgini;
                }
              }
            }
            else {  // if mean_col_support > 0.992970585823
              if ( min_col_support <= 0.966500043869 ) {
                if ( mean_col_support <= 0.993531405926 ) {
                  return 0.00630405457928 < maxgini;
                }
                else {  // if mean_col_support > 0.993531405926
                  return 0.00465493407253 < maxgini;
                }
              }
              else {  // if min_col_support > 0.966500043869
                if ( max_col_coverage <= 0.769047617912 ) {
                  return 0.19669933432 < maxgini;
                }
                else {  // if max_col_coverage > 0.769047617912
                  return 0.0 < maxgini;
                }
              }
            }
          }
          else {  // if mean_col_support > 0.994323551655
            if ( mean_col_coverage <= 0.68717610836 ) {
              if ( min_col_support <= 0.972499966621 ) {
                if ( median_col_coverage <= 0.0181177239865 ) {
                  return 0.0928019036288 < maxgini;
                }
                else {  // if median_col_coverage > 0.0181177239865
                  return 0.00241015227274 < maxgini;
                }
              }
              else {  // if min_col_support > 0.972499966621
                if ( min_col_support <= 0.975499987602 ) {
                  return 0.00163432918168 < maxgini;
                }
                else {  // if min_col_support > 0.975499987602
                  return 0.000843320446517 < maxgini;
                }
              }
            }
            else {  // if mean_col_coverage > 0.68717610836
              if ( median_col_coverage <= 0.267857134342 ) {
                return false;
              }
              else {  // if median_col_coverage > 0.267857134342
                if ( mean_col_support <= 0.997911691666 ) {
                  return 0.00179682281549 < maxgini;
                }
                else {  // if mean_col_support > 0.997911691666
                  return 0.000332609707572 < maxgini;
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
