#include "hpc_helpers.cuh"

HOSTDEVICEQUALIFIER
char encoded_accessor(const char* data, int length, int index){
    const int byte = index / 4;
    const int basepos = index % 4;
    switch((data[byte] >> (3-basepos) * 2) & 0x03){
        case 0x00: return 'A';
        case 0x01: return 'C';
        case 0x02: return 'G';
        case 0x03: return 'T';
        default:return '_'; //cannot happen
    }
}
