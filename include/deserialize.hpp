#ifndef CARE_DESERIALIZE_HPP
#define CARE_DESERIALIZE_HPP

#include <iostream>

namespace care {

// de-serialization helpers

template<typename T>
inline T& read_one(std::ifstream& is, T& v) {
    char tmp[sizeof(v)];
    // this is only to be absolutely 100% standard-compliant no matter how read() is implemented
    // probably absolutely unnecessary but it will be optimized out
    is.read(tmp, sizeof(v));
    std::memcpy(&v, tmp, sizeof(v));
    return v;
}

template<typename T>
inline T read_one(std::ifstream& is) {
    T ret;
    read_one(is, ret);
    return ret;
}

} // namespace care

#endif