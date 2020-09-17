#ifndef CARE_SOLOADER_HPP
#define CARE_SOLOADER_HPP


#include <dlfcn.h> //linux load shared object

#include <functional>
#include <string>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace care{


template<class Signature>
class SoFunction; //not available

template<class Ret, class... Args>
class SoFunction<Ret(Args...)>
{
public:
    void* soHandle = nullptr;
    std::function<Ret(Args...)> f;

    SoFunction(){}

    SoFunction(const std::string& soFilename, const std::string& functionname){
        bool failure = false;

        soHandle = dlopen(soFilename.c_str(), RTLD_NOW);
        if (!soHandle) {
            std::cerr << dlerror() << '\n';
            failure = true;
        }

        void* functionptr = dlsym(soHandle, functionname.c_str());

        const char* error = dlerror();
        if (error != NULL) {
            std::cerr << error << '\n';
            failure = true;
        }

        if(failure){
            throw std::runtime_error("Cannot load function from file :" + soFilename);
        }
        
        f = (Ret(*)(Args...))functionptr;  

    }

    SoFunction(const SoFunction&) = delete;
    SoFunction(SoFunction&&) = default;

    SoFunction& operator=(const SoFunction&) = delete;
    SoFunction& operator=(SoFunction&& rhs){
        closeHandle();
        soHandle = std::exchange(rhs.soHandle, nullptr);
        f = std::move(rhs.f);
        return *this;
    }

    ~SoFunction(){
        closeHandle();
    }

    template<class... T>
    auto operator()(T&&... t) const {
        return f(std::forward<T&&>(t)...);
    }

private:
    void closeHandle(){
        if(soHandle != nullptr){
            int dlcloseretval = dlclose(soHandle);
            if (dlcloseretval != 0) {
                std::cerr << dlerror() << '\n';
            }
        }
    }    
};

}

#endif