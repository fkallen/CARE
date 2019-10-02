#ifndef FILE_HELPERS_HPP
#define FILE_HELPERS_HPP

#include <cstdio>
#include <cassert>

#include <fstream>
#include <iostream>
#include <string>


#define FILE_HELPERS_DEBUG

__inline__
void renameFileSameMount(const std::string& filename, const std::string& newFilename){
#ifdef FILE_HELPERS_DEBUG        
    std::cerr << "Rename " << filename << " to " << newFilename << "\n";
#endif    
    int res = std::rename(filename.c_str(), newFilename.c_str());
    if(res != 0){
        std::perror("rename");
        assert(res == 0);
    }    
}

__inline__
void copyFile(const std::string& filename, const std::string& newFilename){
#ifdef FILE_HELPERS_DEBUG       
    std::cerr << "Copy " << filename << " to " << newFilename << "\n";
#endif    
    std::ifstream src(filename, std::ios::binary);
    std::ofstream dst(newFilename, std::ios::binary);
    assert(bool(src));
    assert(bool(dst));
    dst << src.rdbuf();
    assert(bool(dst));
}

__inline__
void removeFile(const std::string& filename){
#ifdef FILE_HELPERS_DEBUG   
    std::cerr << "Remove " << filename << "\n";
#endif    
    std::ifstream src(filename);
    assert(bool(src));
    int res = std::remove(filename.c_str());
    if(res != 0){
        std::perror("remove");
        assert(res == 0);
    }    
}

__inline__ 
bool fileCanBeOpened(const std::string& filename){
    std::ifstream in(filename);
    return bool(in);
}


#ifdef FILE_HELPERS_DEBUG
#undef FILE_HELPERS_DEBUG
#endif

#endif