#ifndef  kseq_pp_h
#define kseq_pp_h

#include "gziphelpers.hpp"
#include "filereader.hpp"

#include <cstdint>
#include <string>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <iostream>
#include <functional>


namespace kseqpp{

struct KseqPP{    

public:


    KseqPP() = default;

    KseqPP(const std::string& filename)
            : f(std::make_unique<Stream>(filename)){
        // std::cerr << "KseqPP(" << filename << ")\n";
        name.reserve(256);
        comment.reserve(256);
        seq.reserve(256);
        qual.reserve(256);
    }

    int next(){
        int c = 0;
        int r = 0;

        if (last_char == 0) { /* then jump to the next header line */
            while ((c = f->ks_getc()) >= 0 && c != '>' && c != '@'){
                ;
            }
            if (c < 0){
                f->cancel();
                return c; /* end of file or error*/
            }
            last_char = c;
        } /* else: the first header char has been read in the previous call */

        seq.clear();
        qual.clear();
        
        if ((r=f->ks_getuntil(Stream::KS_SEP_SPACE, name, &c)) < 0){
            f->cancel();
            return r;  /* normal exit: EOF or error */
        }
        if (c != '\n'){
            f->ks_getuntil(Stream::KS_SEP_LINE, comment, 0); /* read FASTA/Q comment */
        }

        // if(seq.capacity() == 0){
        //     seq.reserve(256);
        // }

        while ((c = f->ks_getc()) >= 0 && c != '>' && c != '+' && c != '@') {
            if (c == '\n'){
                continue; /* skip empty lines */
            }
            seq.push_back(c);
            f->ks_getuntil2(Stream::KS_SEP_LINE, seq, 0, 1); /* read the rest of the line */
        }

        if (c == '>' || c == '@'){
            last_char = c; /* the first header char has been read */
        }
        if (c != '+'){
            return seq.length(); /* FASTA */
        }
        if(qual.capacity() < seq.capacity()){
            qual.reserve(seq.capacity()); /* allocate memory for qual in case insufficient */
        }
        
        while ((c = f->ks_getc()) >= 0 && c != '\n'){
            ; /* skip the rest of '+' line */
        }

        if (c == -1){
            f->cancel();
            return -2; /* error: no quality string */
        }

        while ((c = f->ks_getuntil2(Stream::KS_SEP_LINE, qual, 0, 1) >= 0 && qual.length() < seq.length())){
            ;
        }
        if (c == -3){
            f->cancel();
            return -3; /* stream error */
        }
        last_char = 0;	/* we have not come to the next header line */
        if(seq.length() != qual.length()){
            std::cerr << "got seq " << seq << "\n got qual " << qual << "\n";
            f->cancel();
            return -2;  /* error: qual string is of a different length */
        }
        
        return seq.length();
    }

    void kseq_rewind(){
        last_char = 0;
        f->is_eof = 0;
        f->begin = 0;
        f->end = 0;
    }

    const std::string& getCurrentName() const{
        return name;
    }

    const std::string& getCurrentComment() const{
        return comment;
    }

    const std::string& getCurrentSequence() const{
        return seq;
    }

    const std::string& getCurrentQuality() const{
        return qual;
    }

private:

    struct kstream_t {
        static constexpr std::size_t bufsize = 16384;

        static constexpr int KS_SEP_SPACE = 0;
        static constexpr int KS_SEP_TAB = 1;
        static constexpr int KS_SEP_LINE = 2;
        static constexpr int KS_SEP_MAX = 2;

        bool running = false;
        bool canContinue = true;
        int begin; 
        int end;
        int is_eof;
        std::vector<char> buf;
        std::unique_ptr<FileReader> filereader;

        kstream_t() = default;

        kstream_t(const std::string& filename) : begin(0), end(0), is_eof(0){

            if(hasGzipHeader(filename)){
                //std::cerr << "assume gz file\n";
                filereader.reset(new GzReader(filename));      
            }else{
                //std::cerr << "assume raw file\n";
                filereader.reset(new RawReader(filename));
            }

            buf.resize(bufsize);
        }

        kstream_t(kstream_t&& rhs){
            *this = std::move(rhs);
        }

        kstream_t& operator=(kstream_t&& rhs){
            buf = std::move(rhs.buf);
            begin = std::move(rhs.begin);
            end = std::move(rhs.end);
            is_eof = std::move(rhs.is_eof);
            canContinue = std::move(rhs.canContinue);
            filereader = std::move(rhs.filereader);

            return *this;
        }

        void cancel(){}

        bool ks_err() const{
            return end == -1;
        }

        bool ks_eof() const{
            return is_eof && begin >= end;
        }

        void ks_rewind(){
            is_eof = 0;
            begin = 0;
            end = 0;
        }

        int fillBuffer(){
            return filereader->read(buf.data(), bufsize);
        }

        int ks_getc(){
            if (ks_err()) return -3;
            if (ks_eof()) return -1;
            if (begin >= end) {
                begin = 0;
                end = fillBuffer();
                if (end == 0){
                    is_eof = 1; 
                    return -1;
                }
                if (end == -1){
                    is_eof = 1; 
                    return -3;
                }
            }
            return (int)buf[begin++];
        }

        int ks_getuntil2(int delimiter, std::string& str, int* dret, int append){
            auto kroundup32 = [](unsigned int x){
                --x;
                x |= x >> 1;
                x |= x >> 2;
                x |= x >> 4;
                x |= x >> 8;
                x |= x >> 16;
                ++x;
                return x;
            } ;

            int gotany = 0;													
            if (dret) *dret = 0;
            if(!append){
                str.clear();
            }																			
            for (;;) {														
                                                                    
                if (ks_err()) return -3;									
                if (begin >= end) {									
                    if (!is_eof) {										
                        begin = 0;										
                        end = fillBuffer();		
                        if (end == 0) {
                            is_eof = 1;
                            break;
                        }else if (end == -1){
                            is_eof = 1; 
                            return -3;
                        }	
                            
                    }else{
                        break;
                    }
                }
                int i = 0;																
                if (delimiter == KS_SEP_LINE) { 
                    for (i = begin; i < end; ++i){
                        if(buf[i] == '\n'){
                            break;
                        }
                    }                        
                } else if (delimiter > KS_SEP_MAX) {						
                    for (i = begin; i < end; ++i){
                        if(buf[i] == delimiter){
                            break;
                        }
                    }								
                } else if (delimiter == KS_SEP_SPACE) {		
                    for (i = begin; i < end; ++i){
                        if(std::isspace(buf[i])){
                            break;
                        }
                    }									
                } else if (delimiter == KS_SEP_TAB) {
                    for (i = begin; i < end; ++i){
                        if(buf[i] != ' ' && std::isspace(buf[i])){
                            break;
                        }
                    }
                }

                const int oldSize = str.length();
                const int oldCapacity = str.capacity();
                if(oldCapacity - oldSize < i - begin){
                    int newCapacity = oldSize + i - begin;
                    newCapacity = kroundup32(newCapacity);
                    str.reserve(newCapacity);                    
                }
                str.resize(oldSize + i - begin);

                gotany = 1;	

                std::copy_n(buf.data() + begin,	i - begin, &str[oldSize]);	
                begin = i + 1;											
                if (i < end) {											
                    if (dret){
                        *dret = buf[i];
                    } 
                    break;													
                }															
            }																
            if (!gotany && ks_eof()){
                return -1;
            }
            if(delimiter == KS_SEP_LINE && str.length() > 1 && str.back() == '\r'){
                str.pop_back();
            }
                                        
            return str.length();													
        }

        int ks_getuntil(int delimiter, std::string& str, int* dret){
            return ks_getuntil2(delimiter, str, dret, 0);
        }
    };

    struct asynckstream_t {
        static constexpr std::size_t bufsize = 16384;

        static constexpr int KS_SEP_SPACE = 0;
        static constexpr int KS_SEP_TAB = 1;
        static constexpr int KS_SEP_LINE = 2;
        static constexpr int KS_SEP_MAX = 2;

        struct ThreadSyncData{
            std::mutex m;
            std::condition_variable cv_producer;
            std::condition_variable cv_consumer;
        };

        bool running = false;
        bool canContinue = true;
        bool tempBufferFilled = false;
        int tempReadBytes;
        int begin; 
        int end;
        int is_eof;
        std::vector<char> buf;
        std::vector<char> tempbuf;
        std::unique_ptr<FileReader> filereader;
        std::unique_ptr<ThreadSyncData> threadSyncData = std::make_unique<ThreadSyncData>();
        std::thread fillerThread{};

        asynckstream_t() = default;

        asynckstream_t(const std::string& filename) : begin(0), end(0), is_eof(0){

            if(hasGzipHeader(filename)){
                //std::cerr << filename << " : assume gz file\n";
                filereader.reset(new ZlibReader(filename));      
            }else{
                //std::cerr << filename << " : assume raw file\n";
                filereader.reset(new ZlibReader(filename));
            }

            buf.resize(bufsize);
            tempbuf.resize(bufsize);

            fillerThread = std::move(std::thread([&](){fillerthreadfunc();}));

            running = true;
        }

        ~asynckstream_t(){
            if(running){
                cancel();
                fillerThread.join();
            }
        }

        asynckstream_t(const asynckstream_t&) = delete;
        asynckstream_t(asynckstream_t&&) = delete;
        asynckstream_t& operator=(const asynckstream_t&) = delete;
        asynckstream_t& operator=(asynckstream_t&&) = delete;

        void fillerthreadfunc(){
            //std::cerr << "launched thread\n";
            std::vector<char> threadBuffer;
            threadBuffer.resize(bufsize);

            int n = 0;
            do{
                n = filereader->read(threadBuffer.data(), bufsize);

                std::unique_lock<std::mutex> ul(threadSyncData->m);
                if(!canContinue){
                    break;
                }
                if(tempBufferFilled){
                    //std::cerr << "filereaderThread: temp buffer still filled\n";
                    threadSyncData->cv_producer.wait(ul, [&](){return !tempBufferFilled || !canContinue;});
                }

                tempBufferFilled = true;
                if(canContinue){
                    std::swap(threadBuffer, tempbuf);                    
                    tempReadBytes = n;
                    //std::cerr << "filereaderThread: temp buffer filled\n";                    
                }else{
                    tempReadBytes = 0;
                }

                threadSyncData->cv_consumer.notify_one();
            }while(n > 0 && canContinue);

            //std::cerr << "finished thread\n";
        }

        void cancel(){
            std::unique_lock<std::mutex> ul(threadSyncData->m);
            canContinue = false;
            threadSyncData->cv_producer.notify_one();
        }

        bool ks_err() const{
            return end == -1;
        }

        bool ks_eof() const{
            return is_eof && begin >= end;
        }

        void ks_rewind(){
            is_eof = 0;
            begin = 0;
            end = 0;
        }

        int fillBuffer(){
            std::unique_lock<std::mutex> ul(threadSyncData->m);
            if(!tempBufferFilled){
                //std::cerr << "main thread: temp buffer still not filled\n";
                threadSyncData->cv_consumer.wait(ul, [&](){return tempBufferFilled;});
            }

            std::swap(buf, tempbuf);
            int numRead = tempReadBytes;
            tempBufferFilled = false;
            //std::cerr << "main thread: temp buffer not filled\n";
            threadSyncData->cv_producer.notify_one();

            return numRead;
        }

        int ks_getc(){
            if (ks_err()) return -3;
            if (ks_eof()) return -1;
            if (begin >= end) {
                begin = 0;
                end = fillBuffer();
                if (end == 0){
                    is_eof = 1; 
                    return -1;
                }
                if (end == -1){
                    is_eof = 1; 
                    return -3;
                }
            }
            return (int)buf[begin++];
        }

        int ks_getuntil2(int delimiter, std::string& str, int* dret, int append){
            auto kroundup32 = [](unsigned int x){
                --x;
                x |= x >> 1;
                x |= x >> 2;
                x |= x >> 4;
                x |= x >> 8;
                x |= x >> 16;
                ++x;
                return x;
            } ;

            int gotany = 0;													
            if (dret) *dret = 0;
            if(!append){
                str.clear();
            }																			
            for (;;) {														
                                                                    
                if (ks_err()) return -3;									
                if (begin >= end) {									
                    if (!is_eof) {										
                        begin = 0;										
                        end = fillBuffer();		
                        if (end == 0) {
                            is_eof = 1;
                            break;
                        }else if (end == -1){
                            is_eof = 1; 
                            return -3;
                        }	
                            
                    }else{
                        break;
                    }
                }
                int i = 0;																
                if (delimiter == KS_SEP_LINE) { 
                    for (i = begin; i < end; ++i){
                        if(buf[i] == '\n'){
                            break;
                        }
                    }                        
                } else if (delimiter > KS_SEP_MAX) {						
                    for (i = begin; i < end; ++i){
                        if(buf[i] == delimiter){
                            break;
                        }
                    }								
                } else if (delimiter == KS_SEP_SPACE) {		
                    for (i = begin; i < end; ++i){
                        if(std::isspace(buf[i])){
                            break;
                        }
                    }									
                } else if (delimiter == KS_SEP_TAB) {
                    for (i = begin; i < end; ++i){
                        if(buf[i] != ' ' && std::isspace(buf[i])){
                            break;
                        }
                    }
                }

                const int oldSize = str.length();
                const int oldCapacity = str.capacity();
                if(oldCapacity - oldSize < i - begin){
                    int newCapacity = oldSize + i - begin;
                    newCapacity = kroundup32(newCapacity);
                    str.reserve(newCapacity);                    
                }
                str.resize(oldSize + i - begin);

                gotany = 1;	

                std::copy_n(buf.data() + begin,	i - begin, &str[oldSize]);	
                begin = i + 1;											
                if (i < end) {											
                    if (dret){
                        *dret = buf[i];
                    } 
                    break;													
                }															
            }																
            if (!gotany && ks_eof()){
                return -1;
            }
            if(delimiter == KS_SEP_LINE && str.length() > 1 && str.back() == '\r'){
                str.pop_back();
            }
                                        
            return str.length();													
        }

        int ks_getuntil(int delimiter, std::string& str, int* dret){
            return ks_getuntil2(delimiter, str, dret, 0);
        }
    };

    //using Stream = kstream_t;
    using Stream = asynckstream_t;

    std::string name{};
    std::string comment{};
    std::string seq{};
    std::string qual{};	

    int last_char{};
    std::unique_ptr<Stream> f{};	

};





} // namespace kseqpp















#endif
