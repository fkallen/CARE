
all: arteval artevalsmallercorrected

arteval:
	g++ -I../include/kseqpp/ -O3 arteval.cpp -lz -lpthread -o arteval
artevalsmallercorrected:
	g++ -I../include/kseqpp/ -O3 artevalsmallercorrected.cpp -lz -lpthread -o artevalsmallercorrected


SEQAN3PATH=/git/seqan3

kmertools: findkmersingenome findmissingkmers

findkmersingenome:
	g++ -std=c++17 -O3 -g -fconcepts -fopenmp \
		-I$(SEQAN3PATH)/include \
		-isystem $(SEQAN3PATH)/submodules/range-v3/include \
		-isystem $(SEQAN3PATH)/submodules/sdsl-lite/include \
		-isystem $(SEQAN3PATH)/submodules/cereal/include \
		findkmersingenome.cpp -lpthread -o findkmersingenome

findmissingkmers: findmissingkmers.cpp
	g++ -std=c++17 -O3 -g findmissingkmers.cpp -o findmissingkmers

clean:
	rm -f arteval artevalsmallercorrected findkmersingenome findmissingkmers