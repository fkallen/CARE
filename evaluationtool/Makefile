
all: arteval artevalsmallercorrected

arteval:
	g++ -I../include/kseqpp/ -O3 arteval.cpp -lz -lpthread -o arteval
artevalsmallercorrected:
	g++ -I../include/kseqpp/ -O3 artevalsmallercorrected.cpp -lz -lpthread -o artevalsmallercorrected

clean:
	rm -f arteval artevalsmallercorrected