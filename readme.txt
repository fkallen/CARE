Build with Makefile.
cpu version: make / make cpu
gpu version: make gpu

Candidate selection:

Candidate reads for correction are found by minhashing.

Given a hash function which hashes k-mers to another k-mer (max 64 bit value, 2 bits per base), 
we calculate hash values of each k-mer in a read and use the smallest hash value (minhash) to identify the read.
Reads with the same minhash as the query are considered as correction candidates.
We store {minhash, read number} in a map.
It is possible to use multiple hash functions (multiple maps). Then, candidates have to share at least one minhash with the query.

The {minhash, read number} tuple is packed into a 64 bit unsigned integer.
The bits can be split arbitrarily in file "inc/minhasher.hpp"

static constexpr std::uint64_t bits_key = 39;
static constexpr std::uint64_t bits_val = 25;

One bit of bits_val is reserved as a flag.

bits_vals need to be chosen large enough to enumerate all reads in the file with (bits_vals-1) bits, 
i.e. with above settings the file must not contain more than 16,777,216 reads.

Then, the maximum k-mer size is effectivly (bits_key + 1) / 2, i.e with above settings maximum k = 20, where one bit is lost (40 - 39).


Alignment:

Hamming distance. only substitution errors. allows to correct candidates, too.
semi-global alignment. substitution + indels.


Correction:

We use an exponential threshold during the voting phase to control when a base is corrected.
If an original base occurs N times and a foreign base occurs M times, the original base is corrected into the foreign base 
if M-N >= aa*pow(xx,N), where aa and xx are free parameters.




Usually we use: 
k=16,14,12
maps=4,6,8
aa=1.0
xx=1.1,1.2,1.3

Observations:

The smaller k, (the more candidates, slower), the more true positive corrections
The more maps, (the more candidates, slower), the more true positive corrections
False positives depend on dataset. we have seen both decrease and increase of FP with k / maps

The larger xx / aa, the less corrections are performed which reduces both TP and FP

