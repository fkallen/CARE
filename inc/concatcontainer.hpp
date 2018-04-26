#ifndef CARE_CONCAT_CONTAINER_HPP
#define CARE_CONCAT_CONTAINER_HPP

#include <array>
#include <cassert>
#include <iterator>

namespace care{

    template<class Iter, int N>
    struct ConcatContainer{

        struct ConcatIterator{
            typedef typename Iter::difference_type difference_type;
            typedef typename Iter::value_type value_type;
            typedef typename Iter::reference reference;
            typedef typename Iter::pointer pointer;
            typedef std::forward_iterator_tag iterator_category;

    		// begin stored on even index, end stored on odd index
    		//begin0, end0, begin1, end1,...
            std::array<Iter, 2*N> baseiters;
            Iter cur;

            int which_orig_range;

            ConcatIterator(){}

            ConcatIterator(const std::array<Iter, 2*N>& arr, Iter cur, int which_orig_range)
    			: baseiters(arr), cur(cur), which_orig_range(which_orig_range)
            {
    			//each range must be non-empty
    			for(int i = 0; i < N; i++)
    				assert(baseiters[2*i] != baseiters[2*i+1]);
            }

            bool operator==(const ConcatIterator& rhs) const{
                return cur == rhs.cur;
            }

            bool operator!=(const ConcatIterator& rhs) const{
                return !(*this == rhs);
            }

            ConcatIterator& operator++(){
                cur++;
    			//if we reached end of a range, switch to begin of next range
                if(cur != end_of_range(N-1) && cur == end_of_range(which_orig_range)){
    				which_orig_range++;
                    cur = begin_of_range(which_orig_range);
                }
                return *this;
            }

            ConcatIterator operator++(int){
                ConcatIterator old(*this);
                ++(*this);
                return old;
            }

            reference operator*() const{
                return *cur;
            }

            pointer operator->() const{
                return cur.operator->();
            }
    	private:
            Iter begin_of_range(int i){
    			return baseiters[2*i];
    		}

            Iter end_of_range(int i){
    			return baseiters[2*i+1];
    		}
        };

        using iterator = ConcatIterator;

        std::array<Iter, 2*N> baseiters;

        template<class... Iters>
    	ConcatContainer(Iter i1, Iter i2, Iters... is):
            baseiters{i1,i2,is...}
        {
            static_assert(sizeof...(Iters) % 2 == 0, "");
        }

        iterator begin() const{
            return iterator(baseiters, baseiters[0], 0);
        }

        iterator end() const{
            return iterator(baseiters, baseiters[2*N-1], N-1);
        }
    };

    template<class Iter, class... Iters, std::size_t N = ((sizeof...(Iters)+2)/2)>
    ConcatContainer<Iter,N> make_concat_container(Iter i1, Iter i2, Iters... is){
        return ConcatContainer<Iter,N>(i1, i2, is...);
    }

}



#endif
