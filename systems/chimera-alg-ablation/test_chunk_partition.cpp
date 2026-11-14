#include "chimera/chunk_partition.hpp"
#include <cassert>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Chimera;
int main() {
    int cases = 0, empty_cases = 0;
    for (int n = 0; n <= 65; ++n) for (int requested = 1; requested <= 80; ++requested) {
        auto [count,size] = ceil_chunk_partition(n,requested);
        assert(count == std::min(n,requested));
        if (!n) { assert(size==0); ++cases; continue; }
        std::vector<int> cover(n,0);
        int previous=0, original_valid=0;
        for(int c=0;c<count;++c) {
            auto [start,end]=clamped_chunk_bounds(n,size,c);
            assert(start==previous && start<=end && end<=n); previous=end;
            for(int i=start;i<end;++i) ++cover[i];
            if(c*size<n) { assert(start==c*size && end==std::min(c*size+size,n)); ++original_valid; }
            else { assert(start==n && end==n); ++empty_cases; }
            for(int target : {1,2,4,10,400}) {
                const int full=std::min(target,n);
                const int carry=std::min(full,end);
                const int budget=std::min(full*(c+1)/count,carry);
                assert(budget>=0 && budget<=carry);
                if(c+1==count) assert(budget==full);
            }
        }
        for(int x:cover) assert(x==1);
        assert(previous==n); ++cases;
    }
    for (int n : {INT_MAX-1,INT_MAX}) for(int requested : {1,2,6,10,31}) {
        auto [count,size]=ceil_chunk_partition(n,requested); int previous=0;
        for(int c=0;c<count;++c) { auto [start,end]=clamped_chunk_bounds(n,size,c);assert(start==previous&&end>=start&&end<=n);previous=end; }
        assert(previous==n);++cases;
    }
    auto [c,s]=ceil_chunk_partition(11,10); assert(c==10&&s==2);
    assert(clamped_chunk_bounds(11,s,9)==std::make_pair(11,11));
    int rejected=0;
    for(auto pair:std::vector<std::pair<int,int>>{{-1,10},{1,0},{1,-2}}) {
        try{ceil_chunk_partition(pair.first,pair.second);}catch(const std::invalid_argument&){++rejected;}
    }
    assert(rejected==3);
    std::cout << "PASS chunk_cases="<<cases<<" empty_tail_ranges="<<empty_cases<<" invalid_cases="<<rejected<<"\n";
}
