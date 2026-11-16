#pragma once
#include "acl/acl.h"
#include <cstdint>
#include <cstddef>
#include <stdexcept>
namespace GmgHost {
class Buffer {
public:
    uint8_t* ptr=nullptr;
    explicit Buffer(size_t bytes) {
        if(aclrtMalloc(reinterpret_cast<void**>(&ptr),bytes,ACL_MEM_MALLOC_HUGE_FIRST)!=ACL_SUCCESS)
            throw std::runtime_error("workspace allocation failed");
    }
    ~Buffer(){if(ptr)aclrtFree(ptr);}
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};
inline void Validate(const TensorGroupInfo& x,const TensorGroupInfo& w,
    const TensorGroupInfo& s,const TensorGroupInfo& g,const TensorGroupInfo& y) {
    const TensorGroupInfo* all[]={&x,&w,&s,&g,&y};
    const int rank[]={2,3,2,1,2},dtype[]={3,3,0,6,1};
    for(int i=0;i<5;++i)
        if(all[i]->numTensors!=1 || !all[i]->tensors ||
           all[i]->tensors[0].numDims!=rank[i] || all[i]->tensors[0].dtype!=dtype[i] ||
           !all[i]->tensors[0].shape)throw std::invalid_argument("invalid tensor metadata");
    const int64_t m=x.tensors[0].shape[0],k=x.tensors[0].shape[1];
    const int64_t e=w.tensors[0].shape[0],n=w.tensors[0].shape[2];
    if(m<1 || m>65536 || k<1 || k>8192 || e<1 || e>128 || n<1 || n>4096 ||
       w.tensors[0].shape[1]!=k || s.tensors[0].shape[0]!=e || s.tensors[0].shape[1]!=n ||
       g.tensors[0].shape[0]!=e || y.tensors[0].shape[0]!=m || y.tensors[0].shape[1]!=n ||
       e*k*n>268435456LL || m*k+e*k*n+4*e*n+8*e+2*m*n>536870912LL)
        throw std::invalid_argument("incompatible tensor shapes");
}
}
