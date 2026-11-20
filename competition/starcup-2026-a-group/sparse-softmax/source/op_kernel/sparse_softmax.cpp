#include "kernel_operator.h"
#include "sparse_softmax_tiling.h"
#include "tiling_key_sparse_softmax.h"
using namespace AscendC;

namespace { constexpr uint32_t CAP=4096; }
template<class IN, uint64_t PATH> class KernelSparseSoftmax {
 using T=float;
public:
 __aicore__ inline void Init(GM_ADDR src,GM_ADDR index,GM_ADDR ptr,GM_ADDR out,const SparseSoftmaxTilingData*data){
  t=data;srcGm.SetGlobalBuffer((__gm__ IN*)src);indexGm.SetGlobalBuffer((__gm__ int32_t*)index);ptrGm.SetGlobalBuffer((__gm__ int32_t*)ptr);outGm.SetGlobalBuffer((__gm__ IN*)out);
  pipe.InitBuffer(inBuf,Align(CAP*sizeof(IN)));pipe.InitBuffer(valBuf,Align(CAP*sizeof(T)));pipe.InitBuffer(tmpBuf,Align(CAP*sizeof(T)));pipe.InitBuffer(outBuf,Align(CAP*sizeof(T)));pipe.InitBuffer(idxBuf,Align((CAP+1)*sizeof(int32_t)));pipe.InitBuffer(outCastBuf,Align(CAP*sizeof(IN)));pipe.InitBuffer(groupBuf,Align(2*CAP*sizeof(T)));block=GetBlockIdx();blocks=GetBlockNum();
 }
 __aicore__ inline void Process(){
  if(t->axisLength==0||t->axisLength>CAP)return;
  if constexpr(PATH==1){
   for(uint64_t task=block;task<t->tasks;task+=blocks){TinyAllPairs8(task);StoreLine(task);}
  }else if constexpr(PATH==2){
   for(uint64_t task=block;task<t->tasks;task+=blocks){
    if(!TinyReduce8(task)){LoadLine(task);LoadPtr();PtrSoftmaxFast();}
    StoreLine(task);
   }
  }else if constexpr(PATH==3){
   for(uint64_t task=block;task<t->tasks;task+=blocks){
    if(t->mode==2u){
     if(!TinyReduce8<true>(task)){LoadLine(task);LoadPtr();PtrSoftmaxFast();}
     StoreLine(task);
    }else if(t->axisLength<=16u){TinyAllPairs16(task);StoreLine(task);}
    else {if(!TinyReduce8(task)){LoadLine(task);IndexSoftmaxFast();}StoreLine(task);}
   }
  }else if constexpr(PATH==4){
   // Minimal variant for the ptr, inner==1, axis<=8, CSR length 2..49 domain:
   // only the small bitmask reduction chain is instantiated, so none of the
   // general PtrSoftmax/LoadLine/LoadPtr/index code ends up in this variant's
   // object.
   for(uint64_t task=block;task<t->tasks;task+=blocks){
    if(!TinyReduce8<true>(task))return;
    StoreLine(task);
   }
  }else if constexpr(PATH==5){
   // Minimal variant for the index, inner==1, 8<axis<=16 domain.
   for(uint64_t task=block;task<t->tasks;task+=blocks){TinyAllPairs16(task);StoreLine(task);}
  }else{
   if(t->mode==2)LoadPtr();
   const bool tiny=(t->innerSize==1&&t->axisLength<=48u);   // 该快路径假定 axisLength<=48，更长走通用实现
   for(uint64_t task=block;task<t->tasks;task+=blocks){
    if(tiny){if(!TinyReduce8(task)){LoadLine(task);if(t->mode==1)IndexSoftmaxFast();else PtrSoftmaxFast();}StoreLine(task);}
    else {LoadLine(task);if(t->mode==1)IndexSoftmaxFast();else PtrSoftmaxFast();StoreLine(task);}
   }
  }
 }
private:
 static __aicore__ inline uint32_t Align(uint32_t n){return (n+31)/32*32;}
 __aicore__ inline uint64_t Offset(uint64_t task,uint32_t axis){uint64_t outer=task/t->innerSize,col=task%t->innerSize;return (outer*t->axisLength+axis)*t->innerSize+col;}
 __aicore__ inline void LoadLine(uint64_t task){
  auto raw=inBuf.Get<IN>();auto vals=valBuf.Get<T>();DataCopyPadExtParams<IN> pad(false,0,0,0);
  SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
  if(t->innerSize==1){
   DataCopyExtParams cp(1,t->axisLength*sizeof(IN),0,0,0);if constexpr(IsSameType<IN,T>::value)DataCopyPad(vals,srcGm[Offset(task,0)],cp,pad);else DataCopyPad(raw,srcGm[Offset(task,0)],cp,pad);
   if(t->mode==1){auto ids=idxBuf.Get<int32_t>();DataCopyPadExtParams<int32_t> ip(false,0,0,0);DataCopyExtParams ci(1,t->axisLength*sizeof(int32_t),0,0,0);uint64_t io=t->indexBroadcast?0:Offset(task,0);DataCopyPad(ids,indexGm[io],ci,ip);}
   SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
   if constexpr(!IsSameType<IN,T>::value)Cast(vals,raw,RoundMode::CAST_NONE,t->axisLength);PipeBarrier<PIPE_V>();
  }else{
   auto ids=idxBuf.Get<int32_t>();
   for(uint32_t i=0;i<t->axisLength;++i){vals.SetValue(i,static_cast<float>(srcGm.GetValue(Offset(task,i))));if(t->mode==1){uint64_t io=t->indexBroadcast?i:Offset(task,i);ids.SetValue(i,indexGm.GetValue(io));}}
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
  }
 }
 __aicore__ inline void LoadPtr(){auto ids=idxBuf.Get<int32_t>();DataCopyExtParams cp(1,t->ptrLength*sizeof(int32_t),0,0,0);DataCopyPadExtParams<int32_t> pad(false,0,0,0);DataCopyPad(ids,ptrGm,cp,pad);SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);}

 // One row per core, one writer. All group membership and reductions stay in V.
 // The 8x8 matrix has one 32-byte block per output element.
 __aicore__ inline void TinyAllPairs8(uint64_t task){
  const uint32_t n=t->axisLength;
  auto raw=inBuf.Get<IN>();auto vals=valBuf.Get<T>();auto ids=idxBuf.Get<int32_t>();
  auto out=outBuf.Get<T>();auto z=tmpBuf.Get<T>();
  auto gi=groupBuf.Get<int32_t>();auto gu=groupBuf.Get<uint32_t>();auto gf=groupBuf.Get<T>();
  auto idi=gi[0];auto idj=gi[64];
  auto xj=gf[256];auto xmasked=gf[320];auto zj=gf[384];auto smasked=gf[448];
  auto mx=gf[512];auto sum=gf[520];auto member=groupBuf.Get<uint8_t>()[528*4];

  // Define all padded columns. Invalid columns have x=-FLT_MAX and z=0,
  // so a matching padded id cannot change the max or sum of a valid row.
  Duplicate(ids,0,8);Duplicate(vals,-3.402823466e38f,8);
  PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
  DataCopyPadExtParams<int32_t> ip(false,0,0,0);DataCopyPadExtParams<IN> pad(false,0,0,0);
  DataCopyExtParams ci(1,n*sizeof(int32_t),0,0,0),cp(1,n*sizeof(IN),0,0,0);
  DataCopyPad(ids,indexGm[t->indexBroadcast?0:Offset(task,0)],ci,ip);
  if constexpr(IsSameType<IN,T>::value)DataCopyPad(vals,srcGm[Offset(task,0)],cp,pad);
  else DataCopyPad(raw,srcGm[Offset(task,0)],cp,pad);
  SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  if constexpr(!IsSameType<IN,T>::value){Cast(vals,raw,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}

  // Repeat the 32-byte source row eight times. CopyRepeatParams strides are
  // in vector blocks: advance the destination by one block and keep the
  // source at block zero. This replaces the offset-generation chain + Gather.
  Brcb(idi,ids,1,BrcbRepeatParams(1,8));
  Copy(idj,ids,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
  Copy(xj,vals,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
  PipeBarrier<PIPE_V>();
  Compare(member,idi,idj,CMPMODE::EQ,64);
  PipeBarrier<PIPE_V>();
  Select(xmasked,member,xj,-3.402823466e38f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
  PipeBarrier<PIPE_V>();
  BlockReduceMax(mx,xmasked,1,64,1,1,8);
  PipeBarrier<PIPE_V>();
  Duplicate(z,0.0f,8);
  PipeBarrier<PIPE_V>();
  Sub(z,vals,mx,n);PipeBarrier<PIPE_V>();Exp(z,z,n);
  PipeBarrier<PIPE_V>();
  Copy(zj,z,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
  PipeBarrier<PIPE_V>();
  Select(smasked,member,zj,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
  PipeBarrier<PIPE_V>();
  BlockReduceSum(sum,smasked,1,64,1,1,8);
  PipeBarrier<PIPE_V>();
  Adds(sum,sum,t->eps,n);PipeBarrier<PIPE_V>();Div(out,z,sum,n);
  PipeBarrier<PIPE_V>();
 }

 // Batched 16x16 equality matrix for index softmax, limited to n<=16.
 // Rows are output elements; columns are candidate members of each group.
 __aicore__ inline void TinyAllPairs16(uint64_t task){
  const uint32_t n=t->axisLength;
  auto raw=inBuf.Get<IN>();auto vals=valBuf.Get<T>();auto ids=idxBuf.Get<int32_t>();
  auto out=outBuf.Get<T>();auto z=tmpBuf.Get<T>();
  auto gi=groupBuf.Get<int32_t>();auto gf=groupBuf.Get<T>();
  // Byte-disjoint, 32-byte-aligned regions in groupBuf.
  auto idi=gi[0];             // [0, 1024)
  auto idj=gi[256];            // [1024, 2048)
  auto xj=gf[512];             // [2048, 3072)
  auto xmasked=gf[768];        // [3072, 4096)
  auto zj=gf[1024];             // [4096, 5120)
  auto smasked=gf[1280];       // [5120, 6144)
  auto mx=gf[1536];             // [6144, 6208)
  auto sum=gf[1552];            // [6208, 6272)
  auto member=groupBuf.Get<uint8_t>()[6272];

  // Define all 16 input lanes before the short DMA; valid lanes overwrite the
  // prefix. This makes the full 16x16 Select inputs defined for n<16.
  Duplicate(ids,0,16);Duplicate(vals,-3.402823466e38f,16);Duplicate(z,0.0f,16);
  PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
  DataCopyPadExtParams<int32_t> ip(false,0,0,0);DataCopyPadExtParams<IN> pad(false,0,0,0);
  DataCopyExtParams ci(1,n*sizeof(int32_t),0,0,0),cp(1,n*sizeof(IN),0,0,0);
  DataCopyPad(ids,indexGm[t->indexBroadcast?0:Offset(task,0)],ci,ip);
  if constexpr(IsSameType<IN,T>::value)DataCopyPad(vals,srcGm[Offset(task,0)],cp,pad);
  else DataCopyPad(raw,srcGm[Offset(task,0)],cp,pad);
  SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  if constexpr(!IsSameType<IN,T>::value){Cast(vals,raw,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}

  // idi[i,j]=ids[i]. The two calls populate the left and right 8-column
  // halves of every row; the repeat stride advances through the 16 rows.
  Brcb(idi,ids,2,BrcbRepeatParams(2,16));
  Brcb(idi[8],ids,2,BrcbRepeatParams(2,16));
  // Expand the left and right 8-column halves of each 16-column row.
  Copy(idj,ids,uint64_t(64),uint8_t(2),CopyRepeatParams(2,0,16,0));
  Copy(idj[8],ids[8],uint64_t(64),uint8_t(2),CopyRepeatParams(2,0,16,0));
  Copy(xj,vals,uint64_t(64),uint8_t(2),CopyRepeatParams(2,0,16,0));
  Copy(xj[8],vals[8],uint64_t(64),uint8_t(2),CopyRepeatParams(2,0,16,0));
  PipeBarrier<PIPE_V>();

  // One membership matrix is reused for both masked reductions.
  Compare(member,idi,idj,CMPMODE::EQ,uint32_t(256));
  PipeBarrier<PIPE_V>();
  Select(xmasked,member,xj,-3.402823466e38f,
         SELMODE::VSEL_TENSOR_SCALAR_MODE,uint32_t(256));
  PipeBarrier<PIPE_V>();
  WholeReduceMax(mx,xmasked,static_cast<int32_t>(n),static_cast<int32_t>(n),1,1,2,
                 ReduceOrder::ORDER_ONLY_VALUE);
  PipeBarrier<PIPE_V>();
  Sub(z,vals,mx,n);Exp(z,z,n);PipeBarrier<PIPE_V>();

  // Repeat z across matrix rows; padded z lanes are already zero.
  Copy(zj,z,uint64_t(64),uint8_t(2),CopyRepeatParams(2,0,16,0));
  Copy(zj[8],z[8],uint64_t(64),uint8_t(2),CopyRepeatParams(2,0,16,0));
  PipeBarrier<PIPE_V>();
  Select(smasked,member,zj,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,uint32_t(256));
  PipeBarrier<PIPE_V>();
  WholeReduceSum(sum,smasked,static_cast<int32_t>(n),static_cast<int32_t>(n),1,1,2);
  PipeBarrier<PIPE_V>();
  Adds(sum,sum,t->eps,n);PipeBarrier<PIPE_V>();Div(out,z,sum,n);PipeBarrier<PIPE_V>();
 }

 // 单行、≤8 元素：用"位掩码 + WholeReduce"求每组归约，再用字节偏移 Gather 取回每元素的组值。
 // 组号必须落在 [0,8) 且组跨度 < CAP，否则返回 false 交给通用路（域内仍正确）。
 template<bool MASK_PTR_GIDS=false>
 __aicore__ inline bool TinyReduce8(uint64_t task){
  const uint32_t n=t->axisLength;
  auto vals=valBuf.Get<T>();auto ids=idxBuf.Get<int32_t>();auto out=outBuf.Get<T>();auto tmp=tmpBuf.Get<T>();
  auto gi=groupBuf.Get<int32_t>();auto gf=groupBuf.Get<float>();auto gu=groupBuf.Get<uint32_t>();
  // Keep each vector and reduction table in a separate 32-byte-aligned UB region.
  // WholeReduce stores each result in an 8-byte slot; up to 48 groups fit here.
  auto gidv=gi[0];auto mvec=gf[64];auto svec=gf[128];
  auto mred=gf[192];auto sred=gf[320];auto gint=gi[448];auto offv=gu[448];
  int32_t lo=0,hi=0;int64_t span=0;
  uint64_t mk[64];for(uint32_t z=0;z<64;++z)mk[z]=0;
  auto raw=inBuf.Get<IN>();DataCopyPadExtParams<IN> pad(false,0,0,0);DataCopyPadExtParams<int32_t> ip(false,0,0,0);
  DataCopyExtParams cp(1,t->axisLength*sizeof(IN),0,0,0);
  if(t->mode==1){
   DataCopyExtParams ci(1,t->axisLength*sizeof(int32_t),0,0,0);
   DataCopyPad(ids,indexGm[t->indexBroadcast?0:Offset(task,0)],ci,ip);
   SetFlag<HardEvent::MTE2_S>(3);
   if constexpr(IsSameType<IN,T>::value)DataCopyPad(vals,srcGm[Offset(task,0)],cp,pad);else DataCopyPad(raw,srcGm[Offset(task,0)],cp,pad);
   WaitFlag<HardEvent::MTE2_S>(3);
  }else{
   // ptr：一趟完成"组掩码 + 字节偏移"，不再展开 gid、不再用 Adds/Muls
   DataCopyExtParams cp0(1,t->ptrLength*sizeof(int32_t),0,0,0);DataCopyPad(ids,ptrGm,cp0,ip);
   SetFlag<HardEvent::MTE2_S>(3);
   if constexpr(IsSameType<IN,T>::value)DataCopyPad(vals,srcGm[Offset(task,0)],cp,pad);else DataCopyPad(raw,srcGm[Offset(task,0)],cp,pad);
   WaitFlag<HardEvent::MTE2_S>(3);
   span=static_cast<int64_t>(t->ptrLength)-1;
   if(span<=0||span>48||span>=static_cast<int64_t>(CAP))return false;
   for(uint32_t g=0;g<(uint32_t)span;++g){
    int32_t a=ids.GetValue(g),b=ids.GetValue(g+1);if(a<0)a=0;if(b>(int32_t)n)b=(int32_t)n;
    if constexpr(MASK_PTR_GIDS){
     if(b>a){
      const uint64_t bits=((1ull<<static_cast<uint32_t>(b))-1ull) ^
                          ((1ull<<static_cast<uint32_t>(a))-1ull);
      mk[g]=bits;
      uint64_t maskBits[1]={bits};
      Duplicate(gint,static_cast<int32_t>(g*8),maskBits,1,1,8);
     }
    }else{
     // 更长的 CSR 轴沿用逐元素组号映射。
     for(int32_t j=a;j<b;++j){mk[g]|=(1ull<<j);gint.SetValue((uint32_t)j,(int32_t)(g*8));}
    }
   }
   lo=0;hi=static_cast<int32_t>(span)-1;
  }
  SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  if constexpr(!IsSameType<IN,T>::value){Cast(vals,raw,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}
  if(t->mode==1){
   SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
   lo=ids.GetValue(0);hi=lo;
   for(uint32_t i=1;i<n;++i){int32_t g=ids.GetValue(i);if(g<lo)lo=g;if(g>hi)hi=g;}
   span=static_cast<int64_t>(hi)-static_cast<int64_t>(lo)+1;
   if(lo<0||span<=0||span>48||span>=static_cast<int64_t>(CAP))return false;
   for(uint32_t i=0;i<n;++i){const uint32_t k=static_cast<uint32_t>(ids.GetValue(i)-lo);mk[k]|=(1ull<<i);}
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
   Adds(gidv,ids,-lo,n);
   Muls(gint,gidv,8,n);
  }else{
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
  }
  PipeBarrier<PIPE_V>();
  for(uint32_t g=0;g<(uint32_t)span;++g){
   if(mk[g]==0)continue;
   uint64_t mg[1];mg[0]=mk[g];
   WholeReduceMax(mred[2*g],vals,mg,1,1,1,1);
  }
  PipeBarrier<PIPE_V>();
  Gather(mvec,mred,offv,0,n);
  Sub(tmp,vals,mvec,n);
  Exp(tmp,tmp,n);
  PipeBarrier<PIPE_V>();
  for(uint32_t g=0;g<(uint32_t)span;++g){
   if(mk[g]==0)continue;
   uint64_t sg[1];sg[0]=mk[g];
   WholeReduceSum(sred[2*g],tmp,sg,1,1,1,1);
  }
  PipeBarrier<PIPE_V>();
  Gather(svec,sred,offv,0,n);
  Adds(svec,svec,t->eps,n);
  Div(out,tmp,svec,n);
  PipeBarrier<PIPE_V>();
  return true;
 }

 __aicore__ inline void SegmentSoftmax(uint32_t a,uint32_t b){
  const uint32_t n=b-a;
  if(n==0)return;
  auto out=outBuf.Get<T>();auto vals=valBuf.Get<T>();auto tmp=tmpBuf.Get<T>();
  if(n==1){out.SetValue(a,1.0f/(1.0f+t->eps));return;}
  float mx=-3.0e38f;
  for(uint32_t j=a;j<b;++j){float x=vals.GetValue(j);if(x>mx)mx=x;}
  for(uint32_t j=0;j<n;++j)tmp.SetValue(j,vals.GetValue(a+j)-mx);
  SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
  Exp(tmp,tmp,n);PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
  float sum=t->eps;
  for(uint32_t j=0;j<n;++j)sum+=tmp.GetValue(j);
  const float inv=1.0f/sum;
  for(uint32_t j=0;j<n;++j)out.SetValue(a+j,tmp.GetValue(j)*inv);
 }
 __aicore__ inline void IndexSoftmaxFast(){
  auto ids=idxBuf.Get<int32_t>();
  SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
  bool sorted=true,allSingleton=true;
  for(uint32_t i=1;i<t->axisLength;++i){
   const int32_t prev=ids.GetValue(i-1),cur=ids.GetValue(i);
   if(cur<prev){sorted=false;break;}
   if(cur==prev)allSingleton=false;
  }
  if(!sorted){DenseGroupSoftmax();return;}
  if(allSingleton){
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
   Duplicate(outBuf.Get<T>(),1.0f/(1.0f+t->eps),t->axisLength);
   PipeBarrier<PIPE_V>();return;
  }
  SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
  Duplicate(outBuf.Get<T>(),0.0f,t->axisLength);PipeBarrier<PIPE_V>();
  for(uint32_t a=0;a<t->axisLength;){
   const int32_t id=ids.GetValue(a);
   uint32_t b=a+1;
   while(b<t->axisLength&&ids.GetValue(b)==id)++b;
   SegmentSoftmax(a,b);
   a=b;
  }
  SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
 }
 __aicore__ inline void DenseGroupSoftmax(){
  auto ids=idxBuf.Get<int32_t>();auto vals=valBuf.Get<T>();auto tmp=tmpBuf.Get<T>();auto out=outBuf.Get<T>();
  int32_t lo=ids.GetValue(0),hi=lo;
  for(uint32_t i=1;i<t->axisLength;++i){int32_t k=ids.GetValue(i);if(k<lo)lo=k;if(k>hi)hi=k;}
  if(static_cast<int64_t>(hi)-static_cast<int64_t>(lo)>=CAP){IndexSoftmax();return;}
  uint32_t groups=static_cast<uint32_t>(static_cast<int64_t>(hi)-static_cast<int64_t>(lo)+1);
  auto gmx=groupBuf.Get<T>();auto gsum=gmx[CAP];
  Duplicate(gmx,-3.0e38f,groups);Duplicate(gsum,t->eps,groups);PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
  for(uint32_t i=0;i<t->axisLength;++i){uint32_t k=static_cast<uint32_t>(ids.GetValue(i)-lo);float x=vals.GetValue(i);if(x>gmx.GetValue(k))gmx.SetValue(k,x);}
  for(uint32_t i=0;i<t->axisLength;++i){uint32_t k=static_cast<uint32_t>(ids.GetValue(i)-lo);tmp.SetValue(i,vals.GetValue(i)-gmx.GetValue(k));}
  SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
  Exp(tmp,tmp,t->axisLength);PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
  for(uint32_t i=0;i<t->axisLength;++i){uint32_t k=static_cast<uint32_t>(ids.GetValue(i)-lo);gsum.SetValue(k,gsum.GetValue(k)+tmp.GetValue(i));}
  for(uint32_t i=0;i<t->axisLength;++i){uint32_t k=static_cast<uint32_t>(ids.GetValue(i)-lo);out.SetValue(i,tmp.GetValue(i)/gsum.GetValue(k));}
  SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
 }
 __aicore__ inline void PtrSoftmaxFast(){
  auto ptr=idxBuf.Get<int32_t>();bool allSingleton=true;
  for(uint32_t g=0;g+1<t->ptrLength;++g){
   if(ptr.GetValue(g+1)-ptr.GetValue(g)>1){allSingleton=false;break;}
  }
  if(allSingleton){
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
   Duplicate(outBuf.Get<T>(),1.0f/(1.0f+t->eps),t->axisLength);
   PipeBarrier<PIPE_V>();return;
  }
  PtrSoftmax();
 }
 __aicore__ inline void IndexSoftmax(){
  auto v=valBuf.Get<T>();auto tmp=tmpBuf.Get<T>();auto out=outBuf.Get<T>();auto ids=idxBuf.Get<int32_t>();Duplicate(out,0.0f,t->axisLength);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
  for(uint32_t i=0;i<t->axisLength;++i){int32_t g=ids.GetValue(i);bool seen=false;for(uint32_t j=0;j<i;++j)if(ids.GetValue(j)==g){seen=true;break;}if(seen)continue;float mx=-3.0e38f;for(uint32_t j=0;j<t->axisLength;++j)if(ids.GetValue(j)==g){float x=v.GetValue(j);if(x>mx)mx=x;}for(uint32_t j=0;j<t->axisLength;++j)tmp.SetValue(j,ids.GetValue(j)==g?v.GetValue(j)-mx:-80.0f);SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);Exp(tmp,tmp,t->axisLength);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);float sum=t->eps;for(uint32_t j=0;j<t->axisLength;++j)if(ids.GetValue(j)==g)sum+=tmp.GetValue(j);float inv=1.0f/sum;for(uint32_t j=0;j<t->axisLength;++j)if(ids.GetValue(j)==g)out.SetValue(j,tmp.GetValue(j)*inv);}
  SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
 }
 __aicore__ inline void PtrSoftmax(){
  auto v=valBuf.Get<T>();auto tmp=tmpBuf.Get<T>();auto out=outBuf.Get<T>();auto ptr=idxBuf.Get<int32_t>();Duplicate(out,0.0f,t->axisLength);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
  for(uint32_t g=0;g+1<t->ptrLength;++g){int32_t a=ptr.GetValue(g),b=ptr.GetValue(g+1);if(a<0)a=0;if(b>(int32_t)t->axisLength)b=t->axisLength;if(b<=a)continue;float mx=-3.0e38f;for(int32_t j=a;j<b;++j){float x=v.GetValue(j);if(x>mx)mx=x;}for(int32_t j=a;j<b;++j)tmp.SetValue(j-a,v.GetValue(j)-mx);SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);Exp(tmp,tmp,b-a);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);float sum=t->eps;for(int32_t j=0;j<b-a;++j)sum+=tmp.GetValue(j);float inv=1.0f/sum;for(int32_t j=a;j<b;++j)out.SetValue(j,tmp.GetValue(j-a)*inv);}
  SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
 }
 __aicore__ inline void StoreLine(uint64_t task){auto o=outBuf.Get<T>();if(t->innerSize==1){auto c=outCastBuf.Get<IN>();if constexpr(!IsSameType<IN,T>::value){Cast(c,o,RoundMode::CAST_RINT,t->axisLength);PipeBarrier<PIPE_V>();}SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);DataCopyExtParams cp(1,t->axisLength*sizeof(IN),0,0,0);if constexpr(IsSameType<IN,T>::value)DataCopyPad(outGm[Offset(task,0)],o,cp);else DataCopyPad(outGm[Offset(task,0)],c,cp);SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);}else{SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);for(uint32_t i=0;i<t->axisLength;++i)outGm.SetValue(Offset(task,i),static_cast<IN>(o.GetValue(i)));}}
 const SparseSoftmaxTilingData*t=nullptr;TPipe pipe;TBuf<TPosition::VECCALC>inBuf,valBuf,tmpBuf,outBuf,idxBuf,outCastBuf,groupBuf;GlobalTensor<IN>srcGm,outGm;GlobalTensor<int32_t>indexGm,ptrGm;uint32_t block=0,blocks=1;
};
// raw-UB fixed entry (PATH6/PATH7): a standalone __aicore__ helper with NO TPipe, NO
// InitBuffer, NO tiling struct and NO dynamic member access. It takes constant UB
// byte addresses through the verified ctor LocalTensor<T>(TPosition,uint32_t
// byteAddr,uint32_t elements) (kernel_tensor_impl.h:818; CreateTensor stores
// bufferAddr=addr and dataLen=tileSize*sizeof(T)) and reuses the 8x8 stable
// max-exp-sum math (TinyAllPairs8) verbatim with every size/offset a literal.
// src/index are read from the real inputs at runtime via SetGlobalBuffer; N and
// eps=1e-16f are compile constants; this helper covers the index input domain
// (index 模式、tasks=1、axis=8、inner=1、广播、eps=1e-16f、fp32/fp16 for PATH6;
// index 模式、tasks=2、axis=4、inner=1、广播、eps=1e-16f、fp32 for PATH7).
// The helper is templated on the row length N (default 8). PATH6 (tasks=1,
// axis=8) serves both the fp32 and the fp16 index domain; PATH7 (tasks=2,
// axis=4) serves the fp32 broadcast-index (2,4) domain. Only the row
// element counts (DMA/Cast/Sub/Exp/Adds/Div/Store) become N; the ids/vals/z
// padding init and every 8x8 matrix op (Brcb/Copy/Compare/BlockReduce) keep the
// 8-lane / 64-element constants, and all LocalTensor sub-offsets stay
// literal. GM row offset = GetBlockIdx()*N; index GM offset stays 0.
// UB layout (byte addresses, all 32B aligned, pairwise disjoint):
//   raw IN @0 (8), vals f32 @32, ids i32 @64, out f32 @96, tmp f32 @128,
//   outcast IN @160, group f32 @192 (832 f32 = 3328 B; sub-offsets are the
//   TinyAllPairs8 constants, all constant, no dynamic LocalTensor indexing).
template<class IN,uint32_t N=8u>
__aicore__ inline void RawUbIndex8AllPairs8(GM_ADDR src,GM_ADDR index,GM_ADDR out){
 using T=float;
 GlobalTensor<IN>srcGm;srcGm.SetGlobalBuffer((__gm__ IN*)src);
 GlobalTensor<int32_t>indexGm;indexGm.SetGlobalBuffer((__gm__ int32_t*)index);
 GlobalTensor<IN>outGm;outGm.SetGlobalBuffer((__gm__ IN*)out);
 LocalTensor<IN>raw(TPosition::VECCALC,0u,8u);
 LocalTensor<T>vals(TPosition::VECCALC,32u,8u);
 LocalTensor<int32_t>ids(TPosition::VECCALC,64u,8u);
 LocalTensor<T>outv(TPosition::VECCALC,96u,8u);
 LocalTensor<T>z(TPosition::VECCALC,128u,8u);
 LocalTensor<IN>outCast(TPosition::VECCALC,160u,8u);
 LocalTensor<int32_t>gi(TPosition::VECCALC,192u,832u);
 LocalTensor<T>gf(TPosition::VECCALC,192u,832u);
 LocalTensor<uint8_t>gb(TPosition::VECCALC,192u,832u*4u);
 auto idi=gi[0];auto idj=gi[64];
 auto xj=gf[256];auto xmasked=gf[320];auto zj=gf[384];auto smasked=gf[448];
 auto mx=gf[512];auto sum=gf[520];auto member=gb[528*4];

 // Define all padded columns. Invalid columns have x=-FLT_MAX and z=0,
 // so a matching padded id cannot change the max or sum of a valid row.
 Duplicate(ids,0,8);Duplicate(vals,-3.402823466e38f,8);
 PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
 DataCopyPadExtParams<int32_t> ip(false,0,0,0);DataCopyPadExtParams<IN> pad(false,0,0,0);
 DataCopyExtParams ci(1,N*sizeof(int32_t),0,0,0),cp(1,N*sizeof(IN),0,0,0);
 const uint32_t base=(N==8u)?0u:GetBlockIdx()*N;
 DataCopyPad(ids,indexGm[0],ci,ip);
 if constexpr(IsSameType<IN,T>::value)DataCopyPad(vals,srcGm[base],cp,pad);
 else DataCopyPad(raw,srcGm[base],cp,pad);
 SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 if constexpr(!IsSameType<IN,T>::value){Cast(vals,raw,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();}

 // Repeat the 32-byte source row eight times (fixed offsets/params, literals).
 Brcb(idi,ids,1,BrcbRepeatParams(1,8));
 Copy(idj,ids,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 Copy(xj,vals,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Compare(member,idi,idj,CMPMODE::EQ,64);
 PipeBarrier<PIPE_V>();
 Select(xmasked,member,xj,-3.402823466e38f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceMax(mx,xmasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Duplicate(z,0.0f,8);
 PipeBarrier<PIPE_V>();
 Sub(z,vals,mx,N);PipeBarrier<PIPE_V>();Exp(z,z,N);
 PipeBarrier<PIPE_V>();
 Copy(zj,z,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Select(smasked,member,zj,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceSum(sum,smasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Adds(sum,sum,1e-16f,N);PipeBarrier<PIPE_V>();Div(outv,z,sum,N);
 PipeBarrier<PIPE_V>();

 // StoreLine inner==1 tail, same DMA/HardEvent/Barrier sequence as the other paths (N elements).
 auto o=outv;auto c=outCast;
 if constexpr(!IsSameType<IN,T>::value){Cast(c,o,RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();}
 SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);
 DataCopyExtParams oc(1,N*sizeof(IN),0,0,0);
 if constexpr(IsSameType<IN,T>::value)DataCopyPad(outGm[base],o,oc);else DataCopyPad(outGm[base],c,oc);
 SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);
}
// raw-UB ptr helper (PATH8 N=8/L=5; shape domain: ptr 模式、
// tasks=1、axis=8、inner=1、ptrLength=5 和 ptr 模式、tasks=2、axis=4、inner=1、
// ptrLength=3，fp32; only the N=8/L=5 instantiation is still dispatched
// here, PATH9 has its own RawPtrPlane2x4 helper): the CSR group-id math
// g_m = #{k : m >= ptr[k]} is computed in the SCALAR domain (8 scalar
// SetValue writes read from the 20 B ptr UB tile; the earlier vector chain
// Cast ptrf + packed row0/S_V + Brcb rowm + Copy ptrm + Compare GE + Select cnt +
// BlockReduceSum + Cast gidv), then the 8x8 stable max-exp-sum AllPairs8 tail, fused
// with the raw-UB fixed-address entry (this path keeps only constant-address
// LocalTensor sub-views, with scalar GetValue on ptri and scalar SetValue on
// gidv; no CreateVecIndex/GetTPipePtr). No TPipe, no
// InitBuffer, no tiling struct, no dynamic LocalTensor indexing: N, L and eps
// are compile-time constants and every UB offset is constant, 32-byte
// aligned and pairwise disjoint, with gf/gi/gb sharing one group base:
//   vals f32 @0 (8), out f32 @32 (8), z f32 @64 (8),
//   group f32 @128 (1024 f32 = 4096 B; group sub-blocks end at 2880 B; only
//   ptri/gidv/idi/idj/... are still used -- rowm/ptrm/cnt/ones/gsum/row0/ptrf are gone).
// Only GM is dynamic: src/out at GetBlockIdx()*N, ptr at 0. Padded vals lanes
// (N..7) are pre-set to -FLT_MAX so padded lanes cannot join a real group (their
// ids are >= L > every real id) and contribute 0 to every max/sum; this path reads only
// ptr lanes 0..L-1, so the ptri right pad is neither defined nor read. Empty and
// uneven groups keep the exact reference
// semantics: the gid of an element is 1 + index of the last boundary <= m,
// which is the id of the unique owning non-empty group; a group with no
// members has no elements and therefore no output.
template<uint32_t N,uint32_t L,class IN>
__aicore__ inline void RawPtr(GM_ADDR src,GM_ADDR ptr,GM_ADDR out){
 using T=float;
 static_assert(IsSameType<IN,T>::value,"this raw-ptr path is fp32-only (tiling keys 8/9 are FLOAT)");
 GlobalTensor<IN>srcGm;srcGm.SetGlobalBuffer((__gm__ IN*)src);
 GlobalTensor<int32_t>ptrGm;ptrGm.SetGlobalBuffer((__gm__ int32_t*)ptr);
 GlobalTensor<IN>outGm;outGm.SetGlobalBuffer((__gm__ IN*)out);
 const uint64_t off=static_cast<uint64_t>(GetBlockIdx())*static_cast<uint64_t>(N);
 LocalTensor<T>vals(TPosition::VECCALC,0u,8u);
 LocalTensor<T>outv(TPosition::VECCALC,32u,8u);
 LocalTensor<T>z(TPosition::VECCALC,64u,8u);
 LocalTensor<T>gf(TPosition::VECCALC,128u,1024u);
 LocalTensor<int32_t>gi(TPosition::VECCALC,128u,1024u);
 LocalTensor<uint8_t>gb(TPosition::VECCALC,128u,4096u);
 // Byte-disjoint, 32-byte-aligned group sub-blocks (fixed constants). The
 // deleted vector gid chain leaves only the tensor views the tail still needs;
 // rowm/ptrm/cnt/ones/gsum/row0/ptrf are no longer declared (they were used only
 // by Cast/Brcb/Copy/Compare/Select/BlockReduceSum/Cast above).
 auto gidv=gi[272];     // [1088,1120) 8 i32 group id vector (scalar-written)
 auto ptri=gi[296];     // [1184,1216) 8 i32 ptr raw (only lanes 0..L-1 read)
 auto idi=gi[304];      // [1216,1472) 64 i32
 auto idj=gi[368];      // [1472,1728) 64 i32
 auto xj=gf[432];       // [1728,1984) 64 f32
 auto xmasked=gf[496];  // [1984,2240) 64 f32
 auto zj=gf[560];       // [2240,2496) 64 f32
 auto smasked=gf[624];  // [2496,2752) 64 f32
 auto mx=gf[688];       // [2752,2784) 8 f32
 auto sum=gf[696];      // [2784,2816) 8 f32
 auto member=gb[2816];  // [2816,2880) 64 u8

 // Define the padded src lanes before the short DMA. No ones fill (the
 // Select that consumed it is gone) and no N+1 ptri fill (the scalar chain below
 // reads only lanes 0..L-1, so the pad lanes of ptri are never read).
 Duplicate(vals,-3.402823466e38f,8);
 PipeBarrier<PIPE_V>();
 SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
 // src and ptr DMA back-to-back, unchanged (the 20 B ptr DMA is not on
 // the critical path, r64). The fence is split so the scalar gid chain only has
 // to wait for the 20 B ptr tile, not for the 32 B src row.
 DataCopyPadExtParams<int32_t> ipp(false,0,0,0);
 DataCopyPadExtParams<IN> pad(false,0,0,0);
 DataCopyExtParams cptr(1,L*sizeof(int32_t),0,0,0),cp(1,N*sizeof(IN),0,0,0);
 DataCopyPad(ptri,ptrGm,cptr,ipp);
 SetFlag<HardEvent::MTE2_S>(0);
 DataCopyPad(vals,srcGm[off],cp,pad);
 SetFlag<HardEvent::MTE2_V>(2);

 // Scalar-domain group ids: gidv[m] = #{k<5 : ptri[k] <= m}, derived on the
 // scalar unit instead of the earlier vector chain (Cast ptrf, packed row0 + S_V,
 // Brcb rowm, Copy ptrm, Compare GE, Select cnt, BlockReduceSum, Cast gidv).
 // Bit-exact for every legal CSR ptr (duplicates, empty groups, non-uniform):
 // the deleted 64-lane chain counted the same L real lanes plus the 3 right-pad
 // lanes, which held N+1 = 9 (larger than every m <= 7, so they never counted).
 // MTE2_S(0) is the file's own DMA-to-UB-then-scalar-GetValue idiom (see LoadPtr
 // in KernelSparseSoftmax), and the src fence MTE2_V(2) is taken after this
 // chain, so the 5 UB loads + 40 compares + 8 stores run in the 32 B src-DMA
 // shadow.
 WaitFlag<HardEvent::MTE2_S>(0);            // 20 B ptr tile has landed in UB
 int32_t p[5];
 #pragma unroll
 for(int32_t k=0;k<5;++k){p[k]=ptri.GetValue(k);}
 #pragma unroll
 for(int32_t m=0;m<8;++m){
  int32_t c=0;
  #pragma unroll
  for(int32_t k=0;k<5;++k){c+=((p[k]<=m)?1:0);}
  gidv.SetValue(m,c);
 }
 SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
 WaitFlag<HardEvent::MTE2_V>(2);           // only now may the tail touch vals

 // Membership/softmax tail copied from TinyAllPairs8 (original untouched), with
 // ids := gidv. Padded rows/columns get ids >= L (real ids are 1..L-1), so a
 // padded lane can never join a real group; padded vals are -FLT_MAX.
 Brcb(idi,gidv,1,BrcbRepeatParams(1,8));
 Copy(idj,gidv,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 Copy(xj,vals,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Compare(member,idi,idj,CMPMODE::EQ,64);
 PipeBarrier<PIPE_V>();
 Select(xmasked,member,xj,-3.402823466e38f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceMax(mx,xmasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Duplicate(z,0.0f,8);
 PipeBarrier<PIPE_V>();
 Sub(z,vals,mx,N);PipeBarrier<PIPE_V>();Exp(z,z,N);
 PipeBarrier<PIPE_V>();
 Copy(zj,z,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Select(smasked,member,zj,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceSum(sum,smasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Adds(sum,sum,1e-16f,N);PipeBarrier<PIPE_V>();Div(outv,z,sum,N);
 PipeBarrier<PIPE_V>();

 // StoreLine inner==1 tail, same DMA/HardEvent/Barrier sequence as the other paths.
 SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);
 DataCopyExtParams oc(1,N*sizeof(IN),0,0,0);
 DataCopyPad(outGm[off],outv,oc);
 SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);
}
// Family member: same raw-UB index algorithm as above but with the row length taken
// from the tiling struct at run time, so every small axis (1..8) is served by this
// one compiled instantiation. The 8-lane padding constants stay literal.
template<class IN>
__aicore__ inline void RawUbIndexAnyAllPairs8(GM_ADDR src,GM_ADDR index,GM_ADDR out,uint32_t n){
 using T=float;
 GlobalTensor<IN>srcGm;srcGm.SetGlobalBuffer((__gm__ IN*)src);
 GlobalTensor<int32_t>indexGm;indexGm.SetGlobalBuffer((__gm__ int32_t*)index);
 GlobalTensor<IN>outGm;outGm.SetGlobalBuffer((__gm__ IN*)out);
 LocalTensor<IN>raw(TPosition::VECCALC,0u,8u);
 LocalTensor<T>vals(TPosition::VECCALC,32u,8u);
 LocalTensor<int32_t>ids(TPosition::VECCALC,64u,8u);
 LocalTensor<T>outv(TPosition::VECCALC,96u,8u);
 LocalTensor<T>z(TPosition::VECCALC,128u,8u);
 LocalTensor<IN>outCast(TPosition::VECCALC,160u,8u);
 LocalTensor<int32_t>gi(TPosition::VECCALC,192u,832u);
 LocalTensor<T>gf(TPosition::VECCALC,192u,832u);
 LocalTensor<uint8_t>gb(TPosition::VECCALC,192u,832u*4u);
 auto idi=gi[0];auto idj=gi[64];
 auto xj=gf[256];auto xmasked=gf[320];auto zj=gf[384];auto smasked=gf[448];
 auto mx=gf[512];auto sum=gf[520];auto member=gb[528*4];

 // Define all padded columns. Invalid columns have x=-FLT_MAX and z=0,
 // so a matching padded id cannot change the max or sum of a valid row.
 Duplicate(ids,0,8);Duplicate(vals,-3.402823466e38f,8);
 PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
 DataCopyPadExtParams<int32_t> ip(false,0,0,0);DataCopyPadExtParams<IN> pad(false,0,0,0);
 DataCopyExtParams ci(1,n*sizeof(int32_t),0,0,0),cp(1,n*sizeof(IN),0,0,0);
 const uint32_t base=GetBlockIdx()*n;
 DataCopyPad(ids,indexGm[0],ci,ip);
 if constexpr(IsSameType<IN,T>::value)DataCopyPad(vals,srcGm[base],cp,pad);
 else DataCopyPad(raw,srcGm[base],cp,pad);
 SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 if constexpr(!IsSameType<IN,T>::value){Cast(vals,raw,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}

 // Repeat the 32-byte source row eight times (fixed offsets/params, literals).
 Brcb(idi,ids,1,BrcbRepeatParams(1,8));
 Copy(idj,ids,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 Copy(xj,vals,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Compare(member,idi,idj,CMPMODE::EQ,64);
 PipeBarrier<PIPE_V>();
 Select(xmasked,member,xj,-3.402823466e38f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceMax(mx,xmasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Duplicate(z,0.0f,8);
 PipeBarrier<PIPE_V>();
 Sub(z,vals,mx,n);PipeBarrier<PIPE_V>();Exp(z,z,n);
 PipeBarrier<PIPE_V>();
 Copy(zj,z,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Select(smasked,member,zj,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceSum(sum,smasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Adds(sum,sum,1e-16f,n);PipeBarrier<PIPE_V>();Div(outv,z,sum,n);
 PipeBarrier<PIPE_V>();

 // StoreLine inner==1 tail, same DMA/HardEvent/Barrier sequence as the other paths (n elements).
 auto o=outv;auto c=outCast;
 if constexpr(!IsSameType<IN,T>::value){Cast(c,o,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();}
 SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);
 DataCopyExtParams oc(1,n*sizeof(IN),0,0,0);
 if constexpr(IsSameType<IN,T>::value)DataCopyPad(outGm[base],o,oc);else DataCopyPad(outGm[base],c,oc);
 SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);
}

// Family member: same CSR group-id math as RawPtr but with row length and ptr length
// read from tiling, covering any 1<=axis<=8 with 2<=ptrLength<=49 in one instantiation.
template<class IN>
__aicore__ inline void RawPtrAny(GM_ADDR src,GM_ADDR ptr,GM_ADDR out,uint32_t n,uint32_t l){
 using T=float;
 static_assert(IsSameType<IN,T>::value,"this raw-ptr path is fp32-only (tiling keys 8/9 are FLOAT)");
 GlobalTensor<IN>srcGm;srcGm.SetGlobalBuffer((__gm__ IN*)src);
 GlobalTensor<int32_t>ptrGm;ptrGm.SetGlobalBuffer((__gm__ int32_t*)ptr);
 GlobalTensor<IN>outGm;outGm.SetGlobalBuffer((__gm__ IN*)out);
 const uint64_t off=static_cast<uint64_t>(GetBlockIdx())*static_cast<uint64_t>(n);
 LocalTensor<T>vals(TPosition::VECCALC,0u,8u);
 LocalTensor<T>outv(TPosition::VECCALC,32u,8u);
 LocalTensor<T>z(TPosition::VECCALC,64u,8u);
 LocalTensor<T>gf(TPosition::VECCALC,128u,1024u);
 LocalTensor<int32_t>gi(TPosition::VECCALC,128u,1024u);
 LocalTensor<uint8_t>gb(TPosition::VECCALC,128u,4096u);
 // Byte-disjoint, 32-byte-aligned group sub-blocks (fixed constants).
 auto rowm=gf[0];       // [0,  256) 64 f32: rowm[m*8+k]=m
 auto ptrm=gf[64];      // [256,512) 64 f32: ptrm[m*8+k]=ptr[k]
 auto cnt=gf[128];      // [512,768) 64 f32: 1/0 comparison matrix
 auto ones=gf[192];     // [768,1024) 64 f32
 auto gsum=gf[256];     // [1024,1056) 8 f32 per-row group ids
 auto gidv=gi[272];     // [1088,1120) 8 i32 group id vector
 auto row0=gf[280];     // [1120,1152) 8 f32 lanes 0..7
 auto ptrf=gf[288];     // [1152,1184) 8 f32 ptr as float
 auto ptri=gi[296];     // [1184,1216) 8 i32 ptr raw (right-padded with n+1)
 auto idi=gi[304];      // [1216,1472) 64 i32
 auto idj=gi[368];      // [1472,1728) 64 i32
 auto xj=gf[432];       // [1728,1984) 64 f32
 auto xmasked=gf[496];  // [1984,2240) 64 f32
 auto zj=gf[560];       // [2240,2496) 64 f32
 auto smasked=gf[624];  // [2496,2752) 64 f32
 auto mx=gf[688];       // [2752,2784) 8 f32
 auto sum=gf[696];      // [2784,2816) 8 f32
 auto member=gb[2816];  // [2816,2880) 64 u8

 // Define the padded src lanes, the padded ptr lanes (n+1) and the Select-ones
 // before the short DMA (the ptr DMA itself does not pad here).
 Duplicate(vals,-3.402823466e38f,8);Duplicate(ones,1.0f,64);
 Duplicate(ptri,static_cast<int32_t>(n+1u),8);
 PipeBarrier<PIPE_V>();
 SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
 // src and ptr DMA back-to-back; lanes l..7 of ptri keep the n+1 fill, so the
 // padded lanes are defined without a scalar fill.
 DataCopyPadExtParams<int32_t> ipp(false,0,0,0);
 DataCopyPadExtParams<IN> pad(false,0,0,0);
 DataCopyExtParams cptr(1,l*sizeof(int32_t),0,0,0),cp(1,n*sizeof(IN),0,0,0);
 DataCopyPad(ptri,ptrGm,cptr,ipp);
 DataCopyPad(vals,srcGm[off],cp,pad);
 SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);

 // Vector-only group ids: g_m = sum_k [m >= ptr[k]].
 Cast(ptrf,ptri,RoundMode::CAST_NONE,8);
 // row0 = 0..7 as 4 little-endian packed 64-bit UB constants written by scalar
 // SetValue into row0's own 32 B (gf[280] -> group base 128 + 280*4 = 1248,
 // 32 B aligned); one S_V flag restores scalar->vector order before the
 // unchanged Brcb/Copy below. The constants are data-independent lane positions
 // only; ptr is still read from GM. The old masked
 // Adds/Duplicate row0 build and CreateVecIndexOneBlk (which calls
 // GetTPipePtr()->FetchEventID(S_V) and faults without a TPipe) are gone.
 PipeBarrier<PIPE_V>();
 LocalTensor<uint64_t> posPacked(TPosition::VECCALC,1248u,4u);
 posPacked.SetValue(0,0x3f80000000000000ull);
 posPacked.SetValue(1,0x4040000040000000ull);
 posPacked.SetValue(2,0x40a0000040800000ull);
 posPacked.SetValue(3,0x40e0000040c00000ull);
 SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
 Brcb(rowm,row0,1,BrcbRepeatParams(1,8));
 Copy(ptrm,ptrf,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Compare(member,rowm,ptrm,CMPMODE::GE,uint32_t(64));
 PipeBarrier<PIPE_V>();
 Select(cnt,member,ones,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,uint32_t(64));
 PipeBarrier<PIPE_V>();
 BlockReduceSum(gsum,cnt,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Cast(gidv,gsum,RoundMode::CAST_RINT,8);
 PipeBarrier<PIPE_V>();

 // Membership/softmax tail copied from TinyAllPairs8 (original untouched), with
 // ids := gidv. Padded rows/columns get ids >= l (real ids are 1..l-1), so a
 // padded lane can never join a real group; padded vals are -FLT_MAX.
 Brcb(idi,gidv,1,BrcbRepeatParams(1,8));
 Copy(idj,gidv,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 Copy(xj,vals,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Compare(member,idi,idj,CMPMODE::EQ,64);
 PipeBarrier<PIPE_V>();
 Select(xmasked,member,xj,-3.402823466e38f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceMax(mx,xmasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Duplicate(z,0.0f,8);
 PipeBarrier<PIPE_V>();
 Sub(z,vals,mx,n);PipeBarrier<PIPE_V>();Exp(z,z,n);
 PipeBarrier<PIPE_V>();
 Copy(zj,z,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Select(smasked,member,zj,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceSum(sum,smasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Adds(sum,sum,1e-16f,n);PipeBarrier<PIPE_V>();Div(outv,z,sum,n);
 PipeBarrier<PIPE_V>();

 // StoreLine inner==1 tail, same DMA/HardEvent/Barrier sequence as the other paths.
 SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);
 DataCopyExtParams oc(1,n*sizeof(IN),0,0,0);
 DataCopyPad(outGm[off],outv,oc);
 SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);
}

// raw-UB 2x4 whole-plane ptr helper (shape domain: ptr 模式、tasks=2、axis=4、
// inner=1、ptr=[0,b,4]、fp32): the legal ptr is [0,b,4], so the only internal
// boundary is b=ptr[1]; it is read once by a scalar ptrGm.GetValue(1) (issued
// right after the src DMA, before MTE2_V), so the ptr DMA, the ptr pad/Cast,
// the byte-offset Gather, the Compare(row0>=ptrB), the 8-lane Select(ones) and
// the Cast to gidv are unnecessary on this path. gsum starts at 1-b, the masked
// Adds (+1 on 0xAA lanes, +2 on 0xCC lanes) rebuild [1,2,3,4,1,2,3,4]-b and
// Maxs(0)/Mins(1) clamp it to the exact binary group id per row: b=0 -> all 1
// and b=4 -> all 0 (empty-group edges included), one 4-lane row per id block.
// idi/idj are the gf[304]/gf[368] float views and the membership Compare is a
// 64-lane float EQ; the stable max-exp-sum tail with its +eps term is unchanged.
// The row number is folded into the id itself
// (id'(lane)=2*group(lane)+row(lane), values 0..3) so the Compare can no longer
// match across rows; no separate row gate tensor, Duplicate(gate,...) calls or
// And(memberU16,memberU16,gate,4) are needed. Everything else (incl.
// the scalar ptrGm.GetValue(1) boundary load and the softmax tail) is unchanged.
__aicore__ inline void RawPtrPlane2x4(GM_ADDR src,GM_ADDR ptr,GM_ADDR out){
 using T=float;
 GlobalTensor<T>srcGm;srcGm.SetGlobalBuffer((__gm__ T*)src);
 GlobalTensor<int32_t>ptrGm;ptrGm.SetGlobalBuffer((__gm__ int32_t*)ptr);
 GlobalTensor<T>outGm;outGm.SetGlobalBuffer((__gm__ T*)out);
 LocalTensor<T>vals(TPosition::VECCALC,0u,8u);
 LocalTensor<T>outv(TPosition::VECCALC,32u,8u);
 LocalTensor<T>z(TPosition::VECCALC,64u,8u);
 LocalTensor<T>gf(TPosition::VECCALC,128u,1024u);
 LocalTensor<uint8_t>gb(TPosition::VECCALC,128u,4096u);
 // Byte-disjoint, 32-byte-aligned group sub-blocks. Only gsum and the float id
 // blocks are needed here; no row0/offset/ptr/ones blocks are used.
 auto gsum=gf[256];     // [1024,1056) 8 f32 per-row group ids
 auto idi=gf[304];      // [1216,1472) 64 f32 id rows
 auto idj=gf[368];      // [1472,1728) 64 f32 id columns
 auto xj=gf[432];       // [1728,1984) 64 f32
 auto xmasked=gf[496];  // [1984,2240) 64 f32
 auto zj=gf[560];       // [2240,2496) 64 f32
 auto smasked=gf[624];  // [2496,2752) 64 f32
 auto mx=gf[688];       // [2752,2784) 8 f32
 auto sum=gf[696];      // [2784,2816) 8 f32
 auto member=gb[2816];  // [2816,2880) 64 u8: Compare's packed 64-bit mask = 8 B

 // Define the padded src lanes before the short DMA. No ptr lanes exist any more.
 Duplicate(vals,-3.402823466e38f,8);
 PipeBarrier<PIPE_V>();
 SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
 // One 32 B src DMA covers both rows (off 0, 8 elements). The boundary is a
 // scalar GM load, overlapped with the src DMA and committed before MTE2_V.
 DataCopyPadExtParams<T> pad(false,0,0,0);
 DataCopyExtParams cp(1,8*sizeof(T),0,0,0);
 DataCopyPad(vals,srcGm[0],cp,pad);
 const int32_t boundary=ptrGm.GetValue(1);
 SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);

 // Vector-only exact binary group ids from the scalar boundary (no ptr DMA, no
 // SetValue/CreateVecIndex/TPipe): start=2-2b, +2 on mask 0xAA lanes, +4 on mask
 // 0xCC lanes -> 2*[1,2,3,4,1,2,3,4]-2b, clamped to [0,2] by Maxs/Mins, then
 // +1 on the four row-1 lanes -> id'(lane)=2*group(lane)+row(lane) in {0,1,2,3}.
 // The row number lives inside the id, so the two rows can never compare equal
 // and no separate row gate tensor (2x Duplicate + And) is needed at all.
 Duplicate(gsum,2.0f*(1.0f-static_cast<float>(boundary)),8);
 PipeBarrier<PIPE_V>();
 uint64_t mA[2]={0xAAull,0ull},mC[2]={0xCCull,0ull};
 Adds<float,true>(gsum,gsum,2.0f,mA,(uint8_t)1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
 Adds<float,true>(gsum,gsum,4.0f,mC,(uint8_t)1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
 Maxs(gsum,gsum,0.0f,8);PipeBarrier<PIPE_V>();
 Mins(gsum,gsum,2.0f,8);PipeBarrier<PIPE_V>();
 uint64_t mR[2]={0xF0ull,0ull};
 Adds<float,true>(gsum,gsum,1.0f,mR,(uint8_t)1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();

 // Membership/softmax tail (raw-UB ptr tail, all N -> 8); the row number is part
 // of the id, so equal ids are automatically same-row and the two rows never merge.
 Brcb(idi,gsum,1,BrcbRepeatParams(1,8));
 Copy(idj,gsum,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 Copy(xj,vals,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Compare(member,idi,idj,CMPMODE::EQ,64);
 PipeBarrier<PIPE_V>();
 Select(xmasked,member,xj,-3.402823466e38f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceMax(mx,xmasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Duplicate(z,0.0f,8);
 PipeBarrier<PIPE_V>();
 Sub(z,vals,mx,8);PipeBarrier<PIPE_V>();Exp(z,z,8);
 PipeBarrier<PIPE_V>();
 Copy(zj,z,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Select(smasked,member,zj,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceSum(sum,smasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Adds(sum,sum,1e-16f,8);PipeBarrier<PIPE_V>();Div(outv,z,sum,8);
 PipeBarrier<PIPE_V>();

 // One 32 B store for the whole plane (both 4-lane rows in a single DMA).
 SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);
 DataCopyExtParams oc(1,8*sizeof(T),0,0,0);
 DataCopyPad(outGm[0],outv,oc);
 SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);
}
// raw-UB whole-plane (2,4) index helper (PATH7): a standalone __aicore__ function
// with NO TPipe, NO InitBuffer, NO tiling struct and NO class/instance state. It is
// the PlaneIndex2x4 algorithm: one 8-lane pass over both
// rows, idsFull[i]=index[i%4] via CreateVecIndex/&3/*4/Gather, membership masked by the
// 64-bit row gate 0xf0f0f0f00f0f0f0f, stable max/exp/sum with eps=1e-16f, single 32 B
// DMA in and out. src/index are read from the real inputs at runtime via
// SetGlobalBuffer; every size/offset is a literal (no dynamic LocalTensor subscripting).
// This helper covers the index 模式、tasks=2、axis=4、inner=1、广播、eps=1e-16f、fp32
// input domain; PATH7 is instantiated for DT_FLOAT only and
// never constructs KernelSparseSoftmax, so the class above (including its unused
// PlaneIndex2x4 method) stays compiled but unreachable on this path.
// UB layout (byte addresses, all 32B aligned, pairwise disjoint):
//   vals f32 @0 (8), out f32 @32 (8), z f32 @64 (8),
//   group views @96 sized 1024 f32 = 4096 B; sub-offsets below are the
//   PlaneIndex2x4 constants and stay constant.
__aicore__ inline void RawPlaneIndex2x4(GM_ADDR src,GM_ADDR index,GM_ADDR out){
 using T=float;
 GlobalTensor<T>srcGm;srcGm.SetGlobalBuffer((__gm__ T*)src);
 GlobalTensor<int32_t>indexGm;indexGm.SetGlobalBuffer((__gm__ int32_t*)index);
 GlobalTensor<T>outGm;outGm.SetGlobalBuffer((__gm__ T*)out);
 LocalTensor<T>vals(TPosition::VECCALC,0u,8u);
 LocalTensor<T>outv(TPosition::VECCALC,32u,8u);
 LocalTensor<T>z(TPosition::VECCALC,64u,8u);
 LocalTensor<int32_t>gi(TPosition::VECCALC,96u,1024u);
 LocalTensor<uint32_t>gu(TPosition::VECCALC,96u,1024u);
 LocalTensor<T>gf(TPosition::VECCALC,96u,1024u);
 LocalTensor<uint16_t>gu16(TPosition::VECCALC,96u,2048u);
 LocalTensor<uint8_t>gb(TPosition::VECCALC,96u,4096u);
 auto idsRaw=gi[544];                            // group+2176, 16 B (4 int32)
 auto idsFull=gi[552];                           // group+2208, 32 B
 auto offI=gi[560];auto offU=gu[560];            // group+2240, 32 B
 auto offU16=gu16[1120];                         // same bytes, 16-bit view
 auto mask3=gu16[1136];                          // group+2272, 32 B
 auto gate=gu16[1152];                           // group+2304, 32 B
 auto memberU16=gu16[1056];                      // member bytes, 16-bit view
 auto idi=gi[0];auto idj=gi[64];
 auto xj=gf[256];auto xmasked=gf[320];auto zj=gf[384];auto smasked=gf[448];
 auto mx=gf[512];auto sum=gf[520];auto member=gb[528*4];

 Duplicate(vals,-3.402823466e38f,8);
 PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
 DataCopyPadExtParams<int32_t> ip(false,0,0,0);DataCopyPadExtParams<T> pad(false,0,0,0);
 DataCopyExtParams ci(1,4*sizeof(int32_t),0,0,0),cp(1,8*sizeof(T),0,0,0);
 DataCopyPad(idsRaw,indexGm[0],ci,ip);
 DataCopyPad(vals,srcGm[0],cp,pad);
 SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);

 // idsFull[i] = idsRaw[i%4]: byte offsets [0,4,8,12,0,4,8,12] held as 4 packed
 // 64-bit UB constants (little-endian = 8 x u32) written by scalar SetValue; one
 // S_V flag restores scalar->vector order before the unchanged Gather below.
 LocalTensor<uint64_t> offPacked(TPosition::VECCALC,96u+2240u,4u);
 offPacked.SetValue(0,0x0000000400000000ull);
 offPacked.SetValue(1,0x0000000c00000008ull);
 offPacked.SetValue(2,0x0000000400000000ull);
 offPacked.SetValue(3,0x0000000c00000008ull);
 SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
 Gather(gu[552],gu[544],offU,0,8);
 PipeBarrier<PIPE_V>();

 // Row gate 0xf0f0f0f00f0f0f0f, first 4 uint16 values (one 32 B base):
 // elements 0..1 = 0x0f0f, elements 2..3 = 0xf0f0.
 uint64_t gm1[2]={3,0};uint64_t gm2[2]={12,0};
 Duplicate(gate,static_cast<uint16_t>(0x0f0f),gm1,1,1,8);
 Duplicate(gate,static_cast<uint16_t>(0xf0f0),gm2,1,1,8);
 PipeBarrier<PIPE_V>();

 // Below: the whole-plane pass (n fixed to 8 valid lanes).
 Brcb(idi,idsFull,1,BrcbRepeatParams(1,8));
 Copy(idj,idsFull,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 Copy(xj,vals,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Compare(member,idi,idj,CMPMODE::EQ,64);
 PipeBarrier<PIPE_V>();
 And(memberU16,memberU16,gate,4);
 PipeBarrier<PIPE_V>();
 Select(xmasked,member,xj,-3.402823466e38f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceMax(mx,xmasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Duplicate(z,0.0f,8);
 PipeBarrier<PIPE_V>();
 Sub(z,vals,mx,8);PipeBarrier<PIPE_V>();Exp(z,z,8);
 PipeBarrier<PIPE_V>();
 Copy(zj,z,uint64_t(64),uint8_t(1),CopyRepeatParams(1,0,8,0));
 PipeBarrier<PIPE_V>();
 Select(smasked,member,zj,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
 PipeBarrier<PIPE_V>();
 BlockReduceSum(sum,smasked,1,64,1,1,8);
 PipeBarrier<PIPE_V>();
 Adds(sum,sum,1e-16f,8);PipeBarrier<PIPE_V>();Div(outv,z,sum,8);
 PipeBarrier<PIPE_V>();
 // One 32 B store for the whole plane (both 4-lane rows in a single DMA).
 SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);
 DataCopyExtParams co(1,8*sizeof(T),0,0,0);
 DataCopyPad(outGm[0],outv,co);
 SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);
}
// ---------------------------------------------------------------------------
// PATH10: axisLength > CAP. The line no longer fits the CAP-sized UB window,
// so it is streamed in three passes that each read at most CAP elements per
// step: 1) group maxima, 2) group sums of exp(x - max), 3) write-back of
// exp(x - max) / (sum + eps). The per-group state lives in the user workspace
// that the host reserves through GetWorkspaceSizes: one slot per block with
// three float tables of LARGE_AXIS_TABLE_GROUPS entries (group max | group sum
// | Kahan compensation of the sum). Index mode groups by the exact int32 id
// value: the line's ids are scanned once for their exact int32 minimum (lo)
// and maximum (hi) and the table index is (id - lo), so the tables fit exactly
// when the id span hi-lo+1 stays inside the capacity. CSR mode groups by the
// interval [ptr[g], ptr[g+1]) and uses the interval ordinal as table index.
// When the span resp. the interval count does not fit, the line degrades to an
// element-exact scan that keeps no table state: every element walks the whole
// axis and only the elements whose group key matches contribute. That scan is
// slow but stays correct for arbitrary int32 ids. No TPipe and no InitBuffer
// on this path: the UB ranges below are fixed, 32 B aligned, pairwise disjoint
// and each holds CAP elements.
//   raw IN @0 | vals f32 @16384 | tmp f32 @32768 | out f32 @49152 |
//   cast IN @65536 | ids i32 @81920
namespace {
constexpr uint32_t UB_RAW=0u,UB_VAL=16384u,UB_TMP=32768u,UB_OUT=49152u,UB_CAST=65536u,UB_IDS=81920u;
}
template<class IN>
struct LargeAxisSoftmax {
 using T=float;
 const SparseSoftmaxTilingData* t=nullptr;
 GlobalTensor<IN>srcGm,outGm;GlobalTensor<int32_t>idxGm,ptrGm;GlobalTensor<T>tabMax,tabSum,tabCmp;
 uint32_t n=0,inner=1,mode=1,bcast=1,groups=0,span=0,pcFb=0;int32_t lo=0,ptrEnd=0;
 uint64_t block=0,blocks=1,lineBase=0;

 __aicore__ inline void Init(GM_ADDR src,GM_ADDR index,GM_ADDR ptr,GM_ADDR out,GM_ADDR ws,const SparseSoftmaxTilingData*d){
  t=d;n=d->axisLength;inner=d->innerSize;mode=d->mode;bcast=d->indexBroadcast;
  srcGm.SetGlobalBuffer((__gm__ IN*)src);outGm.SetGlobalBuffer((__gm__ IN*)out);
  idxGm.SetGlobalBuffer((__gm__ int32_t*)index);ptrGm.SetGlobalBuffer((__gm__ int32_t*)ptr);
  block=GetBlockIdx();blocks=GetBlockNum();
  const uint64_t slot=block*static_cast<uint64_t>(LARGE_AXIS_SLOT_BYTES);
  tabMax.SetGlobalBuffer((__gm__ T*)(ws+slot));
  tabSum.SetGlobalBuffer((__gm__ T*)(ws+slot+LARGE_AXIS_TABLE_GROUPS*sizeof(T)));
  tabCmp.SetGlobalBuffer((__gm__ T*)(ws+slot+2u*LARGE_AXIS_TABLE_GROUPS*sizeof(T)));
  groups=(mode==2u&&t->ptrLength>=2u)?(t->ptrLength-1u):0u;
 }
 __aicore__ inline void Run(){
  for(uint64_t task=block;task<t->tasks;task+=blocks){
   lineBase=(task/inner)*static_cast<uint64_t>(n)*inner+(task%inner);
   if(mode==1u)ScanIdRange();
   else{ptrEnd=0;if(t->ptrLength>=1u){const int32_t e=ptrGm.GetValue(static_cast<int32_t>(t->ptrLength)-1);ptrEnd=(e<0)?0:((e>static_cast<int32_t>(n))?static_cast<int32_t>(n):e);}}
   const uint32_t fit=(mode==1u)?span:groups;
   if(fit>=1u&&fit<=LARGE_AXIS_TABLE_GROUPS)TablePass(fit);else ExactScan();
  }
 }
 // UB views (fixed byte addresses; no TPipe on this path).
 __aicore__ inline LocalTensor<IN>RawBuf() const{return LocalTensor<IN>(TPosition::VECCALC,UB_RAW,CAP);}
 __aicore__ inline LocalTensor<T>ValBuf() const{return LocalTensor<T>(TPosition::VECCALC,UB_VAL,CAP);}
 __aicore__ inline LocalTensor<T>TmpBuf() const{return LocalTensor<T>(TPosition::VECCALC,UB_TMP,CAP);}
 __aicore__ inline LocalTensor<T>OutBuf() const{return LocalTensor<T>(TPosition::VECCALC,UB_OUT,CAP);}
 __aicore__ inline LocalTensor<IN>CastBuf() const{return LocalTensor<IN>(TPosition::VECCALC,UB_CAST,CAP);}
 __aicore__ inline LocalTensor<int32_t>IdsBuf() const{return LocalTensor<int32_t>(TPosition::VECCALC,UB_IDS,CAP);}
 __aicore__ inline uint32_t Chunk(uint32_t i0) const{return (n-i0<CAP)?(n-i0):CAP;}
 __aicore__ inline uint64_t ElemOff(uint32_t i) const{return lineBase+static_cast<uint64_t>(i)*inner;}
 // One axis chunk of source values; scalar and vector reads are both safe after.
 __aicore__ inline void LoadVals(uint32_t i0,uint32_t c){
  auto val=ValBuf();
  if(inner==1u){
   SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
   DataCopyPadExtParams<IN> pad(false,0,0,0);
   DataCopyExtParams cp(1,static_cast<uint32_t>(c*sizeof(IN)),0,0,0);
   if constexpr(IsSameType<IN,T>::value){
    DataCopyPad(val,srcGm[lineBase+i0],cp,pad);
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    SetFlag<HardEvent::MTE2_S>(3);WaitFlag<HardEvent::MTE2_S>(3);
   }else{
    auto raw=RawBuf();DataCopyPad(raw,srcGm[lineBase+i0],cp,pad);
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    Cast(val,raw,RoundMode::CAST_NONE,c);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
   }
  }else{
   const uint64_t e0=ElemOff(i0);
   for(uint32_t j=0;j<c;++j)val.SetValue(j,static_cast<T>(srcGm.GetValue(e0+static_cast<uint64_t>(j)*inner)));
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
  }
 }
 // One axis chunk of index ids (index mode); scalar reads are safe after.
 __aicore__ inline void LoadIds(uint32_t i0,uint32_t c){
  auto ids=IdsBuf();
  if(inner==1u){
   const uint64_t io=bcast?static_cast<uint64_t>(i0):(lineBase+i0);
   DataCopyPadExtParams<int32_t> ip(false,0,0,0);
   DataCopyExtParams ci(1,static_cast<uint32_t>(c*4u),0,0,0);
   DataCopyPad(ids,idxGm[io],ci,ip);
   SetFlag<HardEvent::MTE2_S>(3);WaitFlag<HardEvent::MTE2_S>(3);
  }else{
   for(uint32_t j=0;j<c;++j)ids.SetValue(j,idxGm.GetValue(bcast?static_cast<uint64_t>(i0+j):(lineBase+static_cast<uint64_t>(i0+j)*inner)));
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
  }
 }
 // Exact int32 minimum/maximum of the line's ids -> dense table width.
 __aicore__ inline void ScanIdRange(){
  auto ids=IdsBuf();int32_t mn=0,mx=0;bool first=true;
  for(uint32_t i0=0;i0<n;i0+=CAP){
   const uint32_t c=Chunk(i0);
   LoadIds(i0,c);
   for(uint32_t j=0;j<c;++j){const int32_t v=ids.GetValue(j);if(first){mn=v;mx=v;first=false;}else{if(v<mn)mn=v;if(v>mx)mx=v;}}
   SetFlag<HardEvent::S_MTE2>(4);WaitFlag<HardEvent::S_MTE2>(4);
  }
  lo=mn;
  const int64_t sp=static_cast<int64_t>(mx)-static_cast<int64_t>(mn)+1;
  span=(sp>=1&&sp<=static_cast<int64_t>(0x7fffffff))?static_cast<uint32_t>(sp):0u;
 }
 // Table index of element i, or -1 when it is outside every CSR interval.
 __aicore__ inline int32_t Key(uint32_t i,const LocalTensor<int32_t>&ids,uint32_t j,uint32_t&pc){
  if(mode==1u)return ids.GetValue(j)-lo;
  while(pc<groups&&ptrGm.GetValue(static_cast<int32_t>(pc))<=static_cast<int32_t>(i))++pc;
  if(pc==0u||static_cast<int32_t>(i)>=ptrEnd)return -1;
  return static_cast<int32_t>(pc)-1;
 }
 // Group tables: max = -inf, sum = 0, compensation = 0.
 __aicore__ inline void ClearTables(uint32_t gs){
  auto tmp=TmpBuf();
  SetFlag<HardEvent::S_MTE3>(5);WaitFlag<HardEvent::S_MTE3>(5);
  for(uint32_t k0=0;k0<gs;k0+=CAP){
   const uint32_t c=(gs-k0<CAP)?(gs-k0):CAP;
   DataCopyExtParams cp(1,static_cast<uint32_t>(c*sizeof(T)),0,0,0);
   Duplicate(tmp,-3.402823466e38f,c);PipeBarrier<PIPE_V>();
   SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);
   DataCopyPad(tabMax[k0],tmp,cp);
   SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);
   Duplicate(tmp,0.0f,c);PipeBarrier<PIPE_V>();
   SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);
   DataCopyPad(tabSum[k0],tmp,cp);
   DataCopyPad(tabCmp[k0],tmp,cp);
   SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);
  }
  SetFlag<HardEvent::MTE3_S>(4);WaitFlag<HardEvent::MTE3_S>(4);
 }
 // Three streaming passes with the group tables in the workspace.
 __aicore__ inline void TablePass(uint32_t gs){
  auto val=ValBuf();auto tmp=TmpBuf();auto outv=OutBuf();auto ids=IdsBuf();
  ClearTables(gs);
  uint32_t pc=0;
  for(uint32_t i0=0;i0<n;i0+=CAP){
   const uint32_t c=Chunk(i0);
   LoadVals(i0,c);if(mode==1u)LoadIds(i0,c);
   for(uint32_t j=0;j<c;++j){
    const int32_t k=Key(i0+j,ids,j,pc);
    if(k<0)continue;
    const T v=val.GetValue(j);
    if(v>tabMax.GetValue(k))tabMax.SetValue(k,v);
   }
   SetFlag<HardEvent::S_MTE2>(4);WaitFlag<HardEvent::S_MTE2>(4);
  }
  pc=0;
  for(uint32_t i0=0;i0<n;i0+=CAP){
   const uint32_t c=Chunk(i0);
   LoadVals(i0,c);if(mode==1u)LoadIds(i0,c);
   for(uint32_t j=0;j<c;++j){
    const int32_t k=Key(i0+j,ids,j,pc);
    if(mode==2u)ids.SetValue(j,k);
    tmp.SetValue(j,(k<0)?0.0f:tabMax.GetValue(k));
   }
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
   Sub(outv,val,tmp,c);PipeBarrier<PIPE_V>();Exp(outv,outv,c);PipeBarrier<PIPE_V>();
   SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
   for(uint32_t j=0;j<c;++j){
    const int32_t k=(mode==1u)?(ids.GetValue(j)-lo):ids.GetValue(j);
    if(k<0)continue;
    const T e=outv.GetValue(j);
    const T s=tabSum.GetValue(k);const T cc=tabCmp.GetValue(k);
    const T y=e-cc;const T tt=s+y;
    tabCmp.SetValue(k,(tt-s)-y);tabSum.SetValue(k,tt);
   }
   SetFlag<HardEvent::S_MTE2>(4);WaitFlag<HardEvent::S_MTE2>(4);
  }
  pc=0;
  for(uint32_t i0=0;i0<n;i0+=CAP){
   const uint32_t c=Chunk(i0);
   LoadVals(i0,c);if(mode==1u)LoadIds(i0,c);
   for(uint32_t j=0;j<c;++j){
    const int32_t k=Key(i0+j,ids,j,pc);
    if(mode==2u)ids.SetValue(j,k);
    tmp.SetValue(j,(k<0)?0.0f:tabMax.GetValue(k));
   }
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
   Sub(outv,val,tmp,c);PipeBarrier<PIPE_V>();Exp(outv,outv,c);PipeBarrier<PIPE_V>();
   SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
   for(uint32_t j=0;j<c;++j){
    const int32_t k=(mode==1u)?(ids.GetValue(j)-lo):ids.GetValue(j);
    if(k<0){outv.SetValue(j,0.0f);tmp.SetValue(j,1.0f);}else tmp.SetValue(j,tabSum.GetValue(k)+t->eps);
   }
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
   Div(val,outv,tmp,c);PipeBarrier<PIPE_V>();
   StoreChunk(val,i0,c);
   SetFlag<HardEvent::S_MTE2>(4);WaitFlag<HardEvent::S_MTE2>(4);
  }
 }
 __aicore__ inline void StoreChunk(const LocalTensor<T>&o,uint32_t i0,uint32_t c){
  if(inner==1u){
   auto cast=CastBuf();
   if constexpr(!IsSameType<IN,T>::value){Cast(cast,o,RoundMode::CAST_RINT,c);PipeBarrier<PIPE_V>();}
   SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);
   DataCopyExtParams cp(1,static_cast<uint32_t>(c*sizeof(IN)),0,0,0);
   if constexpr(IsSameType<IN,T>::value)DataCopyPad(outGm[lineBase+i0],o,cp);else DataCopyPad(outGm[lineBase+i0],cast,cp);
   SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);
  }else{
   SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
   const uint64_t e0=ElemOff(i0);
   for(uint32_t j=0;j<c;++j)outGm.SetValue(e0+static_cast<uint64_t>(j)*inner,static_cast<IN>(o.GetValue(j)));
  }
 }
 __aicore__ inline bool SegHit(uint32_t idx,int32_t ia,int32_t ib) const{const int32_t k=static_cast<int32_t>(idx);return k>=ia&&k<ib;}
 __aicore__ inline bool PtrAt(uint32_t i,int32_t&ia,int32_t&ib){
  const int32_t ii=static_cast<int32_t>(i);
  while(pcFb<groups&&ptrGm.GetValue(static_cast<int32_t>(pcFb))<=ii)++pcFb;
  if(pcFb==0u)return false;
  const int32_t g=static_cast<int32_t>(pcFb)-1;
  ia=ptrGm.GetValue(g);ib=ptrGm.GetValue(g+1);
  if(ia<0)ia=0;if(ib>static_cast<int32_t>(n))ib=static_cast<int32_t>(n);
  return ii<ib;
 }
 // Element-exact fallback (no table state): every element scans the whole axis
 // for the members of its own group.
 __aicore__ inline void ExactScan(){
  auto val=ValBuf();auto tmp=TmpBuf();auto ids=IdsBuf();
  pcFb=0;
  for(uint32_t i=0;i<n;++i){
   const uint64_t eo=ElemOff(i);
   int32_t key=0,ia=0,ib=0;bool covered=true;
   if(mode==1u)key=idxGm.GetValue(bcast?static_cast<uint64_t>(i):eo);else covered=PtrAt(i,ia,ib);
   if(!covered){outGm.SetValue(eo,static_cast<IN>(0.0f));continue;}
   T mx=-3.402823466e38f;
   for(uint32_t i0=0;i0<n;i0+=CAP){
    const uint32_t c=Chunk(i0);
    LoadVals(i0,c);if(mode==1u)LoadIds(i0,c);
    for(uint32_t j=0;j<c;++j){
     const bool hit=(mode==1u)?(ids.GetValue(j)==key):SegHit(i0+j,ia,ib);
     if(hit){const T v=val.GetValue(j);if(v>mx)mx=v;}
    }
    SetFlag<HardEvent::S_MTE2>(4);WaitFlag<HardEvent::S_MTE2>(4);
   }
   T s=0.0f,cc=0.0f,ei=1.0f;
   for(uint32_t i0=0;i0<n;i0+=CAP){
    const uint32_t c=Chunk(i0);
    LoadVals(i0,c);if(mode==1u)LoadIds(i0,c);
    Adds(tmp,val,-mx,c);PipeBarrier<PIPE_V>();Exp(tmp,tmp,c);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
    for(uint32_t j=0;j<c;++j){
     const bool hit=(mode==1u)?(ids.GetValue(j)==key):SegHit(i0+j,ia,ib);
     if(!hit)continue;
     const T e=tmp.GetValue(j);
     const T y=e-cc;const T tt=s+y;cc=(tt-s)-y;s=tt;
     if(i0+j==i)ei=e;
    }
    SetFlag<HardEvent::S_MTE2>(4);WaitFlag<HardEvent::S_MTE2>(4);
   }
   outGm.SetValue(eo,static_cast<IN>(ei/(s+t->eps)));
  }
 }
};
// Entry dispatch: PATH6/PATH7 select the raw-UB index helpers and PATH8/PATH9
// the raw-UB ptr paths above. All four skip REGISTER_TILING_DEFAULT /
// GET_TILING_DATA_WITH_STRUCT and never construct KernelSparseSoftmax (so no
// TPipe/InitBuffer). Block dim is still 8; PATH6/PATH8 keep block 0 only
// (tasks=1, one row) and PATH7/PATH9 keep blocks 0..1 (tasks=2, one row
// each); every other instantiation takes the generic entry in the else branch
// below.
// 64-bit id routes (PATH13 = 64-bit index, PATH14 = 64-bit ptr). The id operand
// may be a 64-bit signed value, so every id compare runs on the scalar unit:
// the ids are copied from GM into a UB tile (GlobalTensor<int64_t> +
// DataCopyPad) and read back with GetValue(). No Cast between the 64-bit id
// type and any float type exists anywhere on this route and no vector op touches
// the id tile. The route reads the real src/ids from GM, groups by the exact
// 64-bit id (index) or by the CSR interval (ptr), computes the per group
// max / exp / sum and stores the real output. The output leaves through one
// DMA per tile: element stores would let two blocks clobber each other inside a
// shared cache line, so this route never writes GM element by element when the
// row is contiguous. It is slow by design and exact for every id value.
// Fixed UB byte layout (32 B aligned,
// pairwise disjoint, no TPipe, no InitBuffer):
//   raw IN @0 | vals f32 @4096 | tmp f32 @8192 | out f32 @12288 | cast IN
//   @16384 | ids i64 @20480 | ptr i64 @28672 (every tile holds up to CH64
//   elements; the two id tiles hold CH64 64-bit values each).
namespace {
constexpr uint32_t UB64_RAW=0u,UB64_VAL=4096u,UB64_TMP=8192u,UB64_OUT=12288u,
                   UB64_CAST=16384u,UB64_IDS=20480u,UB64_PTR=28672u,CH64=1024u;
}
template<class IN>
struct Int64Softmax {
 using T=float;
 const SparseSoftmaxTilingData* t=nullptr;
 GlobalTensor<IN>srcGm,outGm;
 GlobalTensor<int64_t>idxGm,ptrGm;
 uint32_t n=0,inner=1,modeId=1,bcast=1,ptrWin=0u,ptrWinC=0u,outBase=0u,outFill=0u;
 uint64_t lineBase=0;
 __aicore__ inline void Init(GM_ADDR src,GM_ADDR index,GM_ADDR ptr,GM_ADDR out,const SparseSoftmaxTilingData*d){
  t=d;n=d->axisLength;inner=d->innerSize;modeId=d->pad0;bcast=d->indexBroadcast;
  srcGm.SetGlobalBuffer((__gm__ IN*)src);outGm.SetGlobalBuffer((__gm__ IN*)out);
  idxGm.SetGlobalBuffer((__gm__ int64_t*)index);ptrGm.SetGlobalBuffer((__gm__ int64_t*)ptr);
 }
 __aicore__ inline void Run(){
  const uint64_t blocks=GetBlockNum();
  for(uint64_t task=GetBlockIdx();task<t->tasks;task+=blocks){
   lineBase=(task/inner)*static_cast<uint64_t>(n)*inner+(task%inner);
   outBase=0u;outFill=0u;
   if(modeId==2u)PtrLine();else IndexLine();
   if(outFill!=0u)FlushOut(outBase,outFill);
  }
 }
 __aicore__ inline LocalTensor<IN>Raw() const{return LocalTensor<IN>(TPosition::VECCALC,UB64_RAW,CH64);}
 __aicore__ inline LocalTensor<T>Val() const{return LocalTensor<T>(TPosition::VECCALC,UB64_VAL,CH64);}
 __aicore__ inline LocalTensor<T>Tmp() const{return LocalTensor<T>(TPosition::VECCALC,UB64_TMP,CH64);}
 __aicore__ inline LocalTensor<int64_t>Ids() const{return LocalTensor<int64_t>(TPosition::VECCALC,UB64_IDS,CH64);}
 __aicore__ inline LocalTensor<int64_t>Ptrs() const{return LocalTensor<int64_t>(TPosition::VECCALC,UB64_PTR,CH64);}
 __aicore__ inline LocalTensor<T>OutBuf() const{return LocalTensor<T>(TPosition::VECCALC,UB64_OUT,CH64);}
 __aicore__ inline LocalTensor<IN>CastBuf() const{return LocalTensor<IN>(TPosition::VECCALC,UB64_CAST,CH64);}
 __aicore__ inline uint32_t Count(uint32_t i0,uint32_t i1) const{return (i1-i0<CH64)?(i1-i0):CH64;}
 __aicore__ inline uint64_t Off(uint32_t i) const{return lineBase+static_cast<uint64_t>(i)*inner;}
 // One tile of source values; scalar readable when it returns.
 __aicore__ inline void LoadVals(uint32_t i0,uint32_t c){
  SetFlag<HardEvent::S_MTE2>(4);WaitFlag<HardEvent::S_MTE2>(4);
  if(inner==1u){
   SetFlag<HardEvent::V_MTE2>(5);WaitFlag<HardEvent::V_MTE2>(5);
   DataCopyPadExtParams<IN> pad(false,0,0,0);
   DataCopyExtParams cp(1,static_cast<uint32_t>(c*sizeof(IN)),0,0,0);
   if constexpr(IsSameType<IN,T>::value){
    DataCopyPad(Val(),srcGm[lineBase+i0],cp,pad);
    SetFlag<HardEvent::MTE2_S>(3);WaitFlag<HardEvent::MTE2_S>(3);
   }else{
    DataCopyPad(Raw(),srcGm[lineBase+i0],cp,pad);
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    Cast(Val(),Raw(),RoundMode::CAST_NONE,c);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
   }
  }else{
   for(uint32_t j=0;j<c;++j)Val().SetValue(j,static_cast<T>(srcGm.GetValue(Off(i0+j))));
   SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
  }
 }
 // One tile of 64-bit ids; every later compare reads it with GetValue().
 __aicore__ inline void LoadIds(uint32_t i0,uint32_t c){
  if(inner==1u){
   SetFlag<HardEvent::S_MTE2>(4);WaitFlag<HardEvent::S_MTE2>(4);
   DataCopyPadExtParams<int64_t> ip(false,0,0,0);
   DataCopyExtParams ci(1,static_cast<uint32_t>(c*sizeof(int64_t)),0,0,0);
   const uint64_t io=bcast?static_cast<uint64_t>(i0):(lineBase+i0);
   DataCopyPad(Ids(),idxGm[io],ci,ip);
   SetFlag<HardEvent::MTE2_S>(3);WaitFlag<HardEvent::MTE2_S>(3);
  }else{
   for(uint32_t j=0;j<c;++j)Ids().SetValue(j,idxGm.GetValue(bcast?static_cast<uint64_t>(i0+j):Off(i0+j)));
   SetFlag<HardEvent::S_MTE2>(4);WaitFlag<HardEvent::S_MTE2>(4);
  }
 }
 __aicore__ inline int64_t IdAt(uint32_t i){
  const uint32_t i0=(i/CH64)*CH64;LoadIds(i0,Count(i0,n));return Ids().GetValue(i-i0);
 }
 // ptr[g] with a cached tile: the group walk reads ptr in ascending order, so
 // one 32 B aligned tile of 64-bit values covers 1024 boundaries.
 __aicore__ inline int64_t PtrAt(uint32_t g){
  if(!(ptrWin<=g&&g<ptrWin+ptrWinC)){
   const uint32_t i0=(g/CH64)*CH64;const uint32_t c=Count(i0,t->ptrLength);
   SetFlag<HardEvent::S_MTE2>(4);WaitFlag<HardEvent::S_MTE2>(4);
   DataCopyPadExtParams<int64_t> ip(false,0,0,0);
   DataCopyExtParams cp(1,static_cast<uint32_t>(c*sizeof(int64_t)),0,0,0);
   DataCopyPad(Ptrs(),ptrGm[i0],cp,ip);
   SetFlag<HardEvent::MTE2_S>(3);WaitFlag<HardEvent::MTE2_S>(3);
   ptrWin=i0;ptrWinC=c;
  }
  return Ptrs().GetValue(g-ptrWin);
 }
 // Outputs are buffered one tile deep and leave through a single DMA per tile
 // when the row is contiguous (inner==1). Each element is written once, in
 // ascending order, so a tile is complete as soon as CH64 of them are in.
 __aicore__ inline void OutSet(uint32_t i,T v){
  OutBuf().SetValue(i-outBase,v);
  if(++outFill==CH64){FlushOut(outBase,CH64);outBase+=CH64;outFill=0u;}
 }
 __aicore__ inline void FlushOut(uint32_t i0,uint32_t c){
  if(c==0u)return;
  if(inner==1u){
   DataCopyExtParams cp(1,static_cast<uint32_t>(c*sizeof(IN)),0,0,0);
   if constexpr(IsSameType<IN,T>::value){
    SetFlag<HardEvent::S_MTE3>(5);WaitFlag<HardEvent::S_MTE3>(5);
    DataCopyPad(outGm[lineBase+i0],OutBuf(),cp);
    SetFlag<HardEvent::MTE3_S>(4);WaitFlag<HardEvent::MTE3_S>(4);
   }else{
    SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
    Cast(CastBuf(),OutBuf(),RoundMode::CAST_RINT,c);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(1);WaitFlag<HardEvent::V_MTE3>(1);
    DataCopyPad(outGm[lineBase+i0],CastBuf(),cp);
    SetFlag<HardEvent::MTE3_V>(6);WaitFlag<HardEvent::MTE3_V>(6);
   }
  }else{
   // strided rows are single block only (inner>1 forces one core in the host),
   // so element stores cannot collide with another block here.
   SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
   for(uint32_t j=0;j<c;++j)outGm.SetValue(Off(i0+j),static_cast<IN>(OutBuf().GetValue(j)));
  }
 }
 // index: element i belongs to the group of its own exact id; both reductions
 // and the final exp() run over the members of that id.
 __aicore__ inline void IndexLine(){
  for(uint32_t i=0;i<n;++i){
   const int64_t key=IdAt(i);
   T mx=-3.402823466e38f;
   for(uint32_t i0=0;i0<n;i0+=CH64){
    const uint32_t c=Count(i0,n);LoadVals(i0,c);LoadIds(i0,c);
    for(uint32_t j=0;j<c;++j)if(Ids().GetValue(j)==key){const T v=Val().GetValue(j);if(v>mx)mx=v;}
   }
   T sum=t->eps,ei=0.0f;
   for(uint32_t i0=0;i0<n;i0+=CH64){
    const uint32_t c=Count(i0,n);LoadVals(i0,c);LoadIds(i0,c);
    for(uint32_t j=0;j<c;++j)Tmp().SetValue(j,(Ids().GetValue(j)==key)?(Val().GetValue(j)-mx):-80.0f);
    SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
    Exp(Tmp(),Tmp(),c);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
    for(uint32_t j=0;j<c;++j)if(Ids().GetValue(j)==key){const T e=Tmp().GetValue(j);sum+=e;if(i0+j==i)ei=e;}
   }
   OutSet(i,ei/sum);
  }
 }
 // ptr: group g owns the clamped interval [ptr[g], ptr[g+1]); intervals that
 // clamp empty are skipped and elements in no interval stay 0, which is exactly
 // the ia32 ptr route semantics. The row is walked once in element order: the
 // interval cursor only moves forward, and one element gets the max / exp / sum
 // of its own interval before it is emitted.
 __aicore__ inline int32_t ClampI(int64_t v) const{
  return (v<0)?0:((v>static_cast<int64_t>(n))?static_cast<int32_t>(n):static_cast<int32_t>(v));
 }
 __aicore__ inline void PtrLine(){
  const uint32_t L=t->ptrLength;
  uint32_t g=0;
  for(uint32_t i=0;i<n;++i){
   uint32_t a=0,u=0;bool found=false;
   while(g+1u<L){
    const int32_t ga=ClampI(PtrAt(g)),gb=ClampI(PtrAt(g+1u));
    if(gb<=ga){++g;continue;}
    if(static_cast<int32_t>(i)<ga)break;
    if(static_cast<int32_t>(i)<gb){a=static_cast<uint32_t>(ga);u=static_cast<uint32_t>(gb);found=true;break;}
    ++g;
   }
   if(!found){OutSet(i,0.0f);continue;}
   T mx=-3.402823466e38f;
   for(uint32_t i0=a;i0<u;i0+=CH64){
    const uint32_t c=Count(i0,u);LoadVals(i0,c);
    for(uint32_t j=0;j<c;++j){const T v=Val().GetValue(j);if(v>mx)mx=v;}
   }
   T sum=t->eps,ei=0.0f;
   for(uint32_t i0=a;i0<u;i0+=CH64){
    const uint32_t c=Count(i0,u);LoadVals(i0,c);
    for(uint32_t j=0;j<c;++j)Tmp().SetValue(j,Val().GetValue(j)-mx);
    SetFlag<HardEvent::S_V>(7);WaitFlag<HardEvent::S_V>(7);
    Exp(Tmp(),Tmp(),c);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(2);WaitFlag<HardEvent::V_S>(2);
    for(uint32_t j=0;j<c;++j){const T e=Tmp().GetValue(j);sum+=e;if(i0+j==i)ei=e;}
   }
   OutSet(i,ei/sum);
  }
 }
};
// Entry dispatch: PATH6/PATH7 select the raw-UB index helpers and PATH8/PATH9

template<uint64_t PATH,typename DT_SRC>__global__ __aicore__ void sparse_softmax(GM_ADDR src,GM_ADDR index,GM_ADDR ptr,GM_ADDR out,GM_ADDR workspace,GM_ADDR tiling){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if constexpr(PATH==6u){
  if(GetBlockIdx()!=0u)return;
  RawUbIndex8AllPairs8<DT_SRC,8u>(src,index,out);
 }else if constexpr(PATH==7u){
  if(GetBlockIdx()!=0u)return;
  RawPlaneIndex2x4(src,index,out);
 }else if constexpr(PATH==8u){
  if(GetBlockIdx()>=1u)return;
  RawPtr<8u,5u,DT_SRC>(src,ptr,out);
 }else if constexpr(PATH==9u){
  if(GetBlockIdx()!=0u)return;
  RawPtrPlane2x4(src,ptr,out);
 }else if constexpr(PATH==10u){
  // Long axes (> CAP) stream through the group tables in the user workspace.
  REGISTER_TILING_DEFAULT(SparseSoftmaxTilingData);GET_TILING_DATA_WITH_STRUCT(SparseSoftmaxTilingData,td,tiling);
  LargeAxisSoftmax<DT_SRC>op;op.Init(src,index,ptr,out,workspace,&td);op.Run();
 }else if constexpr(PATH==11u){
  // Small-shape family member (index/broadcast, inner==1): row length comes from tiling,
  // so this single instantiation serves every 1<=axis<=8 case; one row per owning block.
  REGISTER_TILING_DEFAULT(SparseSoftmaxTilingData);GET_TILING_DATA_WITH_STRUCT(SparseSoftmaxTilingData,td,tiling);
  if(GetBlockIdx()>=td.tasks)return;
  RawUbIndexAnyAllPairs8<DT_SRC>(src,index,out,td.axisLength);
 }else if constexpr(PATH==12u){
  // Small-shape family member (CSR ptr, inner==1): covers 1<=axis<=8 with any
  // 2<=ptrLength<=49 in one instantiation.
  REGISTER_TILING_DEFAULT(SparseSoftmaxTilingData);GET_TILING_DATA_WITH_STRUCT(SparseSoftmaxTilingData,td,tiling);
  if(GetBlockIdx()>=td.tasks)return;
  RawPtrAny<DT_SRC>(src,ptr,out,td.axisLength,td.ptrLength);
 }else if constexpr(PATH==13u||PATH==14u){
  // 64-bit id entry: the tiling pad0 marker (1 = 64-bit index, 2 = 64-bit ptr)
  // picks the index or the CSR engine; the path itself never changes it.
  REGISTER_TILING_DEFAULT(SparseSoftmaxTilingData);GET_TILING_DATA_WITH_STRUCT(SparseSoftmaxTilingData,td,tiling);
  Int64Softmax<DT_SRC>op;op.Init(src,index,ptr,out,&td);op.Run();
 }else{
  REGISTER_TILING_DEFAULT(SparseSoftmaxTilingData);GET_TILING_DATA_WITH_STRUCT(SparseSoftmaxTilingData,td,tiling);KernelSparseSoftmax<DT_SRC,PATH>op;op.Init(src,index,ptr,out,&td);op.Process();
 }
}
