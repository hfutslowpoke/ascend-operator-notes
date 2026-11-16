#ifndef GROUPED_MATMUL_SHORT_MID_IMPL_H
#define GROUPED_MATMUL_SHORT_MID_IMPL_H

// Included by kernel.asc after its common SDK includes and integer aliases.
// Tiny, short-group and medium-K implementations; original order retained.

using TinyMulti_GmmA=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using TinyMulti_GmmB=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using TinyMulti_GmmC=MatmulType<TPosition::GM,CubeFormat::ND,i32>;
__global__ __aicore__ __mix__(1,2)void TinyMulti_grouped_matmul_gelu_custom(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<TinyMulti_GmmA,TinyMulti_GmmB,TinyMulti_GmmC>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*1024);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy,pp;
pipe.InitBuffer(ci,4096);pipe.InitBuffer(zz,4096);pipe.InitBuffer(uu,4096);pipe.InitBuffer(tt,4096);pipe.InitBuffer(ff,4096);pipe.InitBuffer(ss,256);pipe.InitBuffer(yy,2048);pipe.InitBuffer(pp,4096);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();
auto h=yy.Get<half>();auto poly=pp.Get<float>();i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+63)/64;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*64;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<64?N-col:64;mm.SetOrgShape(M,N,K,K,64);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();DataCopy(cint,c,1024);DataCopyExtParams sp{1,(u32)nr*4,0,0,0};DataCopyPad(sc,scale[(i64)e*N+col],sp,DataCopyPadExtParams<float>{true,0,0,0});
auto ev=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(ev);WaitFlag<HardEvent::MTE2_V>(ev);Cast(z,cint,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();for(i32 r=0;r<mr;++r)Mul(z[r*64],z[r*64],sc,64);PipeBarrier<PIPE_V>();auto gmRead=pipe.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(gmRead);WaitFlag<HardEvent::MTE2_S>(gmRead);Abs(u,z,1024);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,1024);PipeBarrier<PIPE_V>();Mul(t,u,u,1024);
PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,1024);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,1024);PipeBarrier<PIPE_V>();Mul(u,u,t,1024);PipeBarrier<PIPE_V>();Muls(u,u,-0.04986778505017909f,1024);PipeBarrier<PIPE_V>();Duplicate(poly,0.0060438121814739504f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.040085832623197183f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.16614812982932259f,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.49992211956100152f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999569069746141f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999996114843803f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Adds(t,poly,1.0f,1024);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,1024);PipeBarrier<PIPE_V>();Mul(f,z,poly,1024);PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,1024);PipeBarrier<PIPE_V>();auto scratch=cint.ReinterpretCast<float>();Muls(poly,t,0.32323232323232326f,1024);
PipeBarrier<PIPE_V>();Adds(poly,poly,-1.4545454545454546f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,t,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,2.1212121212121211f,1024);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,1024);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,1024);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,1024);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,1024);
PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,1024);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,1024);PipeBarrier<PIPE_V>();Mul(f,f,poly,1024);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 r=0;r<mr;++r){
DataCopyExtParams op{1,(u32)nr*2,0,0,0};DataCopyPad(out[(start+row+r)*N+col],h[r*64],op);
}
auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ void TinyMulti_gmm_tiny_hostRows23(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,i32 K,i32 N,i32 E,i32 rowsPerWorker,i32 leadingRows){
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);const i32 worker=GetBlockIdx();const i64 rowFirst=rowsPerWorker*worker+(worker<leadingRows?worker:leadingRows);const i64 rowLast=rowFirst+rowsPerWorker+(worker<leadingRows);InitSocState();
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
GlobalTensor<u64>cache;cache.SetGlobalBuffer((__gm__ u64*)groups);DataCachePreload(cache,(int16_t)0);cache.SetGlobalBuffer((__gm__ u64*)x);DataCachePreload(cache,(int16_t)0);cache.SetGlobalBuffer((__gm__ u64*)w);DataCachePreload(cache,(int16_t)0);cache.SetGlobalBuffer((__gm__ u64*)s);DataCachePreload(cache,(int16_t)0);
#endif
GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);LocalTensor<i32>cint(TPosition::VECCALC,0,64);LocalTensor<float>z(TPosition::VECCALC,256,64),u(TPosition::VECCALC,512,64),t(TPosition::VECCALC,768,64);
LocalTensor<float>f(TPosition::VECCALC,1024,64),poly(TPosition::VECCALC,1280,64);LocalTensor<half>h(TPosition::VECCALC,1536,64);const i32 lanes=64;Duplicate(z,0.0f,lanes);SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);i64 r=rowFirst;for(i32 e=0;e<E&&r<rowLast;++e){
const i64 end=ends.GetValue(e);const i64 stopRow=end<rowLast?end:rowLast;for(;r<stopRow;++r){
for(i32 c=0;c<N;++c){
i32 sum=0;if(K<=4){
i64 xi=r*K;i64 wi=(i64)e*K*N+c;sum=(i32)a.GetValue(xi)*(i32)b.GetValue(wi);if(K>=2)sum+=(i32)a.GetValue(xi+1)*(i32)b.GetValue(wi+N);if(K>=3)sum+=(i32)a.GetValue(xi+2)*(i32)b.GetValue(wi+2*N);if(K>=4)sum+=(i32)a.GetValue(xi+3)*(i32)b.GetValue(wi+3*N);
}else{
for(i32 k=0;k<K;++k){
i32 av=(i32)a.GetValue(r*K+k);i32 bv=(i32)b.GetValue(((i64)e*K+k)*N+c);sum+=av*bv;
}}
float dot=static_cast<float>(sum);float channelScale=scale.GetValue((i64)e*N+c);float dequantized=dot*channelScale;z.SetValue((r-rowFirst)*N+c,dequantized);
}}}
SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);Abs(u,z,64);Maxs(f,z,0.0f,64);PipeBarrier<PIPE_V>();Muls(t,u,-0.412050840144001f,64);PipeBarrier<PIPE_V>();Adds(t,t,-0.712873389614528f,64);PipeBarrier<PIPE_V>();Mul(t,t,u,64);PipeBarrier<PIPE_V>();Adds(t,t,-0.7138483585858206f,64);PipeBarrier<PIPE_V>();Exp(poly,t,64);PipeBarrier<PIPE_V>();Mul(poly,poly,u,64);PipeBarrier<PIPE_V>();Sub(u,f,poly,64);PipeBarrier<PIPE_V>();Cast(h,u,RoundMode::CAST_RINT,lanes);
SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopyExtParams op{1,(u32)((rowLast-rowFirst)*N)*2,0,0,0};DataCopyPad(out[rowFirst*N],h,op);SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
static void TinyMultiRun(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");bool tinyShape=M>=1&&M<=16&&N>=1&&N<=64&&K>=1&&K<=32&&E>=1&&E<=128&&(i64)M*N<=64&&(i64)M*N*K<=256;
if(tinyShape){
u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);if(platform->GetCoreNumAiv()>=1&&ubBytes>=16384){
const i64 byteTiles=((i64)M*N*sizeof(half)+31)/32;const u32 rowWorkers=std::min<i64>(std::min<i64>(M,byteTiles),std::min<i64>(acn,platform->GetCoreNumAiv()));TinyMulti_gmm_tiny_hostRows23<<<rowWorkers,nullptr,stream>>>(x,weight,scale,group_list,y,K,N,E,M/rowWorkers,M%rowWorkers);aclError status=aclrtSynchronizeStream(stream);
if(status!=ACL_SUCCESS)throw std::runtime_error("GMM tiny ACL error "+std::to_string(status));return;
}}
u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());if(!blocks)throw std::runtime_error("no Cube cores");matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);
mt.SetShape(16,64,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(16,64,64);mt.SetBufferSpace(-1,-1,-1);TCubeTiling tiling;if(mt.GetTiling(tiling)!=0)throw std::runtime_error("INT8 tiling failed");void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};
checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*4096,ACL_MEM_MALLOC_HUGE_FIRST));TinyMulti_grouped_matmul_gelu_custom<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E);aclError status=aclrtSynchronizeStream(stream);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
using namespace AscendC;
using TinyScalar_GmmA=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using TinyScalar_GmmB=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using TinyScalar_GmmC=MatmulType<TPosition::GM,CubeFormat::ND,i32>;
__global__ __aicore__ __mix__(1,2)void TinyScalar_grouped_matmul_gelu_custom(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<TinyScalar_GmmA,TinyScalar_GmmB,TinyScalar_GmmC>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*1024);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy,pp;
pipe.InitBuffer(ci,4096);pipe.InitBuffer(zz,4096);pipe.InitBuffer(uu,4096);pipe.InitBuffer(tt,4096);pipe.InitBuffer(ff,4096);pipe.InitBuffer(ss,256);pipe.InitBuffer(yy,2048);pipe.InitBuffer(pp,4096);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();
auto h=yy.Get<half>();auto poly=pp.Get<float>();i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+63)/64;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*64;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<64?N-col:64;mm.SetOrgShape(M,N,K,K,64);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();DataCopy(cint,c,1024);DataCopyExtParams sp{1,(u32)nr*4,0,0,0};DataCopyPad(sc,scale[(i64)e*N+col],sp,DataCopyPadExtParams<float>{true,0,0,0});
auto ev=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(ev);WaitFlag<HardEvent::MTE2_V>(ev);Cast(z,cint,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();for(i32 r=0;r<mr;++r)Mul(z[r*64],z[r*64],sc,64);PipeBarrier<PIPE_V>();auto gmRead=pipe.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(gmRead);WaitFlag<HardEvent::MTE2_S>(gmRead);Abs(u,z,1024);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,1024);PipeBarrier<PIPE_V>();Mul(t,u,u,1024);
PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,1024);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,1024);PipeBarrier<PIPE_V>();Mul(u,u,t,1024);PipeBarrier<PIPE_V>();Muls(u,u,-0.04986778505017909f,1024);PipeBarrier<PIPE_V>();Duplicate(poly,0.0060438121814739504f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.040085832623197183f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.16614812982932259f,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.49992211956100152f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999569069746141f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999996114843803f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Adds(t,poly,1.0f,1024);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,1024);PipeBarrier<PIPE_V>();Mul(f,z,poly,1024);PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,1024);PipeBarrier<PIPE_V>();auto scratch=cint.ReinterpretCast<float>();Muls(poly,t,0.32323232323232326f,1024);
PipeBarrier<PIPE_V>();Adds(poly,poly,-1.4545454545454546f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,t,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,2.1212121212121211f,1024);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,1024);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,1024);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,1024);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,1024);
PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,1024);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,1024);PipeBarrier<PIPE_V>();Mul(f,f,poly,1024);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 r=0;r<mr;++r){
DataCopyExtParams op{1,(u32)nr*2,0,0,0};DataCopyPad(out[(start+row+r)*N+col],h[r*64],op);
}
auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ void TinyScalar_gmm_tiny_midfuse_r2a005(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,i32 M,i32 K,i32 N,i32 E){
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);TPipe pipe;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,yy,pp;pipe.InitBuffer(ci,256);
pipe.InitBuffer(zz,256);pipe.InitBuffer(uu,256);pipe.InitBuffer(tt,256);pipe.InitBuffer(ff,256);pipe.InitBuffer(yy,128);pipe.InitBuffer(pp,256);auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto h=yy.Get<half>();auto poly=pp.Get<float>();auto cint=ci.Get<i32>();const i32 lanes=64;
Duplicate(z,0.0f,lanes);auto vsDot=pipe.FetchEventID(HardEvent::V_S);SetFlag<HardEvent::V_S>(vsDot);WaitFlag<HardEvent::V_S>(vsDot);i64 start=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);for(i64 r=start;r<end;++r){
for(i32 c=0;c<N;++c){
i32 sum=0;if(K<=4){
i64 xi=r*K;i64 wi=(i64)e*K*N+c;sum=(i32)a.GetValue(xi)*(i32)b.GetValue(wi);if(K>=2)sum+=(i32)a.GetValue(xi+1)*(i32)b.GetValue(wi+N);if(K>=3)sum+=(i32)a.GetValue(xi+2)*(i32)b.GetValue(wi+2*N);if(K>=4)sum+=(i32)a.GetValue(xi+3)*(i32)b.GetValue(wi+3*N);
}else{
for(i32 k=0;k<K;++k){
i32 av=(i32)a.GetValue(r*K+k);i32 bv=(i32)b.GetValue(((i64)e*K+k)*N+c);sum+=av*bv;
}}
float dot=static_cast<float>(sum);float channelScale=scale.GetValue((i64)e*N+c);float dequantized=dot*channelScale;z.SetValue(r*N+c,dequantized);
}}
start=end;if(start==M)break;
}
auto svDot=pipe.FetchEventID(HardEvent::S_V);SetFlag<HardEvent::S_V>(svDot);WaitFlag<HardEvent::S_V>(svDot);Abs(u,z,64);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,64);PipeBarrier<PIPE_V>();Mul(t,u,u,64);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,64);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,64);PipeBarrier<PIPE_V>();Mul(u,u,t,64);PipeBarrier<PIPE_V>();Muls(u,u,-0.04986778505017909f,64);PipeBarrier<PIPE_V>();Duplicate(poly,0.0060438121814739504f,64);PipeBarrier<PIPE_V>();Mul(poly,poly,u,64);
PipeBarrier<PIPE_V>();Adds(poly,poly,0.040085832623197183f,64);PipeBarrier<PIPE_V>();Mul(poly,poly,u,64);PipeBarrier<PIPE_V>();Adds(poly,poly,0.16614812982932259f,64);PipeBarrier<PIPE_V>();Mul(poly,poly,u,64);PipeBarrier<PIPE_V>();Adds(poly,poly,0.49992211956100152f,64);PipeBarrier<PIPE_V>();Mul(poly,poly,u,64);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999569069746141f,64);PipeBarrier<PIPE_V>();Mul(poly,poly,u,64);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999996114843803f,64);
PipeBarrier<PIPE_V>();Mul(poly,poly,poly,64);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,64);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,64);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,64);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,64);PipeBarrier<PIPE_V>();Adds(t,poly,1.0f,64);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,64);PipeBarrier<PIPE_V>();Mul(f,z,poly,64);PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
PipeBarrier<PIPE_V>();auto scratch=cint.ReinterpretCast<float>();Muls(poly,t,0.32323232323232326f,64);PipeBarrier<PIPE_V>();Adds(poly,poly,-1.4545454545454546f,64);PipeBarrier<PIPE_V>();Mul(poly,poly,t,64);PipeBarrier<PIPE_V>();Adds(poly,poly,2.1212121212121211f,64);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,64);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,64);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,64);PipeBarrier<PIPE_V>();
Mul(poly,poly,scratch,64);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,64);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,64);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,64);PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,64);PipeBarrier<PIPE_V>();Mul(f,f,poly,64);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,lanes);PipeBarrier<PIPE_V>();auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);DataCopyExtParams op{1,(u32)(M*N)*2,0,0,0};
DataCopyPad(out,h,op);auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}
template<bool Negative>__aicore__ inline float TinyScalarScalarSignCoreR19(float z,float a){
a=a<5.5f?a:5.5f;float t=a*a;t=t*0.044715f;t=t+1.0f;float u=a*t;u=u*-0.012466946262544773f;u=u+0.081f;float u2=u*u;float lo=0.92219369144460805f*u;lo=lo+0.92219369144460805f;float hi=0.15369894857410132f*u;hi=hi+0.46109684572230403f;float p=hi*u2;p=p+lo;p=p*p;p=p*p;p=p*p;p=p*p;p=p*p;
p=p*p;p=p*p;float r=p*0.15224501932798096f;r=r+-0.53285756764793335f;r=r*p;r=r+0.86589354742789182f;r=r*p;r=r+-0.98483496877787691f;r=r*p;r=r+0.99970264644662499f;float numerator=Negative?z*p:z;return numerator*r;
}
__aicore__ inline float TinyScalarScalarSignPathR19(float z){
const bool neg=z<0.0f;float a=neg?-z:z;a=a<4.9550129174188555f?a:4.9550129174188555f;float t=1.0f-a*0.2018158210011117f;float w=(0.48890803302428926f+0.47233697920903633f*a)*a;t=t*t;t=t*t;t=t*t;return(neg?0.0f:z)-w*t;
}
__aicore__ inline uint16_t TinyScalarScalarHalfRint(float value){
union{float f;u32 u;}v;v.f=value;u32 a=v.u&0x7fffffffu,sign=(v.u>>16)&0x8000u;if(a>=0x38800000u)return(uint16_t)(sign|((a+0xfffu-0x38000000u)>>13));
return(uint16_t)sign;
}
__global__ __aicore__ void TinyScalar_gmm_tiny_powerCorr23(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,i32 K,i32 E){
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
GlobalTensor<u64>cache;cache.SetGlobalBuffer((__gm__ u64*)w);DataCachePreload(cache,(int16_t)0);cache.SetGlobalBuffer((__gm__ u64*)s);DataCachePreload(cache,(int16_t)0);
#endif
const i32 firstA=(i32)a.GetValue(0);i32 e=0;if(E<=1){if(ends.GetValue(0)<1)return;}else if(E<=8){e=E-1;while(e>0&&ends.GetValue(e-1)>=1)--e;}else{
i32 hi=E-1;while(e<hi){
i32 mid=(e+hi)/2;if(ends.GetValue(mid)>=1)hi=mid;else e=mid+1;
}}
float channelScale=scale.GetValue(e);i32 sum=0;i64 wi=(i64)e*K;if(K<=4){
sum=firstA*(i32)b.GetValue(wi);if(K>=2)sum+=(i32)a.GetValue(1)*(i32)b.GetValue(wi+1);if(K>=3)sum+=(i32)a.GetValue(2)*(i32)b.GetValue(wi+2);if(K>=4)sum+=(i32)a.GetValue(3)*(i32)b.GetValue(wi+3);
}else{
sum=firstA*(i32)b.GetValue(wi);for(i32 k=1;k<K;++k)sum+=(i32)a.GetValue(k)*(i32)b.GetValue(wi+k);
}
float dot=static_cast<float>(sum);float z=dot*channelScale;float result=TinyScalarScalarSignPathR19(z);uint16_t bits=TinyScalarScalarHalfRint(result);WriteGmByPassDCache((__gm__ uint16_t*)y,bits);PipeBarrier<PIPE_ALL>();
}
static void TinyScalarRun(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");bool tinyShape=M>=1&&M<=16&&N>=1&&N<=64&&K>=1&&K<=32&&E>=1&&E<=128&&(i64)M*N<=64&&(i64)M*N*K<=256;
if(tinyShape){
u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);if(platform->GetCoreNumAiv()>=1&&ubBytes>=16384){
if(M<=1&&N<=1)TinyScalar_gmm_tiny_powerCorr23<<<1,nullptr,stream>>>(x,weight,scale,group_list,y,K,E);else TinyScalar_gmm_tiny_midfuse_r2a005<<<1,nullptr,stream>>>(x,weight,scale,group_list,y,M,K,N,E);aclError status=aclrtSynchronizeStream(stream);if(status!=ACL_SUCCESS)throw std::runtime_error("GMM tiny ACL error "+std::to_string(status));
return;
}}
u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());if(!blocks)throw std::runtime_error("no Cube cores");matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);
mt.SetShape(16,64,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(16,64,64);mt.SetBufferSpace(-1,-1,-1);TCubeTiling tiling;if(mt.GetTiling(tiling)!=0)throw std::runtime_error("INT8 tiling failed");void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};
checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*4096,ACL_MEM_MALLOC_HUGE_FIRST));TinyScalar_grouped_matmul_gelu_custom<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E);aclError status=aclrtSynchronizeStream(stream);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
using namespace AscendC;
using namespace AscendC;
using ShortNarrow_A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using ShortNarrow_B=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using ShortNarrow_C=MatmulType<TPosition::GM,CubeFormat::ND,i32>;
__aicore__ inline void ShortNarrow_PortableGelu(LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>scratch,LocalTensor<half>h,i32 count){
Abs(u,z,count);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,count);PipeBarrier<PIPE_V>();Mul(t,u,u,count);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(u,u,t,count);PipeBarrier<PIPE_V>();Muls(u,u,-0.024933892525089545f,count);PipeBarrier<PIPE_V>();Duplicate(t,0.0013888888888888889f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.0083333333333333332f,count);
PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.041666666666666664f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.16666666666666666f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.5f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);
PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Muls(u,t,-2.0f,count);PipeBarrier<PIPE_V>();Adds(u,u,1.0f,count);PipeBarrier<PIPE_V>();Muls(u,u,0.3333333333333333333f,count);PipeBarrier<PIPE_V>();Duplicate(f,1.0f,count);PipeBarrier<PIPE_V>();for(i32 j=0;j<10;++j){
Mul(f,f,u,count);PipeBarrier<PIPE_V>();Adds(f,f,1.0f,count);PipeBarrier<PIPE_V>();
}
Muls(f,f,0.6666666666666666667f,count);PipeBarrier<PIPE_V>();Mul(scratch,z,t,count);PipeBarrier<PIPE_V>();auto sign=h.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,count);PipeBarrier<PIPE_V>();Select(scratch,sign,scratch,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();Mul(f,f,scratch,count);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
}
__aicore__ inline void ShortNarrow_PortableConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,i64 globalRow,i32 col,i32 e,i32 mr,i32 nr,i32 N,i32 pitch){
i32 count=mr*pitch;if(nr<pitch){
Duplicate(sc,0.0f,pitch);auto zero=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(zero);WaitFlag<HardEvent::V_MTE2>(zero);
}
DataCopy(cint,c,16*pitch);DataCopyPad(sc,scale[(i64)e*N+col],DataCopyExtParams{1,(u32)nr*4,0,0,0},DataCopyPadExtParams<float>{true,0,0,0});auto consumed=pipe.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(consumed);WaitFlag<HardEvent::MTE2_S>(consumed);auto iv=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(iv);WaitFlag<HardEvent::MTE2_V>(iv);Cast(z,cint,RoundMode::CAST_RINT,16*pitch);PipeBarrier<PIPE_V>();for(i32 row=0;row<mr;++row)Mul(z[row*pitch],z[row*pitch],sc,pitch);
PipeBarrier<PIPE_V>();ShortNarrow_PortableGelu(z,u,t,f,cint.ReinterpretCast<float>(),h,count);auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 row=0;row<mr;++row)DataCopyPad(out[(globalRow+row)*N+col],h[row*pitch],DataCopyExtParams{1,(u32)nr*2,0,0,0});auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}
__aicore__ inline void ShortNarrow_PortableFullConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,LocalTensor<float>poly,LocalTensor<uint8_t>normal,LocalTensor<uint8_t>tail0,LocalTensor<uint8_t>tail1,LocalTensor<uint8_t>tail2,LocalTensor<uint8_t>tail3,i64 globalRow,i32 col,i32 e,i32 tileRows,i32 N,i32 tileM,i32 bank){
for(i32 slab=0;slab<tileRows;slab+=16){
i32 mr=tileRows-slab<16?tileRows-slab:16;ShortNarrow_PortableConsume(pipe,c[bank*tileM*256+slab*256],scale,out,cint,z,u,t,f,sc,h,globalRow+slab,col,e,mr,256,N,256);
}}
__global__ __aicore__ __mix__(1,2)void ShortNarrow_AsyncFallback(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileM){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortNarrow_A,ShortNarrow_B,ShortNarrow_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2*tileM*256);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,1024);pipe.InitBuffer(yy,8192);TBuf<TPosition::VECCALC>polyBuffer,normalBuffer;pipe.InitBuffer(polyBuffer,16384);pipe.InitBuffer(normalBuffer,1024);auto poly=polyBuffer.Get<float>();auto normal=normalBuffer.Get<uint8_t>();
TBuf<TPosition::VECCALC>tailBuffer;pipe.InitBuffer(tailBuffer,2048);auto tail0=tailBuffer.Get<uint8_t>();auto tail1=tail0[512];auto tail2=tail0[1024];auto tail3=tail0[1536];auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();
auto h=yy.Get<half>();i64 start=0,tileBase=0;bool pending=false;i32 bank=0,savedExpert=0,savedCol=0,savedRows=0;i64 savedGlobalRow=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=N/256;i64 count=((rows+tileM-1)/tileM)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*tileM,col=(tile%nc)*256;i32 tileRows=rows-row<tileM?rows-row:tileM;i32 producerBank=pending?bank^1:bank;mm.SetOrgShape(M,N,K,K,256);mm.SetSingleShape(tileRows,256,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);if(pending){
mm.IterateAll<false>(c[producerBank*tileM*256],0,false,true);ShortNarrow_PortableFullConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,normal,tail0,tail1,tail2,tail3,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);mm.WaitIterateAll();mm.End();bank=producerBank;
}else{mm.IterateAll(c[producerBank*tileM*256]);mm.End();}
pending=true;savedGlobalRow=start+row;savedCol=col;savedExpert=e;savedRows=tileRows;
}
tileBase+=count;start=end;
}
if(pending){
ShortNarrow_PortableFullConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,normal,tail0,tail1,tail2,tail3,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);
}}}
__aicore__ inline void ShortNarrow_Fp32FastConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,LocalTensor<float>poly,LocalTensor<uint8_t>normal,LocalTensor<uint8_t>tail0,LocalTensor<uint8_t>tail1,LocalTensor<uint8_t>tail2,LocalTensor<uint8_t>tail3,i64 globalRow,i32 col,i32 e,i32 tileRows,i32 N,i32 tileM,i32 bank){
ShortNarrow_PortableFullConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,normal,
tail0,tail1,tail2,tail3,globalRow,col,e,tileRows,N,tileM,bank);
}
__global__ __aicore__ __mix__(1,2)void ShortNarrow_MeasuredFast(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileM){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortNarrow_A,ShortNarrow_B,ShortNarrow_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2*tileM*256);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,1024);pipe.InitBuffer(yy,8192);TBuf<TPosition::VECCALC>polyBuffer,normalBuffer;pipe.InitBuffer(polyBuffer,16384);pipe.InitBuffer(normalBuffer,1024);auto poly=polyBuffer.Get<float>();auto normal=normalBuffer.Get<uint8_t>();
TBuf<TPosition::VECCALC>tailBuffer;pipe.InitBuffer(tailBuffer,2048);auto tail0=tailBuffer.Get<uint8_t>();auto tail1=tail0[512];auto tail2=tail0[1024];auto tail3=tail0[1536];auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();
auto h=yy.Get<half>();i64 start=0,tileBase=0;bool pending=false;i32 bank=0,savedExpert=0,savedCol=0,savedRows=0;i64 savedGlobalRow=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=N/256;i64 count=((rows+tileM-1)/tileM)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*tileM,col=(tile%nc)*256;i32 tileRows=rows-row<tileM?rows-row:tileM;i32 producerBank=pending?bank^1:bank;mm.SetOrgShape(M,N,K,K,256);mm.SetSingleShape(tileRows,256,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);if(pending){
mm.IterateAll<false>(c[producerBank*tileM*256],0,false,true);ShortNarrow_Fp32FastConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,normal,tail0,tail1,tail2,tail3,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);mm.WaitIterateAll();mm.End();bank=producerBank;
}else{mm.IterateAll(c[producerBank*tileM*256]);mm.End();}
pending=true;savedGlobalRow=start+row;savedCol=col;savedExpert=e;savedRows=tileRows;
}
tileBase+=count;start=end;
}
if(pending){
ShortNarrow_Fp32FastConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,normal,tail0,tail1,tail2,tail3,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);
}}}
__global__ __aicore__ __mix__(1,2)void ShortNarrow_Legacy64(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortNarrow_A,ShortNarrow_B,ShortNarrow_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*1024);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,4096);pipe.InitBuffer(zz,4096);pipe.InitBuffer(uu,4096);pipe.InitBuffer(tt,4096);pipe.InitBuffer(ff,4096);pipe.InitBuffer(ss,256);pipe.InitBuffer(yy,2048);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+63)/64;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*64;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<64?N-col:64;mm.SetOrgShape(M,N,K,K,64);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortNarrow_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,64);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortNarrow_Legacy128(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortNarrow_A,ShortNarrow_B,ShortNarrow_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2048);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,8192);pipe.InitBuffer(zz,8192);pipe.InitBuffer(uu,8192);pipe.InitBuffer(tt,8192);pipe.InitBuffer(ff,8192);pipe.InitBuffer(ss,512);pipe.InitBuffer(yy,4096);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+127)/128;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*128;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<128?N-col:128;mm.SetOrgShape(M,N,K,K,128);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortNarrow_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,128);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortNarrow_Legacy256(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 enableFastPath){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortNarrow_A,ShortNarrow_B,ShortNarrow_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*4096);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=N/256+((N%256)+127)/128;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)<N/256?(tile%nc)*256:(N/256)*256+((tile%nc)-N/256)*128;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<((tile%nc)<N/256?256:128)?N-col:((tile%nc)<N/256?256:128);mm.SetOrgShape(M,N,K,K,nr<=64?64:nr<=128?128:256);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);
mm.IterateAll(c);mm.End();ShortNarrow_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,nr<=64?64:nr<=128?128:256);
}
tileBase+=count;start=end;
}}}
using NarrowEvent=HardEvent;
template<bool ScaleRepeat=false,bool F32LogTail=false>__aicore__ inline void ShortNarrowNativeVec(LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<float>poly,LocalTensor<uint8_t>tail0,i32 mr,i32 castCount,i32 outputFree=-1){
Cast(z,cint,RoundMode::CAST_RINT,castCount);PipeBarrier<PIPE_V>();if(ScaleRepeat&&mr>4){
for(i32 q=0;q<4;++q)Mul(z[q*64],z[q*64],sc[q*64],(u64)64,(uint8_t)mr,BinaryRepeatParams{1,1,1,32,32,0});
}else for(i32 r=0;r<mr;++r)Mul(z[r*256],z[r*256],sc,256);PipeBarrier<PIPE_V>();const i32 halfCount=mr*256;if constexpr(F32LogTail){
Abs(u,z,halfCount);Maxs(poly,z,0.0f,halfCount);PipeBarrier<PIPE_V>();Muls(t,u,-0.413330078125f,halfCount);PipeBarrier<PIPE_V>();Adds(t,t,-0.7099609375f,halfCount);PipeBarrier<PIPE_V>();Mul(t,t,u,halfCount);PipeBarrier<PIPE_V>();Adds(t,t,-0.71533203125f,halfCount);PipeBarrier<PIPE_V>();Exp(t,t,halfCount);PipeBarrier<PIPE_V>();Mul(t,t,u,halfCount);PipeBarrier<PIPE_V>();Sub(poly,poly,t,halfCount);PipeBarrier<PIPE_V>();if(outputFree>=0)WaitFlag<NarrowEvent::MTE3_V>((uint8_t)outputFree);Cast(f.ReinterpretCast<half>(),poly,RoundMode::CAST_RINT,halfCount);
PipeBarrier<PIPE_V>();
}else{

ShortNarrow_PortableGelu(z,u,t,poly,cint.ReinterpretCast<float>(),f.ReinterpretCast<half>(),halfCount);
}}
__aicore__ inline void B1401ShortNarrowConsumeAndRelease(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>ci,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<float>poly,LocalTensor<uint8_t>mask,i64 row,i32 e,i32 mr,i32 credit){
DataCopy(ci,c,4096);CrossCoreSetFlag<2,PIPE_MTE2>(credit);DataCopyPad(sc,scale[(i64)e*256],DataCopyExtParams{1,1024,0,0,0},DataCopyPadExtParams<float>{true,0,0,0});auto ev=pipe.FetchEventID(NarrowEvent::MTE2_V);SetFlag<NarrowEvent::MTE2_V>(ev);WaitFlag<NarrowEvent::MTE2_V>(ev);ShortNarrowNativeVec(ci,z,u,t,f,sc,poly,mask,mr,4096);auto vo=pipe.FetchEventID(NarrowEvent::V_MTE3);
SetFlag<NarrowEvent::V_MTE3>(vo);WaitFlag<NarrowEvent::V_MTE3>(vo);DataCopyPad(out[row*256],f.ReinterpretCast<half>(),DataCopyExtParams{(uint16_t)mr,512,0,0,0});auto reuse=pipe.FetchEventID(NarrowEvent::MTE3_MTE2);SetFlag<NarrowEvent::MTE3_MTE2>(reuse);WaitFlag<NarrowEvent::MTE3_MTE2>(reuse);
}
__global__ __aicore__ __mix__(1,2)void B1401NarrowEventarlyCredit(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,i32 M,i32 K,i32 E){
InitSocState();TPipe pipe;GlobalTensor<int8_t>a,b;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
const i32 pairId=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*16384);TBuf<TPosition::A1>a1Buffer;TBuf<TPosition::B1>b1Buffer;TBuf<TPosition::A2>a2Buffer;TBuf<TPosition::B2>b2Buffer;TBuf<TPosition::CO1>c0Buffer;pipe.InitBuffer(a1Buffer,32*K);pipe.InitBuffer(b1Buffer,K*256);pipe.InitBuffer(a2Buffer,32*K);pipe.InitBuffer(b2Buffer,K*256);
pipe.InitBuffer(c0Buffer,32*256*4);auto a1=a1Buffer.Get<int8_t>();auto b1=b1Buffer.Get<int8_t>();auto a2=a2Buffer.Get<int8_t>();auto b2=b2Buffer.Get<int8_t>();auto c0=c0Buffer.Get<i32>();i64 start=0,base=0,seq=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i64 count=(rows+31)/32;for(i64 tile=(pairId-base%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile*32,mr=rows-row<32?rows-row:32,bank=seq&1;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);DataCopy(a1,a[(start+row)*K],Nd2NzParams{1,(uint16_t)mr,(uint16_t)K,0,(uint16_t)K,32,1,0});DataCopy(b1,b[(i64)e*K*256],Nd2NzParams{1,(uint16_t)K,256,0,256,(uint16_t)K,1,0});SetFlag<NarrowEvent::MTE2_MTE1>(EVENT_ID0);WaitFlag<NarrowEvent::MTE2_MTE1>(EVENT_ID0);
for(i32 ki=0;ki<K/32;++ki){
LoadData(a2[ki*512],a1[ki*32*32],LoadData2DParams{0,2,1,0,(uint16_t)(K/32-1),false,0});
}
for(i32 ni=0;ni<8;++ni){
LoadDataWithTranspose(b2[ni*1024],b1[ni*K*32],LoadData2dTransposeParams{0,(uint8_t)(K/32),1,15,0,inc});PipeBarrier<PIPE_MTE1>();
}
SetFlag<NarrowEvent::MTE1_M>(EVENT_ID0);WaitFlag<NarrowEvent::MTE1_M>(EVENT_ID0);Mmad(c0,a2,b2,MmadParams{32,256,(uint16_t)K,0,false,true});PipeBarrier<PIPE_ALL>();SetFlag<NarrowEvent::MTE1_MTE2>(EVENT_ID0);WaitFlag<NarrowEvent::MTE1_MTE2>(EVENT_ID0);SetFlag<NarrowEvent::M_FIX>(EVENT_ID0);WaitFlag<NarrowEvent::M_FIX>(EVENT_ID0);
Fixpipe(c[bank*8192],c0,FixpipeParamsV220{256,(uint16_t)mr,32,256,false,QuantMode_t::NoQuant,0,1,0,0,0});SetFlag<NarrowEvent::FIX_M>(EVENT_ID0);WaitFlag<NarrowEvent::FIX_M>(EVENT_ID0);CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++seq;
}
base+=count;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);
}
if ASCEND_IS_AIV{
const i32 pairId=GetBlockIdx()/2,pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*16384);GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)scaleIn);out.SetGlobalBuffer((__gm__ half*)y);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);
pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,1024);pipe.InitBuffer(yy,8192);TBuf<TPosition::VECCALC>polyBuffer,normalBuffer;pipe.InitBuffer(polyBuffer,16384);pipe.InitBuffer(normalBuffer,1024);auto poly=polyBuffer.Get<float>();TBuf<TPosition::VECCALC>tailBuffer;pipe.InitBuffer(tailBuffer,2048);auto tail0=tailBuffer.Get<uint8_t>();auto cint=ci.Get<i32>();
auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,base=0,seq=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i64 count=(rows+31)/32;for(i64 tile=(pairId-base%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile*32,mr=rows-row<32?rows-row:32,bank=seq&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=GetSubBlockIdx()*16,valid=mr-off;if(valid>16)valid=16;if(valid>0)B1401ShortNarrowConsumeAndRelease(pipe,c[bank*8192+off*256],scale,out,cint,z,u,t,f,sc,poly,tail0,start+row+off,e,valid,8+bank);else CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);
++seq;
}
base+=count;start=end;
}}
PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void B16Pre(GlobalTensor<i32>c,GlobalTensor<float>s,LocalTensor<i32>ci,LocalTensor<float>sc,uint8_t freeIn,uint8_t ready,i32 credit,i32 valid,i32 N=256,bool scaleCached=false){
WaitFlag<NarrowEvent::V_MTE2>(freeIn);CrossCoreWaitFlag<2,PIPE_MTE2>(credit);if(valid>0)DataCopy(ci,c,valid*256);CrossCoreSetFlag<2,PIPE_MTE2>(credit);if(valid>0){
if(N<256){
Duplicate(sc,0.0f,256);PipeBarrier<PIPE_V>();SetFlag<NarrowEvent::V_MTE2>(freeIn);WaitFlag<NarrowEvent::V_MTE2>(freeIn);
}
if(!scaleCached)DataCopy(sc,s,N);
}
SetFlag<NarrowEvent::MTE2_V>(ready);
}
template<bool ScaleRepeat=false,bool F32LogTail=false>__aicore__ inline void B16Post(GlobalTensor<half>out,LocalTensor<i32>ci,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<float>poly,LocalTensor<uint8_t>mask,i64 row,i32 mr,uint8_t in,uint8_t done,uint8_t vo,i32 N=256,bool lateOutputWait=false){
ShortNarrowNativeVec<ScaleRepeat,F32LogTail>(ci,z,u,t,f,sc,poly,mask,mr,mr*256,lateOutputWait?(i32)done:-1);SetFlag<NarrowEvent::V_MTE2>(in);SetFlag<NarrowEvent::V_MTE3>(vo);WaitFlag<NarrowEvent::V_MTE3>(vo);DataCopyPad(out[row*N],f.ReinterpretCast<half>(),DataCopyExtParams{(uint16_t)mr,(u32)N*2,(u32)(256-N)/16,0,0});SetFlag<NarrowEvent::MTE3_V>(done);
}
__aicore__ inline void B17ShortNarrowFill(GlobalTensor<int8_t>a,GlobalTensor<int8_t>b,GlobalTensor<i32>jobs,LocalTensor<int8_t>la,LocalTensor<int8_t>lb,i32 job,i32 K,uint8_t empty,uint8_t filled,i32&row,i32&mr,i32&e,i32 previous=-1,i32 N=256){
if((job&3)==2)DataCacheCleanAndInvalid<i32,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(jobs[8+job*4]);row=jobs.GetValue(8+job*4);mr=jobs.GetValue(9+job*4);e=jobs.GetValue(10+job*4);WaitFlag<NarrowEvent::MTE1_MTE2>(empty);DataCopy(la,a[(i64)row*K],Nd2NzParams{1,(uint16_t)mr,(uint16_t)K,0,(uint16_t)K,32,1,0});
if(e!=previous){
if(N<256){
InitConstValueParams<u32>pad{};pad.repeatTimes=1;pad.blockNum=(uint16_t)(K*8);pad.dstGap=0;pad.initValue=0;InitConstValue(lb.ReinterpretCast<u32>(),pad);PipeBarrier<PIPE_MTE2>();
}
DataCopy(lb,b[(i64)e*K*N],Nd2NzParams{1,(uint16_t)K,(uint16_t)N,0,(uint16_t)N,(uint16_t)K,1,0});
}
SetFlag<NarrowEvent::MTE2_MTE1>(filled);
}
template<bool Banks,bool Prefetch=false,bool Ring3=false,bool ComputeFirst=false,bool Locality=false,bool AlignedRanks=false,bool ScaleRepeat=false,bool GroupA=false,bool BlockReadyFix=false,bool F32LogTail=false,bool ImplicitFirstC=false>__aicore__ inline void ShortNarrowWorklistBanks(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E,i32 N=256){
static_assert(!ImplicitFirstC||(Banks&&Prefetch));InitSocState();TPipe pipe;GlobalTensor<int8_t>a,b;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
const i32 pairId=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*(Ring3?24576:16384));TBuf<TPosition::A1>a1Buffer;TBuf<TPosition::B1>b1Buffer;TBuf<TPosition::A2>a2Buffer;TBuf<TPosition::B2>b2Buffer;TBuf<TPosition::CO1>c0Buffer;pipe.InitBuffer(a1Buffer,(Prefetch?64:32)*K);pipe.InitBuffer(b1Buffer,(Prefetch?512:256)*K);pipe.InitBuffer(a2Buffer,32*K);
pipe.InitBuffer(b2Buffer,K*256);pipe.InitBuffer(c0Buffer,32*256*4);auto a1=a1Buffer.Get<int8_t>();auto b1=b1Buffer.Get<int8_t>();auto a2=a2Buffer.Get<int8_t>();auto b2=b2Buffer.Get<int8_t>();auto c0=c0Buffer.Get<i32>();GlobalTensor<i32>jobs;jobs.SetGlobalBuffer((__gm__ i32*)taskTable+pairId*2048);CrossCoreWaitFlag<2,PIPE_MTE2>(10);auto metadataReady=pipe.FetchEventID(NarrowEvent::MTE2_S);
SetFlag<NarrowEvent::MTE2_S>(metadataReady);WaitFlag<NarrowEvent::MTE2_S>(metadataReady);DataCacheCleanAndInvalid<i32,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(jobs);const i32 jobCount=jobs.GetValue(0);if constexpr(Prefetch){
uint8_t empty[2],filled[2];i32 rows[2],sizes[2],experts[2];i32 resident=-1;bool previousFine=false;uint8_t operandFree=0;if constexpr(BlockReadyFix)operandFree=(uint8_t)pipe.AllocEventID<NarrowEvent::M_MTE1>();for(i32 j=0;j<2;++j){
empty[j]=(uint8_t)pipe.AllocEventID<NarrowEvent::MTE1_MTE2>();filled[j]=(uint8_t)pipe.AllocEventID<NarrowEvent::MTE2_MTE1>();SetFlag<NarrowEvent::MTE1_MTE2>(empty[j]);
}
if(jobCount>0)B17ShortNarrowFill(a,b,jobs,a1,b1,0,K,empty[0],filled[0],rows[0],sizes[0],experts[0],-1,N);for(i32 job=0;job<jobCount;++job){
const i32 bank=job&1,gm=Ring3?job%3:bank,credit=gm+(Ring3?7:8);if(job+1<jobCount){
const i32 next=bank^1;B17ShortNarrowFill(a,b,jobs,a1[next*32*K],b1[next*256*K],job+1,K,empty[next],filled[next],rows[next],sizes[next],experts[next],Locality?experts[bank]:-1,N);
}
const i32 globalRow=rows[bank],mr=sizes[bank];if(!ImplicitFirstC||job>=(Ring3?3:2))CrossCoreWaitFlag<2,PIPE_FIX>(credit);WaitFlag<NarrowEvent::MTE2_MTE1>(filled[bank]);auto activeA=a1[bank*32*K],activeB=b1[bank*256*K];if constexpr(BlockReadyFix){if(previousFine)WaitFlag<NarrowEvent::M_MTE1>(operandFree);}
if constexpr(GroupA){
for(i32 mi=0;mi<2;++mi)LoadData(a2[mi*(K/32)*512],activeA[mi*512],LoadData2DParams{0,(uint8_t)(K/32),2,0,0,false,0});
}else for(i32 ki=0;ki<K/32;++ki){
LoadData(a2[ki*512],activeA[ki*32*32],LoadData2DParams{0,2,1,0,(uint16_t)(K/32-1),false,0});
}
if(!Locality||experts[bank]!=resident){
for(i32 ni=0;ni<8;++ni){
LoadDataWithTranspose(b2[ni*1024],activeB[ni*K*32],LoadData2dTransposeParams{0,(uint8_t)(K/32),1,15,0,inc});PipeBarrier<PIPE_MTE1>();
}
resident=experts[bank];
}
SetFlag<NarrowEvent::MTE1_MTE2>(empty[bank]);SetFlag<NarrowEvent::MTE1_M>(EVENT_ID0);WaitFlag<NarrowEvent::MTE1_M>(EVENT_ID0);const bool fine=BlockReadyFix&&mr==32;Mmad(c0,a2,b2,MmadParams{32,256,(uint16_t)K,(uint8_t)(fine?3:0),false,true});if(fine){SetFlag<NarrowEvent::M_MTE1>(operandFree);}else{
PipeBarrier<PIPE_ALL>();SetFlag<NarrowEvent::M_FIX>(EVENT_ID0);WaitFlag<NarrowEvent::M_FIX>(EVENT_ID0);
}
previousFine=fine;Fixpipe(c[gm*8192],c0,FixpipeParamsV220{256,(uint16_t)mr,32,256,false,QuantMode_t::NoQuant,0,1,0,0,(uint8_t)(fine?3:0)});SetFlag<NarrowEvent::FIX_M>(EVENT_ID0);WaitFlag<NarrowEvent::FIX_M>(EVENT_ID0);CrossCoreSetFlag<2,PIPE_FIX>(credit);
}
if constexpr(BlockReadyFix){if(previousFine)WaitFlag<NarrowEvent::M_MTE1>(operandFree);pipe.ReleaseEventID<NarrowEvent::M_MTE1>(operandFree);}
for(i32 j=0;j<2;++j){
WaitFlag<NarrowEvent::MTE1_MTE2>(empty[j]);pipe.ReleaseEventID<NarrowEvent::MTE1_MTE2>(empty[j]);pipe.ReleaseEventID<NarrowEvent::MTE2_MTE1>(filled[j]);
}}else{
for(i32 job=0;job<jobCount;++job){
if((job&3)==2)DataCacheCleanAndInvalid<i32,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(jobs[8+job*4]);const i32 globalRow=jobs.GetValue(8+job*4),mr=jobs.GetValue(9+job*4),e=jobs.GetValue(10+job*4),bank=job&1;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);DataCopy(a1,a[(i64)globalRow*K],Nd2NzParams{1,(uint16_t)mr,(uint16_t)K,0,(uint16_t)K,32,1,0});
DataCopy(b1,b[(i64)e*K*256],Nd2NzParams{1,(uint16_t)K,256,0,256,(uint16_t)K,1,0});SetFlag<NarrowEvent::MTE2_MTE1>(EVENT_ID0);WaitFlag<NarrowEvent::MTE2_MTE1>(EVENT_ID0);if constexpr(GroupA){
for(i32 mi=0;mi<2;++mi)LoadData(a2[mi*(K/32)*512],a1[mi*512],LoadData2DParams{0,(uint8_t)(K/32),2,0,0,false,0});
}else for(i32 ki=0;ki<K/32;++ki){
LoadData(a2[ki*512],a1[ki*32*32],LoadData2DParams{0,2,1,0,(uint16_t)(K/32-1),false,0});
}
for(i32 ni=0;ni<8;++ni){
LoadDataWithTranspose(b2[ni*1024],b1[ni*K*32],LoadData2dTransposeParams{0,(uint8_t)(K/32),1,15,0,inc});PipeBarrier<PIPE_MTE1>();
}
SetFlag<NarrowEvent::MTE1_M>(EVENT_ID0);WaitFlag<NarrowEvent::MTE1_M>(EVENT_ID0);Mmad(c0,a2,b2,MmadParams{32,256,(uint16_t)K,0,false,true});PipeBarrier<PIPE_ALL>();SetFlag<NarrowEvent::MTE1_MTE2>(EVENT_ID0);WaitFlag<NarrowEvent::MTE1_MTE2>(EVENT_ID0);SetFlag<NarrowEvent::M_FIX>(EVENT_ID0);WaitFlag<NarrowEvent::M_FIX>(EVENT_ID0);
Fixpipe(c[bank*8192],c0,FixpipeParamsV220{256,(uint16_t)mr,32,256,false,QuantMode_t::NoQuant,0,1,0,0,0});SetFlag<NarrowEvent::FIX_M>(EVENT_ID0);WaitFlag<NarrowEvent::FIX_M>(EVENT_ID0);CrossCoreSetFlag<2,PIPE_FIX>(8+bank);
}}
if constexpr(Ring3){if(!ImplicitFirstC||jobCount>0)CrossCoreWaitFlag<2,PIPE_FIX>(7);}
if(!ImplicitFirstC||jobCount>(Ring3?1:0))CrossCoreWaitFlag<2,PIPE_FIX>(8);if(!ImplicitFirstC||jobCount>(Ring3?2:1))CrossCoreWaitFlag<2,PIPE_FIX>(9);
}
if ASCEND_IS_AIV{
const i32 pairId=GetBlockIdx()/2,pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*(Ring3?24576:16384));GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)scaleIn);out.SetGlobalBuffer((__gm__ half*)y);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,Banks?32768:16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);
pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,Banks?2048:1024);pipe.InitBuffer(yy,8192);TBuf<TPosition::VECCALC>polyBuffer,normalBuffer;pipe.InitBuffer(polyBuffer,16384);pipe.InitBuffer(normalBuffer,1024);auto poly=polyBuffer.Get<float>();TBuf<TPosition::VECCALC>tailBuffer;pipe.InitBuffer(tailBuffer,2048);auto tail0=tailBuffer.Get<uint8_t>();
auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();TBuf<TPosition::VECCALC>endpointBuffer,jobBuffer;pipe.InitBuffer(endpointBuffer,1024);pipe.InitBuffer(jobBuffer,8192);auto cachedEnds=endpointBuffer.Get<i64>();
TBuf<TPosition::VECCALC>scaleCacheBuffer;LocalTensor<float>cachedScale;if constexpr(ImplicitFirstC){pipe.InitBuffer(scaleCacheBuffer,16384);cachedScale=scaleCacheBuffer.Get<float>();}
auto localJobs=jobBuffer.Get<i32>();DataCopy(cachedEnds,ends,E);auto endpointsReady=pipe.FetchEventID(NarrowEvent::MTE2_S);SetFlag<NarrowEvent::MTE2_S>(endpointsReady);WaitFlag<NarrowEvent::MTE2_S>(endpointsReady);i32 jobCount=0;i64 plannedStart=0,plannedBase=0;if constexpr(Locality){
bool aligned=false;if constexpr(AlignedRanks){
i32 allEnds=0;for(i32 e=0;e<E;e+=4)allEnds|=(i32)cachedEnds.GetValue(e)|(i32)cachedEnds.GetValue(e+1)|(i32)cachedEnds.GetValue(e+2)|(i32)cachedEnds.GetValue(e+3);aligned=(allEnds&(32-1))==0;
}
if(aligned){
const i32 total=M/32,lo=total*pairId/pairs,hi=total*(pairId+1)/pairs;if(lo<hi){
i32 low=0,high=E;while(low<high){
const i32 mid=(low+high)/2;if(cachedEnds.GetValue(mid)<=lo*32)low=mid+1;else high=mid;
}
i32 e=low,end=(i32)cachedEnds.GetValue(e);for(i32 rank=lo;rank<hi;++rank){
const i32 row=rank*32;while(row>=end){++e;end=(i32)cachedEnds.GetValue(e);}
const i32 offset=8+jobCount*4;localJobs.SetValue(offset,row);localJobs.SetValue(offset+1,32);localJobs.SetValue(offset+2,e);localJobs.SetValue(offset+3,0);++jobCount;
}}}else{
auto prefix=normalBuffer.Get<i32>();i32 total=0,prev=0;for(i32 e=0;e<E;++e){
const i32 end=(i32)cachedEnds.GetValue(e);total+=(end-prev+31)/32;prefix.SetValue(e,total);prev=end;
}
const i32 lo=total*pairId/pairs,hi=total*(pairId+1)/pairs;if(lo<hi){
i32 low=0,high=E;while(low<high){
const i32 mid=(low+high)/2;if(prefix.GetValue(mid)<=lo)low=mid+1;else high=mid;
}
i32 groupStart=low==0?0:(i32)cachedEnds.GetValue(low-1),base=low==0?0:prefix.GetValue(low-1);for(i32 e=low;e<E&&base<hi;++e){
const i32 end=(i32)cachedEnds.GetValue(e),next=prefix.GetValue(e),limit=next<hi?next:hi;for(i32 rank=lo>base?lo:base;rank<limit;++rank){
const i32 row=groupStart+(rank-base)*32,mr=end-row<32?end-row:32,offset=8+jobCount*4;localJobs.SetValue(offset,row);localJobs.SetValue(offset+1,mr);localJobs.SetValue(offset+2,e);localJobs.SetValue(offset+3,0);++jobCount;
}
groupStart=end;base=next;
}}}}else{
for(i32 e=0;e<E;++e){
i64 end=cachedEnds.GetValue(e);i32 rows=end-plannedStart;i64 count=(rows+31)/32;for(i64 tile=(pairId-plannedBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile*32,mr=rows-row<32?rows-row:32,offset=8+jobCount*4;localJobs.SetValue(offset,(i32)plannedStart+row);localJobs.SetValue(offset+1,mr);localJobs.SetValue(offset+2,e);localJobs.SetValue(offset+3,0);++jobCount;
}
plannedBase+=count;plannedStart=end;
}}
localJobs.SetValue(0,jobCount);for(i32 i=1;i<8;++i)localJobs.SetValue(i,0);const i32 usedWords=8+4*jobCount,publishedWords=(usedWords+7)/8*8;for(i32 i=usedWords;i<publishedWords;++i)localJobs.SetValue(i,0);if(GetSubBlockIdx()==0){
GlobalTensor<i32>jobs;jobs.SetGlobalBuffer((__gm__ i32*)taskTable+pairId*2048);auto publish=pipe.FetchEventID(NarrowEvent::S_MTE3);SetFlag<NarrowEvent::S_MTE3>(publish);WaitFlag<NarrowEvent::S_MTE3>(publish);DataCopy(jobs,localJobs,publishedWords);
}
CrossCoreSetFlag<2,PIPE_MTE3>(10);
// Cache only a bounded contiguous expert span assigned by the real worklist.
// Every tile still loads C and follows the original credit/event protocol.
i32 scaleFirst=0,scaleSpan=0;bool scaleCached=false;
if constexpr(ImplicitFirstC){if(N==256&&jobCount>0){
scaleFirst=localJobs.GetValue(10);scaleSpan=localJobs.GetValue(10+(jobCount-1)*4)-scaleFirst+1;
if(scaleSpan>0&&scaleSpan<=16){
DataCopy(cachedScale,scale[(i64)scaleFirst*N],scaleSpan*N);
auto scaleReady=pipe.FetchEventID(NarrowEvent::MTE2_V);SetFlag<NarrowEvent::MTE2_V>(scaleReady);WaitFlag<NarrowEvent::MTE2_V>(scaleReady);scaleCached=true;
}}}
if constexpr(Banks){
uint8_t freeIn[2],ready[2],freeOut[2],vo[2];for(i32 k=0;k<2;++k){
freeIn[k]=(uint8_t)pipe.AllocEventID<NarrowEvent::V_MTE2>();ready[k]=(uint8_t)pipe.AllocEventID<NarrowEvent::MTE2_V>();freeOut[k]=(uint8_t)pipe.AllocEventID<NarrowEvent::MTE3_V>();vo[k]=(uint8_t)pipe.AllocEventID<NarrowEvent::V_MTE3>();SetFlag<NarrowEvent::V_MTE2>(freeIn[k]);SetFlag<NarrowEvent::MTE3_V>(freeOut[k]);
}
if constexpr(!ImplicitFirstC){
if constexpr(Ring3)CrossCoreSetFlag<2,PIPE_MTE2>(7);CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);
}
const i32 off=GetSubBlockIdx()*16;if constexpr(ComputeFirst){
if(jobCount>0){
const i32 step=0;const i32 k=step&1,e=localJobs.GetValue(10+step*4),r=localJobs.GetValue(9+step*4)-off;const i32 gm=Ring3?step%3:k;B16Pre(c[gm*8192+off*256],scale[(i64)e*N],cint[k*4096],(scaleCached?cachedScale[(e-scaleFirst)*256]:sc[k*256]),freeIn[k],ready[k],gm+(Ring3?7:8),r<16?r:16,N,scaleCached);
}
for(i32 job=0;job<jobCount;++job){
const i32 bank=job&1,globalRow=localJobs.GetValue(8+job*4),mr=localJobs.GetValue(9+job*4);WaitFlag<NarrowEvent::MTE2_V>(ready[bank]);i32 valid=mr-off;if(valid>16)valid=16;if(valid>0){
if constexpr(!(ImplicitFirstC&&F32LogTail))WaitFlag<NarrowEvent::MTE3_V>(freeOut[bank]);B16Post<ScaleRepeat,F32LogTail>(out,cint[bank*4096],z,u,t,f[bank*2048],(scaleCached?cachedScale[(localJobs.GetValue(10+job*4)-scaleFirst)*256]:sc[bank*256]),poly,tail0,(i64)globalRow+off,valid,freeIn[bank],freeOut[bank],vo[bank],N,ImplicitFirstC&&F32LogTail);
}else SetFlag<NarrowEvent::V_MTE2>(freeIn[bank]);if(job+1<jobCount){
const i32 step=job+1;const i32 k=step&1,e=localJobs.GetValue(10+step*4),r=localJobs.GetValue(9+step*4)-off;const i32 gm=Ring3?step%3:k;B16Pre(c[gm*8192+off*256],scale[(i64)e*N],cint[k*4096],(scaleCached?cachedScale[(e-scaleFirst)*256]:sc[k*256]),freeIn[k],ready[k],gm+(Ring3?7:8),r<16?r:16,N,scaleCached);
}}}else{
for(i32 step=0;step<=jobCount;++step){
if(step<jobCount){
const i32 k=step&1,e=localJobs.GetValue(10+step*4),r=localJobs.GetValue(9+step*4)-off;const i32 gm=Ring3?step%3:k;B16Pre(c[gm*8192+off*256],scale[(i64)e*N],cint[k*4096],sc[k*256],freeIn[k],ready[k],gm+(Ring3?7:8),r<16?r:16,N);
}
if(step==0)continue;const i32 job=step-1,bank=job&1,globalRow=localJobs.GetValue(8+job*4),mr=localJobs.GetValue(9+job*4);WaitFlag<NarrowEvent::MTE2_V>(ready[bank]);i32 valid=mr-off;if(valid>16)valid=16;if(valid>0){
WaitFlag<NarrowEvent::MTE3_V>(freeOut[bank]);B16Post<ScaleRepeat,F32LogTail>(out,cint[bank*4096],z,u,t,f[bank*2048],sc[bank*256],poly,tail0,(i64)globalRow+off,valid,freeIn[bank],freeOut[bank],vo[bank],N);
}else SetFlag<NarrowEvent::V_MTE2>(freeIn[bank]);
}}
for(i32 k=0;k<2;++k){
WaitFlag<NarrowEvent::V_MTE2>(freeIn[k]);WaitFlag<NarrowEvent::MTE3_V>(freeOut[k]);pipe.ReleaseEventID<NarrowEvent::V_MTE2>(freeIn[k]);pipe.ReleaseEventID<NarrowEvent::MTE2_V>(ready[k]);pipe.ReleaseEventID<NarrowEvent::MTE3_V>(freeOut[k]);pipe.ReleaseEventID<NarrowEvent::V_MTE3>(vo[k]);
}}else{
CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);for(i32 job=0;job<jobCount;++job){
const i32 globalRow=localJobs.GetValue(8+job*4),mr=localJobs.GetValue(9+job*4),e=localJobs.GetValue(10+job*4),bank=job&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=GetSubBlockIdx()*16,valid=mr-off;if(valid>16)valid=16;if(valid>0)B1401ShortNarrowConsumeAndRelease(pipe,c[bank*8192+off*256],scale,out,cint,z,u,t,f,sc,poly,tail0,(i64)globalRow+off,e,valid,8+bank);
else CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);
}}}
PipeBarrier<PIPE_ALL>();
}
__global__ __aicore__ __mix__(1,2)void B1404ShortNarrowDeviceWorklist(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E){ShortNarrowWorklistBanks<false>(x,w,scaleIn,groups,y,scratch,taskTable,M,K,E);}
__global__ __aicore__ __mix__(1,2)void B1606ShortNarrowPipeline(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E){ShortNarrowWorklistBanks<true>(x,w,scaleIn,groups,y,scratch,taskTable,M,K,E);}
__global__ __aicore__ __mix__(1,2)void B1804ShortNarrowCreditRing(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E){ShortNarrowWorklistBanks<true,true,true>(x,w,scaleIn,groups,y,scratch,taskTable,M,K,E);}
__global__ __aicore__ __mix__(1,2)void B1901ShortNarrowBalancedReuse(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E){ShortNarrowWorklistBanks<true,true,true,true,true>(x,w,scaleIn,groups,y,scratch,taskTable,M,K,E);}
__global__ __aicore__ __mix__(1,2)void B2102ShortNarrowAlignedRanks(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E){ShortNarrowWorklistBanks<true,true,true,true,true,true>(x,w,scaleIn,groups,y,scratch,taskTable,M,K,E);}
__global__ __aicore__ __mix__(1,2)void B2104ShortNarrowScaleRepeat(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E){ShortNarrowWorklistBanks<true,true,true,true,true,true,true>(x,w,scaleIn,groups,y,scratch,taskTable,M,K,E);}
__global__ __aicore__ __mix__(1,2)void B2106ShortNarrowGroupA(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E){ShortNarrowWorklistBanks<true,true,true,true,true,true,true,true>(x,w,scaleIn,groups,y,scratch,taskTable,M,K,E);}
__global__ __aicore__ __mix__(1,2)void B2521ShortNarrowBlockReadyFix(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E){ShortNarrowWorklistBanks<true,true,true,true,true,true,true,true,true>(x,w,scaleIn,groups,y,scratch,taskTable,M,K,E);}
__global__ __aicore__ __mix__(1,2)void B2524ShortNarrowF32LogTail(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E){
ShortNarrowWorklistBanks<true,true,true,true,true,true,true,true,true,true>(x,w,scaleIn,groups,y,scratch,taskTable,M,K,E);
}
__global__ __aicore__ __mix__(1,2)void B2546ShortNarrowImplicitFirstC(GM_ADDR x,GM_ADDR w,GM_ADDR scaleIn,GM_ADDR groups,GM_ADDR y,GM_ADDR scratch,GM_ADDR taskTable,i32 M,i32 K,i32 E,i32 N){
ShortNarrowWorklistBanks<true,true,true,true,true,true,true,true,true,true,true>(x,w,scaleIn,groups,y,scratch,taskTable,M,K,E,N);
}
// Extracted from user kernel(1), SHA560d30af; original attribution retained.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include "kernel_operator.h"
#include "adv_api/matmul/matmul_intf.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

// This file is the complete submission. It is included by the official main.asc.
namespace reference_case4 {
using namespace AscendC;
constexpr uint32_t BM = 128;
constexpr uint32_t L0C_INT32_ELEMS = 128U * 1024U / 4U;  // CUBE_L0C_BYTES of int32
constexpr uint32_t BN = 128;      // vector column block
constexpr uint32_t NBC = 256;      // default Cube task column width: one task covers
                                  // ceil(NBC/BN) vector column blocks, so the A tile
                                  // is fetched once for all of them.
constexpr uint32_t NBC_WIDE = 512; // widest single-task row before the split returns
constexpr uint32_t ROWS = 32;
constexpr uint32_t VEC_SIZE = ROWS * BN;
// Flat epilogue: when one Cube task covers the whole output row, a vector block
// is a run of whole rows instead of 32 rows x 128 columns.  Both the C load and
// the y store then cover one contiguous span, i.e. one DMA burst per block
// instead of one per row, and a whole task fits a single block.
constexpr uint32_t WIDE_VEC = 2 * VEC_SIZE;
// A tile that is 256 columns wide needs only half as many epilogue blocks as
// two 128-wide ones for the same elements, and every queue/sync round trip in
// a block is fixed cost.  The task width must be a multiple of 256.
constexpr uint32_t WIDE_COLS = 2 * BN;
constexpr uint32_t WIDE_ELEMS = ROWS * WIDE_COLS;
// Cross-core handshake between one AIC and its two AIVs.  The hardware encodes
// the flag id in 4 bits, and ids 11..14 are reserved by the runtime.
// sqrt(2/pi) and sqrt(2/pi)*0.044715, folded for the GELU cubic term.
constexpr float C1 = 0.7978845608028654f;
constexpr float C1C2 = 0.035677408136361406f;
// The tanh argument is scaled by -2.  Folding that factor into both constants
// removes one vector instruction per block without changing a bit: -2 is a power
// of two, so fl(x * -2c) == -2 * fl(x * c) exactly for every finite x, and the
// same holds for the add (both operands scale exactly).
constexpr float C1N = -2.0f * C1;
constexpr float C1C2N = -2.0f * C1C2;
constexpr uint16_t FLAG_READY = 8;     // 8, 9: ping-pong slots
constexpr uint16_t FLAG_DONE = 15;     // set by both AIV subcores of the group
constexpr uint32_t READY_RING = 8;
constexpr auto MM_CONFIG = GetMDLConfig(false, false, 0, false, false, false, true);

// The scalar header comes first so a core that only needs shapes (every AIV)
// does not have to pull the ~50-word Cube tiling out of GM.
constexpr uint32_t CFG_HEADER_WORDS = 13;  // + packOvl

struct Config {
    uint32_t m;
    uint32_t e;
    uint32_t k;
    uint32_t n;
    uint32_t nReal;  // the output N; `n` is the padded Cube N when padN fired
    uint32_t cores;
    uint32_t nbc;    // Cube task column width (multiple of BN, >= BN)
    uint32_t tiny;   // 1: the product is accumulated on the vector unit
    uint32_t bm;     // Cube row-tile height (128, or smaller when M cannot fill the machine)
    uint32_t manual; // 1: drive the Cube with the hand-written instruction sequence
    uint32_t flat;   // 1: every task covers the whole output row (flat epilogue blocks)
    // Row stride of B in GM.  Equal to `n` whenever B is repacked (or the
    // shape is not padded), but a padded-N shape that reads the original B
    // declares the real stride here and lets the Cube compute the padded
    // width: the surplus columns read the next B row and land in the C
    // padding, where the epilogue drops them.  The last expert is the only
    // one whose surplus read would leave the tensor, so its last column
    // task is slid back to end exactly at nReal (Tasks::Decode).
    uint32_t bStride;
    // 1: the Cube may do its first task's decode and A staging before waiting
    // for the repack flag (the AIVs' packed B), i.e. the pack runs under that
    // prologue instead of in front of it.
    uint32_t packOvl;
    AscendC::tiling::TCubeTiling cube;
};

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Columns of B that a task actually has to stage.  A shape that reads the
// original B at its real row stride carries no data past the real N, and the
// surplus used to be fetched and dropped (251 bytes of B per row on
// 2559/32/1027/773, 33.6 MB against 25.4 MB).  Narrowing the *task* to save
// those bytes is not safe -- the Cube then traps with an L0C read/write
// conflict, deterministically at 128 columns and intermittently at 32/64 -- so
// the task keeps its full width and only the staged slice shrinks; the surplus
// columns then hold stale L1 bytes and land in the C padding, which the
// epilogue never reads.
__aicore__ inline uint32_t StagedCols(const Config& cfg, uint32_t col, uint32_t cols)
{
    uint32_t r = cols;
    if (cfg.bStride != cfg.n && col + cols > cfg.nReal) {
        uint32_t real = (cfg.nReal - col + 31U) & ~31U;
        if (real < r) r = real;
    }
    return r < 32U ? 32U : r;
}

__aicore__ inline uint32_t RoundUp32(uint32_t bytes) { return (bytes + 31U) & ~31U; }

// Vector block width: a residual column block only needs `cols` real columns,
// but the vector math must run on whole 32B rows, so it is packed to a
// multiple of 16 elements (64B for int32, 32B for half) instead of a full BN.
// cat06's last block is one real column out of 128 and used to burn a whole
// block of GELU on 32 x 128 elements.
__aicore__ inline uint32_t CwOf(uint32_t colsLeft, uint32_t bc)
{
    if (colsLeft >= bc) return bc;
    return (colsLeft + 15U) & ~15U;
}
__aicore__ inline uint32_t RoundUp8(uint32_t elements) { return (elements + 7U) & ~7U; }

// C row stride in the workspace.  Same as the 32B-aligned value unless N leaves
// a partial vector block, in which case the row is padded out to a block
// boundary and the Cube computes that last block at full width: a partial
// baseN block costs the Fixpipe one column write per row and MTE2 one B column
// per K step (measured: +38 us mte2 and +42 us fixpipe for a single extra
// column, where a whole extra block costs 4-5 us).  Every C reader - the API
// Cube path, the hand-written path and both epilogues - must use this stride.
__aicore__ inline uint32_t CStride(uint32_t n, uint32_t k)
{
    uint32_t s = (n + 15U) & ~15U;
    uint32_t tail = n % BN;
    if (n > BN && tail != 0 && tail <= BN / 2 && k < 4096) {
        s = ((n + BN - 1U) / BN) * BN;
    }
    return s;
}

// A tiny problem is computed by the vector unit of one core group instead of the
// Cube pipeline (two staged copies plus a cross-core handshake).  Measured with an
// interleaved A/B on a 10-shape grid (reports/experiments/p23_tinygate): it wins
// for M <= 2 with M*K <= 8 (-6% to -13%) and loses from M >= 4, where the Cube's
// 16-row MMA granularity stops wasting work.  The buffer limits are hardware.
constexpr uint32_t TINY_MAX_MN = 2048;
constexpr uint32_t TINY_MAX_KROW = 8192;

inline bool IsTinyShape(uint32_t m, uint32_t e, uint32_t k, uint32_t n)
{
    uint64_t rowBytes = (static_cast<uint64_t>(n) + 31U) & ~31ULL;
    // The row stride is RoundUp16(n) halfs so every per-row y store is 32B
    // aligned; the C buffer must hold m rows at that stride.
    uint64_t rowStride = (static_cast<uint64_t>(n) + 15U) & ~15ULL;
    return m <= 16 && static_cast<uint64_t>(m) * k <= 32 && n <= 256 &&
           static_cast<uint64_t>(m) * rowStride <= TINY_MAX_MN &&
           static_cast<uint64_t>(e) * k * rowBytes <= TINY_MAX_KROW;
}

__aicore__ inline void LoadHeader(GM_ADDR source, Config& destination)
{
    auto src = reinterpret_cast<__gm__ const uint32_t*>(source);
    auto dst = reinterpret_cast<uint32_t*>(&destination);
    // Every word is known to be independent, so all ten loads are issued
    // before the first store: one memory latency instead of ten.  On a small
    // shape this header+init chain is most of the AIC's time.
    // The whole header must be here, `flat` included: Tasks::BuildPrefix uses
    // it to pick the row tile, and a decision that differs between the AIC and
    // the AIVs would leave the two ends of the ready/done handshake disagreeing
    // about how many tasks exist.  It used to stop at `manual` because nothing
    // on this side read `flat`.
    uint32_t v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
    uint32_t v4 = src[4], v5 = src[5], v6 = src[6], v7 = src[7];
    uint32_t v8 = src[8], v9 = src[9], v10 = src[10], v11 = src[11];
    uint32_t v12 = src[12];
    dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
    dst[4] = v4; dst[5] = v5; dst[6] = v6; dst[7] = v7;
    dst[8] = v8; dst[9] = v9; dst[10] = v10; dst[11] = v11;
    dst[12] = v12;
}

// Only the AIC needs the tiling.  Four loads are issued before the first store so
// their latency overlaps instead of being paid one word at a time.
__aicore__ inline void LoadTiling(GM_ADDR source, Config& destination)
{
    auto src = reinterpret_cast<__gm__ const uint32_t*>(source);
    auto dst = reinterpret_cast<uint32_t*>(&destination);
    constexpr uint32_t n = sizeof(Config) / sizeof(uint32_t);
    uint32_t i = CFG_HEADER_WORDS;
    for (; i + 4 <= n; i += 4) {
        uint32_t a = src[i], b = src[i + 1], c = src[i + 2], d = src[i + 3];
        dst[i] = a;
        dst[i + 1] = b;
        dst[i + 2] = c;
        dst[i + 3] = d;
    }
    for (; i < n; ++i) dst[i] = src[i];
}

// Each participant reads these short, data-dependent boundaries on the device.
// AIC cannot allocate Vector UB, so this metadata stays in scalar/stack storage.
struct Tasks {
    int64_t ends[128];
    uint32_t prefix[129];
    uint32_t nTiles;
    TBuf<TPosition::VECCALC> endsBuf;
    // Lazy scan.  On a flat shape (one Cube task per expert) the task index
    // advances monotonically on every core, so the boundary array can be
    // consumed in order as the tiles go: an expert's read then hides behind the
    // tile that came before it instead of all of them delaying the first one.
    // Measured: the full scan is 4.0 us on the point-4 prologue and it sits at
    // the head of the whole kernel's critical path.
    __gm__ const int32_t* endGm = nullptr;   // AIC: the group_list in GM
    LocalTensor<int32_t> endUb;              // AIV: the DMA'd copy
    uint32_t scE = 0, scAcc = 0, scPrev = 0, scMax = 0;
    uint32_t scRow = 0, scRows = 0, scBase = 0;   // the expert currently covering q
    uint32_t scSh = 0;                       // log2(bm) when bm is a power of two
    uint8_t lazy = 0;
    uint8_t scPow2 = 0;

    // The row-tile rule (bm 128 -> 64 for a short, splittable deal) needs the
    // whole boundary array before the first tile, which is exactly what the lazy
    // scan exists to avoid.  Keep the rule only where a host-side lower bound
    // says the deal cannot be short: ceil(m/bm) >= 4*cores means even the most
    // tightly packed group_list needs four tiles per core, so the rule's own
    // gate (acc <= 4*cores) can only hold on the razor case acc == 4*cores.
    // 8192/128/256/256 is exactly on that boundary (acc >= 64 == 4*cores) and
    // takes the lazy path; 8192/32/256/256 (acc >= 64 < 4*20) keeps the rule.
    __aicore__ inline void StartLazy(const Config& cfg)
    {
        lazy = 1;
        nTiles = (cfg.n + cfg.nbc - 1) / cfg.nbc;
        prefix[0] = 0;
        scPow2 = ((cfg.bm & (cfg.bm - 1U)) == 0) ? 1U : 0U;
        scSh = scPow2 ? static_cast<uint32_t>(__builtin_ctz(cfg.bm)) : 0U;
    }

    __aicore__ inline uint32_t FetchEnd(uint32_t e) const
    {
        if (endGm != nullptr) return static_cast<uint32_t>(endGm[2 * e]);
        return static_cast<uint32_t>(endUb.GetValue(2 * e));
    }

    // Advances the frontier until it covers task q.  Idempotent, so a decode
    // may be repeated for the same q.  False means q is past the last task.
    // The frontier keeps only the expert that currently covers q, so the scan
    // writes nothing: the old version stored every boundary into ends[] and every
    // running count into prefix[], which is what the eager path needs for its
    // binary search but the lazy path never reads.
    __aicore__ inline bool Advance(uint32_t q, const Config& cfg)
    {
        while (scAcc <= q) {
            if (scE >= cfg.e) return false;
            uint32_t cur = FetchEnd(scE);
            uint32_t rows = cur - scPrev;
            scRow = scPrev;
            scRows = rows;
            scBase = scAcc;
            scAcc += scPow2 ? (((rows + cfg.bm - 1U) >> scSh) * nTiles)
                            : RowTiles(rows, cfg.bm);
            scPrev = cur;
            ++scE;
        }
        return true;
    }

    // Compile-time split: the lazy frontier is a template argument so a shape
    // that does not use it does not carry the code.  Merely *containing* it cost
    // 1.8 us on 8192/32/256/256 (the same effect the PACK split was made for).
    template <bool LAZY>
    __aicore__ inline bool TaskOk(uint32_t q, const Config& cfg, uint32_t total)
    {
        if constexpr (LAZY) return Advance(q, cfg);
        return q < total;
    }

    __aicore__ inline void BuildPrefix(Config& cfg)
    {
        if (lazy != 0) return;   // StartLazy already fixed nTiles/prefix[0]
        nTiles = (cfg.n + cfg.nbc - 1) / cfg.nbc;
        prefix[0] = 0;
        uint32_t rowsMax = 0;
        uint32_t acc = FillPrefix(cfg, cfg.bm, rowsMax);
        // A 128-row tile is only worth halving when the resulting grid is both
        // short and splittable.  "Splittable" is exactly one expert with more
        // than 64 rows: below that every expert is one tile either way and the
        // switch is pure overhead (+0.26 measured).  "Short" is 2..4 tasks per
        // core: at 4 tasks per core the ragged tail costs up to a quarter of the
        // epilogue; from 8 tasks per core the doubled B traffic is the bigger
        // term and the 128-row grid wins.
        // Measured against the 128-row grid, paired, 16 rounds, over 30
        // group_lists on four shapes: every input inside this window gains
        // 0.9-1.7 us (8192/128/256/256: -0.94 -1.27 -1.11; 8192/64/256/256:
        // -1.73 -1.45) and every input outside it moves by 0.0 +- 0.4 us.  A
        // uniform group_list of 64-row experts has rowsMax 64, so the visible
        // point-4 case keeps its 128-row grid untouched.
        // The N == 256 gate is load-bearing: the same rule at N=128 measured
        // +2.79 us in 10 of 10 rounds, the only reproducible loss in the set.
        if (cfg.flat != 0 && cfg.n == 256U && cfg.bm == 128U && rowsMax > 64U &&
            acc >= 2U * cfg.cores && acc <= 4U * cfg.cores) {
            cfg.bm = 64U;
            FillPrefix(cfg, cfg.bm, rowsMax);
        }
        // A padded shape that reads the original B fetches each row in `nTasks`
        // pieces, and every piece drags in the one 128B sector its neighbour
        // also needs.  On 2559/32/1027/773 four 256-wide pieces take 10.0
        // sectors per row while three 384-wide ones take 9.2 for the same real
        // bytes: AIC MTE2 43.8 -> 38.9 us, kernel -4.4.  A task holds
        // bm x nbc int32 accumulators in L0C, so the wider task has to shrink
        // the row tile, and that is only free when the widest expert still fits
        // in one tile - otherwise the expert's B slice is read once per tile.
        // n == 1024 is the only padded width measured (four 256-column tasks
        // against three of 384, the last of each row landing inside the real
        // columns); AIC and both AIVs read the same group_list, so the decision
        // needs no communication.
        // Dual accumulator half-regions (see RunCubeStream).  They need
        // bm*nbc*4*2 <= L0C, i.e. bm 64 at a 256-wide column task.  Halving bm
        // costs nothing when every expert already fits one 64-row tile - then
        // the row-tile count, the K chunk and the B read width are all
        // unchanged and the only difference is that the fixpipe of task i
        // overlaps the MMA of task i+1.  Measured on 1024/16/2048/1024 in the
        // main line: -1.10 us (4%), y bit-identical; the same rule reaches
        // 2048/32/1024/768 and 4096/64/512/768, whose experts are 64 rows too.
        if (cfg.manual == 2U && cfg.nbc == 256U && cfg.bm == 128U && rowsMax <= 64U &&
            2U * static_cast<uint64_t>(64U) * cfg.nbc <= L0C_INT32_ELEMS) {
            cfg.bm = 64U;
            FillPrefix(cfg, cfg.bm, rowsMax);
        }
        if (cfg.manual == 2U && cfg.n == 1024U && cfg.nReal != cfg.n &&
            cfg.bStride == cfg.nReal && rowsMax != 0U && rowsMax <= 128U) {
            uint32_t bLim = (rowsMax + 15U) / 16U * 16U;
            if (bLim > 80U) {
                // An 81..128-row expert does not fit one 80-row tile, so the
                // wide grid pays an extra B pass for every tile it spills.
                // Measured over the verified point-15 class: one extra row tile
                // across the whole grid is worth -5.8..-6.5 us (three
                // distributions), two extra tiles cost +5.5..+6.1 us.  Count
                // both grids on the real boundaries (no prefix writes, no
                // extra FillPrefix pass: three of those cost 2.4 us here) and
                // keep the wide task only when the spill is that single tile.
                uint32_t t80 = 0, t128 = 0;
                int64_t prev2 = 0;
#pragma unroll 8
                for (uint32_t e = 0; e < cfg.e; ++e) {
                    uint32_t r = static_cast<uint32_t>(ends[e] - prev2);
                    prev2 = ends[e];
                    t80 += (r + 79U) / 80U;
                    t128 += (r + 127U) >> 7;
                }
                if (t80 > t128 + 1U) return;
                bLim = 80U;
            }
            if ((L0C_INT32_ELEMS / bLim) / 32U * 32U >= 384U) {
                // The extra sector per task is only paid back when the task
                // grid still fills the machine: with few experts the wide task
                // leaves cores empty and each of the remaining ones carries
                // more of the row.  Measured on the catalog shape (96 tasks,
                // 4.8 per core); shapes with fewer than four tasks per core
                // keep the 256-wide grid.
                uint32_t saved = nTiles;
                nTiles = (cfg.n + 383U) / 384U;
                uint32_t acc3 = FillPrefix(cfg, bLim, rowsMax);
                if (acc3 >= 4U * cfg.cores) {
                    cfg.bm = bLim;
                    cfg.nbc = 384U;
                } else {
                    nTiles = saved;
                    FillPrefix(cfg, cfg.bm, rowsMax);
                }
            }
        }
    }

    // Fills prefix[] for one row tile and returns the total tile count.  The
    // running maximum of the per-expert row count costs one compare and one
    // select per expert on a loop that is already there; it is what tells
    // BuildPrefix whether halving the tile can split any expert at all.
    __aicore__ inline uint32_t FillPrefix(const Config& cfg, uint32_t bm, uint32_t& rowsMax)
    {
        int64_t prev = 0;
        uint32_t acc = 0;
        rowsMax = 0;
        // cfg.bm is 128 or one of 16/32/64, i.e. always a power of two, so the
        // per-expert ceil division is a shift.  E=128 runtime integer divisions
        // measured ~2.6 us of prologue on every core (AIC and both AIVs), which
        // delays the first Cube tile of the whole kernel.
        if ((bm & (bm - 1U)) == 0) {
            uint32_t sh = static_cast<uint32_t>(__builtin_ctz(bm));
#pragma unroll 8
            for (uint32_t e = 0; e < cfg.e; ++e) {
                uint32_t rows = static_cast<uint32_t>(ends[e] - prev);
                rowsMax = rows > rowsMax ? rows : rowsMax;
                acc += ((rows + bm - 1U) >> sh) * nTiles;
                prefix[e + 1] = acc;
                prev = ends[e];
            }
            return acc;
        }
        for (uint32_t e = 0; e < cfg.e; ++e) {
            uint32_t rows = static_cast<uint32_t>(ends[e] - prev);
            rowsMax = rows > rowsMax ? rows : rowsMax;
            acc += RowTiles(ends[e] - prev, bm);
            prefix[e + 1] = acc;
            prev = ends[e];
        }
        return acc;
    }

    // Vector path: one bulk copy replaces E scalar global loads per AIV.  Every
    // core reads the same boundaries, so the global memory system - not the loop
    // - is the limit; halving the request count is what pays.
    // The AIV needs the whole header, so it arrives by one DMA together with the
    // group_list and shares a single MTE2->Scalar wait.  One scalar read of `e`
    // sizes the group_list copy; seven serialized scalar loads cost ~1.1 us on
    // this path, which is a seventh of the whole tiny kernel.
    template <bool LAZY>
    __aicore__ inline void InitUb(GM_ADDR groups, Config& cfg, TPipe& pipe, GM_ADDR configAddr)
    {
        auto hsrc = reinterpret_cast<__gm__ const uint32_t*>(configAddr);
        uint32_t e = hsrc[1];
        TBuf<TPosition::VECCALC> hdrBuf;
        pipe.InitBuffer(hdrBuf, 64);
        pipe.InitBuffer(endsBuf, 1024);
        auto hdr = hdrBuf.Get<uint32_t>();
        GlobalTensor<uint32_t> ht;
        ht.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(configAddr));
        DataCopyExtParams hp{1, static_cast<uint32_t>(CFG_HEADER_WORDS * sizeof(uint32_t)), 0, 0, 0};
        DataCopyPad(hdr, ht, hp, DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
        auto ub = endsBuf.Get<int32_t>();
        GlobalTensor<int32_t> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groups));
        DataCopyExtParams gp{1, static_cast<uint32_t>(e * 2 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(ub, src, gp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        event_t mte2ToScalar =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
        WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
        auto words = reinterpret_cast<uint32_t*>(&cfg);
        for (uint32_t i = 0; i < CFG_HEADER_WORDS; ++i) words[i] = hdr.GetValue(i);
        if constexpr (LAZY) {
            StartLazy(cfg);
            endUb = ub;
            return;
        }
        // An entry is a cumulative row count, so M <= 65536 means the low 32-bit
        // word of each little-endian int64 already holds the value.
#pragma unroll 8
        for (uint32_t g = 0; g < cfg.e; ++g) {
            ends[g] = static_cast<int64_t>(ub.GetValue(2 * g));
        }
        BuildPrefix(cfg);
    }

    // The group boundaries are independent loads, so sixteen are issued before
    // the first is consumed.  Reading one boundary per iteration costs ~0.12 us
    // per expert (15.6 us for E=128); at eight in flight it is 3.2 us, which
    // sits at the head of the whole kernel's critical path because no Cube tile
    // can start until the boundaries are known.  The entries are little-endian
    // int64 and M fits in the low word, so each request is 4 bytes: the same
    // address stream with half the register pressure, which is what lets the
    // wider window fit.
    template <bool LAZY>
    __aicore__ inline void InitScalar(GM_ADDR groups, Config& cfg)
    {
        if constexpr (LAZY) {
            StartLazy(cfg);
            endGm = reinterpret_cast<__gm__ const int32_t*>(groups);
            return;
        }
        auto p = reinterpret_cast<__gm__ const int32_t*>(groups);
        uint32_t e = 0;
        // The boundary array is 1 KB, i.e. eight 128B lines, and each 16-entry
        // batch below covers exactly one line: read serially that is eight cold
        // fetches, which measured 3.7 us of the point-4 prologue.  Touch one
        // entry per line first so the eight fetches overlap; the loop that
        // follows then hits L2.
        if (cfg.e >= 128U) {
            uint32_t t0 = static_cast<uint32_t>(p[0]);
            uint32_t t1 = static_cast<uint32_t>(p[32]);
            uint32_t t2 = static_cast<uint32_t>(p[64]);
            uint32_t t3 = static_cast<uint32_t>(p[96]);
            uint32_t t4 = static_cast<uint32_t>(p[128]);
            uint32_t t5 = static_cast<uint32_t>(p[160]);
            uint32_t t6 = static_cast<uint32_t>(p[192]);
            uint32_t t7 = static_cast<uint32_t>(p[224]);
            ends[0] = t0;  ends[16] = t1;  ends[32] = t2;  ends[48] = t3;
            ends[64] = t4; ends[80] = t5;  ends[96] = t6;  ends[112] = t7;
        }
        for (; e + 16 <= cfg.e; e += 16) {
            uint32_t v0 = static_cast<uint32_t>(p[2 * e]);
            uint32_t v1 = static_cast<uint32_t>(p[2 * (e + 1)]);
            uint32_t v2 = static_cast<uint32_t>(p[2 * (e + 2)]);
            uint32_t v3 = static_cast<uint32_t>(p[2 * (e + 3)]);
            uint32_t v4 = static_cast<uint32_t>(p[2 * (e + 4)]);
            uint32_t v5 = static_cast<uint32_t>(p[2 * (e + 5)]);
            uint32_t v6 = static_cast<uint32_t>(p[2 * (e + 6)]);
            uint32_t v7 = static_cast<uint32_t>(p[2 * (e + 7)]);
            uint32_t v8 = static_cast<uint32_t>(p[2 * (e + 8)]);
            uint32_t v9 = static_cast<uint32_t>(p[2 * (e + 9)]);
            uint32_t v10 = static_cast<uint32_t>(p[2 * (e + 10)]);
            uint32_t v11 = static_cast<uint32_t>(p[2 * (e + 11)]);
            uint32_t v12 = static_cast<uint32_t>(p[2 * (e + 12)]);
            uint32_t v13 = static_cast<uint32_t>(p[2 * (e + 13)]);
            uint32_t v14 = static_cast<uint32_t>(p[2 * (e + 14)]);
            uint32_t v15 = static_cast<uint32_t>(p[2 * (e + 15)]);
            ends[e] = v0;  ends[e + 1] = v1;   ends[e + 2] = v2;   ends[e + 3] = v3;
            ends[e + 4] = v4;  ends[e + 5] = v5;   ends[e + 6] = v6;   ends[e + 7] = v7;
            ends[e + 8] = v8;  ends[e + 9] = v9;   ends[e + 10] = v10; ends[e + 11] = v11;
            ends[e + 12] = v12; ends[e + 13] = v13; ends[e + 14] = v14; ends[e + 15] = v15;
        }
        for (; e < cfg.e; ++e) ends[e] = static_cast<int64_t>(p[2 * e]);
        BuildPrefix(cfg);
    }

    __aicore__ inline uint32_t Rows(uint32_t e) const
    {
        int64_t start = e == 0 ? 0 : ends[e - 1];
        return static_cast<uint32_t>(ends[e] - start);
    }

    __aicore__ inline uint32_t RowTiles(int64_t rows, uint32_t bm) const
    {
        return ((static_cast<uint32_t>(rows) + bm - 1) / bm) * nTiles;
    }

    template <bool LAZY>
    __aicore__ inline void Decode(uint32_t q, uint32_t& e, uint32_t& row,
                                  uint32_t& col, uint32_t& rows, uint32_t& cols,
                                  const Config& cfg) const
    {
        if constexpr (LAZY) {
            // Advance has already placed the frontier on the expert covering q,
            // and the host only selects this path for nTiles == 1, so a task is
            // one row tile of that expert with the whole row as its columns.
            uint32_t local = q - scBase;
            e = scE - 1U;
            row = scRow + local * cfg.bm;
            col = 0;
            rows = MinU32(cfg.bm, scRows - local * cfg.bm);
            cols = MinU32(cfg.nbc, cfg.n);
            return;
        }
        uint32_t lo = 0, hi = cfg.e;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (prefix[mid + 1] <= q) lo = mid + 1;
            else hi = mid;
        }
        e = lo;
        uint32_t local = q - prefix[e];
        uint32_t start = e == 0 ? 0 : static_cast<uint32_t>(ends[e - 1]);
        // nTiles == 1 on every task whose width is the whole row, which is the
        // common case; skipping the division there keeps two 32-bit divisions
        // out of the per-task decode.
        if (nTiles == 1) {
            row = start + local * cfg.bm;
            col = 0;
        } else {
            row = start + (local / nTiles) * cfg.bm;
            col = (local % nTiles) * cfg.nbc;
        }
        // A padded task computes past the real width.  Every expert but the
        // last reads its surplus columns out of the next expert's rows (in
        // bounds); the last expert would run off the end of the tensor, so its
        // last column task is slid back to end exactly at nReal.  The columns it
        // now shares with the previous task hold the same values either way.
        if (cfg.bStride != cfg.n && cfg.nbc <= cfg.nReal && e + 1U == cfg.e &&
            col + cfg.nbc > cfg.nReal) {
            col = cfg.nReal - cfg.nbc;
        }
        rows = MinU32(cfg.bm, static_cast<uint32_t>(ends[e]) - row);
        cols = MinU32(cfg.nbc, cfg.n - col);
    }
};

// R1: task sequence with the unique multi-tile expert's tasks spread over all
// cores and the single-fractal experts' tasks filling the remainder.  Both the
// AIC and the AIVs build the same sequence from the same group_list, so the
// ready/done handshake stays keyed by the local execution index.
constexpr uint32_t CB16 = 16;              // fractal rows/cols
constexpr uint32_t C0_B8 = 32;             // int8 C0 (K direction)
constexpr uint32_t CUBE_L1_BYTES = 512 * 1024 - 4096;
constexpr uint32_t CUBE_L0A_BYTES = 64 * 1024;
constexpr uint32_t CUBE_L0B_BYTES = 64 * 1024;
constexpr uint32_t CUBE_L0C_BYTES = 128 * 1024;

__aicore__ inline uint32_t AlignUp16(uint32_t v) { return (v + 15U) & ~15U; }
__aicore__ inline uint32_t AlignUp32(uint32_t v) { return (v + 31U) & ~31U; }

struct CubeCtx {
    LocalTensor<int8_t> a1;
    LocalTensor<int8_t> b1;
    LocalTensor<int8_t> a2;
    LocalTensor<int8_t> b2;
    LocalTensor<int32_t> c1;
    uint32_t baseK;
    event_t mte2mte1;
    event_t mte1m;
    event_t mte1mte2;
    event_t mmte1;
    event_t fixm;
};

// A: `rows` x `ck` ND tile -> L1 NZ staging, one 16-row fractal block at a time
// so the tail keeps only the valid rows inside the source tensor.
__aicore__ inline void CubeStageA(const LocalTensor<int8_t>& a1, const GlobalTensor<int8_t>& src,
                                  uint32_t rows, uint32_t ck, uint32_t rowStride)
{
    uint32_t full = rows / CB16 * CB16;
    if (full != 0) {
        Nd2NzParams p;
        p.ndNum = static_cast<uint16_t>(full / CB16);
        p.nValue = CB16;
        p.dValue = static_cast<uint16_t>(ck);
        p.srcNdMatrixStride = static_cast<uint16_t>(CB16 * rowStride);
        p.srcDValue = static_cast<uint16_t>(rowStride);
        p.dstNzC0Stride = CB16;   // one 16x32 int8 fractal
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = static_cast<uint16_t>(CB16 * AlignUp32(ck));
        DataCopy(a1, src, p);
    }
    if (full != rows) {
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = static_cast<uint16_t>(rows - full);
        p.dValue = static_cast<uint16_t>(ck);
        p.srcNdMatrixStride = 0;
        p.srcDValue = static_cast<uint16_t>(rowStride);
        p.dstNzC0Stride = CB16;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        DataCopy(a1[static_cast<uint64_t>(full) * AlignUp32(ck)], src[full * rowStride], p);
    }
}

// B: `ck` x `cols` ND tile -> L1 NZ staging [n1][k1][16][32].
__aicore__ inline void CubeStageB(const LocalTensor<int8_t>& b1, const GlobalTensor<int8_t>& src,
                                  uint32_t ck, uint32_t cols, uint32_t rowStride)
{
    // A task narrower than one C0 block leaves the rest of the B fractal
    // untouched, and an AIC exception was measured on exactly that; zeroing the
    // tile first is what makes it legal.  Wider tasks cover their own staging,
    // so they must not pay for the fill.
    if ((ck % C0_B8) != 0 || cols < C0_B8) {
        // A task narrower than C0 reads ceil(cols/32) = 1 n-group per k-block, so
        // the fill has to cover C0 columns at least; AlignUp16(cols) is 16 for a
        // task narrower than 32 and left the rest of the last fractal unwritten
        // (an AIC exception was measured on exactly that).
        uint32_t zbytes = AlignUp32(AlignUp32(ck) * AlignUp32(cols < C0_B8 ? C0_B8 : cols));
        InitConstValueParams<uint32_t> zp{
            1, static_cast<uint16_t>(zbytes / 512U), 0, 0};
        Fill(b1.template ReinterpretCast<uint32_t>(), zp);
        // The fill and the Nd2Nz below are both MTE2 traffic into the same L1
        // slot, so an MTE2 queue barrier orders them; PIPE_ALL also drained the
        // MMA and fixpipe, which cost 1.8 us per task measured on the 2559/32/
        // 1027/773 shape (K tail chunk: ck = 3).
        PipeBarrier<PIPE_MTE2>();
    }
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = static_cast<uint16_t>(ck);
    p.dValue = static_cast<uint16_t>(cols);
    p.srcNdMatrixStride = 0;
    p.srcDValue = static_cast<uint16_t>(rowStride);
    p.dstNzC0Stride = static_cast<uint16_t>(AlignUp32(ck));
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;
    DataCopy(b1, src, p);
}

// L1 -> L0A: the [m1][k1] fractal order is already L0A's, so each 16-row block
// is a run of `ck/32` consecutive fractals.
__aicore__ inline void CubeTensorA(const LocalTensor<int8_t>& a2, const LocalTensor<int8_t>& a1,
                                   uint32_t rows, uint32_t ck)
{
    LoadData2dParams p;
    p.startIndex = 0;
    p.repeatTimes = static_cast<uint8_t>(AlignUp32(ck) / C0_B8);
    p.srcStride = 1;
    p.sid = 0;
    p.dstGap = 0;
    p.ifTranspose = false;
    p.addrMode = 0;
    uint32_t per = (AlignUp32(ck) / C0_B8) * 512;
    uint32_t m1 = (rows + CB16 - 1) / CB16;
    for (uint32_t i = 0; i < m1; ++i) LoadData(a2[i * per], a1[i * per], p);
}

// Same, but the L1 tile holds the whole K range (`kFull`), so a K sub-block
// starts `k0/32` fractals into each 16-row block instead of at its beginning.
__aicore__ inline void CubeTensorAAt(const LocalTensor<int8_t>& a2, const LocalTensor<int8_t>& a1,
                                     uint32_t rows, uint32_t ck, uint32_t k0, uint32_t kFull)
{
    LoadData2dParams p;
    p.startIndex = 0;
    p.repeatTimes = static_cast<uint8_t>(AlignUp32(ck) / C0_B8);
    p.srcStride = 1;
    p.sid = 0;
    p.dstGap = 0;
    p.ifTranspose = false;
    p.addrMode = 0;
    uint32_t per = (AlignUp32(ck) / C0_B8) * 512;
    uint32_t perFull = (AlignUp32(kFull) / C0_B8) * 512;
    uint32_t off = (k0 / C0_B8) * 512;
    uint32_t m1 = (rows + CB16 - 1) / CB16;
    for (uint32_t i = 0; i < m1; ++i) LoadData(a2[i * per], a1[i * perFull + off], p);
}

// L1 -> L0B: one k-block (32 k values) of every n group at a time, transposed.
__aicore__ inline void CubeTensorB(const LocalTensor<int8_t>& b2, const LocalTensor<int8_t>& b1,
                                   uint32_t cols, uint32_t ck)
{
    uint32_t kc0 = AlignUp32(ck) / C0_B8;
    uint32_t nFra = (cols + C0_B8 - 1) / C0_B8;
    uint32_t nAlign = AlignUp16(cols);
    LoadData2dTransposeParams p;
    p.startIndex = 0;
    p.repeatTimes = static_cast<uint8_t>(nFra);
    p.srcStride = static_cast<uint16_t>(kc0);
    p.dstGap = 1;
    p.dstFracGap = 0;
    p.addrMode = 0;
    for (uint32_t i = 0; i < kc0; ++i) {
        LoadDataWithTranspose(b2[i * nAlign * C0_B8], b1[i * C0_B8 * C0_B8], p);
    }
}

struct CubeBlock {
    uint64_t aOff, bOff, cOff;
    uint32_t rows, cols, ck;
    uint32_t rcols;   // columns actually staged from B (<= cols)
    uint32_t k0;
};

template <bool LAZY>
__aicore__ inline void CubeMakeBlock(CubeBlock& blk, const Tasks& tasks, const Config& cfg,
                                     uint32_t q, uint32_t k0, uint32_t baseK, uint32_t cStride,
                                     uint32_t bStride, uint32_t kCube)
{
    uint32_t e, row, col, rows, cols;
    tasks.template Decode<LAZY>(q, e, row, col, rows, cols, cfg);
    blk.aOff = static_cast<uint64_t>(row) * cfg.k + k0;
    blk.bOff = static_cast<uint64_t>(e) * kCube * bStride + col + static_cast<uint64_t>(k0) * bStride;
    blk.cOff = static_cast<uint64_t>(row) * cStride + col;
    blk.rows = rows;
    blk.cols = cols;
    blk.rcols = StagedCols(cfg, col, cols);
    blk.ck = MinU32(baseK, kCube - k0);
    blk.k0 = k0;
}

__aicore__ inline void CubeFixpipe(const CubeCtx& t, const GlobalTensor<int32_t>& cGm, uint64_t cOff,
                                   uint32_t rows, uint32_t cols, uint32_t cRowStride,
                                   uint64_t cRegOff = 0)
{
    DataCopyCO12DstParams fp;
    fp.nSize = static_cast<uint16_t>(cols);
    fp.mSize = static_cast<uint16_t>(rows);
    fp.quantPre = QuantMode_t::NoQuant;
    fp.dstStride = cRowStride;
    fp.srcStride = static_cast<uint16_t>(AlignUp16(rows));
    fp.nz2ndEn = true;
    fp.unitFlag = 3;   // couples the fixpipe to the final MMA, as the API does
    SetFixpipeNz2ndFlag(1, 1, 1);
    DataCopy(cGm[cOff], t.c1[cRegOff], fp);
}

// One Cube task stream, software pipelined: the Nd2Nz of the next K block is
// issued before the current block's MMA, so the MTE2 staging of the next task
// overlaps the fixpipe of the current one.  Two L1 staging slots make that
// legal; L0A/L0B/L0C stay single-buffered, which costs only the (short) MMA
// time because both sides of that wait are microseconds apart.
template <bool LAZY>
__aicore__ inline void RunCubeManual(GM_ADDR x, GM_ADDR weight, GM_ADDR workspace,
                                     Config& cfg, Tasks& tasks, TPipe& pipe,
                                     uint32_t kpack = 0)
{
    const uint32_t bStride = kpack != 0 ? kpack : cfg.bStride;
    // K as the Cube sees it.  With the repack, B carries `AlignUp32(k)` rows per
    // expert (zero padding), so the Cube can run whole K blocks; A keeps its
    // real row stride, and the bytes it reads past its own K are masked by
    // those zero rows.
    const uint32_t kCube = kpack != 0 ? AlignUp32(cfg.k) : cfg.k;
    GlobalTensor<int8_t> a, b;
    GlobalTensor<int32_t> c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(weight));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));

    uint32_t rowsMax = cfg.bm;
    // Two L0 buffers per operand: baseK is halved so both fit, which lets the MMA of
    // K block i run while block i+1 is loaded into the other buffer.
    uint32_t baseK = MinU32(MinU32(CUBE_L0B_BYTES / (2U * cfg.nbc), CUBE_L0A_BYTES / (2U * rowsMax)),
                            CUBE_L1_BYTES / (2U * (rowsMax + cfg.nbc)));
    baseK = baseK / C0_B8 * C0_B8;
    if (baseK > kCube) baseK = kCube;
    if (baseK < C0_B8) baseK = C0_B8;

    uint32_t aBytes = AlignUp32(rowsMax * AlignUp32(baseK));
    uint32_t bBytes = AlignUp32(AlignUp32(baseK) * AlignUp32(cfg.nbc));

    // Column tasks of one row tile are consecutive in q and share their A tile.
    // Handing each core a contiguous run (instead of every cores-th task) lets
    // the A tile stay in L1 for the whole run; with the round-robin order the
    // same core never sees the same row tile twice.  Shapes with a single column
    // task keep the round-robin order, which spreads the ragged last row tiles.
    uint32_t total = tasks.prefix[cfg.e];
    uint32_t qStart = GetBlockIdx(), qEnd = total, qStep = cfg.cores;
    bool aContig = tasks.nTiles > 1U;
    bool aRes = aContig;
    if (aContig) {
        uint32_t chunk = (total + cfg.cores - 1U) / cfg.cores;
        qStart = GetBlockIdx() * chunk;
        qEnd = MinU32(total, qStart + chunk);
        qStep = 1;
    }
    // The resident A region holds a whole row tile (rows x K); the two pipeline
    // slots then carry only B.  It must still fit beside them in L1.
    uint32_t aFullBytes = 0;
    if (aRes) {
        aFullBytes = AlignUp32(rowsMax * kCube);
        if (2U * bBytes + aFullBytes > CUBE_L1_BYTES) { aRes = false; aFullBytes = 0; }
    }
    uint32_t slotBytes = aRes ? bBytes : (aBytes + bBytes);

    CubeCtx t;
    TBuf<TPosition::A1> l1;
    pipe.InitBuffer(l1, 2 * slotBytes + aFullBytes);
    auto l1t = l1.Get<int8_t>();
    TBuf<TPosition::A2> l0a;
    pipe.InitBuffer(l0a, CUBE_L0A_BYTES);
    t.a2 = l0a.Get<int8_t>();
    TBuf<TPosition::B2> l0b;
    pipe.InitBuffer(l0b, CUBE_L0B_BYTES);
    t.b2 = l0b.Get<int8_t>();
    TBuf<TPosition::CO1> l0c;
    pipe.InitBuffer(l0c, CUBE_L0C_BYTES);
    t.c1 = l0c.Get<int32_t>();

    uint32_t cStride = CStride(cfg.n, cfg.k);
    if (aContig ? (qStart >= qEnd) : !tasks.template TaskOk<LAZY>(qStart, cfg, total)) return;

    // AllocEventID, not FetchEventID: the latter only peeks at the first free id
    // and hands the same one out again, which silently aliases the two slots.
    event_t m2m1[2] = {static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>()),
                       static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>())};
    event_t l1free[2] = {static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>()),
                         static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>())};
    // One event id per L0 buffer pair, so the MMA and the next L0 load overlap.
    event_t mte1m[2] = {static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>()),
                        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>())};
    event_t mmte1[2] = {static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>()),
                        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>())};
    event_t fixm = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::FIX_M>());
    event_t aready = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
    event_t afree = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
    LocalTensor<int8_t> aFull = l1t[static_cast<uint64_t>(2) * slotBytes];
    uint32_t aKey = 0xffffffffU;   // row tile currently staged in the A region
    bool aFresh = false;           // A region staged but its flag not yet waited

    CubeBlock cur, nxt;
    uint32_t q = qStart, k0 = 0, iter = 0, bidx = 0;
    uint32_t pend[2] = {0, 0};   // L1-slot "free" events set but not yet waited
    CubeMakeBlock<LAZY>(cur, tasks, cfg, q, k0, baseK, cStride, bStride, kCube);
    // Stage the first block into slot 0.
    if (aRes) {
        CubeStageA(aFull, a[cur.aOff], cur.rows, cfg.k, cfg.k);
        SetFlag<HardEvent::MTE2_MTE1>(aready);
        aKey = cur.aOff;
        aFresh = true;
        CubeStageB(l1t, b[cur.bOff], cur.ck, cur.rcols, bStride);
    } else {
        CubeStageA(l1t, a[cur.aOff], cur.rows, cur.ck, cfg.k);
        CubeStageB(l1t[aBytes], b[cur.bOff], cur.ck, cur.rcols, bStride);
    }
    SetFlag<HardEvent::MTE2_MTE1>(m2m1[0]);
    SetFlag<HardEvent::M_MTE1>(mmte1[0]);
    SetFlag<HardEvent::M_MTE1>(mmte1[1]);
    SetFlag<HardEvent::FIX_M>(fixm);

    for (;;) {
        uint32_t s = bidx & 1;
        const LocalTensor<int8_t> a1 = l1t[static_cast<uint64_t>(s) * slotBytes];
        const LocalTensor<int8_t> b1 =
            l1t[static_cast<uint64_t>(s) * slotBytes + (aRes ? 0U : aBytes)];
        bool first = cur.k0 == 0;
        bool last = (cur.k0 + cur.ck == kCube);

        WaitFlag<HardEvent::MTE2_MTE1>(m2m1[s]);
        if (aRes && aFresh) {
            WaitFlag<HardEvent::MTE2_MTE1>(aready);
            aFresh = false;
        }
        const uint32_t s2 = bidx & 1;   // L0 buffer pair index
        WaitFlag<HardEvent::M_MTE1>(mmte1[s2]);
        auto a2 = t.a2[static_cast<uint64_t>(s2) * aBytes];
        auto b2 = t.b2[static_cast<uint64_t>(s2) * bBytes];
        if (aRes) {
            CubeTensorAAt(a2, aFull, cur.rows, cur.ck, cur.k0, kCube);
        } else {
            CubeTensorA(a2, a1, cur.rows, cur.ck);
        }
        CubeTensorB(b2, b1, cur.cols, cur.ck);
        SetFlag<HardEvent::MTE1_M>(mte1m[s2]);
        SetFlag<HardEvent::MTE1_MTE2>(l1free[s]);
        ++pend[s];
        if (first) WaitFlag<HardEvent::FIX_M>(fixm);
        WaitFlag<HardEvent::MTE1_M>(mte1m[s2]);

        MmadParams mp;
        // The MMA has no single-row mode here: madM == 1 has to be widened to one
        // fractal (matmul's GetFixedMadM does the same); the fixpipe still writes
        // the single real row.
        mp.m = static_cast<uint16_t>(cur.rows == 1 ? CB16 : cur.rows);
        mp.n = static_cast<uint16_t>(cur.cols);
        mp.k = static_cast<uint16_t>(AlignUp32(cur.ck));
        mp.cmatrixInitVal = first;
        mp.cmatrixSource = false;
        mp.unitFlag = last ? 3 : 2;   // 3 = final accumulation
        Mmad(t.c1, a2, b2, mp);
        SetFlag<HardEvent::M_MTE1>(mmte1[s2]);
        if (last) {
            CubeFixpipe(t, c, cur.cOff, cur.rows, cur.cols, cStride);
            SetFlag<HardEvent::FIX_M>(fixm);
            CrossCoreSetFlag<2, PIPE_FIX>(static_cast<uint16_t>((FLAG_READY + iter) % 11));
            ++iter;
        }

        // Advance to the next K block, then stage it into the other slot so the
        // copy runs while the fixpipe above drains.
        k0 += cur.ck;
        if (k0 >= kCube) {
            k0 = 0;
            q += qStep;
            if (aContig ? (q >= qEnd) : !tasks.template TaskOk<LAZY>(q, cfg, total)) break;
            if (iter >= READY_RING) CrossCoreWaitFlag(FLAG_DONE);
        }
        CubeMakeBlock<LAZY>(nxt, tasks, cfg, q, k0, baseK, cStride, bStride, kCube);
        uint32_t ns = (bidx + 1) & 1;
        if (pend[ns] != 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1free[ns]);
            --pend[ns];
        }
        if (aRes) {
            // aOff - k0 is the row-tile base, i.e. the identity of the A tile.
            uint32_t rowKey = nxt.aOff - nxt.k0;
            if (rowKey != aKey) {
                // One set/wait pair, not a count: an AscendC event id is a single
                // semaphore, so it must never have more than one outstanding set.
                // This drains the MTE1 readers of the old A before overwriting it.
                SetFlag<HardEvent::MTE1_MTE2>(afree);
                WaitFlag<HardEvent::MTE1_MTE2>(afree);
                CubeStageA(aFull, a[rowKey], nxt.rows, cfg.k, cfg.k);
                SetFlag<HardEvent::MTE2_MTE1>(aready);
                aKey = rowKey;
                aFresh = true;
            }
        } else {
            // The last block must not read past the tensor's K end: stage only
            // the real remainder (the fractal layout is the same, because the
            // stride is built from AlignUp32(K) either way) and let the zero
            // rows of the repacked B mask the stale fractal tail.
            uint32_t ack = MinU32(nxt.ck, cfg.k - nxt.k0);
            CubeStageA(l1t[static_cast<uint64_t>(ns) * slotBytes], a[nxt.aOff], nxt.rows, ack, cfg.k);
        }
        CubeStageB(l1t[static_cast<uint64_t>(ns) * slotBytes + (aRes ? 0U : aBytes)],
                   b[nxt.bOff], nxt.ck, nxt.rcols, bStride);
        SetFlag<HardEvent::MTE2_MTE1>(m2m1[ns]);
        cur = nxt;
        ++bidx;
    }
    // Drain the last outstanding producer events so the pipe state is balanced.
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < pend[i]; ++j) WaitFlag<HardEvent::MTE1_MTE2>(l1free[i]);
        pend[i] = 0;
    }
    WaitFlag<HardEvent::M_MTE1>(mmte1[0]);
    WaitFlag<HardEvent::M_MTE1>(mmte1[1]);
    WaitFlag<HardEvent::FIX_M>(fixm);
}

// Load one L0 reduction block (ck x cols) from a B chunk staged as exactly ck
// rows, so the L1 -> L0 transpose reads a packed tile.
template <uint32_t QD, bool PAIR>
class Epilogue {
public:
    __aicore__ inline void Init(GM_ADDR workspace, GM_ADDR scale, GM_ADDR y, TPipe& pipe,
                                const Config& cfg)
    {
        c_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));
        scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
        // Depth 2 on both queues lets the next C block load and the previous y
        // block store overlap the vector math of the current block.
        // Wide blocks on the block path: a task that is exactly 256 columns wide
        // everywhere lets a vector block cover two BN blocks of one row chunk.
        // The whole-row gate is not needed for the padded shapes that do NOT
        // repack (bStride != n): a partial-wide block already has a per-row path,
        // and those shapes have no repack buffer, so the 256-wide slot (173 KB)
        // fits.  The repacking shapes keep the old gate - they measure neutral
        // with it and their repack buffer pushes the UB budget.
        bool wide = cfg.flat == 0 && cfg.nbc == WIDE_COLS &&
                    ((cfg.nReal % WIDE_COLS) == 0 ||
                     (cfg.nReal != cfg.n && cfg.bStride == cfg.nReal));
        // One block per row chunk whenever the alternative is a *narrower* block
        // than the task: a 384-wide task, and a 256-wide one that is not `wide`
        // (its real width is not a multiple of 256, which is every repacked
        // shape whose rows y are misaligned).  A narrower block splits each row
        // of y into several bursts that each drag in the 128B sector at both
        // ends; one block per row chunk writes the whole row in one burst.
        bool wideRow = cfg.flat == 0 && (cfg.nbc == 384U || (cfg.nbc == WIDE_COLS && !wide));
        uint32_t slot = cfg.flat != 0 ? WIDE_VEC : (wideRow ? 16U * cfg.nbc : (wide ? WIDE_ELEMS : VEC_SIZE));
        // 256B of padding rotates a buffer by 8 of the 16 UB bank groups.  With
        // everything allocated back to back all 32B rows land in groups 0..7, so
        // every two-operand vector op reads both sources from the same group
        // (an arch22 read-read conflict).  One pad in front of the staged
        // inputs and one in front of the scratch block leave z alone in 0..7
        // while C, the scale, y and u sit in 8..15.
        uOfs_ = slot + 64;
        rowsPerBlock_ = cfg.flat != 0 ? (WIDE_VEC / cfg.n) : ROWS;
        rowsPerTile_ = wideRow ? 16U : ROWS;
        blockCols_ = cfg.flat != 0 ? cfg.n : (wideRow ? cfg.nbc : (wide ? WIDE_COLS : BN));
        pipe.InitBuffer(padBuf_, 256);
        pipe.InitBuffer(in_, QD, slot * sizeof(int32_t));
        pipe.InitBuffer(out_, QD, slot * sizeof(half));
        pipe.InitBuffer(scaleQ_, 2, 1024 * sizeof(float));
        pipe.InitBuffer(padBuf2_, 256);
        // The block path keeps two live FP32 scratch blocks (the GELU value and
        // the reciprocal argument); every other temporary aliases them.
        pipe.InitBuffer(tmpBuf_, 2U * (slot + 64U) * sizeof(float));
    }

    __aicore__ inline void LoadC(const LocalTensor<int32_t>& c, uint32_t row, uint32_t nr,
                                 uint32_t col, uint32_t cols, uint32_t bc, const Config& cfg)
    {
        uint32_t CS = CStride(cfg.n, cfg.k);
        if (cols < bc) {
            // Partial N tile: the padding columns must be zeroed so no garbage
            // reaches the vector math.  One 2D copy replaces the per-row loop:
            // the UB-side stride unit is 32 bytes and the hardware advances by
            // roundUp32(burst) + stride*32, so the gap is expressible whenever
            // the packed row width is 32B aligned (CwOf makes it so).  cat06's
            // tail is 129 ints of 144: a 516B burst advancing 544B.
            Duplicate(c, int32_t(0), nr * bc);
            PipeBarrier<PIPE_ALL>();
            uint32_t burst = cols * static_cast<uint32_t>(sizeof(int32_t));
            uint32_t adv = (burst + 31U) & ~31U;
            DataCopyExtParams cp{static_cast<uint16_t>(nr), burst,
                                 (CS - cols) * static_cast<uint32_t>(sizeof(int32_t)),
                                 (bc * 4U - adv) / 32U, 0};
            DataCopyPad(c, c_[static_cast<uint64_t>(row) * CS + col], cp,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            return;
        }
        DataCopyExtParams cp{static_cast<uint16_t>(nr),
                             cols * static_cast<uint32_t>(sizeof(int32_t)),
                             (CS - cols) * static_cast<uint32_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(c, c_[static_cast<uint64_t>(row) * CS + col], cp,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    }

    // Flat path: the y block lives in the out_ queue until the next block has
    // been cast, so its V_MTE3 flag is already satisfied when it is dequeued.
    __aicore__ inline void StoreFlatY(const Config& cfg, uint32_t row, uint32_t count)
    {
        auto y = out_.template DeQue<half>();
        DataCopyExtParams yp{1, count * static_cast<uint32_t>(sizeof(half)), 0, 0, 0};
        DataCopyPad(y_[static_cast<uint64_t>(row) * cfg.n], y, yp);
        out_.FreeTensor(y);
    }

    __aicore__ inline void StoreY(const LocalTensor<half>& y, uint32_t row, uint32_t nr,
                                  uint32_t col, uint32_t cols, uint32_t bc, const Config& cfg)
    {
        if (cols < bc) {
            // Same rule on the store side: the UB source advances by
            // roundUp32(burst) + stride*32.  cat06's y tail is 129 halfs of a
            // 144-wide row, i.e. a 258B burst advancing 288B, so the UB stride
            // is zero and only the GM side needs a gap.
            uint32_t burst = cols * static_cast<uint32_t>(sizeof(half));
            uint32_t adv = (burst + 31U) & ~31U;
            DataCopyExtParams yp{static_cast<uint16_t>(nr), burst,
                                 (bc * 2U - adv) / 32U,
                                 (cfg.nReal - cols) * static_cast<uint32_t>(sizeof(half)), 0};
            DataCopyPad(y_[static_cast<uint64_t>(row) * cfg.nReal + col], y, yp);
            return;
        }
        DataCopyExtParams yp{static_cast<uint16_t>(nr),
                             cols * static_cast<uint32_t>(sizeof(half)), 0,
                             (cfg.nReal - cols) * static_cast<uint32_t>(sizeof(half)), 0};
        DataCopyPad(y_[static_cast<uint64_t>(row) * cfg.nReal + col], y, yp);
    }

    // GELU in place over one block: z holds C * scale on entry and the FP32 GELU
    // value on exit, laid out as `rows` rows of `blockCols` columns.
    // u = z * (C1 + C2*z^2) with sqrt(2/pi) and its 0.044715 factor folded into
    // the constants, and the -2 of the tanh argument folded into C1N/C1C2N.
    // Stable signed form: y = z / (1 + exp(-2u)), the same real value as the
    // reference's |u|/branch form (for u < 0, z*e^{-2|u|}/(1+e^{-2|u|}) ==
    // z/(1+e^{-2u})); no sign select is needed and for z >= 8 the denominator is
    // exactly 1.0 in FP32, so no z >= 8 override is needed either.
    __aicore__ inline void Gelu(const LocalTensor<float>& z, const LocalTensor<int32_t>& c,
                                const LocalTensor<float>& scale, uint32_t count, uint32_t rows,
                                uint32_t blockCols)
    {
        auto tmp = tmpBuf_.Get<float>();
        auto u = tmp[uOfs_];
        Cast(z, c, RoundMode::CAST_NONE, count);
        // One source repeat holds 64 fp32 values, so a row of `blockCols`
        // columns takes blockCols/64 calls; the scale keeps its per-column
        // values (src1 repeat stride 0) while z advances one row.
        uint8_t rep = static_cast<uint8_t>(blockCols * static_cast<uint32_t>(sizeof(float)) / 32U);
        for (uint32_t off = 0; off < blockCols; off += 64U) {
            uint32_t m = MinU32(64U, blockCols - off);
            Mul(z[off], z[off], scale[off], static_cast<uint64_t>(m),
                static_cast<uint8_t>(rows), {1, 1, 1, rep, rep, 0});
        }
        Mul(u, z, z, count);
        Muls(u, u, C1C2N, count);
        Adds(u, u, C1N, count);
        Mul(u, z, u, count);
        Exp(u, u, count);
        Adds(u, u, 1.0f, count);
        Div(z, z, u, count);
    }

    // `cols` may span several BN blocks.  Linearising the task's (column block,
    // row chunk) pairs into one loop lets the C prefetch carry across a column
    // block boundary; with one chunk per block a per-call Tile could never
    // prefetch and every C load was fully exposed.
    // `bc` / `rowsPer` are the block geometry for this call.  The narrow-task
    // shapes (one 128-wide Cube task) can merge two adjacent tasks into one
    // 16x256 block, which writes 512B per y row instead of 256B; a task that
    // cannot be merged keeps the 32x128 geometry.
    __aicore__ inline uint32_t BlockCols() const { return blockCols_; }
    __aicore__ inline uint32_t RowsPer() const { return rowsPerTile_; }

    __aicore__ inline void Tile(uint32_t e, uint32_t row, uint32_t col,
                               uint32_t rows, uint32_t cols, const Config& cfg,
                               uint32_t bc, uint32_t rowsPer)
    {
        uint32_t rowChunks = (rows + rowsPer - 1) / rowsPer;
        uint32_t colBlocks = (cols + bc - 1) / bc;
        uint32_t nBlocks = rowChunks * colBlocks;
        if (nBlocks == 0) return;
        for (uint32_t p = 0; p + 1U < QD && p < nBlocks; ++p) {
            uint32_t pcb = p / rowChunks, prb = p % rowChunks;
            auto pre = in_.template AllocTensor<int32_t>();
            uint32_t pcol = MinU32(bc, cols - pcb * bc);
            LoadC(pre, row + prb * rowsPer,
                  MinU32(rowsPer, rows - prb * rowsPer),
                  col + pcb * bc, pcol, CwOf(pcol, bc), cfg);
            in_.EnQue(pre);
        }
        // One scale transfer covers the whole task: a per-column-block transfer
        // had to wait for the previous block's vector math (V->MTE2) every time.
        auto scaleAll = scaleQ_.template AllocTensor<float>();
        if (cols < bc * colBlocks) {
            Duplicate(scaleAll, 0.0f, bc * colBlocks);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyExtParams sp{1, cols * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(scaleAll, scale_[static_cast<uint64_t>(e) * cfg.nReal + col], sp,
                    DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        scaleQ_.EnQue(scaleAll);
        scaleAll = scaleQ_.template DeQue<float>();
        for (uint32_t cb = 0; cb < colBlocks; ++cb) {
            uint32_t ccol = col + cb * bc;
            uint32_t ccols = MinU32(bc, cols - cb * bc);
            auto scale = scaleAll[cb * bc];
        for (uint32_t rb = 0; rb < rowChunks; ++rb) {
            uint32_t r = rb * rowsPer;
            uint32_t nr = MinU32(rowsPer, rows - r);
            uint32_t cw = CwOf(ccols, bc);
            uint32_t count = nr * cw;
            auto c = in_.template DeQue<int32_t>();
            uint32_t blk = cb * rowChunks + rb;
            if (blk + (QD - 1U) < nBlocks) {
                // Keep QD-1 loads in flight so a block's C read never has to
                // complete before the previous block's math is done.
                uint32_t nblk = blk + (QD - 1U);
                uint32_t ncb = nblk / rowChunks;
                uint32_t nrb = nblk % rowChunks;
                auto next = in_.template AllocTensor<int32_t>();
                uint32_t nc = MinU32(bc, cols - ncb * bc);
                LoadC(next, row + nrb * rowsPer, MinU32(rowsPer, rows - nrb * rowsPer),
                      col + ncb * bc, nc, CwOf(nc, bc), cfg);
                in_.EnQue(next);
            }
            auto z = tmpBuf_.Get<float>();
            Gelu(z, c, scale, count, nr, cw);
            // FP32 result -> FP16.  A one-ULP deviation stays inside both bounds
            // while 2^-14 <= |y| < 2 (the FP16 spacing is a normal quantum there,
            // so ULP/|y| <= 2^-10 and ULP < 1e-3 both hold); outside that range an
            // exact ties-to-even repair would be needed, and this path does not
            // carry one.
            auto y = out_.template AllocTensor<half>();
            Cast(y, z, RoundMode::CAST_ROUND, count);
            out_.EnQue(y);
            y = out_.template DeQue<half>();
            StoreY(y, row + r, nr, ccol, ccols, cw, cfg);
            out_.FreeTensor(y);
            in_.FreeTensor(c);
        }
        }
        scaleQ_.FreeTensor(scaleAll);
    }

    // Flat epilogue for tasks that cover the whole output row.
    //
    // Every block is `rowsPerBlock_` whole rows, contiguous in C and in y, so
    // each side is one DMA burst instead of one per row, and one task is a
    // single block.  With one block per task the in-queue has a free slot, so
    // the next task's C load is issued while this task's vector math runs: the
    // AIC produces one tile ahead, so the handshake wait is already satisfied
    // and the C read stops being exposed.
    struct Job {
        uint32_t e, row, rows, nblk, per;
    };

    // `per` is the rows in one block.  The last task of a core group has no
    // successor to prefetch, so it is handed half-size blocks: the second
    // block's C load then hides behind the first block's vector math, and that
    // task is the only AIV work that lands after the AIC's last tile.
    template <bool LAZY>
    __aicore__ inline void MakeJob(Job& j, Tasks& tasks, const Config& cfg, uint32_t q,
                                   uint32_t sub, bool last)
    {
        uint32_t e, row, col, rows, cols;
        uint32_t per = last ? (rowsPerBlock_ > 1 ? rowsPerBlock_ / 2 : 1) : rowsPerBlock_;
        tasks.template Decode<LAZY>(q, e, row, col, rows, cols, cfg);
        // Same split as the block path: half the rows each, and a sub that has no
        // row to take stays idle.  The column split cannot happen here because
        // the host only selects this path for n <= 256, where a one-row task is
        // already below the 512-element single-AIV threshold.
        uint32_t r0 = 0, rn = rows;
        if (rows * cols < 512) {
            if (sub != 0) rn = 0;
        } else {
            uint32_t halfRows = (rows + 1) / 2;
            r0 = sub * halfRows;
            rn = r0 < rows ? MinU32(halfRows, rows - r0) : 0;
        }
        j.e = e;
        j.row = row + r0;
        j.rows = rn;
        j.per = per;
        j.nblk = (rn + per - 1) / per;
    }

    // The "last task of this core" flag halves the block size so the tail tile
    // still has something to prefetch against.  It is decided *after* the job is
    // decoded: in the lazy build the look-ahead moves the frontier, and a decode
    // issued after it would land on the wrong expert.
    __aicore__ inline void MarkLast(Job& j) const
    {
        j.per = rowsPerBlock_ > 1 ? rowsPerBlock_ / 2 : 1;
        j.nblk = (j.rows + j.per - 1) / j.per;
    }

    __aicore__ inline uint32_t BlockRows(const Job& j, uint32_t b) const
    {
        return MinU32(j.per, j.rows - b * j.per);
    }

    __aicore__ inline void LoadFlat(const Job& j, uint32_t b, const Config& cfg)
    {
        uint32_t count = BlockRows(j, b) * cfg.n;
        auto c = in_.template AllocTensor<int32_t>();
        // Same 512B pieces the block path reads: one long burst over the whole
        // task measures 1.5x slower than many 512B bursts on this chip
        // (cat04 aiv_mte2 10.1 us with 512B bursts, 15.2 us as one burst).
        // A short last block (n=64 with an odd row count) does not divide into
        // 512B pieces, so it goes as one burst of its own length.
        uint32_t bytes = count * static_cast<uint32_t>(sizeof(int32_t));
        uint32_t burst = (bytes % 512U == 0) ? 512U : bytes;
        DataCopyExtParams cp{static_cast<uint16_t>(bytes / burst), burst, 0, 0, 0};
        DataCopyPad(c, c_[static_cast<uint64_t>(j.row + b * j.per) *
                              CStride(cfg.n, cfg.k)], cp,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        in_.EnQue(c);
    }

    __aicore__ inline void LoadScale(Job& j, const Config& cfg)
    {
        auto s = scaleQ_.template AllocTensor<float>();
        DataCopyExtParams sp{1, cfg.n * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(s, scale_[static_cast<uint64_t>(j.e) * cfg.n], sp,
                    DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        scaleQ_.EnQue(s);
    }

    template <bool LAZY>
    __aicore__ inline void FlatRun(Tasks& tasks, const Config& cfg, uint32_t core,
                                   uint32_t sub, uint32_t total)
    {
        uint32_t q = core;
        if (!tasks.template TaskOk<LAZY>(q, cfg, total)) return;
        uint32_t iter = 0;
        Job cur, nxt;
        // The y store is issued one block late: the V_MTE3 flag is then already
        // set, so the scalar never waits on its own Cast and the MTE3 copy runs
        // against the next block's vector math.  Same idiom the C prefetch uses.
        bool pendY = false;
        uint32_t pendRow = 0, pendCount = 0;
        CrossCoreWaitFlag(static_cast<uint16_t>((FLAG_READY + iter) % 11));
        MakeJob<LAZY>(cur, tasks, cfg, q, sub, false);
        if (!tasks.template TaskOk<LAZY>(q + cfg.cores, cfg, total)) MarkLast(cur);
        LoadScale(cur, cfg);
        if (cur.nblk != 0) LoadFlat(cur, 0, cfg);
        for (;;) {
            auto scale = scaleQ_.template DeQue<float>();
            for (uint32_t b = 0; b < cur.nblk; ++b) {
                uint32_t nr = BlockRows(cur, b);
                uint32_t count = nr * cfg.n;
                auto c = in_.template DeQue<int32_t>();
                if (b + 1 < cur.nblk) LoadFlat(cur, b + 1, cfg);
                if (b + 1 == cur.nblk) {
                    // The tile's C has landed in UB, so the workspace slot may be
                    // reused.  Raised here, before the next task's prefetch is
                    // issued: a PIPE_MTE2 flag drains what was issued before it,
                    // and waiting for the prefetch too stalled the scalar for
                    // most of the next tile's read time.
                    CrossCoreSetFlag<2, PIPE_MTE2>(FLAG_DONE);
                }
                auto z = tmpBuf_.Get<float>();
                Gelu(z, c, scale, count, nr, cfg.n);
                if (b + 1 == cur.nblk && cur.nblk <= 1 &&
                    tasks.template TaskOk<LAZY>(q + cfg.cores, cfg, total)) {
                    // Prefetch the next task only after this block's vector ops
                    // are issued: the handshake wait then overlaps the math, and
                    // the first task is not delayed by a whole AIC tile.
                    MakeJob<LAZY>(nxt, tasks, cfg, q + cfg.cores, sub, false);
                    if (!tasks.template TaskOk<LAZY>(q + 2U * cfg.cores, cfg, total)) MarkLast(nxt);
                    CrossCoreWaitFlag(static_cast<uint16_t>((FLAG_READY + iter + 1) % 11));
                    LoadScale(nxt, cfg);
                    if (nxt.nblk != 0) LoadFlat(nxt, 0, cfg);
                }
                auto y = out_.template AllocTensor<half>();
                Cast(y, z, RoundMode::CAST_ROUND, count);
                out_.EnQue(y);
                in_.FreeTensor(c);
                if (pendY) StoreFlatY(cfg, pendRow, pendCount);
                pendRow = cur.row + b * cur.per;
                pendCount = count;
                pendY = true;
            }
            scaleQ_.FreeTensor(scale);
            // A sub with no rows takes no block, so nothing raised DONE above.
            if (cur.nblk == 0) CrossCoreSetFlag<2, PIPE_MTE2>(FLAG_DONE);
            ++iter;
            q += cfg.cores;
            if (!tasks.template TaskOk<LAZY>(q, cfg, total)) {
                if (pendY) StoreFlatY(cfg, pendRow, pendCount);
                pendY = false;
                break;
            }
            if (cur.nblk > 1 || cur.nblk == 0) {
                // Multi-block and empty tasks did not prefetch: start the next one.
                MakeJob<LAZY>(nxt, tasks, cfg, q, sub, false);
                if (!tasks.template TaskOk<LAZY>(q + cfg.cores, cfg, total)) MarkLast(nxt);
                CrossCoreWaitFlag(static_cast<uint16_t>((FLAG_READY + iter) % 11));
                LoadScale(nxt, cfg);
                if (nxt.nblk != 0) LoadFlat(nxt, 0, cfg);
            }
            if (pendY) StoreFlatY(cfg, pendRow, pendCount);
            pendY = false;
            cur = nxt;
        }
    }

private:
    GlobalTensor<int32_t> c_;
    GlobalTensor<float> scale_;
    GlobalTensor<half> y_;
    TQue<QuePosition::VECIN, QD> in_;
    TQue<QuePosition::VECOUT, QD> out_;
    TQue<QuePosition::VECIN, 2> scaleQ_;
    TBuf<TPosition::VECCALC> tmpBuf_, padBuf_, padBuf2_;
    uint32_t uOfs_ = VEC_SIZE;
    uint32_t rowsPerBlock_ = ROWS;
    uint32_t rowsPerTile_ = ROWS;
    uint32_t blockCols_ = BN;
};


__global__ __mix__(1,2) void CompletePath(GM_ADDR x,GM_ADDR w,GM_ADDR scale,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR config) {
 Config cfg;TPipe pipe;
 if ASCEND_IS_AIC {
  LoadHeader(config,cfg);Tasks tasks;tasks.InitScalar<true>(groups,cfg);
  RunCubeManual<true>(x,w,workspace,cfg,tasks,pipe);
 }
 if ASCEND_IS_AIV {
  Tasks tasks;tasks.InitUb<true>(groups,cfg,pipe,config);
  Epilogue<2,false> op;op.Init(workspace,scale,y,pipe,cfg);
  op.FlatRun<true>(tasks,cfg,GetBlockIdx()/2,GetBlockIdx()%2,0);
 }
}
static bool TryRun(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR w,const TensorGroupInfo&iw,
 GM_ADDR s,GM_ADDR groups,GM_ADDR y,i64 available,aclrtStream stream) {
 const i64 m=ix.tensors[0].shape[0],k=ix.tensors[0].shape[1],e=iw.tensors[0].shape[0],n=iw.tensors[0].shape[2];
 if(n!=256||k<32||k>256||k%32||e<1||e>128||m<1||m>65536||available<1)return false;
 auto p=platform_ascendc::PlatformAscendCManager::GetInstance();
 const char*soc=aclrtGetSocName();if(!p||!soc||std::string(soc)!="Ascend910_9362")return false;
 uint32_t cores=std::min<i64>(available,std::min(p->GetCoreNumAic(),p->GetCoreNumAiv()/2));
 if(!cores)return false;
 const uint64_t tasksMin=std::min<uint64_t>(m,e);
 if(cores>16&&tasksMin>=2*cores&&tasksMin<=128&&tasksMin%16==0&&tasksMin%cores!=0)cores=16;
 if((m+127)/128<4*cores||m>64*e)return false;
 uint64_t ub=0,l1=0,a0=0,b0=0,c0=0;
 p->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
 p->GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);
 p->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,a0);
 p->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,b0);
 p->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,c0);
 if(ub<196352||l1<98304||a0<65536||b0<65536||c0<131072)return false;
 Config cfg{};cfg.m=m;cfg.e=e;cfg.k=k;cfg.n=n;cfg.nReal=n;cfg.cores=cores;
 cfg.nbc=256;cfg.bm=128;cfg.manual=1;cfg.flat=1;cfg.bStride=n;
 void *workspace=nullptr,*config=nullptr;
 auto check=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("reference Case4 ACL "+std::to_string(err));};
 aclError rc=aclrtMalloc(&workspace,size_t(m)*256*4,ACL_MEM_MALLOC_HUGE_FIRST);
 if(rc!=ACL_SUCCESS)return false;
 rc=aclrtMalloc(&config,sizeof(Config),ACL_MEM_MALLOC_HUGE_FIRST);
 if(rc!=ACL_SUCCESS){aclrtFree(workspace);return false;}
 rc=aclrtMemcpy(config,sizeof(Config),&cfg,sizeof(Config),ACL_MEMCPY_HOST_TO_DEVICE);
 if(rc==ACL_SUCCESS){
  CompletePath<<<cores,nullptr,stream>>>(x,w,s,groups,y,(GM_ADDR)workspace,(GM_ADDR)config);
  rc=aclrtSynchronizeStream(stream);
 }
 aclrtFree(config);aclrtFree(workspace);check(rc);return true;
}
} // namespace reference_case4
static void ShortNarrow_Run(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
if(reference_case4::TryRun(x,ix,weight,iw,scale,group_list,y,acn,stream))return;

const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());
if(!blocks)throw std::runtime_error("no Cube cores");u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);const i64 lowerTiles=((i64)M+15)/16*((N+127)/128);const bool use128=K<=1024&&N>=128&&N<=1024&&lowerTiles>=(i64)blocks*8&&ubBytes>=65536&&(K<=64||lowerTiles>=(i64)blocks*32);const i32 nBlocks256=N/256+((N%256)+127)/128;
const i64 lowerTiles256=((i64)M+15)/16*nBlocks256;i32 columnCycle=N/256;i32 workerCycle=blocks*2;while(workerCycle!=0){
i32 remainder=columnCycle%workerCycle;columnCycle=workerCycle;workerCycle=remainder;
}
const bool dense256Cycle=N%256==0&&columnCycle==1;const bool use256=K<=1024&&N>=32&&N<=1024&&ubBytes>=131072&&lowerTiles256>=(i64)blocks*8&&(K<=64||nBlocks256==1||lowerTiles256>=(i64)blocks*32||dense256Cycle);i32 tileCols=use256?256:use128?128:64;auto makeTiling=[&](i32 cols,TCubeTiling&result){
matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);mt.SetShape(16,cols,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(16,cols,64);mt.SetBufferSpace(-1,-1,-1);
return mt.GetTiling(result);
};TCubeTiling tiling;i32 tilingStatus=makeTiling(tileCols,tiling);if(tilingStatus!=0&&tileCols==256&&use128){tileCols=128;tilingStatus=makeTiling(tileCols,tiling);}
if(tilingStatus!=0&&tileCols!=64){tileCols=64;tilingStatus=makeTiling(tileCols,tiling);}
if(tilingStatus!=0)throw std::runtime_error("INT8 tiling failed");const i32 enableFastPath=(N<=256&&N%32==0&&tileCols==256&&lowerTiles256>=(i64)platform->GetCoreNumAic()*8)?1:0;u64 nativeL1=0,nativeA=0,nativeB=0,nativeC=0;bool earlyReady=false;const char*earlySoc=aclrtGetSocName();if(enableFastPath&&K<=256&&K%64==0&&earlySoc&&std::strcmp(earlySoc,"Ascend910_9362")==0){
platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1,nativeL1);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,nativeA);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,nativeB);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,nativeC);earlyReady=320LL*K<=nativeL1&&32LL*K<=nativeA&&256LL*K<=nativeB&&32LL*256*4<=nativeC;
}
const i64 plannedJobsUpper=(((i64)M+31)/32+E+blocks-1)/blocks;const bool worklistReady=earlyReady&&E<=128&&E%4==0&&plannedJobsUpper<=510;const bool creditRingReady=worklistReady&&ubBytes>=153600&&nativeL1>=576ULL*K;const bool nativeNoWs=tileCols==256&&enableFastPath&&earlyReady&&(N>=256||creditRingReady);
const size_t cScratchBytes=blocks*2*16*tileCols*4*(enableFastPath?2:1)+(creditRingReady?(size_t)blocks*8192*4:0);void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};
if(!nativeNoWs)checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,cScratchBytes+((worklistReady&&nativeNoWs)?(size_t)blocks*8192:0),ACL_MEM_MALLOC_HUGE_FIRST));if(tileCols==256){
if(enableFastPath){
const char*soc=aclrtGetSocName();const bool calibratedSoc=soc&&std::strcmp(soc,"Ascend910_9362")==0;if(calibratedSoc){
if(creditRingReady)B2546ShortNarrowImplicitFirstC<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)scratch,(GM_ADDR)((uint8_t*)scratch+cScratchBytes),M,K,E,N);else if(N<256)ShortNarrow_Legacy256<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,0);
else if(worklistReady&&ubBytes>=137216)B1606ShortNarrowPipeline<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)scratch,(GM_ADDR)((uint8_t*)scratch+cScratchBytes),M,K,E);else if(worklistReady)B1404ShortNarrowDeviceWorklist<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)scratch,(GM_ADDR)((uint8_t*)scratch+cScratchBytes),M,K,E);
else if(earlyReady)B1401NarrowEventarlyCredit<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)scratch,M,K,E);else ShortNarrow_MeasuredFast<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,16);
}else{
ShortNarrow_AsyncFallback<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,16);
}}else{
ShortNarrow_Legacy256<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,enableFastPath);
}}else if(tileCols==128){ShortNarrow_Legacy128<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E);}else{ShortNarrow_Legacy64<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E);}
aclError status=aclrtSynchronizeStream(stream);aclrtFree(scratch);if(workspace)aclrtFree(workspace);checkGmm(status);
}
using ShortWide_A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using ShortWide_B=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using ShortWide_C=MatmulType<TPosition::GM,CubeFormat::ND,i32>;
template<bool Compact=false,bool RepeatScale=false,bool EarlyCredit=false,bool ScaleOnce=false,bool OutOverlap=false,bool WideTail=false,bool Tight16=false,bool LogTail=false>__aicore__ inline void ShortWide_StableConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,i64 globalRow,i32 col,i32 e,i32 mr,i32 nr,i32 N,i32 credit=-1,bool loadScale=true,uint8_t outFree=0,uint8_t outReady=0){
const i32 gmPitch=nr<=64?64:nr<=128?128:256;const i32 pitch=Compact&&nr>128&&nr<=192?(Tight16?(nr+15)/16*16:192):gmPitch;const i32 liveCount=mr*pitch,count=Tight16?(liveCount+63)/64*64:liveCount;const i32 frame=WideTail&&nr>128&&nr<=192?32:16;if((!ScaleOnce||loadScale)&&nr<pitch){
Duplicate(sc,0.0f,pitch);auto zero=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(zero);WaitFlag<HardEvent::V_MTE2>(zero);
}
if(pitch==gmPitch)DataCopy(cint,c,frame*pitch);else DataCopyPad(cint,c,DataCopyExtParams{(uint16_t)frame,(u32)pitch*4,(u32)(gmPitch-pitch)*4,0,0},DataCopyPadExtParams<i32>{false,0,0,0});if constexpr(EarlyCredit){if(credit>=0)CrossCoreSetFlag<2,PIPE_MTE2>(credit);}
if(!ScaleOnce||loadScale)DataCopyPad(sc,scale[(i64)e*N+col],DataCopyExtParams{1,(u32)nr*4,0,0,0},DataCopyPadExtParams<float>{true,0,0,0});auto consumed=pipe.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(consumed);WaitFlag<HardEvent::MTE2_S>(consumed);auto iv=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(iv);WaitFlag<HardEvent::MTE2_V>(iv);Cast(z,cint,RoundMode::CAST_RINT,frame*pitch);PipeBarrier<PIPE_V>();
if(RepeatScale&&mr>pitch/64){
const uint8_t stride=pitch/8;for(i32 q=0;q<pitch;q+=64)Mul(z[q],z[q],sc[q],(u64)(Tight16&&pitch-q<64?pitch-q:64),(uint8_t)mr,BinaryRepeatParams{1,1,1,stride,stride,0});
}else for(i32 r=0;r<mr;++r)Mul(z[r*pitch],z[r*pitch],sc,pitch);if constexpr(Tight16){if(count!=liveCount)Duplicate(z[liveCount],0.0f,count-liveCount);}
PipeBarrier<PIPE_V>();if constexpr(LogTail){
Abs(u,z,count);Maxs(f,z,0.0f,count);PipeBarrier<PIPE_V>();Muls(t,u,-0.413330078125f,count);PipeBarrier<PIPE_V>();Adds(t,t,-0.7099609375f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,-0.71533203125f,count);PipeBarrier<PIPE_V>();Exp(t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Sub(f,f,t,count);PipeBarrier<PIPE_V>();
}else{
Abs(u,z,count);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,count);PipeBarrier<PIPE_V>();Mul(t,u,u,count);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(u,u,t,count);PipeBarrier<PIPE_V>();Muls(u,u,-1.5957691216057308f,count);PipeBarrier<PIPE_V>();Exp(t,u,count);PipeBarrier<PIPE_V>();Mul(f,z,t,count);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,count);
PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Div(f,f,t,count);PipeBarrier<PIPE_V>();
}
if constexpr(OutOverlap)WaitFlag<HardEvent::MTE3_V>(outFree);Cast(h,f,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();if constexpr(OutOverlap){
auto inputReuse=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(inputReuse);WaitFlag<HardEvent::V_MTE2>(inputReuse);
}
auto vo=OutOverlap?outReady:pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);if constexpr(OutOverlap&&Tight16&&LogTail){
const i32 alignedNr=(nr+15)/16*16;
DataCopyPad(out[globalRow*N+col],h,DataCopyExtParams{(uint16_t)mr,(u32)nr*2,(u32)(pitch-alignedNr)/16,(u32)(N-nr)*2,0});
}else{for(i32 r=0;r<mr;++r)DataCopyPad(out[(globalRow+r)*N+col],h[r*pitch],DataCopyExtParams{1,(u32)nr*2,0,0,0});}if constexpr(OutOverlap)SetFlag<HardEvent::MTE3_V>(outFree);else{
auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}}
__global__ __aicore__ __mix__(1,2)void ShortWide_VectorRequest(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortWide_A,ShortWide_B,ShortWide_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*16*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*tileCols;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<tileCols?N-col:tileCols;mm.SetOrgShape(M,N,K,K,nr<=64?64:nr<=128?128:256);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortWide_StableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortWide_StageKernel(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
InitSocState();SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;GlobalTensor<int8_t>a,b,ap,bp;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ap.SetGlobalBuffer((__gm__ int8_t*)staging);bp.SetGlobalBuffer((__gm__ int8_t*)staging+(i64)M*Kp);ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
const i32 pairId=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);MatmulImpl<ShortWide_A,ShortWide_B,ShortWide_C>mm;mm.Init(&tiling,&pipe);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);mm.SetOrgShape(M,Np,Kp,Kp,pitch);mm.SetSingleShape(mr,nr,Kp);mm.SetTensorA(ap[(start+row)*Kp]);mm.SetTensorB(bp[(i64)e*Kp*Np+col]);mm.IterateAll(c[bank*64*tileCols]);
mm.End();CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);
}
if ASCEND_IS_AIV{
const i32 id=GetBlockIdx(),workers=GetBlockNum()*2;const i32 pairId=id/2,pairs=GetBlockNum();GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)s);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);
pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;
pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();auto pack=ci.Get<u32>();auto bytes=ci.Get<int8_t>();auto pv=pipe.FetchEventID(HardEvent::V_MTE2);
auto pm=pipe.FetchEventID(HardEvent::MTE2_MTE3);auto pr=pipe.FetchEventID(HardEvent::MTE3_V);const i32 xRows=16384/Kp,wRows=16384/Np;for(i32 row=id*xRows;row<M;row+=workers*xRows){
i32 rows=M-row<xRows?M-row:xRows;Duplicate(pack,(u32)0,rows*Kp/4);SetFlag<HardEvent::V_MTE2>(pv);WaitFlag<HardEvent::V_MTE2>(pv);DataCopyPad(bytes,a[(i64)row*K],DataCopyExtParams{(uint16_t)rows,(u32)K,0,(u32)(Kp-((K+31)/32)*32)/32,0},DataCopyPadExtParams<int8_t>{true,0,0,0});SetFlag<HardEvent::MTE2_MTE3>(pm);WaitFlag<HardEvent::MTE2_MTE3>(pm);DataCopy(ap[(i64)row*Kp],bytes,rows*Kp);SetFlag<HardEvent::MTE3_V>(pr);
WaitFlag<HardEvent::MTE3_V>(pr);
}
i32 chunks=(Kp+wRows-1)/wRows;for(i64 chunk=id;chunk<(i64)E*chunks;chunk+=workers){
i32 e=chunk/chunks,kk=(chunk%chunks)*wRows;i32 targetRows=Kp-kk<wRows?Kp-kk:wRows;i32 rows=K-kk<targetRows?K-kk:targetRows;Duplicate(pack,(u32)0,targetRows*Np/4);SetFlag<HardEvent::V_MTE2>(pv);WaitFlag<HardEvent::V_MTE2>(pv);if(rows>0){
DataCopyPad(bytes,b[((i64)e*K+kk)*N],DataCopyExtParams{(uint16_t)rows,(u32)N,0,(u32)(Np-((N+31)/32)*32)/32,0},DataCopyPadExtParams<int8_t>{true,0,0,0});SetFlag<HardEvent::MTE2_MTE3>(pm);WaitFlag<HardEvent::MTE2_MTE3>(pm);
}else{
auto zm=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(zm);WaitFlag<HardEvent::V_MTE3>(zm);
}
DataCopy(bp[((i64)e*Kp+kk)*Np],bytes,targetRows*Np);SetFlag<HardEvent::MTE3_V>(pr);WaitFlag<HardEvent::MTE3_V>(pr);
}
PipeBarrier<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=GetSubBlockIdx()*32,valid=mr-off;if(valid>32)valid=32;for(i32 slab=0;slab<valid;slab+=16){
i32 sr=valid-slab<16?valid-slab:16;ShortWide_StableConsume(pipe,c[bank*64*tileCols+(off+slab)*pitch],scale,out,cint,z,u,t,f,sc,h,start+row+off+slab,col,e,sr,nr,N);
}
CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);++sequence;
}
tileBase+=count;start=end;
}}
PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void ShortWide_PortableGelu(LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>scratch,LocalTensor<half>h,i32 count){
Abs(u,z,count);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,count);PipeBarrier<PIPE_V>();Mul(t,u,u,count);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(u,u,t,count);PipeBarrier<PIPE_V>();Muls(u,u,-0.024933892525089545f,count);PipeBarrier<PIPE_V>();Duplicate(t,0.0013888888888888889f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.0083333333333333332f,count);
PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.041666666666666664f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.16666666666666666f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.5f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);
PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Muls(u,t,-2.0f,count);PipeBarrier<PIPE_V>();Adds(u,u,1.0f,count);PipeBarrier<PIPE_V>();Muls(u,u,0.3333333333333333333f,count);PipeBarrier<PIPE_V>();Duplicate(f,1.0f,count);PipeBarrier<PIPE_V>();for(i32 j=0;j<10;++j){
Mul(f,f,u,count);PipeBarrier<PIPE_V>();Adds(f,f,1.0f,count);PipeBarrier<PIPE_V>();
}
Muls(f,f,0.6666666666666666667f,count);PipeBarrier<PIPE_V>();Mul(scratch,z,t,count);PipeBarrier<PIPE_V>();auto sign=h.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,count);PipeBarrier<PIPE_V>();Select(scratch,sign,scratch,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();Mul(f,f,scratch,count);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
}
__aicore__ inline void ShortWide_PortableConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,i64 globalRow,i32 col,i32 e,i32 mr,i32 nr,i32 N,i32 pitch){
i32 count=mr*pitch;if(nr<pitch){
Duplicate(sc,0.0f,pitch);auto zero=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(zero);WaitFlag<HardEvent::V_MTE2>(zero);
}
DataCopy(cint,c,16*pitch);DataCopyPad(sc,scale[(i64)e*N+col],DataCopyExtParams{1,(u32)nr*4,0,0,0},DataCopyPadExtParams<float>{true,0,0,0});auto consumed=pipe.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(consumed);WaitFlag<HardEvent::MTE2_S>(consumed);auto iv=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(iv);WaitFlag<HardEvent::MTE2_V>(iv);Cast(z,cint,RoundMode::CAST_RINT,16*pitch);PipeBarrier<PIPE_V>();for(i32 row=0;row<mr;++row)Mul(z[row*pitch],z[row*pitch],sc,pitch);
PipeBarrier<PIPE_V>();ShortWide_PortableGelu(z,u,t,f,cint.ReinterpretCast<float>(),h,count);auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 row=0;row<mr;++row)DataCopyPad(out[(globalRow+row)*N+col],h[row*pitch],DataCopyExtParams{1,(u32)nr*2,0,0,0});auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}
__global__ __aicore__ __mix__(1,2)void ShortWide_Legacy64(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortWide_A,ShortWide_B,ShortWide_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*1024);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,4096);pipe.InitBuffer(zz,4096);pipe.InitBuffer(uu,4096);pipe.InitBuffer(tt,4096);pipe.InitBuffer(ff,4096);pipe.InitBuffer(ss,256);pipe.InitBuffer(yy,2048);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+63)/64;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*64;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<64?N-col:64;mm.SetOrgShape(M,N,K,K,64);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortWide_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,64);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortWide_Legacy128(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortWide_A,ShortWide_B,ShortWide_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2048);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,8192);pipe.InitBuffer(zz,8192);pipe.InitBuffer(uu,8192);pipe.InitBuffer(tt,8192);pipe.InitBuffer(ff,8192);pipe.InitBuffer(ss,512);pipe.InitBuffer(yy,4096);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+127)/128;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*128;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<128?N-col:128;mm.SetOrgShape(M,N,K,K,128);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortWide_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,128);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortWide_Legacy256(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 enableFastPath){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortWide_A,ShortWide_B,ShortWide_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*4096);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=N/256+((N%256)+127)/128;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)<N/256?(tile%nc)*256:(N/256)*256+((tile%nc)-N/256)*128;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<((tile%nc)<N/256?256:128)?N-col:((tile%nc)<N/256?256:128);mm.SetOrgShape(M,N,K,K,nr<=64?64:nr<=128?128:256);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);
mm.IterateAll(c);mm.End();ShortWide_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,nr<=64?64:nr<=128?128:256);
}
tileBase+=count;start=end;
}}}
__aicore__ inline void B17ShortWidePackIn(GlobalTensor<int8_t>src,LocalTensor<u32>pack,LocalTensor<int8_t>bytes,i32 rows,i32 targetRows,i32 cols,i32 pitch,uint8_t free,uint8_t vr,uint8_t mr,u32 srcGap=0){
WaitFlag<HardEvent::MTE3_V>(free);Duplicate(pack,(u32)0,targetRows*pitch/4);SetFlag<HardEvent::V_MTE2>(vr);WaitFlag<HardEvent::V_MTE2>(vr);if(rows>0)DataCopyPad(bytes,src,DataCopyExtParams{(uint16_t)rows,(u32)cols,srcGap,(u32)(pitch-((cols+31)/32)*32)/32,0},DataCopyPadExtParams<int8_t>{true,0,0,0});SetFlag<HardEvent::MTE2_MTE3>(mr);
}
__aicore__ inline void B17ShortWidePackOut(GlobalTensor<int8_t>dst,LocalTensor<int8_t>src,i32 count,uint8_t ready,uint8_t free){
WaitFlag<HardEvent::MTE2_MTE3>(ready);DataCopy(dst,src,count);SetFlag<HardEvent::MTE3_V>(free);
}
template<bool Panels,bool Compact=false,bool DirectA=false,bool Rotate=false,bool RepeatScale=false,bool EarlyCredit=false,bool ScaleOnce=false,bool OutOverlap=false,int RequestRows=64,bool WideTail=false,bool Tight16=false,bool LogTail=false>__aicore__ inline void B18ShortWidePanelBody(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
InitSocState();SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;GlobalTensor<int8_t>a,b,ap,bp;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ap.SetGlobalBuffer((__gm__ int8_t*)staging);bp.SetGlobalBuffer((__gm__ int8_t*)staging+(DirectA?0:(i64)M*Kp));ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
const i32 pairId=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*2*RequestRows*tileCols);MatmulImpl<ShortWide_A,ShortWide_B,ShortWide_C>mm;mm.Init(&tiling,&pipe);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+RequestRows-1)/RequestRows)*nc;for(i64 tile=(pairId-(tileBase+(Rotate?e:0))%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*RequestRows,col=tile%nc*tileCols;i32 mr=rows-row<RequestRows?rows-row:RequestRows,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);mm.SetOrgShape(M,Panels?tileCols:Np,DirectA?K:Kp,Kp,pitch);mm.SetSingleShape(mr,nr,DirectA?K:Kp);
if constexpr(DirectA)mm.SetTensorA(a[(start+row)*K]);else mm.SetTensorA(ap[(start+row)*Kp]);mm.SetTensorB(bp[Panels?((i64)e*(Np/tileCols)+col/tileCols)*Kp*tileCols:(i64)e*Kp*Np+col]);mm.IterateAll(c[bank*RequestRows*tileCols]);mm.End();CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);
}
if ASCEND_IS_AIV{
const i32 id=GetBlockIdx(),workers=GetBlockNum()*2;const i32 pairId=id/2,pairs=GetBlockNum();GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)s);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*2*RequestRows*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,32768);pipe.InitBuffer(zz,WideTail?24576:16384);
pipe.InitBuffer(uu,WideTail?24576:16384);pipe.InitBuffer(tt,WideTail?24576:16384);pipe.InitBuffer(ff,WideTail?24576:16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;pipe.InitBuffer(diagPoly,WideTail?24576:16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();auto pack=ci.Get<u32>();
auto bytes=ci.Get<int8_t>();uint8_t freePack[2],vReady[2],mReady[2];for(i32 j=0;j<2;++j){
freePack[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE3_V>();vReady[j]=(uint8_t)pipe.AllocEventID<HardEvent::V_MTE2>();mReady[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE2_MTE3>();SetFlag<HardEvent::MTE3_V>(freePack[j]);
}
bool pendingPack=false;i32 savedBank=0,savedCount=0;i64 savedDest=0,packSeq=0;const i32 wp=Panels?tileCols:Np,xRows=16384/Kp,wRows=16384/wp;if constexpr(!DirectA){
for(i32 row=id*xRows;row<M;row+=workers*xRows){
const i32 rows=M-row<xRows?M-row:xRows,slot=packSeq&1;B17ShortWidePackIn(a[(i64)row*K],pack[slot*4096],bytes[slot*16384],rows,rows,K,Kp,freePack[slot],vReady[slot],mReady[slot]);if(pendingPack)B17ShortWidePackOut(ap[savedDest],bytes[savedBank*16384],savedCount,mReady[savedBank],freePack[savedBank]);
savedDest=(i64)row*Kp;savedCount=rows*Kp;savedBank=slot;pendingPack=true;++packSeq;
}}
if constexpr(Panels){
const i32 chunks=(Kp+wRows-1)/wRows,activePanels=(N+tileCols-1)/tileCols;for(i64 chunk=id;chunk<(i64)E*activePanels*chunks;chunk+=workers){
const i32 ep=chunk/chunks,e=ep/activePanels,p=ep%activePanels;const i32 kk=(chunk%chunks)*wRows,slot=packSeq&1,col=p*tileCols,nr=N-col<tileCols?N-col:tileCols;const i32 targetRows=Kp-kk<wRows?Kp-kk:wRows,rows=K-kk<targetRows?K-kk:targetRows;GlobalTensor<int8_t>source=b;if(rows>0)source=b[((i64)e*K+kk)*N+col];
B17ShortWidePackIn(source,pack[slot*4096],bytes[slot*16384],rows,targetRows,nr,tileCols,freePack[slot],vReady[slot],mReady[slot],(u32)(N-nr));if(pendingPack)B17ShortWidePackOut(ap[savedDest],bytes[savedBank*16384],savedCount,mReady[savedBank],freePack[savedBank]);savedDest=(DirectA?0:(i64)M*Kp)+(((i64)e*(Np/tileCols)+p)*Kp+kk)*tileCols;
savedCount=targetRows*tileCols;savedBank=slot;pendingPack=true;++packSeq;
}}else{
i32 chunks=(Kp+wRows-1)/wRows;for(i64 chunk=id;chunk<(i64)E*chunks;chunk+=workers){
const i32 e=chunk/chunks,kk=(chunk%chunks)*wRows,slot=packSeq&1;const i32 targetRows=Kp-kk<wRows?Kp-kk:wRows,rows=K-kk<targetRows?K-kk:targetRows;GlobalTensor<int8_t>source=b;if(rows>0)source=b[((i64)e*K+kk)*N];B17ShortWidePackIn(source,pack[slot*4096],bytes[slot*16384],rows,targetRows,N,Np,freePack[slot],vReady[slot],mReady[slot]);
if(pendingPack)B17ShortWidePackOut(ap[savedDest],bytes[savedBank*16384],savedCount,mReady[savedBank],freePack[savedBank]);savedDest=(DirectA?0:(i64)M*Kp)+((i64)e*Kp+kk)*Np;savedCount=targetRows*Np;savedBank=slot;pendingPack=true;++packSeq;
}}
if(pendingPack)B17ShortWidePackOut(ap[savedDest],bytes[savedBank*16384],savedCount,mReady[savedBank],freePack[savedBank]);for(i32 j=0;j<2;++j){
WaitFlag<HardEvent::MTE3_V>(freePack[j]);pipe.ReleaseEventID<HardEvent::MTE3_V>(freePack[j]);pipe.ReleaseEventID<HardEvent::V_MTE2>(vReady[j]);pipe.ReleaseEventID<HardEvent::MTE2_MTE3>(mReady[j]);
}
PipeBarrier<PIPE_ALL>();SyncAll<true>();uint8_t outFree[2]={},outReady[2]={};i64 outSeq=0;if constexpr(OutOverlap)for(i32 j=0;j<2;++j){
outFree[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE3_V>();outReady[j]=(uint8_t)pipe.AllocEventID<HardEvent::V_MTE3>();SetFlag<HardEvent::MTE3_V>(outFree[j]);
}
CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+RequestRows-1)/RequestRows)*nc;for(i64 tile=(pairId-(tileBase+(Rotate?e:0))%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*RequestRows,col=tile%nc*tileCols;i32 mr=rows-row<RequestRows?rows-row:RequestRows,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=GetSubBlockIdx()*(RequestRows/2),valid=mr-off;if(valid>RequestRows/2)valid=RequestRows/2;
const i32 frame=WideTail&&nr>128&&nr<=192?32:16;for(i32 slab=0;slab<valid;slab+=frame){
i32 sr=valid-slab<frame?valid-slab:frame;const i32 ob=OutOverlap?outSeq++&1:0;ShortWide_StableConsume<Compact,RepeatScale,EarlyCredit,ScaleOnce,OutOverlap,WideTail,Tight16,LogTail>(pipe,c[bank*RequestRows*tileCols+(off+slab)*pitch],scale,out,cint,z,u,t,f,sc,OutOverlap?diagPolyTensor.ReinterpretCast<half>()[ob*(WideTail?6144:4096)]:h,start+row+off+slab,col,e,sr,nr,N,slab+frame>=valid?8+bank:-1,slab==0,outFree[ob],outReady[ob]);
}
if(!EarlyCredit||valid<=0)CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
if constexpr(OutOverlap)for(i32 j=0;j<2;++j){
WaitFlag<HardEvent::MTE3_V>(outFree[j]);pipe.ReleaseEventID<HardEvent::MTE3_V>(outFree[j]);pipe.ReleaseEventID<HardEvent::V_MTE3>(outReady[j]);
}}
PipeBarrier<PIPE_ALL>();
}
__global__ __aicore__ __mix__(1,2)void B1722ShortWideK32Padding(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){B18ShortWidePanelBody<false>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);}
__global__ __aicore__ __mix__(1,2)void B1810ShortWideWeightPanels(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){B18ShortWidePanelBody<true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);}
__global__ __aicore__ __mix__(1,2)void B1811ShortWideCompactEpilogue(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){B18ShortWidePanelBody<true,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);}
__global__ __aicore__ __mix__(1,2)void B1813ShortWideDirectA(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){B18ShortWidePanelBody<true,true,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);}
__global__ __aicore__ __mix__(1,2)void B1816ShortWideStripeBalance(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){B18ShortWidePanelBody<true,true,true,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);}
__global__ __aicore__ __mix__(1,2)void B1819ShortWideRepeatScale(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){B18ShortWidePanelBody<true,true,true,true,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);}
__global__ __aicore__ __mix__(1,2)void B1840ShortWideEarlyCredit(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
B18ShortWidePanelBody<true,true,true,true,true,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);
}
__global__ __aicore__ __mix__(1,2)void B1842ShortWideScaleOnce(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
B18ShortWidePanelBody<true,true,true,true,true,true,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);
}
__global__ __aicore__ __mix__(1,2)void B1844ShortWideOutputOverlap(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
B18ShortWidePanelBody<true,true,true,true,true,true,true,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);
}
__global__ __aicore__ __mix__(1,2)void B1910ShortWideRequest128(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
B18ShortWidePanelBody<true,true,true,true,true,true,true,true,128>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);
}
__global__ __aicore__ __mix__(1,2)void B2110ShortWideWideTail(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
B18ShortWidePanelBody<true,true,true,true,true,true,true,true,128,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);
}
__global__ __aicore__ __mix__(1,2)void B2501ShortWideTight16(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
B18ShortWidePanelBody<true,true,true,true,true,true,true,true,128,true,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols,staging,Kp,Np);
}
__aicore__ inline void ShortWideDirectPadNd2Nz(LocalTensor<int8_t>dst,i32 staticHeight,i32 staticWidth,i32 height,i32 width){
const i32 widthC0=(width+31)/32,staticWidthC0=staticWidth/32;if(height<staticHeight){
InitConstValueParams<u32>p{};p.repeatTimes=(uint16_t)widthC0;p.blockNum=(uint16_t)(staticHeight-height);p.dstGap=(uint16_t)height;p.initValue=0;InitConstValue(dst[height*32].ReinterpretCast<u32>(),p);
}
if(widthC0<staticWidthC0){
InitConstValueParams<u32>p{};p.repeatTimes=1;p.blockNum=(uint16_t)((staticWidthC0-widthC0)*staticHeight);p.dstGap=0;p.initValue=0;InitConstValue(dst[widthC0*staticHeight*32].ReinterpretCast<u32>(),p);
}}
__aicore__ inline void ShortWideDirectCachedLoadL0(LocalTensor<int8_t>a1,LocalTensor<int8_t>b1,LocalTensor<int8_t>a2,LocalTensor<int8_t>b2,i32 kk,i32 Kp,uint8_t free,uint8_t ready){
WaitFlag<HardEvent::M_MTE1>(free);LoadData2DParams gather{};gather.startIndex=0;gather.repeatTimes=8;gather.srcStride=1;gather.sid=0;gather.dstGap=1;gather.ifTranspose=false;gather.addrMode=0;for(i32 q=0;q<2;++q){const i32 aSource=kk*128+q*4096,aDest=q*512;LoadData(a2[aDest],a1[aSource],gather);}
LoadData2dTransposeParams trans{};trans.startIndex=0;trans.repeatTimes=2;trans.srcStride=1;trans.dstGap=15;trans.dstFracGap=0;for(i32 n=0;n<8;++n){const i32 bSource=n*Kp*32+kk*32,bDest=n*1024;LoadDataWithTranspose(b2[bDest],b1[bSource],trans);}
SetFlag<HardEvent::MTE1_M>(ready);
}
__aicore__ inline void ShortWideDirectCached128Body(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR ws,GM_ADDR cs,i32 M,i32 K,i32 N,i32 E,i32 Kp,i32 Np,bool ap){
InitSocState();SetSysWorkspaceForce(ws);clearWorkspace(ws);TPipe pipe;const i32 nc=Np/256;GlobalTensor<int8_t>a,b;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ends.SetGlobalBuffer((__gm__ i64*)groups);i64 total=0,pr=0;for(i32 e=0;e<E;++e){
const i64 end=ends.GetValue(e);total+=((end-pr+127)/128)*nc;pr=end;
}
if ASCEND_IS_AIC{
if(N<1){MXPAD1024 MXPAD512 MXPAD64 MXPAD8}
const i32 pair=GetBlockIdx(),pairs=GetBlockNum();const i64 units=total>>ap;const i64 lo=(units*pair/pairs)<<ap,hi=(units*(pair+1)/pairs)<<ap;c.SetGlobalBuffer((__gm__ i32*)cs+(i64)pair*65536);TBuf<TPosition::A1>aa;TBuf<TPosition::B1>bb;TBuf<TPosition::A2>ab;TBuf<TPosition::B2>ba;TBuf<TPosition::CO1>cc;pipe.InitBuffer(aa,128*Kp);pipe.InitBuffer(bb,(ap?512:256)*Kp);
pipe.InitBuffer(ab,16384);pipe.InitBuffer(ba,32768);pipe.InitBuffer(cc,131072);auto a1=aa.Get<int8_t>();auto b1=bb.Get<int8_t>();auto a2=ab.Get<int8_t>();auto b2=ba.Get<int8_t>();auto c0=cc.Get<i32>();uint8_t freeL0[2],readyL0[2];for(i32 j=0;j<2;++j){
freeL0[j]=(uint8_t)pipe.AllocEventID<HardEvent::M_MTE1>();readyL0[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE1_M>();SetFlag<HardEvent::M_MTE1>(freeL0[j]);
}
auto lr=pipe.FetchEventID(HardEvent::MTE2_MTE1),lf=pipe.FetchEventID(HardEvent::MTE1_MTE2);auto cr=pipe.FetchEventID(HardEvent::M_FIX),cf=pipe.FetchEventID(HardEvent::FIX_M);SetFlag<HardEvent::MTE1_MTE2>(lf);i64 start=0,base=0,seq=0,pp=-1;for(i32 e=0;e<E;++e){
const i64 end=ends.GetValue(e);const i32 rows=end-start,rt=(rows+127)/128;const i32 span=ap?2*rt:rt;for(i32 j=0;j<(ap?1:nc);++j){
const i64 pb=base+(i64)j*rt;const i64 beg=lo>pb?lo:pb,lim=hi<pb+span?hi:pb+span;for(i64 rank=beg;rank<lim;++rank){
const i32 k=rank-pb,p=ap?(k&1):j;const i32 row=(k>>ap)*128,mr=rows-row<128?rows-row:128,bank=seq&1;const i64 panel=(i64)e*(Np/256)+p;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);WaitFlag<HardEvent::MTE1_MTE2>(lf);if(!ap||p==0){
ShortWideDirectPadNd2Nz(a1,128,Kp,mr,K);Nd2NzParams aCopy{};aCopy.ndNum=1;aCopy.nValue=(uint16_t)mr;aCopy.dValue=(uint16_t)K;aCopy.srcNdMatrixStride=0;aCopy.srcDValue=(uint16_t)K;aCopy.dstNzC0Stride=128;aCopy.dstNzNStride=1;aCopy.dstNzMatrixStride=0;const i64 as=(start+row)*K;DataCopy(a1,a[as],aCopy);
}
auto bp=b1[ap?p*256*Kp:0];if(ap?rank-beg<2:pp!=panel){
const i32 col=p*256,nr=N-col<256?N-col:256;ShortWideDirectPadNd2Nz(bp,Kp,256,K,nr);Nd2NzParams bCopy{};bCopy.ndNum=1;bCopy.nValue=(uint16_t)K;bCopy.dValue=(uint16_t)nr;bCopy.srcNdMatrixStride=0;bCopy.srcDValue=(uint16_t)N;bCopy.dstNzC0Stride=(uint16_t)Kp;bCopy.dstNzNStride=1;bCopy.dstNzMatrixStride=0;
const i64 bs=(i64)e*K*N+col;DataCopy(bp,b[bs],bCopy);pp=panel;
}
SetFlag<HardEvent::MTE2_MTE1>(lr);WaitFlag<HardEvent::MTE2_MTE1>(lr);ShortWideDirectCachedLoadL0(a1,bp,a2,b2,0,Kp,freeL0[0],readyL0[0]);for(i32 kc=0;kc<Kp/64;++kc){
const i32 d=kc&1;if(kc+1<Kp/64){
const i32 next=d^1;ShortWideDirectCachedLoadL0(a1,bp,a2[next*8192],b2[next*16384],(kc+1)*64,Kp,freeL0[next],readyL0[next]);
}
WaitFlag<HardEvent::MTE1_M>(readyL0[d]);Mmad(c0,a2[d*8192],b2[d*16384],MmadParams{128,256,64,0,false,kc==0});SetFlag<HardEvent::M_MTE1>(freeL0[d]);
}
SetFlag<HardEvent::MTE1_MTE2>(lf);SetFlag<HardEvent::M_FIX>(cr);WaitFlag<HardEvent::M_FIX>(cr);Fixpipe(c[bank*32768],c0,FixpipeParamsV220{256,(uint16_t)mr,128,256,false,QuantMode_t::NoQuant,0,1,0,0,0});SetFlag<HardEvent::FIX_M>(cf);WaitFlag<HardEvent::FIX_M>(cf);CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++seq;
}}
base+=(i64)rt*nc;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);WaitFlag<HardEvent::MTE1_MTE2>(lf);for(i32 j=0;j<2;++j){
WaitFlag<HardEvent::M_MTE1>(freeL0[j]);pipe.ReleaseEventID<HardEvent::M_MTE1>(freeL0[j]);pipe.ReleaseEventID<HardEvent::MTE1_M>(readyL0[j]);
}}
if ASCEND_IS_AIV{
if(N<1){MXPAD256 MXPAD32 MXPAD8 MXPAD4 MXPAD2}
const i32 id=GetBlockIdx(),pair=id/2,pairs=GetBlockNum();const i64 units=total>>ap;const i64 lo=(units*pair/pairs)<<ap,hi=(units*(pair+1)/pairs)<<ap;GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)s);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)cs+(i64)pair*65536);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss;
pipe.InitBuffer(ci,32768);pipe.InitBuffer(zz,24576);pipe.InitBuffer(uu,24576);pipe.InitBuffer(tt,24576);pipe.InitBuffer(ff,24576);pipe.InitBuffer(ss,4096);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();TBuf<TPosition::VECCALC>hb;pipe.InitBuffer(hb,24576);
auto ht=hb.Get<float>();uint8_t outFree[2]={},outReady[2]={};i64 outSeq=0;for(i32 j=0;j<2;++j){
outFree[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE3_V>();outReady[j]=(uint8_t)pipe.AllocEventID<HardEvent::V_MTE3>();SetFlag<HardEvent::MTE3_V>(outFree[j]);
}
CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,base=0,seq=0;for(i32 e=0;e<E;++e){
const i64 end=ends.GetValue(e);const i32 rows=end-start,rt=(rows+127)/128;const i32 span=ap?2*rt:rt;for(i32 j=0;j<(ap?1:nc);++j){
const i64 pb=base+(i64)j*rt;const i64 beg=lo>pb?lo:pb,lim=hi<pb+span?hi:pb+span;for(i64 rank=beg;rank<lim;++rank){
const i32 k=rank-pb,p=ap?(k&1):j;const i32 row=(k>>ap)*128,mr=rows-row<128?rows-row:128,col=p*256,nr=N-col<256?N-col:256,bank=seq&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=GetSubBlockIdx()*64,valid=mr-off;if(valid>64)valid=64;const i32 frame=nr>128&&nr<=192?32:16;for(i32 slab=0;slab<valid;slab+=frame){
const i32 sr=valid-slab<frame?valid-slab:frame,ob=outSeq++&1;ShortWide_StableConsume<true,true,true,true,true,true,true,true>(pipe,c[bank*32768+(off+slab)*256],scale,out,cint,z,u,t,f,sc,ht.ReinterpretCast<half>()[ob*6144],start+row+off+slab,col,e,sr,nr,N,slab+frame>=valid?8+bank:-1,slab==0,outFree[ob],outReady[ob]);
}
if(valid<=0)CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);++seq;
}}
base+=(i64)rt*nc;start=end;
}
for(i32 j=0;j<2;++j){
WaitFlag<HardEvent::MTE3_V>(outFree[j]);pipe.ReleaseEventID<HardEvent::MTE3_V>(outFree[j]);pipe.ReleaseEventID<HardEvent::V_MTE3>(outReady[j]);
}}
PipeBarrier<PIPE_ALL>();
}
__global__ __aicore__ __mix__(1,2)void B2506ShortWideLogTail(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR ws,GM_ADDR cs,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){ShortWideDirectCached128Body(x,w,s,groups,y,ws,cs,M,K,N,E,Kp,Np,tileCols==512);}
static bool ShortWideTryDirectCachedRun(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];if(M<=0||K<=0||N<=256||E<=0||K>65535||N>65535||acn<1||(i64)M<128LL*E||N%256<=128||N%256>192)return false;
auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform)return false;
const char*soc=aclrtGetSocName();if(!soc||std::strcmp(soc,"Ascend910_9362")!=0)return false;
const u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());if(!blocks||platform->GetCoreNumAiv()<2*blocks)return false;
u64 ub=0,l1=0,l0a=0,l0b=0,l0c=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,l0a);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);const i64 kp64=((i64)K+63)/64*64,np64=((i64)N+255)/256*256;if(kp64<64||kp64>16384||kp64/32>255||kp64*8>65535)return false;
if(ub<159744||l1<(u64)kp64*384||l0a<16384||l0b<32768||l0c<131072)return false;
const i32 Kp=kp64,Np=np64;const u64 cBytes=(u64)blocks*65536*sizeof(i32);if(cBytes>64ULL*1024*1024)return false;
const size_t sys=platform->GetLibApiWorkSpaceSize();if(sys==0)return false;
void*workspace=nullptr,*scratch=nullptr;aclError status=aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST);if(status==ACL_SUCCESS)status=aclrtMalloc(&scratch,cBytes,ACL_MEM_MALLOC_HUGE_FIRST);if(status!=ACL_SUCCESS){
if(scratch)aclrtFree(scratch);if(workspace)aclrtFree(workspace);return false;
}
TCubeTiling unusedTiling{};B2506ShortWideLogTail<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,unusedTiling,M,K,N,E,Np<=512&&l1>=640*kp64?512:256,(GM_ADDR)nullptr,Kp,Np);status=aclrtSynchronizeStream(stream);aclrtFree(scratch);aclrtFree(workspace);if(status!=ACL_SUCCESS)throw std::runtime_error("ShortWide direct cached ACL error "+std::to_string(status));
return true;
}
static void ShortWide_Run(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
if(ShortWideTryDirectCachedRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream))return;
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());
if(!blocks)throw std::runtime_error("no Cube cores");u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);const i64 lowerTiles=((i64)M+15)/16*((N+127)/128);const bool use128=K<=1024&&N>=128&&N<=1024&&lowerTiles>=(i64)blocks*8&&ubBytes>=65536&&(K<=64||lowerTiles>=(i64)blocks*32);const i32 nBlocks256=N/256+((N%256)+127)/128;
const i64 lowerTiles256=((i64)M+15)/16*nBlocks256;i32 columnCycle=N/256;i32 workerCycle=blocks*2;while(workerCycle!=0){
i32 remainder=columnCycle%workerCycle;columnCycle=workerCycle;workerCycle=remainder;
}
const bool dense256Cycle=N%256==0&&columnCycle==1;const bool use256=K<=1024&&N>=256&&N<=1024&&ubBytes>=131072&&lowerTiles256>=(i64)blocks*8&&(K<=64||nBlocks256==1||lowerTiles256>=(i64)blocks*32||dense256Cycle);i32 tileCols=use256?256:use128?128:64;auto makeTiling=[&](i32 cols,TCubeTiling&result){
matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);mt.SetShape(16,cols,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(16,cols,64);mt.SetBufferSpace(-1,-1,-1);
return mt.GetTiling(result);
};TCubeTiling tiling;i32 tilingStatus=makeTiling(tileCols,tiling);if(tilingStatus!=0&&tileCols==256&&use128){tileCols=128;tilingStatus=makeTiling(tileCols,tiling);}
if(tilingStatus!=0&&tileCols!=64){tileCols=64;tilingStatus=makeTiling(tileCols,tiling);}
if(tilingStatus!=0)throw std::runtime_error("INT8 tiling failed");const bool tryWide=(i64)M>=(i64)E*128&&tileCols==256&&N%256>128&&N%256<=192;const i32 enableFastPath=0;const i32 oldKp=(K+63)/64*64,Np=(N+255)/256*256;i32 Kp=(K+31)/32*32;bool compactAccepted=true;i64 stagedBytes=(i64)M*Kp+(i64)E*Kp*Np;
const i64 oldStageBound=(i64)M*oldKp+(i64)E*oldKp*Np;const bool useStage=(K%32!=0||N%32!=0)&&oldStageBound<=134217728LL;void*staging=nullptr;void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};
checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*(useStage?(tryWide?128:64):16)*tileCols*4*(enableFastPath?2:1),ACL_MEM_MALLOC_HUGE_FIRST));const char*b505soc=aclrtGetSocName();if(b505soc&&std::strcmp(b505soc,"Ascend910_9362")==0&&ubBytes>=131072){
if(useStage){
matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);u64 l0a=0,l0b=0,l0c=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,l0a);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);
platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);i32 baseM=2*64*64<=l0a&&2*tileCols*64<=l0b&&64*tileCols*4<=l0c?64:16;mt.SetShape(64,tileCols,Kp);mt.SetOrgShape(M,Np,Kp);mt.SetFixSplit(baseM,tileCols,64);mt.SetBufferSpace(-1,-1,-1);TCubeTiling packedTiling;int tilingStatus64=mt.GetTiling(packedTiling);if(tilingStatus64!=0&&baseM==64){mt.SetFixSplit(16,tileCols,64);tilingStatus64=mt.GetTiling(packedTiling);}
if(tilingStatus64!=0&&Kp!=oldKp){
Kp=oldKp;stagedBytes=oldStageBound;compactAccepted=false;mt.SetShape(64,tileCols,Kp);mt.SetOrgShape(M,Np,Kp);mt.SetFixSplit(baseM,tileCols,64);tilingStatus64=mt.GetTiling(packedTiling);if(tilingStatus64!=0&&baseM==64){mt.SetFixSplit(16,tileCols,64);tilingStatus64=mt.GetTiling(packedTiling);}}
if(tilingStatus64==0){
TCubeTiling wideTiling;bool wideReady=false;if(tryWide&&compactAccepted&&2LL*128*64<=l0a&&2LL*tileCols*64<=l0b&&128LL*tileCols*4<=l0c){
mt.SetShape(128,tileCols,Kp);mt.SetOrgShape(M,Np,Kp);mt.SetFixSplit(128,tileCols,64);if(mt.GetTiling(wideTiling)==0)wideReady=wideTiling.baseM==128&&wideTiling.baseN==tileCols&&2LL*128*wideTiling.baseK<=l0a&&2LL*tileCols*wideTiling.baseK<=l0b&&128LL*tileCols*4<=l0c;
}
const bool directPanelA=compactAccepted&&N>tileCols&&Np%tileCols==0&&(wideReady||(tileCols==256&&N%256>128&&N%256<=192));if(directPanelA)stagedBytes-=(i64)M*Kp;checkGmm(aclrtMalloc(&staging,stagedBytes,ACL_MEM_MALLOC_HUGE_FIRST));if(compactAccepted){
if(N>tileCols&&Np%tileCols==0){
if(wideReady){
if(ubBytes>=171008)B2501ShortWideTight16<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,wideTiling,M,K,N,E,tileCols,(GM_ADDR)staging,Kp,Np);else B1910ShortWideRequest128<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,wideTiling,M,K,N,E,tileCols,(GM_ADDR)staging,Kp,Np);
}else if(tileCols==256&&N%256>128&&N%256<=192)B1844ShortWideOutputOverlap<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,packedTiling,M,K,N,E,tileCols,(GM_ADDR)staging,Kp,Np);else B1810ShortWideWeightPanels<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,packedTiling,M,K,N,E,tileCols,(GM_ADDR)staging,Kp,Np);
}else B1722ShortWideK32Padding<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,packedTiling,M,K,N,E,tileCols,(GM_ADDR)staging,Kp,Np);
}else ShortWide_StageKernel<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,packedTiling,M,K,N,E,tileCols,(GM_ADDR)staging,Kp,Np);
}else{
ShortWide_VectorRequest<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileCols);
}}else{
ShortWide_VectorRequest<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileCols);
}}else{
if(tileCols==256){ShortWide_Legacy256<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,0);}else if(tileCols==128){ShortWide_Legacy128<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E);}else{ShortWide_Legacy64<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E);}}
aclError status=aclrtSynchronizeStream(stream);if(staging)aclrtFree(staging);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
using ShortMid_A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using ShortMid_B=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using ShortMid_C=MatmulType<TPosition::GM,CubeFormat::ND,i32>;
__aicore__ inline void ShortMid_StableConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,i64 globalRow,i32 col,i32 e,i32 mr,i32 nr,i32 N){
const i32 pitch=nr<=64?64:nr<=128?128:256;const i32 count=mr*pitch;if(nr<pitch){
Duplicate(sc,0.0f,pitch);auto zero=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(zero);WaitFlag<HardEvent::V_MTE2>(zero);
}
DataCopy(cint,c,16*pitch);DataCopyPad(sc,scale[(i64)e*N+col],DataCopyExtParams{1,(u32)nr*4,0,0,0},DataCopyPadExtParams<float>{true,0,0,0});auto consumed=pipe.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(consumed);WaitFlag<HardEvent::MTE2_S>(consumed);auto iv=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(iv);WaitFlag<HardEvent::MTE2_V>(iv);Cast(z,cint,RoundMode::CAST_RINT,16*pitch);PipeBarrier<PIPE_V>();for(i32 r=0;r<mr;++r)Mul(z[r*pitch],z[r*pitch],sc,pitch);
PipeBarrier<PIPE_V>();Abs(u,z,count);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,count);PipeBarrier<PIPE_V>();Mul(t,u,u,count);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(u,u,t,count);PipeBarrier<PIPE_V>();Muls(u,u,-1.5957691216057308f,count);PipeBarrier<PIPE_V>();Exp(t,u,count);PipeBarrier<PIPE_V>();Mul(f,z,t,count);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,count);
PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Div(f,f,t,count);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 r=0;r<mr;++r)DataCopyPad(out[(globalRow+r)*N+col],h[r*pitch],DataCopyExtParams{1,(u32)nr*2,0,0,0});auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);
SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}
__global__ __aicore__ __mix__(1,2)void ShortMid_VectorRequest(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortMid_A,ShortMid_B,ShortMid_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*16*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*tileCols;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<tileCols?N-col:tileCols;mm.SetOrgShape(M,N,K,K,nr<=64?64:nr<=128?128:256);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortMid_StableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortMid_StageKernel(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
InitSocState();SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;GlobalTensor<int8_t>a,b,ap,bp;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ap.SetGlobalBuffer((__gm__ int8_t*)staging);bp.SetGlobalBuffer((__gm__ int8_t*)staging+(i64)M*Kp);ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
const i32 pairId=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);MatmulImpl<ShortMid_A,ShortMid_B,ShortMid_C>mm;mm.Init(&tiling,&pipe);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);mm.SetOrgShape(M,Np,Kp,Kp,pitch);mm.SetSingleShape(mr,nr,Kp);mm.SetTensorA(ap[(start+row)*Kp]);mm.SetTensorB(bp[(i64)e*Kp*Np+col]);mm.IterateAll(c[bank*64*tileCols]);
mm.End();CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);
}
if ASCEND_IS_AIV{
const i32 id=GetBlockIdx(),workers=GetBlockNum()*2;const i32 pairId=id/2,pairs=GetBlockNum();GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)s);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);
pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;
pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();auto pack=ci.Get<u32>();auto bytes=ci.Get<int8_t>();auto pv=pipe.FetchEventID(HardEvent::V_MTE2);
auto pm=pipe.FetchEventID(HardEvent::MTE2_MTE3);auto pr=pipe.FetchEventID(HardEvent::MTE3_V);const i32 xRows=16384/Kp,wRows=16384/Np;for(i32 row=id*xRows;row<M;row+=workers*xRows){
i32 rows=M-row<xRows?M-row:xRows;Duplicate(pack,(u32)0,rows*Kp/4);SetFlag<HardEvent::V_MTE2>(pv);WaitFlag<HardEvent::V_MTE2>(pv);DataCopyPad(bytes,a[(i64)row*K],DataCopyExtParams{(uint16_t)rows,(u32)K,0,(u32)(Kp-((K+31)/32)*32)/32,0},DataCopyPadExtParams<int8_t>{true,0,0,0});SetFlag<HardEvent::MTE2_MTE3>(pm);WaitFlag<HardEvent::MTE2_MTE3>(pm);DataCopy(ap[(i64)row*Kp],bytes,rows*Kp);SetFlag<HardEvent::MTE3_V>(pr);
WaitFlag<HardEvent::MTE3_V>(pr);
}
i32 chunks=(Kp+wRows-1)/wRows;for(i64 chunk=id;chunk<(i64)E*chunks;chunk+=workers){
i32 e=chunk/chunks,kk=(chunk%chunks)*wRows;i32 targetRows=Kp-kk<wRows?Kp-kk:wRows;i32 rows=K-kk<targetRows?K-kk:targetRows;Duplicate(pack,(u32)0,targetRows*Np/4);SetFlag<HardEvent::V_MTE2>(pv);WaitFlag<HardEvent::V_MTE2>(pv);if(rows>0){
DataCopyPad(bytes,b[((i64)e*K+kk)*N],DataCopyExtParams{(uint16_t)rows,(u32)N,0,(u32)(Np-((N+31)/32)*32)/32,0},DataCopyPadExtParams<int8_t>{true,0,0,0});SetFlag<HardEvent::MTE2_MTE3>(pm);WaitFlag<HardEvent::MTE2_MTE3>(pm);
}else{
auto zm=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(zm);WaitFlag<HardEvent::V_MTE3>(zm);
}
DataCopy(bp[((i64)e*Kp+kk)*Np],bytes,targetRows*Np);SetFlag<HardEvent::MTE3_V>(pr);WaitFlag<HardEvent::MTE3_V>(pr);
}
PipeBarrier<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=GetSubBlockIdx()*32,valid=mr-off;if(valid>32)valid=32;for(i32 slab=0;slab<valid;slab+=16){
i32 sr=valid-slab<16?valid-slab:16;ShortMid_StableConsume(pipe,c[bank*64*tileCols+(off+slab)*pitch],scale,out,cint,z,u,t,f,sc,h,start+row+off+slab,col,e,sr,nr,N);
}
CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);++sequence;
}
tileBase+=count;start=end;
}}
PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void ShortMid_PortableGelu(LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>scratch,LocalTensor<half>h,i32 count){
Abs(u,z,count);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,count);PipeBarrier<PIPE_V>();Mul(t,u,u,count);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(u,u,t,count);PipeBarrier<PIPE_V>();Muls(u,u,-0.024933892525089545f,count);PipeBarrier<PIPE_V>();Duplicate(t,0.0013888888888888889f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.0083333333333333332f,count);
PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.041666666666666664f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.16666666666666666f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.5f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);
PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Muls(u,t,-2.0f,count);PipeBarrier<PIPE_V>();Adds(u,u,1.0f,count);PipeBarrier<PIPE_V>();Muls(u,u,0.3333333333333333333f,count);PipeBarrier<PIPE_V>();Duplicate(f,1.0f,count);PipeBarrier<PIPE_V>();for(i32 j=0;j<10;++j){
Mul(f,f,u,count);PipeBarrier<PIPE_V>();Adds(f,f,1.0f,count);PipeBarrier<PIPE_V>();
}
Muls(f,f,0.6666666666666666667f,count);PipeBarrier<PIPE_V>();Mul(scratch,z,t,count);PipeBarrier<PIPE_V>();auto sign=h.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,count);PipeBarrier<PIPE_V>();Select(scratch,sign,scratch,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();Mul(f,f,scratch,count);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
}
__aicore__ inline void ShortMid_PortableConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,i64 globalRow,i32 col,i32 e,i32 mr,i32 nr,i32 N,i32 pitch){
i32 count=mr*pitch;if(nr<pitch){
Duplicate(sc,0.0f,pitch);auto zero=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(zero);WaitFlag<HardEvent::V_MTE2>(zero);
}
DataCopy(cint,c,16*pitch);DataCopyPad(sc,scale[(i64)e*N+col],DataCopyExtParams{1,(u32)nr*4,0,0,0},DataCopyPadExtParams<float>{true,0,0,0});auto consumed=pipe.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(consumed);WaitFlag<HardEvent::MTE2_S>(consumed);auto iv=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(iv);WaitFlag<HardEvent::MTE2_V>(iv);Cast(z,cint,RoundMode::CAST_RINT,16*pitch);PipeBarrier<PIPE_V>();for(i32 row=0;row<mr;++row)Mul(z[row*pitch],z[row*pitch],sc,pitch);
PipeBarrier<PIPE_V>();ShortMid_PortableGelu(z,u,t,f,cint.ReinterpretCast<float>(),h,count);auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 row=0;row<mr;++row)DataCopyPad(out[(globalRow+row)*N+col],h[row*pitch],DataCopyExtParams{1,(u32)nr*2,0,0,0});auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}
__global__ __aicore__ __mix__(1,2)void ShortMid_Legacy64(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortMid_A,ShortMid_B,ShortMid_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*1024);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,4096);pipe.InitBuffer(zz,4096);pipe.InitBuffer(uu,4096);pipe.InitBuffer(tt,4096);pipe.InitBuffer(ff,4096);pipe.InitBuffer(ss,256);pipe.InitBuffer(yy,2048);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+63)/64;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*64;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<64?N-col:64;mm.SetOrgShape(M,N,K,K,64);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortMid_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,64);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortMid_Legacy128(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortMid_A,ShortMid_B,ShortMid_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2048);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,8192);pipe.InitBuffer(zz,8192);pipe.InitBuffer(uu,8192);pipe.InitBuffer(tt,8192);pipe.InitBuffer(ff,8192);pipe.InitBuffer(ss,512);pipe.InitBuffer(yy,4096);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+127)/128;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*128;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<128?N-col:128;mm.SetOrgShape(M,N,K,K,128);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortMid_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,128);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortMid_Legacy256(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 enableFastPath){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortMid_A,ShortMid_B,ShortMid_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*4096);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=N/256+((N%256)+127)/128;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)<N/256?(tile%nc)*256:(N/256)*256+((tile%nc)-N/256)*128;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<((tile%nc)<N/256?256:128)?N-col:((tile%nc)<N/256?256:128);mm.SetOrgShape(M,N,K,K,nr<=64?64:nr<=128?128:256);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);
mm.IterateAll(c);mm.End();ShortMid_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,nr<=64?64:nr<=128?128:256);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void B1101ShortMidPair64(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){
InitSocState();SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;GlobalTensor<int8_t>a,b,ap,bp;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ap.SetGlobalBuffer((__gm__ int8_t*)x);bp.SetGlobalBuffer((__gm__ int8_t*)w);ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
const i32 pairId=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);MatmulImpl<ShortMid_A,ShortMid_B,ShortMid_C>mm;mm.Init(&tiling,&pipe);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);mm.SetOrgShape(M,N,K,K,pitch);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(ap[(start+row)*K]);mm.SetTensorB(bp[(i64)e*K*N+col]);mm.IterateAll(c[bank*64*tileCols]);
mm.End();CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);
}
if ASCEND_IS_AIV{
const i32 id=GetBlockIdx(),workers=GetBlockNum()*2;const i32 pairId=id/2,pairs=GetBlockNum();GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)s);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);
pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;
pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=GetSubBlockIdx()*32,valid=mr-off;if(valid>32)valid=32;for(i32 slab=0;slab<valid;slab+=16){
i32 sr=valid-slab<16?valid-slab:16;ShortMid_StableConsume(pipe,c[bank*64*tileCols+(off+slab)*pitch],scale,out,cint,z,u,t,f,sc,h,start+row+off+slab,col,e,sr,nr,N);
}
CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);++sequence;
}
tileBase+=count;start=end;
}}
PipeBarrier<PIPE_ALL>();
}
template<int Width=256>__aicore__ inline void B17ShortMidPost(GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,i64 globalRow,i32 col,i32 mr,i32 N,uint8_t freeIn,uint8_t freeOut,uint8_t vo){
const i32 pitch=Width,nr=Width,count=mr*Width;Cast(z,cint,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();if(mr>4){
for(i32 q=0;q<Width/64;++q)Mul(z[q*64],z[q*64],sc[q*64],(u64)64,(uint8_t)mr,BinaryRepeatParams{1,1,1,(uint8_t)(Width/8),(uint8_t)(Width/8),0});
}else for(i32 r=0;r<mr;++r)Mul(z[r*pitch],z[r*pitch],sc,pitch);PipeBarrier<PIPE_V>();Abs(u,z,count);PipeBarrier<PIPE_V>();Mul(t,u,u,count);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(u,u,t,count);PipeBarrier<PIPE_V>();Muls(u,u,-1.5957691216057308f,count);PipeBarrier<PIPE_V>();Exp(t,u,count);PipeBarrier<PIPE_V>();Mul(f,z,t,count);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();
CompareScalar(sign,z,0.0f,CMPMODE::LT,count);PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Div(f,f,t,count);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(freeIn);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 r=0;r<mr;++r)DataCopyPad(out[(globalRow+r)*N+col],h[r*Width],DataCopyExtParams{1,(u32)Width*2,0,0,0});
SetFlag<HardEvent::MTE3_V>(freeOut);
}
template<bool BPanelLayout=false>__aicore__ inline void B20ShortMidL0(LocalTensor<int8_t>a1,LocalTensor<int8_t>b1,LocalTensor<int8_t>a2,LocalTensor<int8_t>b2,i32 kk,i32 K,uint8_t free,uint8_t ready){
WaitFlag<HardEvent::M_MTE1>(free);for(i32 i=0;i<4;++i)LoadData(a2[i*512],a1[(kk/32+i)*64*32],LoadData2DParams{0,4,1,0,3,false,0});if constexpr(BPanelLayout)LoadDataWithTranspose(b2,b1[kk*256],LoadData2dTransposeParams{0,32,1,1,0,inc});else for(i32 i=0;i<8;++i)LoadDataWithTranspose(b2[i*1024],b1[i*K*32+kk*32],LoadData2dTransposeParams{0,4,1,15,0,inc});
SetFlag<HardEvent::MTE1_M>(ready);
}
template<bool Native,bool Columns=false,bool BPanelLayout=false>__aicore__ inline void B20ShortMidNativeBody(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){
static_assert(!BPanelLayout||Native);InitSocState();SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;GlobalTensor<int8_t>a,b,ap,bp;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ap.SetGlobalBuffer((__gm__ int8_t*)x);bp.SetGlobalBuffer((__gm__ int8_t*)w);ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
if constexpr(Native){
const i32 pair=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pair*128*tileCols);TBuf<TPosition::A1>aa;TBuf<TPosition::B1>bb;TBuf<TPosition::A2>ab;TBuf<TPosition::B2>ba;TBuf<TPosition::CO1>cc;pipe.InitBuffer(aa,64*K);pipe.InitBuffer(bb,256*K);pipe.InitBuffer(ab,16384);pipe.InitBuffer(ba,65536);pipe.InitBuffer(cc,65536);auto a1=aa.Get<int8_t>(),b1=bb.Get<int8_t>(),a2=ab.Get<int8_t>(),b2=ba.Get<int8_t>();
auto c0=cc.Get<i32>();uint8_t freeL0[2],readyL0[2];for(i32 j=0;j<2;++j){
freeL0[j]=(uint8_t)pipe.AllocEventID<HardEvent::M_MTE1>();readyL0[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE1_M>();SetFlag<HardEvent::M_MTE1>(freeL0[j]);
}
auto l1ready=pipe.FetchEventID(HardEvent::MTE2_MTE1),l1free=pipe.FetchEventID(HardEvent::MTE1_MTE2);auto cready=pipe.FetchEventID(HardEvent::M_FIX),cfree=pipe.FetchEventID(HardEvent::FIX_M);SetFlag<HardEvent::MTE1_MTE2>(l1free);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
const i64 end=ends.GetValue(e);const i32 rows=end-start,nc=N/256;const i64 count=((rows+63)/64)*nc;for(i64 tile=(pair-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
const i32 row=tile/nc*64,col=tile%nc*256,mr=rows-row<64?rows-row:64,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);WaitFlag<HardEvent::MTE1_MTE2>(l1free);DataCopy(a1,a[(start+row)*K],Nd2NzParams{1,(uint16_t)mr,(uint16_t)K,0,(uint16_t)K,64,1,0});if constexpr(BPanelLayout)DataCopy(b1,b[(i64)e*K*N+col],Nd2NzParams{(uint16_t)(K/32),32,256,(uint16_t)(32*N),(uint16_t)N,32,1,8192});
else DataCopy(b1,b[(i64)e*K*N+col],Nd2NzParams{1,(uint16_t)K,256,0,(uint16_t)N,(uint16_t)K,1,0});SetFlag<HardEvent::MTE2_MTE1>(l1ready);WaitFlag<HardEvent::MTE2_MTE1>(l1ready);B20ShortMidL0<BPanelLayout>(a1,b1,a2,b2,0,K,freeL0[0],readyL0[0]);for(i32 ki=0;ki<K/128;++ki){
const i32 slot=ki&1;if(ki+1<K/128){
const i32 next=slot^1;B20ShortMidL0<BPanelLayout>(a1,b1,a2[next*8192],b2[next*32768],(ki+1)*128,K,freeL0[next],readyL0[next]);
}
WaitFlag<HardEvent::MTE1_M>(readyL0[slot]);Mmad(c0,a2[slot*8192],b2[slot*32768],MmadParams{64,256,128,0,false,ki==0});SetFlag<HardEvent::M_MTE1>(freeL0[slot]);
}
SetFlag<HardEvent::MTE1_MTE2>(l1free);SetFlag<HardEvent::M_FIX>(cready);WaitFlag<HardEvent::M_FIX>(cready);Fixpipe(c[bank*64*tileCols],c0,FixpipeParamsV220{256,(uint16_t)mr,64,256,false,QuantMode_t::NoQuant,0,1,0,0,0});SetFlag<HardEvent::FIX_M>(cfree);WaitFlag<HardEvent::FIX_M>(cfree);CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);WaitFlag<HardEvent::MTE1_MTE2>(l1free);for(i32 j=0;j<2;++j){
WaitFlag<HardEvent::M_MTE1>(freeL0[j]);pipe.ReleaseEventID<HardEvent::M_MTE1>(freeL0[j]);pipe.ReleaseEventID<HardEvent::MTE1_M>(readyL0[j]);
}}else{
const i32 pairId=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);MatmulImpl<ShortMid_A,ShortMid_B,ShortMid_C>mm;mm.Init(&tiling,&pipe);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);mm.SetOrgShape(M,N,K,K,pitch);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(ap[(start+row)*K]);mm.SetTensorB(bp[(i64)e*K*N+col]);mm.IterateAll(c[bank*64*tileCols]);
mm.End();CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);
}}
if ASCEND_IS_AIV{
const i32 id=GetBlockIdx(),workers=GetBlockNum()*2;const i32 pairId=id/2,pairs=GetBlockNum();GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)s);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,32768);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);
pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,16384);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;
pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();uint8_t freeIn[2],ready[2],freeOut[2],vo[2];for(i32 j=0;j<2;++j){
freeIn[j]=(uint8_t)pipe.AllocEventID<HardEvent::V_MTE2>();ready[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE2_V>();freeOut[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE3_V>();vo[j]=(uint8_t)pipe.AllocEventID<HardEvent::V_MTE3>();SetFlag<HardEvent::V_MTE2>(freeIn[j]);SetFlag<HardEvent::MTE3_V>(freeOut[j]);
}
i64 fragment=0,savedRow=0;i32 savedCol=0,savedRows=0,savedBank=0;bool pending=false;CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=Columns?0:GetSubBlockIdx()*32,valid=mr-off;if(!Columns&&valid>32)valid=32;const i32 co=Columns?GetSubBlockIdx()*128:0,width=Columns?128:256;
for(i32 slab=0;slab<valid;slab+=16){
const i32 sr=valid-slab<16?valid-slab:16,slot=fragment&1;WaitFlag<HardEvent::V_MTE2>(freeIn[slot]);if constexpr(Columns)DataCopyPad(cint[slot*4096],c[bank*64*tileCols+slab*256+co],DataCopyExtParams{(uint16_t)sr,512,512,0,0},DataCopyPadExtParams<i32>{false,0,0,0});else DataCopy(cint[slot*4096],c[bank*64*tileCols+(off+slab)*256],sr*256);if(slab+16>=valid)CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);
DataCopy(sc[slot*256],scale[(i64)e*N+col+co],width);SetFlag<HardEvent::MTE2_V>(ready[slot]);if(pending){
WaitFlag<HardEvent::MTE2_V>(ready[savedBank]);WaitFlag<HardEvent::MTE3_V>(freeOut[savedBank]);B17ShortMidPost<Columns?128:256>(out,cint[savedBank*4096],z,u,t,f,sc[savedBank*256],h[savedBank*4096],savedRow,savedCol,savedRows,N,freeIn[savedBank],freeOut[savedBank],vo[savedBank]);
}
savedRow=start+row+off+slab;savedCol=col+co;savedRows=sr;savedBank=slot;pending=true;++fragment;
}
if(valid<=0)CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
if(pending){
WaitFlag<HardEvent::MTE2_V>(ready[savedBank]);WaitFlag<HardEvent::MTE3_V>(freeOut[savedBank]);B17ShortMidPost<Columns?128:256>(out,cint[savedBank*4096],z,u,t,f,sc[savedBank*256],h[savedBank*4096],savedRow,savedCol,savedRows,N,freeIn[savedBank],freeOut[savedBank],vo[savedBank]);
}
for(i32 j=0;j<2;++j){
WaitFlag<HardEvent::V_MTE2>(freeIn[j]);WaitFlag<HardEvent::MTE3_V>(freeOut[j]);pipe.ReleaseEventID<HardEvent::V_MTE2>(freeIn[j]);pipe.ReleaseEventID<HardEvent::MTE2_V>(ready[j]);pipe.ReleaseEventID<HardEvent::MTE3_V>(freeOut[j]);pipe.ReleaseEventID<HardEvent::V_MTE3>(vo[j]);
}}
PipeBarrier<PIPE_ALL>();
}
__global__ __aicore__ __mix__(1,2)void B1702ShortMidPipeline(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){B20ShortMidNativeBody<false>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols);}
__global__ __aicore__ __mix__(1,2)void B2002ShortMidRepeatScale(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){B20ShortMidNativeBody<true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols);}
__global__ __aicore__ __mix__(1,2)void B2004ShortMidColumns(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){B20ShortMidNativeBody<true,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols);}
__aicore__ inline void ShortMidRollingLoadB(LocalTensor<int8_t>dst,GlobalTensor<int8_t>src,i64 base,i32 K,i32 N,i32 panel){
const i32 kk=panel*256,kr=K-kk<256?K-kk:256;Nd2NzParams p{};p.ndNum=(uint16_t)(kr/32);p.nValue=32;p.dValue=256;p.srcNdMatrixStride=(uint16_t)(32*N);p.srcDValue=(uint16_t)N;p.dstNzC0Stride=32;p.dstNzNStride=1;p.dstNzMatrixStride=8192;DataCopy(dst,src[base+(i64)kk*N],p);
}
__aicore__ inline void ShortMidRollingLoadPhase(LocalTensor<int8_t>a1,LocalTensor<int8_t>b1,LocalTensor<int8_t>a2,LocalTensor<int8_t>b2,GlobalTensor<int8_t>b,i64 bbase,i32 K,i32 N,i32 ki,i32 chunks,i32 panels,uint8_t freeL0,uint8_t readyL0,uint8_t bReady,uint8_t bFree,uint8_t aFree){
const i32 panel=ki/2,bank=panel%3,part=ki&1,kk=ki*128;WaitFlag<HardEvent::M_MTE1>(freeL0);if(!part)WaitFlag<HardEvent::MTE2_MTE1>(bReady);for(i32 i=0;i<4;++i)LoadData(a2[i*512],a1[(kk/32+i)*64*32],LoadData2DParams{0,4,1,0,3,false,0});LoadDataWithTranspose(b2,b1[bank*65536+part*32768],LoadData2dTransposeParams{0,32,1,1,0,inc});SetFlag<HardEvent::MTE1_M>(readyL0);
if(part||ki+1==chunks){
SetFlag<HardEvent::MTE1_MTE2>(bFree);if(panel+3<panels){
WaitFlag<HardEvent::MTE1_MTE2>(bFree);ShortMidRollingLoadB(b1[bank*65536],b,bbase,K,N,panel+3);SetFlag<HardEvent::MTE2_MTE1>(bReady);
}}
if(ki+1==chunks)SetFlag<HardEvent::MTE1_MTE2>(aFree);
}
template<bool Native,bool Columns=false,bool BPanelLayout=false>__aicore__ inline void ShortMidRollingBBody(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){
static_assert(!BPanelLayout||Native);InitSocState();SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;GlobalTensor<int8_t>a,b,ap,bp;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ap.SetGlobalBuffer((__gm__ int8_t*)x);bp.SetGlobalBuffer((__gm__ int8_t*)w);ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
const i32 pair=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pair*128*tileCols);TBuf<TPosition::A1>aa;TBuf<TPosition::B1>bb;TBuf<TPosition::A2>ab;TBuf<TPosition::B2>ba;TBuf<TPosition::CO1>cc;pipe.InitBuffer(aa,64*K);pipe.InitBuffer(bb,3*65536);pipe.InitBuffer(ab,16384);pipe.InitBuffer(ba,65536);pipe.InitBuffer(cc,65536);auto a1=aa.Get<int8_t>(),b1=bb.Get<int8_t>(),a2=ab.Get<int8_t>(),b2=ba.Get<int8_t>();
auto c0=cc.Get<i32>();uint8_t freeL0[2],readyL0[2],bReady[3],bFree[3];for(i32 j=0;j<2;++j){
freeL0[j]=(uint8_t)pipe.AllocEventID<HardEvent::M_MTE1>();readyL0[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE1_M>();SetFlag<HardEvent::M_MTE1>(freeL0[j]);
}
const auto aReady=pipe.AllocEventID<HardEvent::MTE2_MTE1>();const auto aFree=pipe.AllocEventID<HardEvent::MTE1_MTE2>();for(i32 j=0;j<3;++j){
bReady[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE2_MTE1>();bFree[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE1_MTE2>();SetFlag<HardEvent::MTE1_MTE2>(bFree[j]);
}
const auto cReady=pipe.AllocEventID<HardEvent::M_FIX>();const auto cFree=pipe.AllocEventID<HardEvent::FIX_M>();SetFlag<HardEvent::MTE1_MTE2>(aFree);const i32 chunks=K/128,panels=(chunks+1)/2;i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
const i64 end=ends.GetValue(e);const i32 rows=end-start,nc=N/256;const i64 count=((rows+63)/64)*nc;for(i64 tile=(pair-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
const i32 row=tile/nc*64,col=tile%nc*256,mr=rows-row<64?rows-row:64,bank=sequence&1;const i64 bbase=(i64)e*K*N+col;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);WaitFlag<HardEvent::MTE1_MTE2>(aFree);if(mr<64){
InitConstValueParams<u32>pad{};pad.repeatTimes=(uint16_t)(K/32);pad.blockNum=(uint16_t)(64-mr);pad.dstGap=(uint16_t)mr;pad.initValue=0;InitConstValue(a1[mr*32].ReinterpretCast<u32>(),pad);
}
DataCopy(a1,a[(start+row)*K],Nd2NzParams{1,(uint16_t)mr,(uint16_t)K,0,(uint16_t)K,64,1,0});SetFlag<HardEvent::MTE2_MTE1>(aReady);for(i32 initial=0;initial<3&&initial<panels;++initial){
WaitFlag<HardEvent::MTE1_MTE2>(bFree[initial]);ShortMidRollingLoadB(b1[initial*65536],b,bbase,K,N,initial);SetFlag<HardEvent::MTE2_MTE1>(bReady[initial]);
}
WaitFlag<HardEvent::MTE2_MTE1>(aReady);ShortMidRollingLoadPhase(a1,b1,a2,b2,b,bbase,K,N,0,chunks,panels,freeL0[0],readyL0[0],bReady[0],bFree[0],(uint8_t)aFree);for(i32 ki=0;ki<chunks;++ki){
const i32 slot=ki&1;if(ki+1<chunks){
const i32 next=slot^1,nextB=((ki+1)/2)%3;ShortMidRollingLoadPhase(a1,b1,a2[next*8192],b2[next*32768],b,bbase,K,N,ki+1,chunks,panels,freeL0[next],readyL0[next],bReady[nextB],bFree[nextB],(uint8_t)aFree);
}
WaitFlag<HardEvent::MTE1_M>(readyL0[slot]);Mmad(c0,a2[slot*8192],b2[slot*32768],MmadParams{64,256,128,0,false,ki==0});SetFlag<HardEvent::M_MTE1>(freeL0[slot]);
}
SetFlag<HardEvent::M_FIX>(cReady);WaitFlag<HardEvent::M_FIX>(cReady);Fixpipe(c[bank*64*tileCols],c0,FixpipeParamsV220{256,(uint16_t)mr,64,256,false,QuantMode_t::NoQuant,0,1,0,0,0});SetFlag<HardEvent::FIX_M>(cFree);WaitFlag<HardEvent::FIX_M>(cFree);CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);WaitFlag<HardEvent::MTE1_MTE2>(aFree);for(i32 j=0;j<3;++j){
WaitFlag<HardEvent::MTE1_MTE2>(bFree[j]);pipe.ReleaseEventID<HardEvent::MTE2_MTE1>(bReady[j]);pipe.ReleaseEventID<HardEvent::MTE1_MTE2>(bFree[j]);
}
for(i32 j=0;j<2;++j){
WaitFlag<HardEvent::M_MTE1>(freeL0[j]);pipe.ReleaseEventID<HardEvent::M_MTE1>(freeL0[j]);pipe.ReleaseEventID<HardEvent::MTE1_M>(readyL0[j]);
}
pipe.ReleaseEventID<HardEvent::MTE2_MTE1>(aReady);pipe.ReleaseEventID<HardEvent::MTE1_MTE2>(aFree);pipe.ReleaseEventID<HardEvent::M_FIX>(cReady);pipe.ReleaseEventID<HardEvent::FIX_M>(cFree);
}
if ASCEND_IS_AIV{
const i32 id=GetBlockIdx(),workers=GetBlockNum()*2;const i32 pairId=id/2,pairs=GetBlockNum();GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)s);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,32768);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);
pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,16384);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;
pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();uint8_t freeIn[2],ready[2],freeOut[2],vo[2];for(i32 j=0;j<2;++j){
freeIn[j]=(uint8_t)pipe.AllocEventID<HardEvent::V_MTE2>();ready[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE2_V>();freeOut[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE3_V>();vo[j]=(uint8_t)pipe.AllocEventID<HardEvent::V_MTE3>();SetFlag<HardEvent::V_MTE2>(freeIn[j]);SetFlag<HardEvent::MTE3_V>(freeOut[j]);
}
i64 fragment=0,savedRow=0;i32 savedCol=0,savedRows=0,savedBank=0;bool pending=false;CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=Columns?0:GetSubBlockIdx()*32,valid=mr-off;if(!Columns&&valid>32)valid=32;const i32 co=Columns?GetSubBlockIdx()*128:0,width=Columns?128:256;
for(i32 slab=0;slab<valid;slab+=16){
const i32 sr=valid-slab<16?valid-slab:16,slot=fragment&1;WaitFlag<HardEvent::V_MTE2>(freeIn[slot]);if constexpr(Columns)DataCopyPad(cint[slot*4096],c[bank*64*tileCols+slab*256+co],DataCopyExtParams{(uint16_t)sr,512,512,0,0},DataCopyPadExtParams<i32>{false,0,0,0});else DataCopy(cint[slot*4096],c[bank*64*tileCols+(off+slab)*256],sr*256);if(slab+16>=valid)CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);
DataCopy(sc[slot*256],scale[(i64)e*N+col+co],width);SetFlag<HardEvent::MTE2_V>(ready[slot]);if(pending){
WaitFlag<HardEvent::MTE2_V>(ready[savedBank]);WaitFlag<HardEvent::MTE3_V>(freeOut[savedBank]);B17ShortMidPost<Columns?128:256>(out,cint[savedBank*4096],z,u,t,f,sc[savedBank*256],h[savedBank*4096],savedRow,savedCol,savedRows,N,freeIn[savedBank],freeOut[savedBank],vo[savedBank]);
}
savedRow=start+row+off+slab;savedCol=col+co;savedRows=sr;savedBank=slot;pending=true;++fragment;
}
if(valid<=0)CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
if(pending){
WaitFlag<HardEvent::MTE2_V>(ready[savedBank]);WaitFlag<HardEvent::MTE3_V>(freeOut[savedBank]);B17ShortMidPost<Columns?128:256>(out,cint[savedBank*4096],z,u,t,f,sc[savedBank*256],h[savedBank*4096],savedRow,savedCol,savedRows,N,freeIn[savedBank],freeOut[savedBank],vo[savedBank]);
}
for(i32 j=0;j<2;++j){
WaitFlag<HardEvent::V_MTE2>(freeIn[j]);WaitFlag<HardEvent::MTE3_V>(freeOut[j]);pipe.ReleaseEventID<HardEvent::V_MTE2>(freeIn[j]);pipe.ReleaseEventID<HardEvent::MTE2_V>(ready[j]);pipe.ReleaseEventID<HardEvent::MTE3_V>(freeOut[j]);pipe.ReleaseEventID<HardEvent::V_MTE3>(vo[j]);
}}
PipeBarrier<PIPE_ALL>();
}
__global__ __aicore__ __mix__(1,2)void B2230ShortMidBPanelLayout(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){ShortMidRollingBBody<true,true,true>(x,w,s,groups,y,workspace,scratch,tiling,M,K,N,E,tileCols);}
static void ShortMid_Run(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());
if(!blocks)throw std::runtime_error("no Cube cores");u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);const i64 lowerTiles=((i64)M+15)/16*((N+127)/128);const bool use128=K<=1024&&N>=128&&N<=1024&&lowerTiles>=(i64)blocks*8&&ubBytes>=65536&&(K<=64||lowerTiles>=(i64)blocks*32);const i32 nBlocks256=N/256+((N%256)+127)/128;
const i64 lowerTiles256=((i64)M+15)/16*nBlocks256;i32 columnCycle=N/256;i32 workerCycle=blocks*2;while(workerCycle!=0){
i32 remainder=columnCycle%workerCycle;columnCycle=workerCycle;workerCycle=remainder;
}
const bool dense256Cycle=N%256==0&&columnCycle==1;const bool use256=K<=1024&&N>=256&&N<=1024&&ubBytes>=131072&&lowerTiles256>=(i64)blocks*8&&(K<=64||nBlocks256==1||lowerTiles256>=(i64)blocks*32||dense256Cycle);i32 tileCols=use256?256:use128?128:64;auto makeTiling=[&](i32 cols,TCubeTiling&result){
matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);mt.SetShape(16,cols,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(16,cols,64);mt.SetBufferSpace(-1,-1,-1);
return mt.GetTiling(result);
};TCubeTiling tiling;i32 tilingStatus=makeTiling(tileCols,tiling);if(tilingStatus!=0&&tileCols==256&&use128){tileCols=128;tilingStatus=makeTiling(tileCols,tiling);}
if(tilingStatus!=0&&tileCols!=64){tileCols=64;tilingStatus=makeTiling(tileCols,tiling);}
if(tilingStatus!=0)throw std::runtime_error("INT8 tiling failed");const i32 enableFastPath=0;const i32 Kp=(K+63)/64*64,Np=(N+255)/256*256;const i64 stagedBytes=(i64)M*Kp+(i64)E*Kp*Np;const bool useStage=(K%32!=0||N%32!=0)&&stagedBytes<=134217728LL;TCubeTiling pairTiling;bool pairReady=false;u64 l0a=0,l0b=0,l0c=0;
const char*pairSoc=aclrtGetSocName();if(K%32==0&&N%32==0&&ubBytes>=131072&&pairSoc&&std::strcmp(pairSoc,"Ascend910_9362")==0&&lowerTiles256>=(i64)platform->GetCoreNumAic()*8){
matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);mt.SetShape(64,tileCols,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(64,tileCols,64);mt.SetBufferSpace(-1,-1,-1);
platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,l0a);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);if(mt.GetTiling(pairTiling)==0)pairReady=pairTiling.baseM==64&&pairTiling.baseN==tileCols&&2LL*64*pairTiling.baseK<=l0a&&2LL*tileCols*pairTiling.baseK<=l0b&&64LL*tileCols*4<=l0c;
}
u64 nativeL1=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1,nativeL1);const bool nativeReady=pairReady&&tileCols==256&&N%256==0&&K%128==0&&K<=65504&&320ULL*K<=nativeL1&&l0a>=16384&&l0b>=65536&&l0c>=65536&&ubBytes>=138240;void*staging=nullptr;void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();
auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*((useStage||pairReady)?64:16)*tileCols*4*(enableFastPath?2:1),ACL_MEM_MALLOC_HUGE_FIRST));const char*b505soc=aclrtGetSocName();if(b505soc&&std::strcmp(b505soc,"Ascend910_9362")==0&&ubBytes>=131072){
if(pairReady){
if(nativeReady){
if(K%128==0&&K/32<=4095&&32LL*N<=65535&&K/32<=255&&nativeL1>=64ULL*K+3*65536ULL)B2230ShortMidBPanelLayout<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,pairTiling,M,K,N,E,tileCols);else B2004ShortMidColumns<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,pairTiling,M,K,N,E,tileCols);
}else if(tileCols==256&&N%256==0&&ubBytes>=138240)B1702ShortMidPipeline<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,pairTiling,M,K,N,E,tileCols);else B1101ShortMidPair64<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,pairTiling,M,K,N,E,tileCols);
}else if(useStage){
matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);mt.SetShape(64,tileCols,Kp);mt.SetOrgShape(M,Np,Kp);mt.SetFixSplit(16,tileCols,64);mt.SetBufferSpace(-1,-1,-1);
TCubeTiling packedTiling;if(mt.GetTiling(packedTiling)==0){
checkGmm(aclrtMalloc(&staging,stagedBytes,ACL_MEM_MALLOC_HUGE_FIRST));ShortMid_StageKernel<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,packedTiling,M,K,N,E,tileCols,(GM_ADDR)staging,Kp,Np);
}else{
ShortMid_VectorRequest<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileCols);
}}else{
ShortMid_VectorRequest<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileCols);
}}else{
if(tileCols==256){ShortMid_Legacy256<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,0);}else if(tileCols==128){ShortMid_Legacy128<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E);}else{ShortMid_Legacy64<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E);}}
aclError status=aclrtSynchronizeStream(stream);if(staging)aclrtFree(staging);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
using ShortMany_A=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using ShortMany_B=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using ShortMany_C=MatmulType<TPosition::GM,CubeFormat::ND,i32>;
__aicore__ inline void ShortMany_StableConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,i64 globalRow,i32 col,i32 e,i32 mr,i32 nr,i32 N){
const i32 pitch=nr<=64?64:nr<=128?128:256;const i32 count=mr*pitch;if(nr<pitch){
Duplicate(sc,0.0f,pitch);auto zero=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(zero);WaitFlag<HardEvent::V_MTE2>(zero);
}
DataCopy(cint,c,16*pitch);DataCopyPad(sc,scale[(i64)e*N+col],DataCopyExtParams{1,(u32)nr*4,0,0,0},DataCopyPadExtParams<float>{true,0,0,0});auto consumed=pipe.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(consumed);WaitFlag<HardEvent::MTE2_S>(consumed);auto iv=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(iv);WaitFlag<HardEvent::MTE2_V>(iv);Cast(z,cint,RoundMode::CAST_RINT,16*pitch);PipeBarrier<PIPE_V>();for(i32 r=0;r<mr;++r)Mul(z[r*pitch],z[r*pitch],sc,pitch);
PipeBarrier<PIPE_V>();Abs(u,z,count);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,count);PipeBarrier<PIPE_V>();Mul(t,u,u,count);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(u,u,t,count);PipeBarrier<PIPE_V>();Muls(u,u,-1.5957691216057308f,count);PipeBarrier<PIPE_V>();Exp(t,u,count);PipeBarrier<PIPE_V>();Mul(f,z,t,count);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,count);
PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Div(f,f,t,count);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 r=0;r<mr;++r)DataCopyPad(out[(globalRow+r)*N+col],h[r*pitch],DataCopyExtParams{1,(u32)nr*2,0,0,0});auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);
SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}
__global__ __aicore__ __mix__(1,2)void ShortMany_VectorRequest(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortMany_A,ShortMany_B,ShortMany_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*16*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*tileCols;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<tileCols?N-col:tileCols;mm.SetOrgShape(M,N,K,K,nr<=64?64:nr<=128?128:256);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortMany_StableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortMany_StageKernel(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
InitSocState();SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;GlobalTensor<int8_t>a,b,ap,bp;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ap.SetGlobalBuffer((__gm__ int8_t*)staging);bp.SetGlobalBuffer((__gm__ int8_t*)staging+(i64)M*Kp);ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
const i32 pairId=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);MatmulImpl<ShortMany_A,ShortMany_B,ShortMany_C>mm;mm.Init(&tiling,&pipe);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);mm.SetOrgShape(M,Np,Kp,Kp,pitch);mm.SetSingleShape(mr,nr,Kp);mm.SetTensorA(ap[(start+row)*Kp]);mm.SetTensorB(bp[(i64)e*Kp*Np+col]);mm.IterateAll(c[bank*64*tileCols]);
mm.End();CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);
}
if ASCEND_IS_AIV{
const i32 id=GetBlockIdx(),workers=GetBlockNum()*2;const i32 pairId=id/2,pairs=GetBlockNum();GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)s);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);
pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;
pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();auto pack=ci.Get<u32>();auto bytes=ci.Get<int8_t>();auto pv=pipe.FetchEventID(HardEvent::V_MTE2);
auto pm=pipe.FetchEventID(HardEvent::MTE2_MTE3);auto pr=pipe.FetchEventID(HardEvent::MTE3_V);const i32 xRows=16384/Kp,wRows=16384/Np;for(i32 row=id*xRows;row<M;row+=workers*xRows){
i32 rows=M-row<xRows?M-row:xRows;Duplicate(pack,(u32)0,rows*Kp/4);SetFlag<HardEvent::V_MTE2>(pv);WaitFlag<HardEvent::V_MTE2>(pv);DataCopyPad(bytes,a[(i64)row*K],DataCopyExtParams{(uint16_t)rows,(u32)K,0,(u32)(Kp-((K+31)/32)*32)/32,0},DataCopyPadExtParams<int8_t>{true,0,0,0});SetFlag<HardEvent::MTE2_MTE3>(pm);WaitFlag<HardEvent::MTE2_MTE3>(pm);DataCopy(ap[(i64)row*Kp],bytes,rows*Kp);SetFlag<HardEvent::MTE3_V>(pr);
WaitFlag<HardEvent::MTE3_V>(pr);
}
i32 chunks=(Kp+wRows-1)/wRows;for(i64 chunk=id;chunk<(i64)E*chunks;chunk+=workers){
i32 e=chunk/chunks,kk=(chunk%chunks)*wRows;i32 targetRows=Kp-kk<wRows?Kp-kk:wRows;i32 rows=K-kk<targetRows?K-kk:targetRows;Duplicate(pack,(u32)0,targetRows*Np/4);SetFlag<HardEvent::V_MTE2>(pv);WaitFlag<HardEvent::V_MTE2>(pv);if(rows>0){
DataCopyPad(bytes,b[((i64)e*K+kk)*N],DataCopyExtParams{(uint16_t)rows,(u32)N,0,(u32)(Np-((N+31)/32)*32)/32,0},DataCopyPadExtParams<int8_t>{true,0,0,0});SetFlag<HardEvent::MTE2_MTE3>(pm);WaitFlag<HardEvent::MTE2_MTE3>(pm);
}else{
auto zm=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(zm);WaitFlag<HardEvent::V_MTE3>(zm);
}
DataCopy(bp[((i64)e*Kp+kk)*Np],bytes,targetRows*Np);SetFlag<HardEvent::MTE3_V>(pr);WaitFlag<HardEvent::MTE3_V>(pr);
}
PipeBarrier<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=GetSubBlockIdx()*32,valid=mr-off;if(valid>32)valid=32;for(i32 slab=0;slab<valid;slab+=16){
i32 sr=valid-slab<16?valid-slab:16;ShortMany_StableConsume(pipe,c[bank*64*tileCols+(off+slab)*pitch],scale,out,cint,z,u,t,f,sc,h,start+row+off+slab,col,e,sr,nr,N);
}
CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);++sequence;
}
tileBase+=count;start=end;
}}
PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void ShortMany_PortableGelu(LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>scratch,LocalTensor<half>h,i32 count){
Abs(u,z,count);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,count);PipeBarrier<PIPE_V>();Mul(t,u,u,count);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(u,u,t,count);PipeBarrier<PIPE_V>();Muls(u,u,-0.024933892525089545f,count);PipeBarrier<PIPE_V>();Duplicate(t,0.0013888888888888889f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.0083333333333333332f,count);
PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.041666666666666664f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.16666666666666666f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,0.5f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);
PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,t,count);PipeBarrier<PIPE_V>();Muls(u,t,-2.0f,count);PipeBarrier<PIPE_V>();Adds(u,u,1.0f,count);PipeBarrier<PIPE_V>();Muls(u,u,0.3333333333333333333f,count);PipeBarrier<PIPE_V>();Duplicate(f,1.0f,count);PipeBarrier<PIPE_V>();for(i32 j=0;j<10;++j){
Mul(f,f,u,count);PipeBarrier<PIPE_V>();Adds(f,f,1.0f,count);PipeBarrier<PIPE_V>();
}
Muls(f,f,0.6666666666666666667f,count);PipeBarrier<PIPE_V>();Mul(scratch,z,t,count);PipeBarrier<PIPE_V>();auto sign=h.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,count);PipeBarrier<PIPE_V>();Select(scratch,sign,scratch,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();Mul(f,f,scratch,count);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
}
__aicore__ inline void ShortMany_PortableConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,i64 globalRow,i32 col,i32 e,i32 mr,i32 nr,i32 N,i32 pitch){
i32 count=mr*pitch;if(nr<pitch){
Duplicate(sc,0.0f,pitch);auto zero=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(zero);WaitFlag<HardEvent::V_MTE2>(zero);
}
DataCopy(cint,c,16*pitch);DataCopyPad(sc,scale[(i64)e*N+col],DataCopyExtParams{1,(u32)nr*4,0,0,0},DataCopyPadExtParams<float>{true,0,0,0});auto consumed=pipe.FetchEventID(HardEvent::MTE2_S);SetFlag<HardEvent::MTE2_S>(consumed);WaitFlag<HardEvent::MTE2_S>(consumed);auto iv=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(iv);WaitFlag<HardEvent::MTE2_V>(iv);Cast(z,cint,RoundMode::CAST_RINT,16*pitch);PipeBarrier<PIPE_V>();for(i32 row=0;row<mr;++row)Mul(z[row*pitch],z[row*pitch],sc,pitch);
PipeBarrier<PIPE_V>();ShortMany_PortableGelu(z,u,t,f,cint.ReinterpretCast<float>(),h,count);auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 row=0;row<mr;++row)DataCopyPad(out[(globalRow+row)*N+col],h[row*pitch],DataCopyExtParams{1,(u32)nr*2,0,0,0});auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}
__global__ __aicore__ __mix__(1,2)void ShortMany_Legacy64(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortMany_A,ShortMany_B,ShortMany_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*1024);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,4096);pipe.InitBuffer(zz,4096);pipe.InitBuffer(uu,4096);pipe.InitBuffer(tt,4096);pipe.InitBuffer(ff,4096);pipe.InitBuffer(ss,256);pipe.InitBuffer(yy,2048);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+63)/64;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*64;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<64?N-col:64;mm.SetOrgShape(M,N,K,K,64);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortMany_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,64);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortMany_Legacy128(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortMany_A,ShortMany_B,ShortMany_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2048);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,8192);pipe.InitBuffer(zz,8192);pipe.InitBuffer(uu,8192);pipe.InitBuffer(tt,8192);pipe.InitBuffer(ff,8192);pipe.InitBuffer(ss,512);pipe.InitBuffer(yy,4096);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+127)/128;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)*128;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<128?N-col:128;mm.SetOrgShape(M,N,K,K,128);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();ShortMany_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,128);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortMany_Legacy256(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 enableFastPath){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<ShortMany_A,ShortMany_B,ShortMany_C>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*4096);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=N/256+((N%256)+127)/128;i64 count=((rows+15)/16)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*16,col=(tile%nc)<N/256?(tile%nc)*256:(N/256)*256+((tile%nc)-N/256)*128;i32 mr=rows-row<16?rows-row:16;i32 nr=N-col<((tile%nc)<N/256?256:128)?N-col:((tile%nc)<N/256?256:128);mm.SetOrgShape(M,N,K,K,nr<=64?64:nr<=128?128:256);mm.SetSingleShape(mr,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);
mm.IterateAll(c);mm.End();ShortMany_PortableConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,start+row,col,e,mr,nr,N,nr<=64?64:nr<=128?128:256);
}
tileBase+=count;start=end;
}}}
__global__ __aicore__ __mix__(1,2)void ShortMany_AlignedPairs(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols,GM_ADDR staging,i32 Kp,i32 Np){
InitSocState();SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;GlobalTensor<int8_t>a,b,ap,bp;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ap.SetGlobalBuffer((__gm__ int8_t*)(staging?staging:x));bp.SetGlobalBuffer(staging?((__gm__ int8_t*)staging+(i64)M*Kp):(__gm__ int8_t*)w);ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
const i32 pairId=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);MatmulImpl<ShortMany_A,ShortMany_B,ShortMany_C>mm;mm.Init(&tiling,&pipe);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);mm.SetOrgShape(M,Np,Kp,Kp,pitch);mm.SetSingleShape(mr,nr,Kp);mm.SetTensorA(ap[(start+row)*Kp]);mm.SetTensorB(bp[(i64)e*Kp*Np+col]);mm.IterateAll(c[bank*64*tileCols]);
mm.End();CrossCoreSetFlag<2,PIPE_FIX>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);
}
if ASCEND_IS_AIV{
const i32 id=GetBlockIdx(),workers=GetBlockNum()*2;const i32 pairId=id/2,pairs=GetBlockNum();GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)s);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);
pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,8192);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;
pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();if(staging){
auto pack=ci.Get<u32>();auto bytes=ci.Get<int8_t>();auto pv=pipe.FetchEventID(HardEvent::V_MTE2);auto pm=pipe.FetchEventID(HardEvent::MTE2_MTE3);auto pr=pipe.FetchEventID(HardEvent::MTE3_V);const i32 xRows=16384/Kp,wRows=16384/Np;for(i32 row=id*xRows;row<M;row+=workers*xRows){
i32 rows=M-row<xRows?M-row:xRows;Duplicate(pack,(u32)0,rows*Kp/4);SetFlag<HardEvent::V_MTE2>(pv);WaitFlag<HardEvent::V_MTE2>(pv);DataCopyPad(bytes,a[(i64)row*K],DataCopyExtParams{(uint16_t)rows,(u32)K,0,(u32)(Kp-((K+31)/32)*32)/32,0},DataCopyPadExtParams<int8_t>{true,0,0,0});SetFlag<HardEvent::MTE2_MTE3>(pm);WaitFlag<HardEvent::MTE2_MTE3>(pm);DataCopy(ap[(i64)row*Kp],bytes,rows*Kp);SetFlag<HardEvent::MTE3_V>(pr);
WaitFlag<HardEvent::MTE3_V>(pr);
}
i32 chunks=(Kp+wRows-1)/wRows;for(i64 chunk=id;chunk<(i64)E*chunks;chunk+=workers){
i32 e=chunk/chunks,kk=(chunk%chunks)*wRows;i32 targetRows=Kp-kk<wRows?Kp-kk:wRows;i32 rows=K-kk<targetRows?K-kk:targetRows;Duplicate(pack,(u32)0,targetRows*Np/4);SetFlag<HardEvent::V_MTE2>(pv);WaitFlag<HardEvent::V_MTE2>(pv);if(rows>0){
DataCopyPad(bytes,b[((i64)e*K+kk)*N],DataCopyExtParams{(uint16_t)rows,(u32)N,0,(u32)(Np-((N+31)/32)*32)/32,0},DataCopyPadExtParams<int8_t>{true,0,0,0});SetFlag<HardEvent::MTE2_MTE3>(pm);WaitFlag<HardEvent::MTE2_MTE3>(pm);
}else{
auto zm=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(zm);WaitFlag<HardEvent::V_MTE3>(zm);
}
DataCopy(bp[((i64)e*Kp+kk)*Np],bytes,targetRows*Np);SetFlag<HardEvent::MTE3_V>(pr);WaitFlag<HardEvent::MTE3_V>(pr);
}
PipeBarrier<PIPE_ALL>();SyncAll<true>();
}
CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=GetSubBlockIdx()*32,valid=mr-off;if(valid>32)valid=32;for(i32 slab=0;slab<valid;slab+=16){
i32 sr=valid-slab<16?valid-slab:16;ShortMany_StableConsume(pipe,c[bank*64*tileCols+(off+slab)*pitch],scale,out,cint,z,u,t,f,sc,h,start+row+off+slab,col,e,sr,nr,N);
}
CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);++sequence;
}
tileBase+=count;start=end;
}}
PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void B17ShortManyPost(GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,i64 globalRow,i32 col,i32 mr,i32 N,uint8_t freeIn,uint8_t freeOut,uint8_t vo){
const i32 pitch=256,nr=256,count=mr*256;Cast(z,cint,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();if(mr>4){
for(i32 q=0;q<4;++q)Mul(z[q*64],z[q*64],sc[q*64],(u64)64,(uint8_t)mr,BinaryRepeatParams{1,1,1,32,32,0});
}else for(i32 r=0;r<mr;++r)Mul(z[r*pitch],z[r*pitch],sc,pitch);PipeBarrier<PIPE_V>();Abs(u,z,count);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,count);PipeBarrier<PIPE_V>();Mul(t,u,u,count);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Mul(u,u,t,count);PipeBarrier<PIPE_V>();Muls(u,u,-1.5957691216057308f,count);PipeBarrier<PIPE_V>();Exp(t,u,count);PipeBarrier<PIPE_V>();Mul(f,z,t,count);
PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,count);PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(freeIn);Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();Div(f,f,t,count);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 r=0;r<mr;++r)DataCopyPad(out[(globalRow+r)*N+col],h[r*256],DataCopyExtParams{1,512,0,0,0});
SetFlag<HardEvent::MTE3_V>(freeOut);
}
__aicore__ inline void B2541ShortManyLogTailPostImpl(GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,i64 globalRow,i32 col,i32 mr,i32 N,uint8_t freeIn,uint8_t freeOut,uint8_t vo){
const i32 pitch=256,nr=256,count=mr*256;Cast(z,cint,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();if(mr>4){
for(i32 q=0;q<4;++q)Mul(z[q*64],z[q*64],sc[q*64],(u64)64,(uint8_t)mr,BinaryRepeatParams{1,1,1,32,32,0});
}else for(i32 r=0;r<mr;++r)Mul(z[r*pitch],z[r*pitch],sc,pitch);PipeBarrier<PIPE_V>();Abs(u,z,count);Maxs(f,z,0.0f,count);PipeBarrier<PIPE_V>();Muls(t,u,-0.413330078125f,count);PipeBarrier<PIPE_V>();Adds(t,t,-0.7099609375f,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();Adds(t,t,-0.71533203125f,count);PipeBarrier<PIPE_V>();Exp(t,t,count);PipeBarrier<PIPE_V>();Mul(t,t,u,count);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(freeIn);
Sub(f,f,t,count);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 r=0;r<mr;++r)DataCopyPad(out[(globalRow+r)*N+col],h[r*256],DataCopyExtParams{1,512,0,0,0});SetFlag<HardEvent::MTE3_V>(freeOut);
}
__aicore__ inline void ShortManyNativeL0(LocalTensor<int8_t>a1,LocalTensor<int8_t>b1,LocalTensor<int8_t>a2,LocalTensor<int8_t>b2,i32 kk,uint8_t free,uint8_t ready){
WaitFlag<HardEvent::M_MTE1>(free);for(i32 i=0;i<4;++i)LoadData(a2[i*512],a1[(kk/32+i)*64*32],LoadData2DParams{0,4,1,0,3,false,0});LoadDataWithTranspose(b2,b1[kk*256],LoadData2dTransposeParams{0,32,1,1,0,inc});SetFlag<HardEvent::MTE1_M>(ready);
}
__aicore__ inline void ShortManyNativeFinish(GlobalTensor<i32>c,LocalTensor<int8_t>a1,LocalTensor<int8_t>b1,LocalTensor<int8_t>a2,LocalTensor<int8_t>b2,LocalTensor<i32>c0,i32 K,i32 mr,i32 bank,i32 tileCols,uint8_t l1ready,uint8_t l1free,uint8_t f0,uint8_t f1,uint8_t r0,uint8_t r1,uint8_t cready,uint8_t cfree){
const uint8_t freeL0[2]={f0,f1},readyL0[2]={r0,r1};CrossCoreWaitFlag<2,PIPE_FIX>(8+bank);WaitFlag<HardEvent::MTE2_MTE1>(l1ready);ShortManyNativeL0(a1,b1,a2,b2,0,freeL0[0],readyL0[0]);for(i32 ki=0;ki<K/128;++ki){
const i32 slot=ki&1;if(ki+1<K/128){const i32 next=slot^1;ShortManyNativeL0(a1,b1,a2[next*8192],b2[next*32768],(ki+1)*128,freeL0[next],readyL0[next]);}
WaitFlag<HardEvent::MTE1_M>(readyL0[slot]);Mmad(c0,a2[slot*8192],b2[slot*32768],MmadParams{64,256,128,0,false,ki==0});SetFlag<HardEvent::M_MTE1>(freeL0[slot]);
}
SetFlag<HardEvent::MTE1_MTE2>(l1free);SetFlag<HardEvent::M_FIX>(cready);WaitFlag<HardEvent::M_FIX>(cready);Fixpipe(c[bank*64*tileCols],c0,FixpipeParamsV220{256,(uint16_t)mr,64,256,false,QuantMode_t::NoQuant,0,1,0,0,0});SetFlag<HardEvent::FIX_M>(cfree);WaitFlag<HardEvent::FIX_M>(cfree);CrossCoreSetFlag<2,PIPE_FIX>(8+bank);
}
__global__ __aicore__ __mix__(1,2)void B2541ShortManyLogTail(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileCols){
InitSocState();SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;GlobalTensor<int8_t>a,b,ap,bp;GlobalTensor<i64>ends;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);ap.SetGlobalBuffer((__gm__ int8_t*)x);bp.SetGlobalBuffer((__gm__ int8_t*)w);ends.SetGlobalBuffer((__gm__ i64*)groups);if ASCEND_IS_AIC{
if(N<1){MXPAD1024 MXPAD512 MXPAD128 MXPAD64 MXPAD16 MXPAD8 MXPAD4 MXPAD2}
const i32 pair=GetBlockIdx(),pairs=GetBlockNum();c.SetGlobalBuffer((__gm__ i32*)scratch+pair*128*tileCols);TBuf<TPosition::A1>aa;TBuf<TPosition::B1>bb;TBuf<TPosition::A2>ab;TBuf<TPosition::B2>ba;TBuf<TPosition::CO1>cc;pipe.InitBuffer(aa,128*K);pipe.InitBuffer(bb,512*K);pipe.InitBuffer(ab,16384);pipe.InitBuffer(ba,65536);pipe.InitBuffer(cc,65536);auto a1=aa.Get<int8_t>(),b1=bb.Get<int8_t>(),a2=ab.Get<int8_t>(),b2=ba.Get<int8_t>();
auto c0=cc.Get<i32>();uint8_t freeL0[2],readyL0[2],l1ready[2],l1free[2];for(i32 j=0;j<2;++j){
freeL0[j]=(uint8_t)pipe.AllocEventID<HardEvent::M_MTE1>();readyL0[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE1_M>();SetFlag<HardEvent::M_MTE1>(freeL0[j]);l1ready[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE2_MTE1>();l1free[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE1_MTE2>();SetFlag<HardEvent::MTE1_MTE2>(l1free[j]);
}
auto cready=pipe.FetchEventID(HardEvent::M_FIX),cfree=pipe.FetchEventID(HardEvent::FIX_M);i64 start=0,tileBase=0,sequence=0;i32 previousSlot=0,previousMr=0,previousBank=0;bool pending=false;for(i32 e=0;e<E;++e){
const i64 end=ends.GetValue(e);const i32 rows=end-start,nc=N/256;const i64 count=((rows+63)/64)*nc;for(i64 tile=(pair-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
const i32 row=tile/nc*64,col=tile%nc*256,mr=rows-row<64?rows-row:64,bank=sequence&1;WaitFlag<HardEvent::MTE1_MTE2>(l1free[bank]);DataCopy(a1[bank*64*K],a[(start+row)*K],Nd2NzParams{1,(uint16_t)mr,(uint16_t)K,0,(uint16_t)K,64,1,0});if(mr<64){
PipeBarrier<PIPE_MTE2>();InitConstValueParams<u32>zp{};zp.repeatTimes=(uint16_t)(K/32);zp.blockNum=(uint16_t)(64-mr);zp.dstGap=(uint16_t)mr;zp.initValue=0;InitConstValue(a1[bank*64*K+mr*32].ReinterpretCast<u32>(),zp);
}
DataCopy(b1[bank*256*K],b[(i64)e*K*N+col],Nd2NzParams{(uint16_t)(K/32),32,256,(uint16_t)(32*N),(uint16_t)N,32,1,8192});SetFlag<HardEvent::MTE2_MTE1>(l1ready[bank]);if(pending)ShortManyNativeFinish(c,a1[previousSlot*64*K],b1[previousSlot*256*K],a2,b2,c0,K,previousMr,previousBank,tileCols,l1ready[previousSlot],l1free[previousSlot],freeL0[0],freeL0[1],readyL0[0],readyL0[1],cready,cfree);
previousSlot=bank;previousMr=mr;previousBank=bank;pending=true;++sequence;
}
tileBase+=count;start=end;
}
if(pending)ShortManyNativeFinish(c,a1[previousSlot*64*K],b1[previousSlot*256*K],a2,b2,c0,K,previousMr,previousBank,tileCols,l1ready[previousSlot],l1free[previousSlot],freeL0[0],freeL0[1],readyL0[0],readyL0[1],cready,cfree);CrossCoreWaitFlag<2,PIPE_FIX>(8);CrossCoreWaitFlag<2,PIPE_FIX>(9);for(i32 j=0;j<2;++j){
WaitFlag<HardEvent::MTE1_MTE2>(l1free[j]);WaitFlag<HardEvent::M_MTE1>(freeL0[j]);pipe.ReleaseEventID<HardEvent::M_MTE1>(freeL0[j]);pipe.ReleaseEventID<HardEvent::MTE1_M>(readyL0[j]);pipe.ReleaseEventID<HardEvent::MTE2_MTE1>(l1ready[j]);pipe.ReleaseEventID<HardEvent::MTE1_MTE2>(l1free[j]);
}}
if ASCEND_IS_AIV{
const i32 id=GetBlockIdx(),workers=GetBlockNum()*2;const i32 pairId=id/2,pairs=GetBlockNum();GlobalTensor<float>scale;GlobalTensor<half>out;scale.SetGlobalBuffer((__gm__ float*)s);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+pairId*128*tileCols);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;pipe.InitBuffer(ci,32768);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);
pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,4096);pipe.InitBuffer(yy,16384);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();TBuf<TPosition::VECCALC>diagPoly,diagNormal,diagTail;
pipe.InitBuffer(diagPoly,16384);pipe.InitBuffer(diagNormal,1024);pipe.InitBuffer(diagTail,2048);auto diagPolyTensor=diagPoly.Get<float>();auto diagNormalTensor=diagNormal.Get<uint8_t>();auto diagTailTensor=diagTail.Get<uint8_t>();uint8_t freeIn[2],ready[2],freeOut[2],vo[2];for(i32 j=0;j<2;++j){
freeIn[j]=(uint8_t)pipe.AllocEventID<HardEvent::V_MTE2>();ready[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE2_V>();freeOut[j]=(uint8_t)pipe.AllocEventID<HardEvent::MTE3_V>();vo[j]=(uint8_t)pipe.AllocEventID<HardEvent::V_MTE3>();SetFlag<HardEvent::V_MTE2>(freeIn[j]);SetFlag<HardEvent::MTE3_V>(freeOut[j]);
}
i64 fragment=0,savedRow=0;i32 savedCol=0,savedRows=0,savedBank=0;bool pending=false;CrossCoreSetFlag<2,PIPE_MTE2>(8);CrossCoreSetFlag<2,PIPE_MTE2>(9);i64 start=0,tileBase=0,sequence=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+tileCols-1)/tileCols;i64 count=((rows+63)/64)*nc;for(i64 tile=(pairId-tileBase%pairs+pairs)%pairs;tile<count;tile+=pairs){
i32 row=tile/nc*64,col=tile%nc*tileCols;i32 mr=rows-row<64?rows-row:64,nr=N-col<tileCols?N-col:tileCols;i32 pitch=nr<=64?64:nr<=128?128:256,bank=sequence&1;CrossCoreWaitFlag<2,PIPE_MTE2>(8+bank);i32 off=GetSubBlockIdx()*32,valid=mr-off;if(valid>32)valid=32;for(i32 slab=0;slab<valid;slab+=16){
const i32 sr=valid-slab<16?valid-slab:16,slot=fragment&1;WaitFlag<HardEvent::V_MTE2>(freeIn[slot]);DataCopy(cint[slot*4096],c[bank*64*tileCols+(off+slab)*256],sr*256);if(slab+16>=valid)CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);DataCopy(sc[slot*256],scale[(i64)e*N+col],256);SetFlag<HardEvent::MTE2_V>(ready[slot]);if(pending){
WaitFlag<HardEvent::MTE2_V>(ready[savedBank]);WaitFlag<HardEvent::MTE3_V>(freeOut[savedBank]);B2541ShortManyLogTailPostImpl(out,cint[savedBank*4096],z,u,t,f,sc[savedBank*256],h[savedBank*4096],savedRow,savedCol,savedRows,N,freeIn[savedBank],freeOut[savedBank],vo[savedBank]);
}
savedRow=start+row+off+slab;savedCol=col;savedRows=sr;savedBank=slot;pending=true;++fragment;
}
if(valid<=0)CrossCoreSetFlag<2,PIPE_MTE2>(8+bank);++sequence;
}
tileBase+=count;start=end;
}
if(pending){
WaitFlag<HardEvent::MTE2_V>(ready[savedBank]);WaitFlag<HardEvent::MTE3_V>(freeOut[savedBank]);B2541ShortManyLogTailPostImpl(out,cint[savedBank*4096],z,u,t,f,sc[savedBank*256],h[savedBank*4096],savedRow,savedCol,savedRows,N,freeIn[savedBank],freeOut[savedBank],vo[savedBank]);
}
for(i32 j=0;j<2;++j){
WaitFlag<HardEvent::V_MTE2>(freeIn[j]);WaitFlag<HardEvent::MTE3_V>(freeOut[j]);pipe.ReleaseEventID<HardEvent::V_MTE2>(freeIn[j]);pipe.ReleaseEventID<HardEvent::MTE2_V>(ready[j]);pipe.ReleaseEventID<HardEvent::MTE3_V>(freeOut[j]);pipe.ReleaseEventID<HardEvent::V_MTE3>(vo[j]);
}}
PipeBarrier<PIPE_ALL>();
}
static void ShortMany_Run(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());
if(!blocks)throw std::runtime_error("no Cube cores");u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);const i64 lowerTiles=((i64)M+15)/16*((N+127)/128);const bool use128=K<=1024&&N>=128&&N<=1024&&lowerTiles>=(i64)blocks*8&&ubBytes>=65536&&(K<=64||lowerTiles>=(i64)blocks*32);const i32 nBlocks256=N/256+((N%256)+127)/128;
const i64 lowerTiles256=((i64)M+15)/16*nBlocks256;i32 columnCycle=N/256;i32 workerCycle=blocks*2;while(workerCycle!=0){
i32 remainder=columnCycle%workerCycle;columnCycle=workerCycle;workerCycle=remainder;
}
const bool dense256Cycle=N%256==0&&columnCycle==1;const bool use256=K<=1024&&N>=256&&N<=1024&&ubBytes>=131072&&lowerTiles256>=(i64)blocks*8&&(K<=64||nBlocks256==1||lowerTiles256>=(i64)blocks*32||dense256Cycle);i32 tileCols=use256?256:use128?128:64;u64 pairL1=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1,pairL1);
const char*pairSoc=aclrtGetSocName();const bool alignedPairs=K%32==0&&N%32==0&&ubBytes>=131072&&pairSoc&&std::strcmp(pairSoc,"Ascend910_9362")==0&&2LL*(64+tileCols)*K<=(i64)pairL1&&lowerTiles256>=(i64)platform->GetCoreNumAic()*8;u64 l0a=0,l0b=0,l0c=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,l0a);
platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);auto makeTiling=[&](i32 cols,TCubeTiling&result){
matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);mt.SetShape(alignedPairs?64:16,cols,K);mt.SetOrgShape(M,N,K);i32 baseM=alignedPairs&&2*64*64<=l0a&&2*cols*64<=l0b&&64*cols*4<=l0c?64:16;
mt.SetFixSplit(baseM,cols,64);mt.SetBufferSpace(-1,-1,-1);int status=mt.GetTiling(result);if(status!=0&&baseM==64){mt.SetFixSplit(16,cols,64);status=mt.GetTiling(result);}
return status;
};TCubeTiling tiling;i32 tilingStatus=makeTiling(tileCols,tiling);if(tilingStatus!=0&&tileCols==256&&use128){tileCols=128;tilingStatus=makeTiling(tileCols,tiling);}
if(tilingStatus!=0&&tileCols!=64){tileCols=64;tilingStatus=makeTiling(tileCols,tiling);}
if(tilingStatus!=0)throw std::runtime_error("INT8 tiling failed");const i32 enableFastPath=0;const i32 Kp=(K+63)/64*64,Np=(N+255)/256*256;const i64 stagedBytes=(i64)M*Kp+(i64)E*Kp*Np;const bool useStage=(K%32!=0||N%32!=0)&&stagedBytes<=134217728LL;void*staging=nullptr;void*workspace=nullptr,*scratch=nullptr;
size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*((useStage||alignedPairs)?64:16)*tileCols*4*(enableFastPath?2:1),ACL_MEM_MALLOC_HUGE_FIRST));const char*b505soc=aclrtGetSocName();
if(b505soc&&std::strcmp(b505soc,"Ascend910_9362")==0&&ubBytes>=131072){
if(useStage){
matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);mt.SetShape(64,tileCols,Kp);mt.SetOrgShape(M,Np,Kp);mt.SetFixSplit(16,tileCols,64);mt.SetBufferSpace(-1,-1,-1);
TCubeTiling packedTiling;if(mt.GetTiling(packedTiling)==0){
checkGmm(aclrtMalloc(&staging,stagedBytes,ACL_MEM_MALLOC_HUGE_FIRST));ShortMany_StageKernel<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,packedTiling,M,K,N,E,tileCols,(GM_ADDR)staging,Kp,Np);
}else{
ShortMany_VectorRequest<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileCols);
}}else if(alignedPairs){
if(tileCols==256&&N%256==0&&ubBytes>=138240&&K%128==0&&640LL*K<=(i64)pairL1&&l0a>=16384&&l0b>=65536&&l0c>=65536&&32LL*N<=65535)B2541ShortManyLogTail<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileCols);else ShortMany_AlignedPairs<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileCols,nullptr,K,N);
}else{
ShortMany_VectorRequest<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileCols);
}}else{
if(tileCols==256){ShortMany_Legacy256<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,0);}else if(tileCols==128){ShortMany_Legacy128<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E);}else{ShortMany_Legacy64<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E);}}
aclError status=aclrtSynchronizeStream(stream);if(staging)aclrtFree(staging);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
using namespace AscendC;
using GmmMidK_GmmA=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using GmmMidK_GmmB=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
using GmmMidK_GmmC=MatmulType<TPosition::GM,CubeFormat::ND,i32>;
__aicore__ inline i64 GmmMidK_GmmFloatToQ(float x){
union{float f;u32 bits;}v;v.f=x;u32 a=v.bits&0x7fffffffU;if(!a)return 0;
int shift=(int)((a>>23)&255)-127+27;u64 mant=(a&0x7fffffU)|0x800000U;u64 q=shift>=0?mant<<shift:(shift>-64?mant>>-shift:0);return v.bits>>31?-(i64)q:(i64)q;
}
__aicore__ inline float GmmMidK_GmmQToFloat(i64 x){
union{float f;u32 bits;}v;if(!x){v.bits=0;return v.f;}
u64 a=x<0?-x:x;int top=63-__builtin_clzll(a),shift=top-23;u64 mant;if(shift>0){
mant=a>>shift;u64 rem=a&((1ULL<<shift)-1),halfWay=1ULL<<(shift-1);if(rem>halfWay||(rem==halfWay&&(mant&1)))++mant;if(mant==(1ULL<<24)){mant>>=1;++top;}}else mant=a<<-shift;v.bits=(x<0?0x80000000U:0)|((u32)(top-50+127)<<23)|((u32)mant&0x7fffffU);return v.f;
}
__aicore__ inline i64 GmmMidK_GmmQMul(i64 a,i64 b){
bool neg=(a<0)!=(b<0);u64 aa=a<0?-a:a,bb=b<0?-b:b;const u64 mask=(1ULL<<25)-1;u64 ah=aa>>25,al=aa&mask,bh=bb>>25,bl=bb&mask;u64 v=ah*bh+((ah*bl+al*bh+((al*bl)>>25)+(1ULL<<24))>>25);return neg?-(i64)v:(i64)v;
}
__aicore__ inline i64 GmmMidK_GmmQMulSmallNegative(i64 a,u64 rh,u64 rl){
u64 aa=a<0?-a:a;u64 v=(aa*rh+((aa*rl)>>25)+(1ULL<<24))>>25;return a<0?(i64)v:-(i64)v;
}
__aicore__ inline i64 GmmMidK_GmmQMulTinyNegative(i64 a,u64 r){
u64 aa=a<0?-a:a;u64 v=(aa*r+(1ULL<<49))>>50;return a<0?(i64)v:-(i64)v;
}
__aicore__ inline i64 GmmMidK_GmmQExpNeg(i64 a){
const i64 Q=1LL<<50,ln2=780414346020670LL;i64 n=a/ln2,r=a-n*ln2;if(n>53)return 0;
i64 term=-r,sum=Q-r;for(int i=2;i<=10;++i){term=GmmMidK_GmmQMul(term,-r)/i;sum+=term;}
u64 rh=(u64)r>>25,rl=(u64)r&((1ULL<<25)-1);for(int i=11;i<=13;++i){term=GmmMidK_GmmQMulSmallNegative(term,rh,rl)/i;sum+=term;}
for(int i=14;i<=15;++i){term=GmmMidK_GmmQMulTinyNegative(term,(u64)r)/i;sum+=term;}
return sum>>n;
}
__aicore__ inline float GmmMidK_GmmRefine(float z){
if(z>=8.0f)return z;
if(z<=-8.0f)return 0.0f;
const i64 Q=1LL<<50;i64 zq=GmmMidK_GmmFloatToQ(z);i64 cubic=GmmMidK_GmmQMul(GmmMidK_GmmQMul(zq,zq),zq);i64 inner=GmmMidK_GmmQMul(898338152679114LL,zq+GmmMidK_GmmQMul(50344614334468LL,cubic));i64 tail=GmmMidK_GmmQExpNeg(2*(inner<0?-inner:inner));i64 den=Q+tail;float df=GmmMidK_GmmQToFloat(den);
i64 inv=GmmMidK_GmmFloatToQ(1.0f/df);inv=GmmMidK_GmmQMul(inv,2*Q-GmmMidK_GmmQMul(den,inv));inv=GmmMidK_GmmQMul(inv,2*Q-GmmMidK_GmmQMul(den,inv));i64 value=GmmMidK_GmmQMul(inner>=0?zq:GmmMidK_GmmQMul(zq,tail),inv);union{float f;u32 bits;}v;v.f=GmmMidK_GmmQToFloat(value);i64 back=GmmMidK_GmmFloatToQ(v.f);
u32 ab=v.bits&0x7fffffffU;u64 backAbs=back<0?-back:back;bool mid=ab>=0x38800000U?((ab&8191U)==4096U):((backAbs&((1ULL<<26)-1))==(1ULL<<25));if(mid){
if(back>value)v.bits+=v.f>0?-1:1;else if(back<value)v.bits+=v.f>0?1:-1;else if(v.f==z&&z>4.0f&&z<7.09f)--v.bits;
}
return v.f;
}
__global__ __aicore__ __mix__(1,2)void GmmMidK_grouped_matmul_gelu_custom(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileM){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<GmmMidK_GmmA,GmmMidK_GmmB,GmmMidK_GmmC>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*tileM*64);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,4096);pipe.InitBuffer(zz,4096);pipe.InitBuffer(uu,4096);pipe.InitBuffer(tt,4096);pipe.InitBuffer(ff,4096);pipe.InitBuffer(ss,256);pipe.InitBuffer(yy,2048);auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();
i64 start=0,tileBase=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+63)/64;i64 count=((rows+tileM-1)/tileM)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*tileM,col=(tile%nc)*64;i32 tileRows=rows-row<tileM?rows-row:tileM;i32 nr=N-col<64?N-col:64;mm.SetOrgShape(M,N,K,K,64);mm.SetSingleShape(tileRows,nr,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);mm.IterateAll(c);mm.End();for(i32 slab=0;slab<tileRows;slab+=16){
i32 mr=tileRows-slab<16?tileRows-slab:16;DataCopy(cint,c[slab*64],1024);DataCopyExtParams sp{1,(u32)nr*4,0,0,0};DataCopyPad(sc,scale[(i64)e*N+col],sp,DataCopyPadExtParams<float>{true,0,0,0});auto ev=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(ev);WaitFlag<HardEvent::MTE2_V>(ev);Cast(z,cint,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();for(i32 r=0;r<mr;++r)Mul(z[r*64],z[r*64],sc,64);
PipeBarrier<PIPE_V>();Mul(u,z,z,1024);PipeBarrier<PIPE_V>();Muls(u,u,0.044715f,1024);PipeBarrier<PIPE_V>();Adds(u,u,1.0f,1024);PipeBarrier<PIPE_V>();Mul(u,u,z,1024);PipeBarrier<PIPE_V>();Muls(u,u,1.5957691216057308f,1024);PipeBarrier<PIPE_V>();Abs(t,u,1024);PipeBarrier<PIPE_V>();Muls(t,t,-1.0f,1024);PipeBarrier<PIPE_V>();Exp(t,t,1024);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,1024);PipeBarrier<PIPE_V>();Mins(f,u,0.0f,1024);PipeBarrier<PIPE_V>();Exp(f,f,1024);PipeBarrier<PIPE_V>();Mul(f,f,z,1024);PipeBarrier<PIPE_V>();
Div(f,f,t,1024);PipeBarrier<PIPE_V>();auto m0=ci.Get<uint8_t>();auto m1=m0[128];auto m2=m0[256];auto m3=m0[384];Duplicate(t.ReinterpretCast<i32>(),(i32)8191,1024);PipeBarrier<PIPE_V>();And(u.ReinterpretCast<uint16_t>(),f.ReinterpretCast<uint16_t>(),t.ReinterpretCast<uint16_t>(),2048);PipeBarrier<PIPE_V>();Cast(u,u.ReinterpretCast<i32>(),RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();CompareScalar(m0,u,4064.0f,CMPMODE::GE,1024);CompareScalar(m1,u,4128.0f,CMPMODE::LE,1024);
Abs(t,f,1024);PipeBarrier<PIPE_V>();CompareScalar(m2,t,0.00006103515625f,CMPMODE::LT,1024);CompareScalar(m3,t,0.0f,CMPMODE::GT,1024);PipeBarrier<PIPE_V>();And(m0.ReinterpretCast<uint16_t>(),m0.ReinterpretCast<uint16_t>(),m1.ReinterpretCast<uint16_t>(),64);And(m2.ReinterpretCast<uint16_t>(),m2.ReinterpretCast<uint16_t>(),m3.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();Or(m0.ReinterpretCast<uint16_t>(),m0.ReinterpretCast<uint16_t>(),m2.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();auto vs=pipe.FetchEventID(HardEvent::V_S);
SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);u64 validColumns=nr==64?~0ULL:((1ULL<<nr)-1);for(int word=0;word<mr;++word){
u64 bits=m0.ReinterpretCast<u64>().GetValue(word)&validColumns;while(bits){
int lane=word*64+__builtin_ctzll(bits);bits&=bits-1;f.SetValue(lane,GmmMidK_GmmRefine(z.GetValue(lane)));
}}
auto sv=pipe.FetchEventID(HardEvent::S_V);SetFlag<HardEvent::S_V>(sv);WaitFlag<HardEvent::S_V>(sv);Cast(h,f,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 r=0;r<mr;++r){
DataCopyExtParams op{1,(u32)nr*2,0,0,0};DataCopyPad(out[(start+row+slab+r)*N+col],h[r*64],op);
}
auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}}
tileBase+=count;start=end;
}}}
static void GmmMidKRun(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());
if(!blocks)throw std::runtime_error("no Cube cores");matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);i32 tileM=32;mt.SetShape(tileM,64,K);mt.SetOrgShape(M,N,K);
mt.SetFixSplit(tileM,64,64);mt.SetBufferSpace(-1,-1,-1);TCubeTiling tiling;if(mt.GetTiling(tiling)!=0){
tileM=16;mt.SetShape(tileM,64,K);mt.SetFixSplit(tileM,64,64);if(mt.GetTiling(tiling)!=0)throw std::runtime_error("INT8 tiling failed");
}
void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*tileM*64*sizeof(i32),ACL_MEM_MALLOC_HUGE_FIRST));GmmMidK_grouped_matmul_gelu_custom<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileM);
aclError status=aclrtSynchronizeStream(stream);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
__aicore__ inline void GmmMidTailBoundedConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,LocalTensor<float>poly,LocalTensor<uint8_t>normal,LocalTensor<uint8_t>tail0,LocalTensor<uint8_t>tail1,LocalTensor<uint8_t>tail2,LocalTensor<uint8_t>tail3,i64 globalRow,i32 col,i32 e,i32 tileRows,i32 N,i32 tileM,i32 bank){
i32 nr=64;for(i32 slab=0;slab<tileRows;slab+=16){
i32 mr=tileRows-slab<16?tileRows-slab:16;DataCopy(cint,c[bank*tileM*64+slab*64],1024);DataCopyExtParams sp{1,(u32)nr*4,0,0,0};DataCopyPad(sc,scale[(i64)e*N+col],sp,DataCopyPadExtParams<float>{true,0,0,0});auto ev=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(ev);WaitFlag<HardEvent::MTE2_V>(ev);Cast(z,cint,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();for(i32 r=0;r<mr;++r)Mul(z[r*64],z[r*64],sc,64);
PipeBarrier<PIPE_V>();Mul(u,z,z,1024);PipeBarrier<PIPE_V>();Muls(u,u,0.044715f,1024);PipeBarrier<PIPE_V>();Adds(u,u,1.0f,1024);PipeBarrier<PIPE_V>();Mul(u,u,z,1024);PipeBarrier<PIPE_V>();Muls(u,u,1.5957691216057308f,1024);PipeBarrier<PIPE_V>();Abs(t,u,1024);PipeBarrier<PIPE_V>();Muls(t,t,-1.0f,1024);PipeBarrier<PIPE_V>();Exp(t,t,1024);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,1024);PipeBarrier<PIPE_V>();Mins(f,u,0.0f,1024);PipeBarrier<PIPE_V>();Exp(f,f,1024);PipeBarrier<PIPE_V>();Mul(f,f,z,1024);PipeBarrier<PIPE_V>();
Div(f,f,t,1024);PipeBarrier<PIPE_V>();Mins(u,z,1.8f,1024);PipeBarrier<PIPE_V>();Maxs(u,u,-1.8f,1024);PipeBarrier<PIPE_V>();Mul(t,u,u,1024);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,1024);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,1024);PipeBarrier<PIPE_V>();Mul(u,u,t,1024);PipeBarrier<PIPE_V>();Muls(u,u,0.7978845608028654f,1024);PipeBarrier<PIPE_V>();Mul(t,u,u,1024);PipeBarrier<PIPE_V>();Adds(poly,t,378.0f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,t,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,17325.0f,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,t,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,135135.0f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();auto den=cint.ReinterpretCast<float>();Muls(den,t,28.0f,1024);PipeBarrier<PIPE_V>();Adds(den,den,3150.0f,1024);PipeBarrier<PIPE_V>();Mul(den,den,t,1024);PipeBarrier<PIPE_V>();Adds(den,den,62370.0f,1024);PipeBarrier<PIPE_V>();Mul(den,den,t,1024);PipeBarrier<PIPE_V>();Adds(den,den,135135.0f,1024);
PipeBarrier<PIPE_V>();Div(t,poly,den,1024);PipeBarrier<PIPE_V>();Mul(u,t,den,1024);PipeBarrier<PIPE_V>();Sub(u,u,poly,1024);PipeBarrier<PIPE_V>();Abs(u,u,1024);PipeBarrier<PIPE_V>();Abs(poly,poly,1024);PipeBarrier<PIPE_V>();Muls(poly,poly,0.000001f,1024);PipeBarrier<PIPE_V>();auto certified=normal[128];Compare(certified,u,poly,CMPMODE::LE,1024);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,1024);PipeBarrier<PIPE_V>();Mul(t,t,z,1024);PipeBarrier<PIPE_V>();Muls(t,t,0.5f,1024);
PipeBarrier<PIPE_V>();Abs(u,z,1024);PipeBarrier<PIPE_V>();auto normalUpper=cint.ReinterpretCast<uint8_t>();CompareScalar(normal,u,0.001f,CMPMODE::GE,1024);CompareScalar(normalUpper,u,1.8f,CMPMODE::LE,1024);PipeBarrier<PIPE_V>();And(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),normalUpper.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();And(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),64);
PipeBarrier<PIPE_V>();Select(f,normal,t,f,SELMODE::VSEL_TENSOR_TENSOR_MODE,1024);PipeBarrier<PIPE_V>();Not(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();Not(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();Mins(u,z,-1.8f,1024);PipeBarrier<PIPE_V>();Maxs(u,u,-5.5f,1024);PipeBarrier<PIPE_V>();Mul(t,u,u,1024);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,1024);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,1024);PipeBarrier<PIPE_V>();Mul(u,u,t,1024);
PipeBarrier<PIPE_V>();Muls(u,u,0.04986778505017909f,1024);PipeBarrier<PIPE_V>();Duplicate(poly,2.7557319223985888e-07f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,2.7557319223985893e-06f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,2.4801587301587302e-05f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.00019841269841269841f,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.0013888888888888889f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.0083333333333333332f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.041666666666666664f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.16666666666666666f,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.5f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,1.f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,1.f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Adds(t,poly,1.0f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,z,1024);PipeBarrier<PIPE_V>();Adds(u,poly,0.0f,1024);PipeBarrier<PIPE_V>();Div(poly,poly,t,1024);PipeBarrier<PIPE_V>();auto tailScratch=cint.ReinterpretCast<float>();Mul(tailScratch,poly,t,1024);PipeBarrier<PIPE_V>();Sub(tailScratch,tailScratch,u,1024);PipeBarrier<PIPE_V>();Abs(tailScratch,tailScratch,1024);
PipeBarrier<PIPE_V>();Abs(u,u,1024);PipeBarrier<PIPE_V>();Muls(u,u,0.000001f,1024);PipeBarrier<PIPE_V>();Compare(tail3,tailScratch,u,CMPMODE::LE,1024);PipeBarrier<PIPE_V>();Abs(t,poly,1024);PipeBarrier<PIPE_V>();Muls(t,t,16777216.0f,1024);PipeBarrier<PIPE_V>();Mins(t,t,2048.0f,1024);PipeBarrier<PIPE_V>();Cast(tailScratch.ReinterpretCast<i32>(),t,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();Cast(tailScratch,tailScratch.ReinterpretCast<i32>(),RoundMode::CAST_RINT,1024);
PipeBarrier<PIPE_V>();Sub(tailScratch,tailScratch,t,1024);PipeBarrier<PIPE_V>();Abs(tailScratch,tailScratch,1024);PipeBarrier<PIPE_V>();Muls(tailScratch,tailScratch,-1.0f,1024);PipeBarrier<PIPE_V>();Adds(tailScratch,tailScratch,0.5f,1024);PipeBarrier<PIPE_V>();Muls(t,t,0.0008f,1024);PipeBarrier<PIPE_V>();Compare(tail0,tailScratch,t,CMPMODE::GT,1024);PipeBarrier<PIPE_V>();Abs(u,poly,1024);PipeBarrier<PIPE_V>();CompareScalar(tail1,u,0.00006103515625f,CMPMODE::GE,1024);
PipeBarrier<PIPE_V>();Or(tail0.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();And(tail0.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),tail3.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();CompareScalar(tail1,z,-5.5f,CMPMODE::GE,1024);CompareScalar(tail2,z,-1.8f,CMPMODE::LE,1024);PipeBarrier<PIPE_V>();And(tail0.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();And(tail0.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),tail2.ReinterpretCast<uint16_t>(),64);
PipeBarrier<PIPE_V>();Select(f,tail0,poly,f,SELMODE::VSEL_TENSOR_TENSOR_MODE,1024);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();CompareScalar(tail0,z,-5.5f,CMPMODE::LE,1024);PipeBarrier<PIPE_V>();Duplicate(poly,0.0f,1024);PipeBarrier<PIPE_V>();Select(f,tail0,poly,f,SELMODE::VSEL_TENSOR_TENSOR_MODE,1024);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),64);
PipeBarrier<PIPE_V>();Not(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();auto m0=cint.ReinterpretCast<uint8_t>();auto m1=m0[128];auto m2=m0[256];auto m3=m0[384];Duplicate(t.ReinterpretCast<i32>(),(i32)8191,1024);PipeBarrier<PIPE_V>();And(u.ReinterpretCast<uint16_t>(),f.ReinterpretCast<uint16_t>(),t.ReinterpretCast<uint16_t>(),2048);PipeBarrier<PIPE_V>();Cast(u,u.ReinterpretCast<i32>(),RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();CompareScalar(m0,u,4064.0f,CMPMODE::GE,1024);
CompareScalar(m1,u,4128.0f,CMPMODE::LE,1024);Abs(t,f,1024);PipeBarrier<PIPE_V>();CompareScalar(m2,t,0.00006103515625f,CMPMODE::LT,1024);CompareScalar(m3,t,0.0f,CMPMODE::GT,1024);PipeBarrier<PIPE_V>();And(m0.ReinterpretCast<uint16_t>(),m0.ReinterpretCast<uint16_t>(),m1.ReinterpretCast<uint16_t>(),64);And(m2.ReinterpretCast<uint16_t>(),m2.ReinterpretCast<uint16_t>(),m3.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();Or(m0.ReinterpretCast<uint16_t>(),m0.ReinterpretCast<uint16_t>(),m2.ReinterpretCast<uint16_t>(),64);
PipeBarrier<PIPE_V>();And(m0.ReinterpretCast<uint16_t>(),m0.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),64);PipeBarrier<PIPE_V>();auto vs=pipe.FetchEventID(HardEvent::V_S);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);u64 validColumns=nr==64?~0ULL:((1ULL<<nr)-1);for(int word=0;word<mr;++word){
u64 bits=m0.ReinterpretCast<u64>().GetValue(word)&validColumns;while(bits){
int lane=word*64+__builtin_ctzll(bits);bits&=bits-1;f.SetValue(lane,GmmMidK_GmmRefine(z.GetValue(lane)));
}}
auto sv=pipe.FetchEventID(HardEvent::S_V);SetFlag<HardEvent::S_V>(sv);WaitFlag<HardEvent::S_V>(sv);Cast(h,f,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);for(i32 r=0;r<mr;++r){
DataCopyExtParams op{1,(u32)nr*2,0,0,0};DataCopyPad(out[(globalRow+slab+r)*N+col],h[r*64],op);
}
auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}}
__global__ __aicore__ __mix__(1,2)void GmmMidTailBounded_grouped_matmul_gelu_custom(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileM){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<GmmMidK_GmmA,GmmMidK_GmmB,GmmMidK_GmmC>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2*tileM*64);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,4096);pipe.InitBuffer(zz,4096);pipe.InitBuffer(uu,4096);pipe.InitBuffer(tt,4096);pipe.InitBuffer(ff,4096);pipe.InitBuffer(ss,256);pipe.InitBuffer(yy,2048);TBuf<TPosition::VECCALC>polyBuffer,normalBuffer;pipe.InitBuffer(polyBuffer,4096);pipe.InitBuffer(normalBuffer,256);auto poly=polyBuffer.Get<float>();auto normal=normalBuffer.Get<uint8_t>();
TBuf<TPosition::VECCALC>tailBuffer;pipe.InitBuffer(tailBuffer,512);auto tail0=tailBuffer.Get<uint8_t>();auto tail1=tail0[128];auto tail2=tail0[256];auto tail3=tail0[384];auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();
auto h=yy.Get<half>();i64 start=0,tileBase=0;bool pending=false;i32 bank=0,savedExpert=0,savedCol=0,savedRows=0;i64 savedGlobalRow=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=N/64;i64 count=((rows+tileM-1)/tileM)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*tileM,col=(tile%nc)*64;i32 tileRows=rows-row<tileM?rows-row:tileM;i32 producerBank=pending?bank^1:bank;mm.SetOrgShape(M,N,K,K,64);mm.SetSingleShape(tileRows,64,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);if(pending){
mm.IterateAll<false>(c[producerBank*tileM*64],0,false,true);GmmMidTailBoundedConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,normal,tail0,tail1,tail2,tail3,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);mm.WaitIterateAll();mm.End();bank=producerBank;
}else{mm.IterateAll(c[producerBank*tileM*64]);mm.End();}
pending=true;savedGlobalRow=start+row;savedCol=col;savedExpert=e;savedRows=tileRows;
}
tileBase+=count;start=end;
}
if(pending){
GmmMidTailBoundedConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,normal,tail0,tail1,tail2,tail3,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);
}}}
static void GmmMidTailBoundedRun(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];if(N%64!=0){GmmMidKRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());if(!blocks)throw std::runtime_error("no Cube cores");matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);
mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);i32 tileM=32;mt.SetShape(tileM,64,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(tileM,64,64);mt.SetBufferSpace(-1,-1,-1);TCubeTiling tiling;if(mt.GetTiling(tiling)!=0){GmmMidKRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*2*tileM*64*sizeof(i32),ACL_MEM_MALLOC_HUGE_FIRST));GmmMidTailBounded_grouped_matmul_gelu_custom<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileM);
aclError status=aclrtSynchronizeStream(stream);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
__aicore__ inline void GmmMidUnifiedRectConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,LocalTensor<float>poly,LocalTensor<uint8_t>normal,LocalTensor<uint8_t>tail0,LocalTensor<uint8_t>tail1,LocalTensor<uint8_t>tail2,LocalTensor<uint8_t>tail3,i64 globalRow,i32 col,i32 e,i32 tileRows,i32 N,i32 tileM,i32 bank){
i32 nr=256;for(i32 slab=0;slab<tileRows;slab+=16){
i32 mr=tileRows-slab<16?tileRows-slab:16;DataCopy(cint,c[bank*tileM*256+slab*256],4096);DataCopyExtParams sp{1,(u32)nr*4,0,0,0};DataCopyPad(sc,scale[(i64)e*N+col],sp,DataCopyPadExtParams<float>{true,0,0,0});auto ev=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(ev);WaitFlag<HardEvent::MTE2_V>(ev);Cast(z,cint,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();for(i32 r=0;r<mr;++r)Mul(z[r*256],z[r*256],sc,256);
PipeBarrier<PIPE_V>();Abs(u,z,4096);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,4096);PipeBarrier<PIPE_V>();Mul(t,u,u,4096);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,4096);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,4096);PipeBarrier<PIPE_V>();Mul(u,u,t,4096);PipeBarrier<PIPE_V>();Muls(u,u,-0.04986778505017909f,4096);PipeBarrier<PIPE_V>();Duplicate(poly,2.4801587301587302e-05f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.00019841269841269841f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.0013888888888888889f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.0083333333333333332f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.041666666666666664f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.16666666666666666f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.5f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,1.f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,1.f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Adds(t,poly,1.0f,4096);PipeBarrier<PIPE_V>();Muls(u,poly,0.00008f,4096);PipeBarrier<PIPE_V>();Adds(u,u,0.000001f,4096);PipeBarrier<PIPE_V>();CompareScalar(tail0,z,0.0f,CMPMODE::LT,4096);PipeBarrier<PIPE_V>();Mul(f,z,poly,4096);PipeBarrier<PIPE_V>();Select(f,tail0,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Adds(poly,f,0.0f,4096);PipeBarrier<PIPE_V>();Div(f,f,t,4096);PipeBarrier<PIPE_V>();auto scratch=cint.ReinterpretCast<float>();
Mul(scratch,f,t,4096);PipeBarrier<PIPE_V>();Sub(scratch,scratch,poly,4096);PipeBarrier<PIPE_V>();Abs(scratch,scratch,4096);PipeBarrier<PIPE_V>();Abs(poly,poly,4096);PipeBarrier<PIPE_V>();Muls(poly,poly,0.0000002f,4096);PipeBarrier<PIPE_V>();Compare(tail3,scratch,poly,CMPMODE::LE,4096);PipeBarrier<PIPE_V>();Duplicate(t,0.00008f,4096);PipeBarrier<PIPE_V>();Select(u,tail0,t,u,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Abs(t,f,4096);
PipeBarrier<PIPE_V>();Maxs(poly,t,0.00006103515625f,4096);PipeBarrier<PIPE_V>();Duplicate(scratch.ReinterpretCast<i32>(),(i32)0x7f800000,4096);PipeBarrier<PIPE_V>();And(poly.ReinterpretCast<uint16_t>(),poly.ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),8192);PipeBarrier<PIPE_V>();Duplicate(scratch.ReinterpretCast<i32>(),(i32)0x7f000000,4096);PipeBarrier<PIPE_V>();Sub(poly.ReinterpretCast<i32>(),scratch.ReinterpretCast<i32>(),poly.ReinterpretCast<i32>(),4096);PipeBarrier<PIPE_V>();Muls(poly,poly,1024.0f,4096);
PipeBarrier<PIPE_V>();Mul(t,t,poly,4096);PipeBarrier<PIPE_V>();Mins(t,t,2048.0f,4096);PipeBarrier<PIPE_V>();Mul(u,u,t,4096);PipeBarrier<PIPE_V>();Muls(u,u,2.0f,4096);PipeBarrier<PIPE_V>();Cast(scratch.ReinterpretCast<i32>(),t,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();Cast(scratch,scratch.ReinterpretCast<i32>(),RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();Sub(scratch,scratch,t,4096);PipeBarrier<PIPE_V>();Abs(scratch,scratch,4096);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,4096);
PipeBarrier<PIPE_V>();Adds(scratch,scratch,0.5f,4096);PipeBarrier<PIPE_V>();Compare(normal,scratch,u,CMPMODE::GT,4096);PipeBarrier<PIPE_V>();Abs(t,f,4096);PipeBarrier<PIPE_V>();CompareScalar(tail1,t,0.00006103515625f,CMPMODE::GE,4096);CompareScalar(tail2,t,1.99f,CMPMODE::LT,4096);PipeBarrier<PIPE_V>();And(tail1.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),tail2.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),256);
PipeBarrier<PIPE_V>();And(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail3.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();CompareScalar(tail1,z,-5.5f,CMPMODE::LE,4096);PipeBarrier<PIPE_V>();Duplicate(poly,0.0f,4096);PipeBarrier<PIPE_V>();Select(f,tail1,poly,f,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Abs(t,z,4096);PipeBarrier<PIPE_V>();CompareScalar(tail1,t,0.00000001490116119384765625f,CMPMODE::LE,4096);
PipeBarrier<PIPE_V>();Select(f,tail1,poly,f,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();CompareScalar(tail1,z,7.1f,CMPMODE::GE,4096);PipeBarrier<PIPE_V>();Select(f,tail1,z,f,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Duplicate(scratch.ReinterpretCast<i32>(),(i32)8191,4096);
PipeBarrier<PIPE_V>();And(poly.ReinterpretCast<uint16_t>(),z.ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),8192);PipeBarrier<PIPE_V>();Cast(poly,poly.ReinterpretCast<i32>(),RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();CompareScalar(tail0,poly,4096.0f,CMPMODE::EQ,4096);CompareScalar(tail1,z,5.5f,CMPMODE::GE,4096);CompareScalar(tail2,z,7.0f,CMPMODE::LE,4096);PipeBarrier<PIPE_V>();And(tail0.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),256);
PipeBarrier<PIPE_V>();And(tail0.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),tail2.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Adds(poly.ReinterpretCast<i32>(),z.ReinterpretCast<i32>(),(i32)-1,4096);PipeBarrier<PIPE_V>();Select(f,tail0,poly,f,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Not(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),256);
PipeBarrier<PIPE_V>();auto m0=normal;auto vs=pipe.FetchEventID(HardEvent::V_S);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);u64 validColumns=~0ULL;for(int word=0;word<mr*4;++word){
u64 bits=m0.ReinterpretCast<u64>().GetValue(word)&validColumns;while(bits){
int lane=word*64+__builtin_ctzll(bits);bits&=bits-1;f.SetValue(lane,GmmMidK_GmmRefine(z.GetValue(lane)));
}}
auto sv=pipe.FetchEventID(HardEvent::S_V);SetFlag<HardEvent::S_V>(sv);WaitFlag<HardEvent::S_V>(sv);Cast(h,f,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);DataCopyExtParams op{(uint16_t)mr,(u32)nr*2,0,(u32)(N-nr)*2,0};DataCopyPad(out[(globalRow+slab)*N+col],h,op);auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}}
__global__ __aicore__ __mix__(1,2)void GmmMidUnifiedRect_grouped_matmul_gelu_custom(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileM){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<GmmMidK_GmmA,GmmMidK_GmmB,GmmMidK_GmmC>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2*tileM*256);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,1024);pipe.InitBuffer(yy,8192);TBuf<TPosition::VECCALC>polyBuffer,normalBuffer;pipe.InitBuffer(polyBuffer,16384);pipe.InitBuffer(normalBuffer,1024);auto poly=polyBuffer.Get<float>();auto normal=normalBuffer.Get<uint8_t>();
TBuf<TPosition::VECCALC>tailBuffer;pipe.InitBuffer(tailBuffer,2048);auto tail0=tailBuffer.Get<uint8_t>();auto tail1=tail0[512];auto tail2=tail0[1024];auto tail3=tail0[1536];auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();
auto h=yy.Get<half>();i64 start=0,tileBase=0;bool pending=false;i32 bank=0,savedExpert=0,savedCol=0,savedRows=0;i64 savedGlobalRow=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=N/256;i64 count=((rows+tileM-1)/tileM)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*tileM,col=(tile%nc)*256;i32 tileRows=rows-row<tileM?rows-row:tileM;i32 producerBank=pending?bank^1:bank;mm.SetOrgShape(M,N,K,K,256);mm.SetSingleShape(tileRows,256,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);if(pending){
mm.IterateAll<false>(c[producerBank*tileM*256],0,false,true);GmmMidUnifiedRectConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,normal,tail0,tail1,tail2,tail3,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);mm.WaitIterateAll();mm.End();bank=producerBank;
}else{mm.IterateAll(c[producerBank*tileM*256]);mm.End();}
pending=true;savedGlobalRow=start+row;savedCol=col;savedExpert=e;savedRows=tileRows;
}
tileBase+=count;start=end;
}
if(pending){
GmmMidUnifiedRectConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,normal,tail0,tail1,tail2,tail3,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);
}}}
static void GmmMidUnifiedRectRun(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];if(N%256!=0){GmmMidTailBoundedRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);if(ubBytes<143360){GmmMidTailBoundedRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());if(!blocks)throw std::runtime_error("no Cube cores");matmul_tiling::MatmulApiTiling mt(*platform);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);
i32 tileM=32;mt.SetShape(tileM,256,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(tileM,256,128);mt.SetBufferSpace(-1,-1,-1);TCubeTiling tiling;if(mt.GetTiling(tiling)!=0){GmmMidTailBoundedRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*2*tileM*256*sizeof(i32),ACL_MEM_MALLOC_HUGE_FIRST));GmmMidUnifiedRect_grouped_matmul_gelu_custom<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileM);
aclError status=aclrtSynchronizeStream(stream);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
__aicore__ inline float GmmSquareTail31Refine(float z){
if(z>-5.5f&&z<-4.0f){
i64 zq=GmmMidK_GmmFloatToQ(z);i64 cubic=GmmMidK_GmmQMul(GmmMidK_GmmQMul(zq,zq),zq);i64 inner=GmmMidK_GmmQMul(898338152679114LL,zq+GmmMidK_GmmQMul(50344614334468LL,cubic));i64 a=-2*inner,n=a/780414346020670LL;i64 r=a-n*780414346020670LL;i64 r31=(r+(1LL<<18))>>19;i64 p=592LL;p=((p*r31+(1LL<<30))>>31)+(-5918LL);
p=((p*r31+(1LL<<30))>>31)+(53261LL);p=((p*r31+(1LL<<30))>>31)+(-426088LL);p=((p*r31+(1LL<<30))>>31)+(2982616LL);p=((p*r31+(1LL<<30))>>31)+(-17895697LL);p=((p*r31+(1LL<<30))>>31)+(89478485LL);p=((p*r31+(1LL<<30))>>31)+(-357913941LL);p=((p*r31+(1LL<<30))>>31)+(1073741824LL);p=((p*r31+(1LL<<30))>>31)+(-2147483648LL);
p=((p*r31+(1LL<<30))>>31)+(2147483648LL);union{float f;u32 bits;}tail;tail.f=GmmMidK_GmmQToFloat(p<<19);tail.bits-=(u32)n<<23;float f=(z*tail.f)*(1.0f-tail.f);float q=-f*16777216.0f;i32 whole=(i32)q;float fraction=q-(float)whole;float distance=fraction>0.5f?fraction-0.5f:0.5f-fraction;
if(distance>q*0.000001f+0.000001f)return f;
}
return GmmMidK_GmmRefine(z);
}
__aicore__ inline void GmmSquareWordQConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,LocalTensor<float>poly,LocalTensor<i32>wordIndex,LocalTensor<uint8_t>normal,LocalTensor<uint8_t>tail0,LocalTensor<uint8_t>tail1,LocalTensor<uint8_t>tail2,LocalTensor<uint8_t>tail3,i64 globalRow,i32 col,i32 e,i32 tileRows,i32 N,i32 tileM,i32 bank){
i32 nr=256;for(i32 slab=0;slab<tileRows;slab+=16){
i32 mr=tileRows-slab<16?tileRows-slab:16;DataCopy(cint,c[bank*tileM*256+slab*256],4096);DataCopyExtParams sp{1,(u32)nr*4,0,0,0};DataCopyPad(sc,scale[(i64)e*N+col],sp,DataCopyPadExtParams<float>{true,0,0,0});auto ev=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(ev);WaitFlag<HardEvent::MTE2_V>(ev);Cast(z,cint,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();for(int chunk=0;chunk<4;++chunk){
Mul(z[chunk*64],z[chunk*64],sc[chunk*64],(u64)64,(uint8_t)mr,BinaryRepeatParams{1,1,1,32,32,0});
}
PipeBarrier<PIPE_V>();Abs(u,z,4096);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,4096);PipeBarrier<PIPE_V>();Mul(t,u,u,4096);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,4096);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,4096);PipeBarrier<PIPE_V>();Mul(u,u,t,4096);PipeBarrier<PIPE_V>();Muls(u,u,-0.04986778505017909f,4096);PipeBarrier<PIPE_V>();Duplicate(poly,0.0060438121814739504f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.040085832623197183f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.16614812982932259f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.49992211956100152f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999569069746141f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999996114843803f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Adds(t,poly,1.0f,4096);PipeBarrier<PIPE_V>();Muls(u,poly,0.00008f,4096);PipeBarrier<PIPE_V>();Adds(u,u,0.000001f,4096);PipeBarrier<PIPE_V>();CompareScalar(tail0,z,0.0f,CMPMODE::LT,4096);PipeBarrier<PIPE_V>();Mul(f,z,poly,4096);
PipeBarrier<PIPE_V>();Select(f,tail0,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Adds(poly,f,0.0f,4096);PipeBarrier<PIPE_V>();Div(f,f,t,4096);PipeBarrier<PIPE_V>();auto scratch=cint.ReinterpretCast<float>();Mul(scratch,f,t,4096);PipeBarrier<PIPE_V>();Sub(scratch,scratch,poly,4096);PipeBarrier<PIPE_V>();Abs(scratch,scratch,4096);PipeBarrier<PIPE_V>();Abs(poly,poly,4096);PipeBarrier<PIPE_V>();Muls(poly,poly,0.0000002f,4096);PipeBarrier<PIPE_V>();Compare(tail3,scratch,poly,CMPMODE::LE,4096);
PipeBarrier<PIPE_V>();Duplicate(t,0.00008f,4096);PipeBarrier<PIPE_V>();Select(u,tail0,t,u,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Abs(t,f,4096);PipeBarrier<PIPE_V>();Maxs(poly,t,0.00006103515625f,4096);PipeBarrier<PIPE_V>();Duplicate(scratch.ReinterpretCast<i32>(),(i32)0x7f800000,4096);PipeBarrier<PIPE_V>();And(poly.ReinterpretCast<uint16_t>(),poly.ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),8192);PipeBarrier<PIPE_V>();Duplicate(scratch.ReinterpretCast<i32>(),(i32)-2080374784,4096);
PipeBarrier<PIPE_V>();Sub(poly.ReinterpretCast<i32>(),scratch.ReinterpretCast<i32>(),poly.ReinterpretCast<i32>(),4096);PipeBarrier<PIPE_V>();Mul(t,t,poly,4096);PipeBarrier<PIPE_V>();Mul(u,u,t,4096);PipeBarrier<PIPE_V>();Muls(u,u,2.0f,4096);PipeBarrier<PIPE_V>();Cast(scratch.ReinterpretCast<i32>(),t,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();Cast(scratch,scratch.ReinterpretCast<i32>(),RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();Sub(scratch,scratch,t,4096);PipeBarrier<PIPE_V>();Abs(scratch,scratch,4096);
PipeBarrier<PIPE_V>();Add(scratch,scratch,u,4096);PipeBarrier<PIPE_V>();CompareScalar(normal,scratch,0.5f,CMPMODE::LT,4096);PipeBarrier<PIPE_V>();Abs(t,f,4096);PipeBarrier<PIPE_V>();CompareScalar(tail1,t,0.00006103515625f,CMPMODE::GE,4096);CompareScalar(tail2,t,1.99f,CMPMODE::LT,4096);PipeBarrier<PIPE_V>();And(tail1.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),tail2.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),256);
PipeBarrier<PIPE_V>();And(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail3.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();CompareScalar(tail1,z,-5.5f,CMPMODE::LE,4096);PipeBarrier<PIPE_V>();Abs(t,z,4096);PipeBarrier<PIPE_V>();CompareScalar(tail2,t,0.00000001490116119384765625f,CMPMODE::LE,4096);PipeBarrier<PIPE_V>();Or(tail1.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),tail2.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Duplicate(poly,0.0f,4096);
PipeBarrier<PIPE_V>();Select(f,tail1,poly,f,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();CompareScalar(tail1,z,7.1f,CMPMODE::GE,4096);PipeBarrier<PIPE_V>();Select(f,tail1,z,f,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Duplicate(scratch.ReinterpretCast<i32>(),(i32)8191,4096);
PipeBarrier<PIPE_V>();And(poly.ReinterpretCast<uint16_t>(),z.ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),8192);PipeBarrier<PIPE_V>();CompareScalar(tail0,poly.ReinterpretCast<i32>(),(i32)4096,CMPMODE::EQ,4096);CompareScalar(tail1,z,5.5f,CMPMODE::GE,4096);CompareScalar(tail2,z,7.0f,CMPMODE::LE,4096);PipeBarrier<PIPE_V>();And(tail0.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),tail1.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();And(tail0.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),tail2.ReinterpretCast<uint16_t>(),256);
PipeBarrier<PIPE_V>();Adds(poly.ReinterpretCast<i32>(),z.ReinterpretCast<i32>(),(i32)-1,4096);PipeBarrier<PIPE_V>();Select(f,tail0,poly,f,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);PipeBarrier<PIPE_V>();Or(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),tail0.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Not(normal.ReinterpretCast<uint16_t>(),normal.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Cast(u,normal.ReinterpretCast<i32>(),RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();CompareScalar(tail0,u,0.0f,CMPMODE::NE,128);
PipeBarrier<PIPE_V>();auto patRead=pipe.FetchEventID(HardEvent::V_S);SetFlag<HardEvent::V_S>(patRead);WaitFlag<HardEvent::V_S>(patRead);u64 second=tail0.ReinterpretCast<u64>().GetValue(1);tail0.ReinterpretCast<u64>().SetValue(4,second);auto patWrite=pipe.FetchEventID(HardEvent::S_V);SetFlag<HardEvent::S_V>(patWrite);WaitFlag<HardEvent::S_V>(patWrite);u64 wordCount=0,indexCount=0;GatherMask(poly.ReinterpretCast<i32>(),normal.ReinterpretCast<i32>(),tail0.ReinterpretCast<u32>(),false,0,GatherMaskParams{1,2,8,1},wordCount);
PipeBarrier<PIPE_V>();GatherMask(u.ReinterpretCast<i32>(),wordIndex,tail0.ReinterpretCast<u32>(),false,0,GatherMaskParams{1,2,8,1},indexCount);PipeBarrier<PIPE_V>();auto vs=pipe.FetchEventID(HardEvent::V_S);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);for(int i=0;i<(int)wordCount;++i){
int word=u.ReinterpretCast<i32>().GetValue(i);if(word>=mr*8)continue;u32 bits=poly.ReinterpretCast<u32>().GetValue(i);while(bits){
int lane=word*32+__builtin_ctz(bits);bits&=bits-1;f.SetValue(lane,GmmSquareTail31Refine(z.GetValue(lane)));
}}
auto sv=pipe.FetchEventID(HardEvent::S_V);SetFlag<HardEvent::S_V>(sv);WaitFlag<HardEvent::S_V>(sv);Cast(h,f,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();auto vo=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(vo);WaitFlag<HardEvent::V_MTE3>(vo);DataCopyExtParams op{(uint16_t)mr,(u32)nr*2,0,(u32)(N-nr)*2,0};DataCopyPad(out[(globalRow+slab)*N+col],h,op);auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}}
__global__ __aicore__ __mix__(1,2)void GmmSquareWordQ_grouped_matmul_gelu_custom(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileM){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<GmmMidK_GmmA,GmmMidK_GmmB,GmmMidK_GmmC,GmmMidK_GmmC,CFG_MDL>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2*tileM*256);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,1024);pipe.InitBuffer(yy,8192);TBuf<TPosition::VECCALC>polyBuffer,normalBuffer;pipe.InitBuffer(polyBuffer,16384);pipe.InitBuffer(normalBuffer,1024);auto poly=polyBuffer.Get<float>();auto normal=normalBuffer.Get<uint8_t>();
TBuf<TPosition::VECCALC>tailBuffer;pipe.InitBuffer(tailBuffer,2048);auto tail0=tailBuffer.Get<uint8_t>();auto tail1=tail0[512];auto tail2=tail0[1024];auto tail3=tail0[1536];auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();
auto h=yy.Get<half>();TBuf<TPosition::VECCALC>indexBuffer;pipe.InitBuffer(indexBuffer,512);auto wordIndex=indexBuffer.Get<i32>();CreateVecIndex(wordIndex,(i32)0,128);PipeBarrier<PIPE_V>();i64 start=0,tileBase=0;bool pending=false;i32 bank=0,savedExpert=0,savedCol=0,savedRows=0;i64 savedGlobalRow=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=N/256;i64 count=((rows+tileM-1)/tileM)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*tileM,col=(tile%nc)*256;i32 tileRows=rows-row<tileM?rows-row:tileM;i32 producerBank=pending?bank^1:bank;mm.SetOrgShape(M,N,K,K,256);mm.SetSingleShape(tileRows,256,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);if(pending){
mm.IterateAll<false>(c[producerBank*tileM*256],0,false,true);GmmSquareWordQConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,wordIndex,normal,tail0,tail1,tail2,tail3,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);mm.WaitIterateAll();mm.End();bank=producerBank;
}else{mm.IterateAll(c[producerBank*tileM*256]);mm.End();}
pending=true;savedGlobalRow=start+row;savedCol=col;savedExpert=e;savedRows=tileRows;
}
tileBase+=count;start=end;
}
if(pending){
GmmSquareWordQConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,wordIndex,normal,tail0,tail1,tail2,tail3,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);
}}}
static void GmmSquareWordQRun(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];if(N%256!=0){GmmMidTailBoundedRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);if(ubBytes<143872){GmmMidTailBoundedRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());if(!blocks)throw std::runtime_error("no Cube cores");matmul_tiling::MatmulApiTiling mt(*platform);mt.SetMatmulConfigParams(1);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);
i32 tileM=32;mt.SetShape(tileM,256,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(tileM,256,128);mt.SetBufferSpace(-1,-1,-1);TCubeTiling tiling;if(mt.GetTiling(tiling)!=0){GmmMidTailBoundedRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*2*tileM*256*sizeof(i32),ACL_MEM_MALLOC_HUGE_FIRST));GmmSquareWordQ_grouped_matmul_gelu_custom<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileM);
aclError status=aclrtSynchronizeStream(stream);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
__aicore__ inline void GmmSquarePanelConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,LocalTensor<float>poly,i64 globalRow,i32 col,i32 e,i32 tileRows,i32 N,i32 tileM,i32 bank){
for(int slab=0;slab<tileRows;slab+=16){
int mr=tileRows-slab<16?tileRows-slab:16;DataCopy(cint,c[bank*tileM*256+slab*256],4096);DataCopyExtParams sp{1,1024,0,0,0};DataCopyPad(sc,scale[(i64)e*N+col],sp,DataCopyPadExtParams<float>{true,0,0,0});auto ev=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(ev);WaitFlag<HardEvent::MTE2_V>(ev);Cast(z,cint,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();for(int chunk=0;chunk<4;++chunk){
Mul(z[chunk*64],z[chunk*64],sc[chunk*64],(u64)64,(uint8_t)mr,BinaryRepeatParams{1,1,1,32,32,0});
}
PipeBarrier<PIPE_V>();Abs(u,z,4096);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,4096);PipeBarrier<PIPE_V>();Mul(t,u,u,4096);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,4096);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,4096);PipeBarrier<PIPE_V>();Mul(u,u,t,4096);PipeBarrier<PIPE_V>();Muls(u,u,-0.04986778505017909f,4096);PipeBarrier<PIPE_V>();Duplicate(poly,0.0060438121814739504f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.040085832623197183f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.16614812982932259f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.49992211956100152f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999569069746141f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999996114843803f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Adds(t,poly,1.0f,4096);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,4096);PipeBarrier<PIPE_V>();Mul(f,z,poly,4096);PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);
PipeBarrier<PIPE_V>();auto scratch=cint.ReinterpretCast<float>();Muls(poly,t,0.32323232323232326f,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,-1.4545454545454546f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,t,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,2.1212121212121211f,4096);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,4096);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,4096);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,4096);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,4096);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,4096);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,4096);PipeBarrier<PIPE_V>();Mul(f,f,poly,4096);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();auto output=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(output);WaitFlag<HardEvent::V_MTE3>(output);
DataCopyExtParams op{(uint16_t)mr,512,0,(u32)(N-256)*2,0};DataCopyPad(out[(globalRow+slab)*N+col],h,op);auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}}
__global__ __aicore__ __mix__(1,2)void GmmSquarePanel_grouped_matmul_gelu_custom(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileM){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<GmmMidK_GmmA,GmmMidK_GmmB,GmmMidK_GmmC,GmmMidK_GmmC,CFG_MDL>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2*tileM*256);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,1024);pipe.InitBuffer(yy,8192);TBuf<TPosition::VECCALC>polyBuffer;pipe.InitBuffer(polyBuffer,16384);auto poly=polyBuffer.Get<float>();auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();
auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();i64 start=0,tileBase=0;bool pending=false;i32 bank=0,savedExpert=0,savedCol=0,savedRows=0;i64 savedGlobalRow=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=N/256;i64 count=((rows+tileM-1)/tileM)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*tileM,col=(tile%nc)*256;i32 tileRows=rows-row<tileM?rows-row:tileM;i32 producerBank=pending?bank^1:bank;mm.SetOrgShape(M,N,K,K,256);mm.SetSingleShape(tileRows,256,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);if(pending){
mm.IterateAll<false>(c[producerBank*tileM*256],0,false,true);GmmSquarePanelConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);mm.WaitIterateAll();mm.End();bank=producerBank;
}else{mm.IterateAll(c[producerBank*tileM*256]);mm.End();}
pending=true;savedGlobalRow=start+row;savedCol=col;savedExpert=e;savedRows=tileRows;
}
tileBase+=count;start=end;
}
if(pending){GmmSquarePanelConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);}}}
static void GmmSquarePanelRun(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];if(N%256!=0){GmmMidTailBoundedRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);if(ubBytes<143872){GmmMidTailBoundedRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());if(!blocks)throw std::runtime_error("no Cube cores");matmul_tiling::MatmulApiTiling mt(*platform);mt.SetMatmulConfigParams(1);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);
i32 tileM=64;mt.SetShape(tileM,256,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(tileM,256,128);mt.SetBufferSpace(-1,-1,-1);TCubeTiling tiling;if(mt.GetTiling(tiling)!=0){GmmMidTailBoundedRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);return;}
void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*2*tileM*256*sizeof(i32),ACL_MEM_MALLOC_HUGE_FIRST));GmmSquarePanel_grouped_matmul_gelu_custom<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileM);
aclError status=aclrtSynchronizeStream(stream);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
__aicore__ inline void GmmMidPanel256Consume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,LocalTensor<float>poly,i64 globalRow,i32 col,i32 e,i32 tileRows,i32 N,i32 tileM,i32 bank){
for(int slab=0;slab<tileRows;slab+=16){
int mr=tileRows-slab<16?tileRows-slab:16;DataCopy(cint,c[bank*tileM*256+slab*256],4096);int nr=N-col<256?N-col:256;Duplicate(sc,0.0f,256);PipeBarrier<PIPE_V>();auto scaleReady=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(scaleReady);WaitFlag<HardEvent::V_MTE2>(scaleReady);DataCopyExtParams sp{1,(u32)nr*4,0,0,0};DataCopyPad(sc,scale[(i64)e*N+col],sp,DataCopyPadExtParams<float>{true,0,0,0});
auto ev=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(ev);WaitFlag<HardEvent::MTE2_V>(ev);Cast(z,cint,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();for(int chunk=0;chunk<4;++chunk){
Mul(z[chunk*64],z[chunk*64],sc[chunk*64],(u64)64,(uint8_t)mr,BinaryRepeatParams{1,1,1,32,32,0});
}
PipeBarrier<PIPE_V>();Abs(u,z,4096);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,4096);PipeBarrier<PIPE_V>();Mul(t,u,u,4096);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,4096);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,4096);PipeBarrier<PIPE_V>();Mul(u,u,t,4096);PipeBarrier<PIPE_V>();Muls(u,u,-0.04986778505017909f,4096);PipeBarrier<PIPE_V>();Duplicate(poly,0.0060438121814739504f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.040085832623197183f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.16614812982932259f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.49992211956100152f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999569069746141f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999996114843803f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Adds(t,poly,1.0f,4096);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,4096);PipeBarrier<PIPE_V>();Mul(f,z,poly,4096);PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);
PipeBarrier<PIPE_V>();auto scratch=cint.ReinterpretCast<float>();Muls(poly,t,0.32323232323232326f,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,-1.4545454545454546f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,t,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,2.1212121212121211f,4096);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,4096);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,4096);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,4096);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,4096);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,4096);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,4096);PipeBarrier<PIPE_V>();Mul(f,f,poly,4096);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();auto output=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(output);WaitFlag<HardEvent::V_MTE3>(output);
DataCopyExtParams op{(uint16_t)mr,(u32)nr*2,(u32)(256-((nr+15)/16)*16)/16,(u32)(N-nr)*2,0};DataCopyPad(out[(globalRow+slab)*N+col],h,op);auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}}
__global__ __aicore__ __mix__(1,2)void GmmMidPanel256_grouped_matmul_gelu_custom(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileM){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<GmmMidK_GmmA,GmmMidK_GmmB,GmmMidK_GmmC,GmmMidK_GmmC,CFG_MDL>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2*tileM*256);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,1024);pipe.InitBuffer(yy,8192);TBuf<TPosition::VECCALC>polyBuffer;pipe.InitBuffer(polyBuffer,16384);auto poly=polyBuffer.Get<float>();auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();
auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();i64 start=0,tileBase=0;bool pending=false;i32 bank=0,savedExpert=0,savedCol=0,savedRows=0;i64 savedGlobalRow=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+255)/256;i64 count=((rows+tileM-1)/tileM)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*tileM,col=(tile%nc)*256;i32 tileRows=rows-row<tileM?rows-row:tileM;i32 producerBank=pending?bank^1:bank;mm.SetOrgShape(M,N,K,K,256);mm.SetSingleShape(tileRows,N-col<256?N-col:256,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);if(pending){
mm.IterateAll<false>(c[producerBank*tileM*256],0,false,true);GmmMidPanel256Consume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);mm.WaitIterateAll();mm.End();bank=producerBank;
}else{mm.IterateAll(c[producerBank*tileM*256]);mm.End();}
pending=true;savedGlobalRow=start+row;savedCol=col;savedExpert=e;savedRows=tileRows;
}
tileBase+=count;start=end;
}
if(pending){GmmMidPanel256Consume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);}}}
static void GmmMidPanel256Run(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream,bool wideFallback=false){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];auto fallback=[&](){
if(wideFallback)GmmMidKRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);else GmmMidUnifiedRectRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);
};auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);if(ubBytes<143872){fallback();return;}
u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());if(!blocks)throw std::runtime_error("no Cube cores");matmul_tiling::MatmulApiTiling mt(*platform);mt.SetMatmulConfigParams(1);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);
i32 tileM=64;mt.SetShape(tileM,256,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(tileM,256,128);mt.SetBufferSpace(-1,-1,-1);TCubeTiling tiling;if(mt.GetTiling(tiling)!=0){fallback();return;}
void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*2*tileM*256*sizeof(i32),ACL_MEM_MALLOC_HUGE_FIRST));GmmMidPanel256_grouped_matmul_gelu_custom<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileM);
aclError status=aclrtSynchronizeStream(stream);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
__aicore__ inline void GmmMidR6NarrowConsume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,LocalTensor<float>poly,i64 globalRow,i32 col,i32 e,i32 tileRows,i32 N,i32 tileM,i32 bank){
for(int slab=0;slab<tileRows;slab+=16){
int mr=tileRows-slab<16?tileRows-slab:16;DataCopyExtParams cp{(uint16_t)mr,256,768,0,0};DataCopyPad(cint,c[bank*tileM*256+slab*256],cp,DataCopyPadExtParams<i32>{false,0,0,0});int nr=N-col<256?N-col:256;Duplicate(sc,0.0f,64);PipeBarrier<PIPE_V>();auto scaleReady=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(scaleReady);WaitFlag<HardEvent::V_MTE2>(scaleReady);DataCopyExtParams sp{1,(u32)nr*4,0,0,0};DataCopyPad(sc,scale[(i64)e*N+col],sp,DataCopyPadExtParams<float>{true,0,0,0});
auto ev=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(ev);WaitFlag<HardEvent::MTE2_V>(ev);Cast(z,cint,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();for(int chunk=0;chunk<1;++chunk){
Mul(z[chunk*64],z[chunk*64],sc[chunk*64],(u64)64,(uint8_t)mr,BinaryRepeatParams{1,1,1,8,8,0});
}
PipeBarrier<PIPE_V>();Abs(u,z,1024);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,1024);PipeBarrier<PIPE_V>();Mul(t,u,u,1024);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,1024);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,1024);PipeBarrier<PIPE_V>();Mul(u,u,t,1024);PipeBarrier<PIPE_V>();Muls(u,u,-0.04986778505017909f,1024);PipeBarrier<PIPE_V>();Duplicate(poly,0.0060438121814739504f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.040085832623197183f,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.16614812982932259f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.49992211956100152f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999569069746141f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,u,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999996114843803f,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,1024);PipeBarrier<PIPE_V>();Adds(t,poly,1.0f,1024);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,1024);PipeBarrier<PIPE_V>();Mul(f,z,poly,1024);PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,1024);
PipeBarrier<PIPE_V>();auto scratch=cint.ReinterpretCast<float>();Muls(poly,t,0.32323232323232326f,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,-1.4545454545454546f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,t,1024);PipeBarrier<PIPE_V>();Adds(poly,poly,2.1212121212121211f,1024);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,1024);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,1024);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,1024);
PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,1024);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,1024);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,1024);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,1024);PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,1024);PipeBarrier<PIPE_V>();Mul(f,f,poly,1024);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();auto output=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(output);WaitFlag<HardEvent::V_MTE3>(output);
DataCopyExtParams op{(uint16_t)mr,(u32)nr*2,(u32)(64-((nr+15)/16)*16)/16,(u32)(N-nr)*2,0};DataCopyPad(out[(globalRow+slab)*N+col],h,op);auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}}
__aicore__ inline void GmmMidR6Consume(TPipe&pipe,GlobalTensor<i32>c,GlobalTensor<float>scale,GlobalTensor<half>out,LocalTensor<i32>cint,LocalTensor<float>z,LocalTensor<float>u,LocalTensor<float>t,LocalTensor<float>f,LocalTensor<float>sc,LocalTensor<half>h,LocalTensor<float>poly,i64 globalRow,i32 col,i32 e,i32 tileRows,i32 N,i32 tileM,i32 bank){
if(N-col<=64){GmmMidR6NarrowConsume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,globalRow,col,e,tileRows,N,tileM,bank);return;}
for(int slab=0;slab<tileRows;slab+=16){
int mr=tileRows-slab<16?tileRows-slab:16;DataCopy(cint,c[bank*tileM*256+slab*256],4096);int nr=N-col<256?N-col:256;Duplicate(sc,0.0f,256);PipeBarrier<PIPE_V>();auto scaleReady=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(scaleReady);WaitFlag<HardEvent::V_MTE2>(scaleReady);DataCopyExtParams sp{1,(u32)nr*4,0,0,0};DataCopyPad(sc,scale[(i64)e*N+col],sp,DataCopyPadExtParams<float>{true,0,0,0});
auto ev=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(ev);WaitFlag<HardEvent::MTE2_V>(ev);Cast(z,cint,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();for(int chunk=0;chunk<4;++chunk){
Mul(z[chunk*64],z[chunk*64],sc[chunk*64],(u64)64,(uint8_t)mr,BinaryRepeatParams{1,1,1,32,32,0});
}
PipeBarrier<PIPE_V>();Abs(u,z,4096);PipeBarrier<PIPE_V>();Mins(u,u,5.5f,4096);PipeBarrier<PIPE_V>();Mul(t,u,u,4096);PipeBarrier<PIPE_V>();Muls(t,t,0.044715f,4096);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,4096);PipeBarrier<PIPE_V>();Mul(u,u,t,4096);PipeBarrier<PIPE_V>();Muls(u,u,-0.04986778505017909f,4096);PipeBarrier<PIPE_V>();Duplicate(poly,0.0060438121814739504f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.040085832623197183f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.16614812982932259f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.49992211956100152f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999569069746141f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,u,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,0.99999996114843803f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,poly,4096);PipeBarrier<PIPE_V>();Adds(t,poly,1.0f,4096);PipeBarrier<PIPE_V>();auto sign=cint.ReinterpretCast<uint8_t>();CompareScalar(sign,z,0.0f,CMPMODE::LT,4096);PipeBarrier<PIPE_V>();Mul(f,z,poly,4096);PipeBarrier<PIPE_V>();Select(f,sign,f,z,SELMODE::VSEL_TENSOR_TENSOR_MODE,4096);
PipeBarrier<PIPE_V>();auto scratch=cint.ReinterpretCast<float>();Muls(poly,t,0.32323232323232326f,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,-1.4545454545454546f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,t,4096);PipeBarrier<PIPE_V>();Adds(poly,poly,2.1212121212121211f,4096);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,4096);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,4096);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,4096);
PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,4096);PipeBarrier<PIPE_V>();Mul(scratch,t,poly,4096);PipeBarrier<PIPE_V>();Muls(scratch,scratch,-1.0f,4096);PipeBarrier<PIPE_V>();Adds(scratch,scratch,2.0f,4096);PipeBarrier<PIPE_V>();Mul(poly,poly,scratch,4096);PipeBarrier<PIPE_V>();Mul(f,f,poly,4096);PipeBarrier<PIPE_V>();Cast(h,f,RoundMode::CAST_RINT,4096);PipeBarrier<PIPE_V>();auto output=pipe.FetchEventID(HardEvent::V_MTE3);SetFlag<HardEvent::V_MTE3>(output);WaitFlag<HardEvent::V_MTE3>(output);
DataCopyExtParams op{(uint16_t)mr,(u32)nr*2,(u32)(256-((nr+15)/16)*16)/16,(u32)(N-nr)*2,0};DataCopyPad(out[(globalRow+slab)*N+col],h,op);auto reuse=pipe.FetchEventID(HardEvent::MTE3_MTE2);SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
}}
__global__ __aicore__ __mix__(1,2)void GmmMidR6_grouped_matmul_gelu_custom(GM_ADDR x,GM_ADDR w,GM_ADDR s,GM_ADDR groups,GM_ADDR y,GM_ADDR workspace,GM_ADDR scratch,TCubeTiling tiling,i32 M,i32 K,i32 N,i32 E,i32 tileM){
SetSysWorkspaceForce(workspace);clearWorkspace(workspace);TPipe pipe;Matmul<GmmMidK_GmmA,GmmMidK_GmmB,GmmMidK_GmmC,GmmMidK_GmmC,CFG_MDL>mm;REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&tiling);if ASCEND_IS_AIV{
const i32 id=GetBlockIdx();const i32 workers=GetBlockNum()*2;GlobalTensor<int8_t>a,b;GlobalTensor<float>scale;GlobalTensor<i64>ends;GlobalTensor<half>out;GlobalTensor<i32>c;a.SetGlobalBuffer((__gm__ int8_t*)x);b.SetGlobalBuffer((__gm__ int8_t*)w);scale.SetGlobalBuffer((__gm__ float*)s);ends.SetGlobalBuffer((__gm__ i64*)groups);out.SetGlobalBuffer((__gm__ half*)y);c.SetGlobalBuffer((__gm__ i32*)scratch+id*2*tileM*256);TBuf<TPosition::VECCALC>ci,zz,uu,tt,ff,ss,yy;
pipe.InitBuffer(ci,16384);pipe.InitBuffer(zz,16384);pipe.InitBuffer(uu,16384);pipe.InitBuffer(tt,16384);pipe.InitBuffer(ff,16384);pipe.InitBuffer(ss,1024);pipe.InitBuffer(yy,8192);TBuf<TPosition::VECCALC>polyBuffer;pipe.InitBuffer(polyBuffer,16384);auto poly=polyBuffer.Get<float>();auto cint=ci.Get<i32>();auto z=zz.Get<float>();auto u=uu.Get<float>();
auto t=tt.Get<float>();auto f=ff.Get<float>();auto sc=ss.Get<float>();auto h=yy.Get<half>();i64 start=0,tileBase=0;bool pending=false;i32 bank=0,savedExpert=0,savedCol=0,savedRows=0;i64 savedGlobalRow=0;for(i32 e=0;e<E;++e){
i64 end=ends.GetValue(e);i32 rows=end-start;i32 nc=(N+255)/256;i64 count=((rows+tileM-1)/tileM)*nc;for(i64 tile=(id-tileBase%workers+workers)%workers;tile<count;tile+=workers){
i32 row=(tile/nc)*tileM,col=(tile%nc)*256;i32 tileRows=rows-row<tileM?rows-row:tileM;i32 producerBank=pending?bank^1:bank;mm.SetOrgShape(M,N,K,K,256);mm.SetSingleShape(tileRows,N-col<256?N-col:256,K);mm.SetTensorA(a[(start+row)*K]);mm.SetTensorB(b[(i64)e*K*N+col]);if(pending){
mm.IterateAll<false>(c[producerBank*tileM*256],0,false,true);GmmMidR6Consume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);mm.WaitIterateAll();mm.End();bank=producerBank;
}else{mm.IterateAll(c[producerBank*tileM*256]);mm.End();}
pending=true;savedGlobalRow=start+row;savedCol=col;savedExpert=e;savedRows=tileRows;
}
tileBase+=count;start=end;
}
if(pending){GmmMidR6Consume(pipe,c,scale,out,cint,z,u,t,f,sc,h,poly,savedGlobalRow,savedCol,savedExpert,savedRows,N,tileM,bank);}}}
static void GmmMidR6Run(GM_ADDR x,const TensorGroupInfo&ix,GM_ADDR weight,const TensorGroupInfo&iw,GM_ADDR scale,const TensorGroupInfo&isc,GM_ADDR group_list,const TensorGroupInfo&ig,GM_ADDR y,const TensorGroupInfo&iy,i64 acn,aclrtStream stream,bool wideFallback=false){
const i32 M=ix.tensors[0].shape[0],K=ix.tensors[0].shape[1];const i32 E=iw.tensors[0].shape[0],N=iw.tensors[0].shape[2];if(N%32==0){GmmMidPanel256Run(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream,wideFallback);return;}
auto fallback=[&](){
if(wideFallback)GmmMidKRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);else GmmMidUnifiedRectRun(x,ix,weight,iw,scale,isc,group_list,ig,y,iy,acn,stream);
};auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();if(!platform||acn<1)throw std::runtime_error("invalid core budget/platform");u64 ubBytes=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);if(ubBytes<143872){fallback();return;}
u32 blocks=std::min<i64>(acn,platform->GetCoreNumAic());if(!blocks)throw std::runtime_error("no Cube cores");matmul_tiling::MatmulApiTiling mt(*platform);mt.SetMatmulConfigParams(1);mt.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT8);mt.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32);
i32 tileM=(N%32!=0&&(i64)M>64LL*E&&(i64)M<=128LL*E&&(i64)E*((N+255)/256)>=4LL*platform->GetCoreNumAic()?128:64);mt.SetShape(tileM,256,K);mt.SetOrgShape(M,N,K);mt.SetFixSplit(tileM,256,128);mt.SetBufferSpace(-1,-1,-1);TCubeTiling tiling;if(mt.GetTiling(tiling)!=0){
tileM=64;mt.SetShape(tileM,256,K);mt.SetFixSplit(tileM,256,128);if(mt.GetTiling(tiling)!=0){fallback();return;}}
void*workspace=nullptr,*scratch=nullptr;size_t sys=platform->GetLibApiWorkSpaceSize();auto checkGmm=[](aclError err){if(err!=ACL_SUCCESS)throw std::runtime_error("GMM ACL error "+std::to_string(err));};checkGmm(aclrtMalloc(&workspace,sys,ACL_MEM_MALLOC_HUGE_FIRST));checkGmm(aclrtMalloc(&scratch,blocks*2*2*tileM*256*sizeof(i32),ACL_MEM_MALLOC_HUGE_FIRST));GmmMidR6_grouped_matmul_gelu_custom<<<blocks,nullptr,stream>>>(x,weight,scale,group_list,y,(GM_ADDR)workspace,(GM_ADDR)scratch,tiling,M,K,N,E,tileM);
aclError status=aclrtSynchronizeStream(stream);aclrtFree(scratch);aclrtFree(workspace);checkGmm(status);
}
using namespace AscendC;

#endif // GROUPED_MATMUL_SHORT_MID_IMPL_H
