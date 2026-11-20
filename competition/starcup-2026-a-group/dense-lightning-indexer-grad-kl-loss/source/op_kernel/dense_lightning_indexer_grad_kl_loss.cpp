#include "kernel_operator.h"
#include "dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_key_dense_lightning_indexer_grad_kl_loss.h"
using namespace AscendC;
namespace {
constexpr int32_t MAIN_CACHE_COLS = 16;
constexpr int32_t CHUNK = 2048;
constexpr float   LOSS_EPS = 1e-8f;
constexpr int32_t GTILE = 16;
constexpr int32_t NIDX1_CAP = 64;
constexpr int32_t N1_CAP = 128;
constexpr int32_t RESIDENT_CAP = 4096;
constexpr bool kLeanRawB1 = true;
constexpr bool kLeanRawB2 = true;
constexpr bool kHeadRawP7 = true;
constexpr bool kHeadRawP6 = true;
constexpr bool kHeadRawP5 = true;
}
template <typename DTQ, bool WIDTH64, bool BATCH_TWO>
class KernelLeanPublic {
public:
using T = float;
using IN_T = DTQ;
using OUT_T = DTQ;
static constexpr uint32_t kN1 = 32u;
static constexpr uint32_t kNJ = 4u;
static constexpr uint32_t kNG = 8u;
static constexpr uint32_t kD = WIDTH64 ? 64u : 128u;
static constexpr uint32_t kJTP = 8u;
static constexpr uint32_t kMaskElems = 64u;
static constexpr uint32_t kDHalf = kD / 64u;
static constexpr bool kProdAll = (kNJ <= 4u);
static constexpr uint32_t kB = BATCH_TWO ? 2u : 1u;
static constexpr uint32_t kWsScatterBytes = kB * kNJ * kD * sizeof(T);
static constexpr uint32_t Align512(uint32_t x) { return (x + 511u) / 512u * 512u; }
static constexpr uint32_t Align32(uint32_t x) { return (x + 31u) / 32u * 32u; }
static constexpr uint32_t kInBytes = sizeof(IN_T);
static constexpr uint32_t kOutBytes = sizeof(OUT_T);
static constexpr uint32_t kF32Bytes = sizeof(T);
static constexpr uint32_t offQIn = 0u;
static constexpr uint32_t offKIn = offQIn + Align512(kN1 * kD * kInBytes);
static constexpr uint32_t offQiIn = offKIn + Align512(kNJ * kN1 * kD * kInBytes);
static constexpr uint32_t offKiIn = offQiIn + Align512(kNG * kD * kInBytes);
static constexpr uint32_t offWIn = offKiIn + Align512(kNJ * kD * kInBytes);
static constexpr uint32_t offQF = offWIn + Align512(kNG * kInBytes);
static constexpr uint32_t offKRowF = offQF + Align512(kN1 * kD * kF32Bytes);
static constexpr uint32_t offProdF = offKRowF + Align512(kN1 * kD * kF32Bytes);
static constexpr uint32_t offProdAll = offProdF + Align512(kN1 * kD * kF32Bytes);
static constexpr uint32_t offLogF = offProdAll + Align512(kNJ * kN1 * kD * kF32Bytes);
static constexpr uint32_t offPVec = offLogF + Align512(kNJ * kN1 * kF32Bytes);
static constexpr uint32_t offVec = offPVec + Align512(kJTP * kF32Bytes);
static constexpr uint32_t offQiF = offVec + Align512(3u * kN1 * kF32Bytes);
static constexpr uint32_t offKiF = offQiF + Align512(kNG * kD * kF32Bytes);
static constexpr uint32_t offSimF = offKiF + Align512(kNJ * kD * kF32Bytes);
static constexpr uint32_t offDSF = offSimF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offDS1F = offDSF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offCoefL1F = offDS1F + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offWTileF = offCoefL1F + Align512(kJTP * kMaskElems * kF32Bytes);
static constexpr uint32_t offOnesF = offWTileF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offTmpJTF = offOnesF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offTmp2F = offTmpJTF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offKeepF = offTmp2F + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offAccF = offKeepF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offDqF = offAccF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offDkF = offDqF + Align512(kNG * kD * kF32Bytes);
static constexpr uint32_t offWork = offDkF + Align512(kNJ * kD * kF32Bytes);
static constexpr uint32_t offSmall = offWork + Align512(4096u);
static constexpr uint32_t offMask = offSmall + Align512(16u * kJTP * kF32Bytes);
static constexpr uint32_t offOut = offMask + Align512(512u);
static constexpr uint32_t kArenaBytes =
offOut + Align512((kNG + kNJ + 16u) * kD * kOutBytes);
__aicore__ inline KernelLeanPublic() {}
__aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
GM_ADDR weights, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex,
GM_ADDR dWeights, GM_ADDR loss, GM_ADDR workspace,
const DenseLightningIndexerGradKlLossTilingData *td) {
t = td;
blockIdx_ = BATCH_TWO ? GetBlockIdx() : 0u;
queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
queryIndexGm.SetGlobalBuffer((__gm__ IN_T *)queryIndex);
keyIndexGm.SetGlobalBuffer((__gm__ IN_T *)keyIndex);
weights16Gm.SetGlobalBuffer((__gm__ IN_T *)weights);
dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
dWeights16Gm.SetGlobalBuffer((__gm__ IN_T *)dWeights);
lossGmRef.SetGlobalBuffer((__gm__ T *)loss);
pipe.InitBuffer(leanBuf_, Align32(kArenaBytes));
pipe.InitBuffer(smallBuf, 256u);
if (blockNum_ > 1u) {
__gm__ uint8_t *ws = (__gm__ uint8_t *)workspace;
const uint64_t scatterBytes = static_cast<uint64_t>(kWsScatterBytes);
const uint64_t syncOffset = t->wsOffsetScatter + ((scatterBytes + 511u) / 512u) * 512u;
syncGm.SetGlobalBuffer((__gm__ int32_t *)(ws + syncOffset));
pipe.InitBuffer(syncBuf, blockNum_ * 32u);
}
}
__aicore__ inline void Process() { RunLeanProcess(); }
private:
__aicore__ inline void RunLeanProcess() {
lossSum_ = 0.0f;
if constexpr (WIDTH64) {
if (blockNum_ > 1u) { InitSoftSync(); }
}
if (blockNum_ > 1u && blockIdx_ == 0u) {
lossGmRef.SetValue(0, 0.0F);
DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lossGmRef);
}
if (blockNum_ > 1u) {
if constexpr (WIDTH64) {
SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_);
} else {
SyncAll<true>();
}
}
{
LocalTensor<uint8_t> lb0 = leanBuf_.Get<uint8_t>();
Duplicate(lb0[offAccF].ReinterpretCast<T>(), static_cast<T>(0.0f), kJTP);
LocalTensor<T> kb = lb0[offKeepF].ReinterpretCast<T>();
LocalTensor<T> keepV = kb;
LocalTensor<T> compV = kb[kJTP];
Duplicate(keepV, static_cast<T>(0.0f), kJTP);
Adds(keepV, keepV, static_cast<T>(1.0f), kNJ);
Duplicate(compV, static_cast<T>(1.0f), kJTP);
Adds(compV, compV, static_cast<T>(-1.0f), kNJ);
PipeBarrier<PIPE_V>();
}
lossSum_ += RunLeanTask(blockIdx_, 0u);
if constexpr (!BATCH_TWO) { return; }
if constexpr (WIDTH64) {
if (blockNum_ > 1u) { SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_); }
}
LocalTensor<T> ls = smallBuf.Get<T>();
{
LocalTensor<uint8_t> lb = leanBuf_.Get<uint8_t>();
LocalTensor<T> acc = lb[offAccF].ReinterpretCast<T>();
LocalTensor<uint8_t> wk = lb[offWork].ReinterpretCast<uint8_t>();
uint32_t shAcc[2] = {1u, kJTP};
ReduceSum<T, Pattern::Reduce::AR, true>(ls, acc, wk, shAcc, true);
PipeBarrier<PIPE_V>();
}
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
if (blockNum_ > 1u) { SetAtomicAdd<float>(); }
DataCopyExtParams cpl(1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
DataCopyPad(lossGmRef, ls, cpl);
if (blockNum_ > 1u) { SetAtomicNone(); }
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
__aicore__ inline float RunLeanTask(uint32_t b, uint32_t s1) {
const uint32_t N1 = kN1, NJ = kNJ, NG = kNG, D = kD;
const uint32_t n1D = N1 * D, njD = NJ * D, JTP = kJTP;
const uint32_t dHalf = kDHalf;
LocalTensor<uint8_t> buf = leanBuf_.Get<uint8_t>();
LocalTensor<IN_T> qIn = buf[offQIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> kIn = buf[offKIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> qiIn = buf[offQiIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> kiIn = buf[offKiIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> wIn = buf[offWIn].ReinterpretCast<IN_T>();
LocalTensor<T> qF = buf[offQF].ReinterpretCast<T>();
LocalTensor<T> kRowF = buf[offKRowF].ReinterpretCast<T>();
LocalTensor<T> prodF = buf[offProdF].ReinterpretCast<T>();
LocalTensor<T> logF = buf[offLogF].ReinterpretCast<T>();
LocalTensor<T> pVec = buf[offPVec].ReinterpretCast<T>();
LocalTensor<T> mxF = buf[offVec].ReinterpretCast<T>();
LocalTensor<T> sumF = mxF[N1];
LocalTensor<T> tmpF = mxF[2u * N1];
LocalTensor<T> qiF = buf[offQiF].ReinterpretCast<T>();
LocalTensor<T> kiF = buf[offKiF].ReinterpretCast<T>();
LocalTensor<T> simM = buf[offSimF].ReinterpretCast<T>();
LocalTensor<T> coefM = buf[offDSF].ReinterpretCast<T>();
LocalTensor<T> dST = buf[offDS1F].ReinterpretCast<T>();
LocalTensor<T> coefL1 = buf[offCoefL1F].ReinterpretCast<T>();
LocalTensor<T> wTile = buf[offWTileF].ReinterpretCast<T>();
LocalTensor<T> onesV = buf[offOnesF].ReinterpretCast<T>();
LocalTensor<T> dlogL1 = buf[offTmpJTF].ReinterpretCast<T>();
LocalTensor<T> iTile = buf[offTmp2F].ReinterpretCast<T>();
LocalTensor<T> lossAcc = buf[offAccF].ReinterpretCast<T>();
LocalTensor<T> prodAll = (kProdAll ? buf[offProdAll] : buf[offProdF]).ReinterpretCast<T>();
LocalTensor<T> dqF = buf[offDqF].ReinterpretCast<T>();
LocalTensor<T> dkF = buf[offDkF].ReinterpretCast<T>();
LocalTensor<uint8_t> work = buf[offWork].ReinterpretCast<uint8_t>();
LocalTensor<T> small = buf[offSmall].ReinterpretCast<T>();
LocalTensor<T> iVec = small;
LocalTensor<T> yVec = small[2u * JTP];
LocalTensor<T> dlog = small[3u * JTP];
LocalTensor<T> lp = small[4u * JTP];
LocalTensor<T> ly = small[5u * JTP];
LocalTensor<T> pmax = small[6u * JTP];
LocalTensor<T> ymax = small[7u * JTP];
LocalTensor<T> wRow = small[8u * JTP];
LocalTensor<T> mxRep = small[9u * JTP];
LocalTensor<T> zRep = small[10u * JTP];
LocalTensor<T> keepV = buf[offKeepF].ReinterpretCast<T>();
LocalTensor<T> compV = keepV[JTP];
LocalTensor<T> tailV = keepV[2u * JTP];
LocalTensor<uint8_t> mask = buf[offMask].ReinterpretCast<uint8_t>();
LocalTensor<OUT_T> outStage = buf[offOut].ReinterpretCast<OUT_T>();
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, IN_T(0));
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qIn, queryGm[(static_cast<uint64_t>(b) * 1u + s1) * n1D],
DataCopyExtParams(1, n1D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(kIn, keyGm[static_cast<uint64_t>(b) * NJ * n1D],
DataCopyExtParams(1, NJ * n1D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(qiIn, queryIndexGm[(static_cast<uint64_t>(b) * 1u + s1) * NG * D],
DataCopyExtParams(1, NG * D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(kiIn, keyIndexGm[static_cast<uint64_t>(b) * njD],
DataCopyExtParams(1, njD * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(wIn, weights16Gm[(static_cast<uint64_t>(b) * 1u + s1) * NG],
DataCopyExtParams(1, NG * sizeof(IN_T), 0, 0, 0), pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
Cast(qF, qIn, RoundMode::CAST_NONE, n1D);
Cast(qiF, qiIn, RoundMode::CAST_NONE, NG * D);
Cast(kiF, kiIn, RoundMode::CAST_NONE, njD);
Duplicate(wRow, static_cast<T>(0.0f), JTP);
Cast(wRow, wIn, RoundMode::CAST_NONE, NG);
Duplicate(onesV, static_cast<T>(1.0f), JTP);
PipeBarrier<PIPE_V>();
Mul(wTile, wRow, onesV, static_cast<uint64_t>(JTP), static_cast<uint8_t>(JTP),
BinaryRepeatParams(1, 1, 1, 1, 0, 0));
Duplicate(mxF, static_cast<T>(-3.0e38f), N1);
Duplicate(sumF, static_cast<T>(0.0f), N1);
uint32_t shRowOne[2] = {N1, D};
if constexpr (!WIDTH64) {
Cast(prodAll, kIn, RoundMode::CAST_NONE, NJ * n1D);
}
for (uint32_t j = 0; j < NJ; ++j) {
if constexpr (WIDTH64) {
Cast(kRowF, kIn[j * n1D], RoundMode::CAST_NONE, n1D);
Mul(kRowF, kRowF, qF, n1D);
if (kProdAll) {
DataCopy(prodAll[j * n1D], kRowF, n1D);
} else {
ReduceSum<T, Pattern::Reduce::AR, true>(logF[j * N1], kRowF, work,
shRowOne, true);
}
} else {
Mul(prodAll[j * n1D], prodAll[j * n1D], qF, n1D);
}
}
if (kProdAll) {
uint32_t shRowAll[2] = {NJ * N1, D};
ReduceSum<T, Pattern::Reduce::AR, true>(logF, prodAll, work, shRowAll, true);
}
Muls(logF, logF, t->scaleValue, NJ * N1);
for (uint32_t j = 0; j < NJ; ++j) {
Max(mxF, mxF, logF[j * N1], N1);
}
{
const uint64_t maskRow = static_cast<uint64_t>(N1);
const uint8_t repNJ = static_cast<uint8_t>(NJ);
const uint8_t rowBlk = static_cast<uint8_t>(N1 / 8u);
Sub(logF, logF, mxF, maskRow, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
Exp(logF, logF, NJ * N1);
for (uint32_t j = 0; j < NJ; ++j) {
Add(sumF, sumF, logF[j * N1], N1);
}
Div(logF, logF, sumF, maskRow, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
}
Duplicate(pVec, static_cast<T>(0.0f), JTP);
uint32_t shAll[2] = {NJ, N1};
ReduceSum<T, Pattern::Reduce::AR, true>(pVec, logF, work, shAll, true);
Muls(pVec, pVec, static_cast<T>(1.0f / 32.0f), JTP);
Duplicate(simM, static_cast<T>(0.0f), kMaskElems);
PipeBarrier<PIPE_V>();
uint32_t shK[2] = {NG, D};
if (kProdAll) {
if (NG != JTP) {
Duplicate(prodAll, static_cast<T>(0.0f), NJ * JTP * D);
PipeBarrier<PIPE_V>();
}
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodAll[(j * JTP) * D + off], kiF[j * D + off], qiF[off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8), 0,
static_cast<uint8_t>(D / 8)));
}
}
uint32_t shKAll[2] = {NJ * JTP, D};
ReduceSum<T, Pattern::Reduce::AR, true>(simM, prodAll, work, shKAll, true);
} else {
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodF[off], kiF[j * D + off], qiF[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8), 0,
static_cast<uint8_t>(D / 8)));
}
ReduceSum<T, Pattern::Reduce::AR, true>(simM[j * JTP], prodF, work, shK, true);
PipeBarrier<PIPE_V>();
}
}
Maxs(simM, simM, static_cast<T>(0.0f), kMaskElems);
PipeBarrier<PIPE_V>();
Mul(dST, simM, wTile, kMaskElems);
PipeBarrier<PIPE_V>();
uint32_t shI0[2] = {NJ, JTP};
Duplicate(iVec, static_cast<T>(0.0f), JTP);
PipeBarrier<PIPE_V>();
ReduceSum<T, Pattern::Reduce::AR, true>(iVec, dST, work, shI0, true);
PipeBarrier<PIPE_V>();
uint32_t sh1[2] = {1u, NJ};
ReduceMax<T, Pattern::Reduce::AR, true>(tmpF, iVec, work, sh1, true);
PipeBarrier<PIPE_V>();
Brcb(mxRep, tmpF, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Sub(iVec, iVec, mxRep, JTP);
Muls(tailV, compV, static_cast<T>(3.0e38f), JTP);
Sub(iVec, iVec, tailV, JTP);
Exp(iVec, iVec, JTP);
ReduceSum<T, Pattern::Reduce::AR, true>(tmpF, iVec, work, sh1, true);
PipeBarrier<PIPE_V>();
Brcb(zRep, tmpF, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Div(yVec, iVec, zRep, JTP);
Maxs(pmax, pVec, static_cast<T>(LOSS_EPS), JTP);
Maxs(ymax, yVec, static_cast<T>(LOSS_EPS), JTP);
Log(lp, pmax, JTP);
Log(ly, ymax, JTP);
Sub(iTile, lp, ly, JTP);
Mul(iTile, iTile, pmax, JTP);
Add(lossAcc, lossAcc, iTile, JTP);
PipeBarrier<PIPE_V>();
Sub(dlog, yVec, pVec, JTP);
PipeBarrier<PIPE_V>();
Brcb(dlogL1, dlog, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Mul(coefM, dlogL1, wTile, kMaskElems);
CompareScalar(mask, simM, static_cast<T>(0.0f), CMPMODE::GT, kMaskElems);
PipeBarrier<PIPE_V>();
Select(coefM, mask, coefM, static_cast<T>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
kMaskElems);
PipeBarrier<PIPE_V>();
Brcb(coefL1, coefM, static_cast<uint8_t>(NJ), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
Mul(dST, simM, dlogL1, kMaskElems);
PipeBarrier<PIPE_V>();
Select(dST, mask, dST, static_cast<T>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
kMaskElems);
PipeBarrier<PIPE_V>();
uint32_t shDW[2] = {NJ, JTP};
ReduceSum<T, Pattern::Reduce::RA, true>(wRow, dST, work, shDW, true);
PipeBarrier<PIPE_V>();
Duplicate(dqF, static_cast<T>(0.0f), NG * D);
PipeBarrier<PIPE_V>();
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
MulAddDst(dqF[off], coefL1[j * 64u], kiF[j * D + off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D / 8), 1, 0));
}
PipeBarrier<PIPE_V>();
}
uint32_t shDKAll[2] = {NG, NJ * D};
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodAll[j * D + off], coefL1[j * 64u], qiF[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(NJ * D / 8), 1,
static_cast<uint8_t>(D / 8)));
}
}
ReduceSum<T, Pattern::Reduce::RA, true>(dkF[0], prodAll, work, shDKAll, true);
PipeBarrier<PIPE_V>();
if constexpr (!BATCH_TWO) {
LocalTensor<T> lsE50 = smallBuf.Get<T>();
uint32_t shAccE50[2] = {1u, kJTP};
ReduceSum<T, Pattern::Reduce::AR, true>(lsE50, lossAcc, work, shAccE50, true);
PipeBarrier<PIPE_V>();
}
LocalTensor<OUT_T> outQ = outStage;
LocalTensor<OUT_T> outK = outStage[NG * D];
LocalTensor<OUT_T> outW = outStage[NG * D + njD];
Cast(outQ, dqF, RoundMode::CAST_RINT, NG * D);
Cast(outK, dkF, RoundMode::CAST_RINT, njD);
Cast(outW, wRow, RoundMode::CAST_RINT, NG);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyPad(dQueryIndexGm[(static_cast<uint64_t>(b) * 1u + s1) * NG * D], outQ,
DataCopyExtParams(1, NG * D * sizeof(OUT_T), 0, 0, 0));
DataCopyPad(dKeyIndexGm[static_cast<uint64_t>(b) * njD], outK,
DataCopyExtParams(1, njD * sizeof(OUT_T), 0, 0, 0));
DataCopyPad(dWeights16Gm[(static_cast<uint64_t>(b) * 1u + s1) * NG], outW,
DataCopyExtParams(1, NG * sizeof(OUT_T), 0, 0, 0));
if constexpr (!BATCH_TWO) {
LocalTensor<T> lsE50 = smallBuf.Get<T>();
DataCopyExtParams cplE50(1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
DataCopyPad(lossGmRef, lsE50, cplE50);
}
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
return 0.0f;
}
__aicore__ inline void InitSoftSync() {
auto local = syncBuf.Get<int32_t>();
const uint32_t count = blockNum_ * 8u;
Duplicate(local, 0, count);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, count * sizeof(int32_t), 0, 0, 0);
DataCopyPad(syncGm, local, cp);
SetFlag<HardEvent::MTE3_S>(1); WaitFlag<HardEvent::MTE3_S>(1);
}
const DenseLightningIndexerGradKlLossTilingData *t = nullptr;
TPipe pipe;
TBuf<TPosition::VECCALC> leanBuf_;
TBuf<TPosition::VECCALC> smallBuf;
TBuf<TPosition::VECCALC> syncBuf;
GlobalTensor<IN_T> queryGm, keyGm, queryIndexGm, keyIndexGm;
GlobalTensor<IN_T> weights16Gm;
GlobalTensor<OUT_T> dQueryIndexGm, dKeyIndexGm;
GlobalTensor<IN_T> dWeights16Gm;
GlobalTensor<T> lossGmRef;
GlobalTensor<int32_t> syncGm;
float lossSum_ = 0.0f;
uint32_t blockIdx_ = 0;
static constexpr uint32_t blockNum_ = kB;
static constexpr int32_t EV_MTE2V = 0;
static constexpr int32_t EV_VMTE3 = 1;
static constexpr int32_t EV_VMTE2 = 5;
static constexpr int32_t EV_MTE3V = 6;
};
template <typename DTQ, bool WIDTH64>
class KernelLeanRawB1 {
public:
using T = float;
using IN_T = DTQ;
using OUT_T = DTQ;
using K = KernelLeanPublic<DTQ, WIDTH64, false>;
static constexpr uint32_t kN1 = K::kN1;
static constexpr uint32_t kNJ = K::kNJ;
static constexpr uint32_t kNG = K::kNG;
static constexpr uint32_t kD = K::kD;
static constexpr uint32_t kJTP = K::kJTP;
static constexpr uint32_t kMaskElems = K::kMaskElems;
static constexpr uint32_t kDHalf = K::kDHalf;
static constexpr bool kProdAll = K::kProdAll;
static constexpr bool BATCH_TWO = false;
static constexpr uint32_t kArenaBytes = K::kArenaBytes;
static constexpr uint32_t kSmallOff = K::Align32(K::kArenaBytes);
static constexpr uint32_t kSmallBytes = 256u;
static constexpr uint32_t offQIn = K::offQIn;
static constexpr uint32_t offKIn = K::offKIn;
static constexpr uint32_t offQiIn = K::offQiIn;
static constexpr uint32_t offKiIn = K::offKiIn;
static constexpr uint32_t offWIn = K::offWIn;
static constexpr uint32_t offQF = K::offQF;
static constexpr uint32_t offKRowF = K::offKRowF;
static constexpr uint32_t offProdF = K::offProdF;
static constexpr uint32_t offProdAll = K::offProdAll;
static constexpr uint32_t offLogF = K::offLogF;
static constexpr uint32_t offPVec = K::offPVec;
static constexpr uint32_t offVec = K::offVec;
static constexpr uint32_t offQiF = K::offQiF;
static constexpr uint32_t offKiF = K::offKiF;
static constexpr uint32_t offSimF = K::offSimF;
static constexpr uint32_t offDSF = K::offDSF;
static constexpr uint32_t offDS1F = K::offDS1F;
static constexpr uint32_t offCoefL1F = K::offCoefL1F;
static constexpr uint32_t offWTileF = K::offWTileF;
static constexpr uint32_t offOnesF = K::offOnesF;
static constexpr uint32_t offTmpJTF = K::offTmpJTF;
static constexpr uint32_t offTmp2F = K::offTmp2F;
static constexpr uint32_t offKeepF = K::offKeepF;
static constexpr uint32_t offAccF = K::offAccF;
static constexpr uint32_t offDqF = K::offDqF;
static constexpr uint32_t offDkF = K::offDkF;
static constexpr uint32_t offWork = K::offWork;
static constexpr uint32_t offSmall = K::offSmall;
static constexpr uint32_t offMask = K::offMask;
static constexpr uint32_t offOut = K::offOut;
static constexpr uint32_t kScaleOffset = static_cast<uint32_t>(
__builtin_offsetof(DenseLightningIndexerGradKlLossTilingData, scaleValue));
static_assert(kScaleOffset + sizeof(T) <= sizeof(DenseLightningIndexerGradKlLossTilingData),
"scale_value offset must stay inside the frozen 128B tiling ABI");
static_assert(kScaleOffset % sizeof(T) == 0u, "scale_value offset must be 4B aligned");
static_assert(kSmallOff + kSmallBytes <= 192u * 1024u, "raw Lean B1 UB layout exceeds 192KiB");
__aicore__ inline KernelLeanRawB1() {}
__aicore__ inline LocalTensor<uint8_t> UbArena() const {
return LocalTensor<uint8_t>(TPosition::VECCALC, 0u, kArenaBytes);
}
__aicore__ inline LocalTensor<T> SmallBuf() const {
return LocalTensor<T>(TPosition::VECCALC, kSmallOff, kSmallBytes / sizeof(T));
}
__aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
GM_ADDR weights, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex,
GM_ADDR dWeights, GM_ADDR loss, GM_ADDR tiling) {
queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
queryIndexGm.SetGlobalBuffer((__gm__ IN_T *)queryIndex);
keyIndexGm.SetGlobalBuffer((__gm__ IN_T *)keyIndex);
weights16Gm.SetGlobalBuffer((__gm__ IN_T *)weights);
dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
dWeights16Gm.SetGlobalBuffer((__gm__ IN_T *)dWeights);
lossGmRef.SetGlobalBuffer((__gm__ T *)loss);
GlobalTensor<T> scaleGm;
scaleGm.SetGlobalBuffer((__gm__ T *)(tiling + kScaleOffset));
scale_ = scaleGm.GetValue(0);
}
__aicore__ inline void Process() {
const uint32_t N1 = kN1, NJ = kNJ, NG = kNG, D = kD;
const uint32_t n1D = N1 * D, njD = NJ * D, JTP = kJTP;
const uint32_t dHalf = kDHalf;
LocalTensor<uint8_t> buf = UbArena();
LocalTensor<IN_T> qIn = buf[offQIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> kIn = buf[offKIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> qiIn = buf[offQiIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> kiIn = buf[offKiIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> wIn = buf[offWIn].ReinterpretCast<IN_T>();
LocalTensor<T> qF = buf[offQF].ReinterpretCast<T>();
LocalTensor<T> kRowF = buf[offKRowF].ReinterpretCast<T>();
LocalTensor<T> prodF = buf[offProdF].ReinterpretCast<T>();
LocalTensor<T> logF = buf[offLogF].ReinterpretCast<T>();
LocalTensor<T> pVec = buf[offPVec].ReinterpretCast<T>();
LocalTensor<T> mxF = buf[offVec].ReinterpretCast<T>();
LocalTensor<T> sumF = mxF[N1];
LocalTensor<T> tmpF = mxF[2u * N1];
LocalTensor<T> qiF = buf[offQiF].ReinterpretCast<T>();
LocalTensor<T> kiF = buf[offKiF].ReinterpretCast<T>();
LocalTensor<T> simM = buf[offSimF].ReinterpretCast<T>();
LocalTensor<T> coefM = buf[offDSF].ReinterpretCast<T>();
LocalTensor<T> dST = buf[offDS1F].ReinterpretCast<T>();
LocalTensor<T> coefL1 = buf[offCoefL1F].ReinterpretCast<T>();
LocalTensor<T> wTile = buf[offWTileF].ReinterpretCast<T>();
LocalTensor<T> onesV = buf[offOnesF].ReinterpretCast<T>();
LocalTensor<T> dlogL1 = buf[offTmpJTF].ReinterpretCast<T>();
LocalTensor<T> iTile = buf[offTmp2F].ReinterpretCast<T>();
LocalTensor<T> lossAcc = buf[offAccF].ReinterpretCast<T>();
LocalTensor<T> prodAll = (kProdAll ? buf[offProdAll] : buf[offProdF]).ReinterpretCast<T>();
LocalTensor<T> dqF = buf[offDqF].ReinterpretCast<T>();
LocalTensor<T> dkF = buf[offDkF].ReinterpretCast<T>();
LocalTensor<uint8_t> work = buf[offWork].ReinterpretCast<uint8_t>();
LocalTensor<T> small = buf[offSmall].ReinterpretCast<T>();
LocalTensor<T> iVec = small;
LocalTensor<T> yVec = small[2u * JTP];
LocalTensor<T> dlog = small[3u * JTP];
LocalTensor<T> lp = small[4u * JTP];
LocalTensor<T> ly = small[5u * JTP];
LocalTensor<T> pmax = small[6u * JTP];
LocalTensor<T> ymax = small[7u * JTP];
LocalTensor<T> wRow = small[8u * JTP];
LocalTensor<T> mxRep = small[9u * JTP];
LocalTensor<T> zRep = small[10u * JTP];
LocalTensor<T> keepV = buf[offKeepF].ReinterpretCast<T>();
LocalTensor<T> compV = keepV[JTP];
LocalTensor<T> tailV = keepV[2u * JTP];
LocalTensor<uint8_t> mask = buf[offMask].ReinterpretCast<uint8_t>();
LocalTensor<OUT_T> outStage = buf[offOut].ReinterpretCast<OUT_T>();
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, IN_T(0));
DataCopyPad(qIn, queryGm[0],
DataCopyExtParams(1, n1D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(kIn, keyGm[0],
DataCopyExtParams(1, NJ * n1D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(qiIn, queryIndexGm[0],
DataCopyExtParams(1, NG * D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(kiIn, keyIndexGm[0],
DataCopyExtParams(1, njD * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(wIn, weights16Gm[0],
DataCopyExtParams(1, NG * sizeof(IN_T), 0, 0, 0), pad);
{
Duplicate(buf[offAccF].ReinterpretCast<T>(), static_cast<T>(0.0f), kJTP);
LocalTensor<T> kb = buf[offKeepF].ReinterpretCast<T>();
LocalTensor<T> compV = kb[kJTP];
Duplicate(compV, static_cast<T>(1.0f), kJTP);
Adds(compV, compV, static_cast<T>(-1.0f), kNJ);
}
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
Cast(qF, qIn, RoundMode::CAST_NONE, n1D);
Cast(qiF, qiIn, RoundMode::CAST_NONE, NG * D);
Cast(kiF, kiIn, RoundMode::CAST_NONE, njD);
Duplicate(wRow, static_cast<T>(0.0f), JTP);
Cast(wRow, wIn, RoundMode::CAST_NONE, NG);
Duplicate(onesV, static_cast<T>(1.0f), JTP);
Mul(wTile, wRow, onesV, static_cast<uint64_t>(JTP), static_cast<uint8_t>(JTP),
BinaryRepeatParams(1, 1, 1, 1, 0, 0));
Duplicate(mxF, static_cast<T>(-3.0e38f), N1);
Duplicate(sumF, static_cast<T>(0.0f), N1);
Cast(prodAll, kIn, RoundMode::CAST_NONE, NJ * n1D);
for (uint32_t j = 0; j < NJ; ++j) {
Mul(prodAll[j * n1D], prodAll[j * n1D], qF, n1D);
}
if (kProdAll) {
uint32_t shRowAll[2] = {NJ * N1, D};
ReduceSum<T, Pattern::Reduce::AR, true>(logF, prodAll, work, shRowAll, true);
}
Muls(logF, logF, scale_, NJ * N1);
for (uint32_t j = 0; j < NJ; ++j) {
Max(mxF, mxF, logF[j * N1], N1);
}
{
const uint64_t maskRow = static_cast<uint64_t>(N1);
const uint8_t repNJ = static_cast<uint8_t>(NJ);
const uint8_t rowBlk = static_cast<uint8_t>(N1 / 8u);
Sub(logF, logF, mxF, maskRow, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
Exp(logF, logF, NJ * N1);
for (uint32_t j = 0; j < NJ; ++j) {
Add(sumF, sumF, logF[j * N1], N1);
}
Div(logF, logF, sumF, maskRow, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
}
Duplicate(pVec, static_cast<T>(0.0f), JTP);
uint32_t shAll[2] = {NJ, N1};
ReduceSum<T, Pattern::Reduce::AR, true>(pVec, logF, work, shAll, true);
Muls(pVec, pVec, static_cast<T>(1.0f / 32.0f), JTP);
if constexpr (!kProdAll) {
Duplicate(simM, static_cast<T>(0.0f), kMaskElems);
}
uint32_t shK[2] = {NG, D};
if (kProdAll) {
if (NG != JTP) {
Duplicate(prodAll, static_cast<T>(0.0f), NJ * JTP * D);
}
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodAll[(j * JTP) * D + off], kiF[j * D + off], qiF[off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8), 0,
static_cast<uint8_t>(D / 8)));
}
}
uint32_t shKAll[2] = {NJ * JTP, D};
ReduceSum<T, Pattern::Reduce::AR, true>(simM, prodAll, work, shKAll, true);
} else {
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodF[off], kiF[j * D + off], qiF[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8), 0,
static_cast<uint8_t>(D / 8)));
}
ReduceSum<T, Pattern::Reduce::AR, true>(simM[j * JTP], prodF, work, shK, true);
}
}
Maxs(simM, simM, static_cast<T>(0.0f), kMaskElems);
Mul(dST, simM, wTile, kMaskElems);
uint32_t shI0[2] = {NJ, JTP};
Duplicate(iVec, static_cast<T>(0.0f), JTP);
ReduceSum<T, Pattern::Reduce::AR, true>(iVec, dST, work, shI0, true);
uint32_t sh1[2] = {1u, NJ};
ReduceMax<T, Pattern::Reduce::AR, true>(tmpF, iVec, work, sh1, true);
Brcb(mxRep, tmpF, 1, BrcbRepeatParams(1, 1));
Sub(iVec, iVec, mxRep, JTP);
Muls(tailV, compV, static_cast<T>(3.0e38f), JTP);
Sub(iVec, iVec, tailV, JTP);
Exp(iVec, iVec, JTP);
ReduceSum<T, Pattern::Reduce::AR, true>(tmpF, iVec, work, sh1, true);
Brcb(zRep, tmpF, 1, BrcbRepeatParams(1, 1));
Div(yVec, iVec, zRep, JTP);
Maxs(pmax, pVec, static_cast<T>(LOSS_EPS), JTP);
Maxs(ymax, yVec, static_cast<T>(LOSS_EPS), JTP);
Log(lp, pmax, JTP);
Log(ly, ymax, JTP);
Sub(iTile, lp, ly, JTP);
Mul(iTile, iTile, pmax, JTP);
Add(lossAcc, lossAcc, iTile, JTP);
Sub(dlog, yVec, pVec, JTP);
Brcb(dlogL1, dlog, 1, BrcbRepeatParams(1, 1));
Mul(coefM, dlogL1, wTile, kMaskElems);
CompareScalar(mask, simM, static_cast<T>(0.0f), CMPMODE::GT, kMaskElems);
Select(coefM, mask, coefM, static_cast<T>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
kMaskElems);
Brcb(coefL1, coefM, static_cast<uint8_t>(NJ), BrcbRepeatParams(1, 8));
Mul(dST, simM, dlogL1, kMaskElems);
Select(dST, mask, dST, static_cast<T>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
kMaskElems);
uint32_t shDW[2] = {NJ, JTP};
ReduceSum<T, Pattern::Reduce::RA, true>(wRow, dST, work, shDW, true);
Duplicate(dqF, static_cast<T>(0.0f), NG * D);
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
MulAddDst(dqF[off], coefL1[j * 64u], kiF[j * D + off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D / 8), 1, 0));
}
}
uint32_t shDKAll[2] = {NG, NJ * D};
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodAll[j * D + off], coefL1[j * 64u], qiF[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(NJ * D / 8), 1,
static_cast<uint8_t>(D / 8)));
}
}
ReduceSum<T, Pattern::Reduce::RA, true>(dkF[0], prodAll, work, shDKAll, true);
if constexpr (!BATCH_TWO) {
LocalTensor<T> lsE50 = SmallBuf();
uint32_t shAccE50[2] = {1u, kJTP};
ReduceSum<T, Pattern::Reduce::AR, true>(lsE50, lossAcc, work, shAccE50, true);
}
LocalTensor<OUT_T> outQ = outStage;
LocalTensor<OUT_T> outK = outStage[NG * D];
LocalTensor<OUT_T> outW = outStage[NG * D + njD];
Cast(outQ, dqF, RoundMode::CAST_RINT, NG * D);
Cast(outK, dkF, RoundMode::CAST_RINT, njD);
Cast(outW, wRow, RoundMode::CAST_RINT, NG);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyPad(dQueryIndexGm[0], outQ,
DataCopyExtParams(1, NG * D * sizeof(OUT_T), 0, 0, 0));
DataCopyPad(dKeyIndexGm[0], outK,
DataCopyExtParams(1, njD * sizeof(OUT_T), 0, 0, 0));
DataCopyPad(dWeights16Gm[0], outW,
DataCopyExtParams(1, NG * sizeof(OUT_T), 0, 0, 0));
if constexpr (!BATCH_TWO) {
LocalTensor<T> lsE50 = SmallBuf();
DataCopyExtParams cplE50(1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
DataCopyPad(lossGmRef, lsE50, cplE50);
}
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
return;
}
private:
GlobalTensor<IN_T> queryGm, keyGm, queryIndexGm, keyIndexGm;
GlobalTensor<IN_T> weights16Gm;
GlobalTensor<OUT_T> dQueryIndexGm, dKeyIndexGm;
GlobalTensor<IN_T> dWeights16Gm;
GlobalTensor<T> lossGmRef;
float scale_ = 0.0f;
static constexpr int32_t EV_MTE2V = 0;
static constexpr int32_t EV_VMTE3 = 1;
static constexpr int32_t EV_MTE3V = 6;
};
template <typename DTQ, bool WIDTH64>
class KernelLeanPublicDynamic;
template <typename DTQ>
class KernelLeanRawB2 {
public:
using T = float;
using IN_T = DTQ;
using OUT_T = DTQ;
using K = KernelLeanPublicDynamic<DTQ, false>;
static_assert(K::kD == 128u, "raw Lean B2 只服务 D=128 的 key 2561");
static constexpr bool WIDTH64 = false;
static constexpr uint32_t kN1 = K::kN1;
static constexpr uint32_t kNJ = K::kNJ;
static constexpr uint32_t kNG = K::kNG;
static constexpr uint32_t kD = K::kD;
static constexpr uint32_t kJTP = K::kJTP;
static constexpr uint32_t kMaskElems = K::kMaskElems;
static constexpr uint32_t kDHalf = K::kDHalf;
static constexpr bool kProdAll = K::kProdAll;
static constexpr uint32_t kArenaBytes = K::kArenaBytes;
static constexpr uint32_t kSmallOff = K::Align32(K::kArenaBytes);
static constexpr uint32_t kSmallBytes = 256u;
static constexpr uint32_t offQIn = K::offQIn;
static constexpr uint32_t offKIn = K::offKIn;
static constexpr uint32_t offQiIn = K::offQiIn;
static constexpr uint32_t offKiIn = K::offKiIn;
static constexpr uint32_t offWIn = K::offWIn;
static constexpr uint32_t offQF = K::offQF;
static constexpr uint32_t offKRowF = K::offKRowF;
static constexpr uint32_t offProdF = K::offProdF;
static constexpr uint32_t offProdAll = K::offProdAll;
static constexpr uint32_t offLogF = K::offLogF;
static constexpr uint32_t offPVec = K::offPVec;
static constexpr uint32_t offVec = K::offVec;
static constexpr uint32_t offQiF = K::offQiF;
static constexpr uint32_t offKiF = K::offKiF;
static constexpr uint32_t offSimF = K::offSimF;
static constexpr uint32_t offDSF = K::offDSF;
static constexpr uint32_t offDS1F = K::offDS1F;
static constexpr uint32_t offCoefL1F = K::offCoefL1F;
static constexpr uint32_t offWTileF = K::offWTileF;
static constexpr uint32_t offOnesF = K::offOnesF;
static constexpr uint32_t offTmpJTF = K::offTmpJTF;
static constexpr uint32_t offTmp2F = K::offTmp2F;
static constexpr uint32_t offKeepF = K::offKeepF;
static constexpr uint32_t offAccF = K::offAccF;
static constexpr uint32_t offDqF = K::offDqF;
static constexpr uint32_t offDkF = K::offDkF;
static constexpr uint32_t offWork = K::offWork;
static constexpr uint32_t offSmall = K::offSmall;
static constexpr uint32_t offMask = K::offMask;
static constexpr uint32_t offOut = K::offOut;
static constexpr uint32_t kScaleOffset = static_cast<uint32_t>(
__builtin_offsetof(DenseLightningIndexerGradKlLossTilingData, scaleValue));
static_assert(kScaleOffset + sizeof(T) <= sizeof(DenseLightningIndexerGradKlLossTilingData),
"scale_value offset must stay inside the frozen 128B tiling ABI");
static_assert(kScaleOffset % sizeof(T) == 0u, "scale_value offset must be 4B aligned");
static_assert(kSmallOff + kSmallBytes <= 192u * 1024u, "raw Lean B2 UB layout exceeds 192KiB");
__aicore__ inline KernelLeanRawB2() {}
__aicore__ inline LocalTensor<uint8_t> UbArena() const {
return LocalTensor<uint8_t>(TPosition::VECCALC, 0u, kArenaBytes);
}
__aicore__ inline LocalTensor<T> SmallBuf() const {
return LocalTensor<T>(TPosition::VECCALC, kSmallOff, kSmallBytes / sizeof(T));
}
__aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
GM_ADDR weights, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex,
GM_ADDR dWeights, GM_ADDR loss, GM_ADDR tiling) {
queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
queryIndexGm.SetGlobalBuffer((__gm__ IN_T *)queryIndex);
keyIndexGm.SetGlobalBuffer((__gm__ IN_T *)keyIndex);
weights16Gm.SetGlobalBuffer((__gm__ IN_T *)weights);
dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
dWeights16Gm.SetGlobalBuffer((__gm__ IN_T *)dWeights);
lossGmRef.SetGlobalBuffer((__gm__ T *)loss);
GlobalTensor<T> scaleGm;
scaleGm.SetGlobalBuffer((__gm__ T *)(tiling + kScaleOffset));
scale_ = scaleGm.GetValue(0);
}
__aicore__ inline void Process() {
const uint32_t N1 = kN1, NJ = kNJ, NG = kNG, D = kD;
const uint32_t n1D = N1 * D, njD = NJ * D, JTP = kJTP;
const uint32_t dHalf = kDHalf;
const uint32_t core = GetBlockIdx();
LocalTensor<uint8_t> buf = UbArena();
LocalTensor<IN_T> qIn = buf[offQIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> kIn = buf[offKIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> qiIn = buf[offQiIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> kiIn = buf[offKiIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> wIn = buf[offWIn].ReinterpretCast<IN_T>();
LocalTensor<T> qF = buf[offQF].ReinterpretCast<T>();
LocalTensor<T> kRowF = buf[offKRowF].ReinterpretCast<T>();
LocalTensor<T> prodF = buf[offProdF].ReinterpretCast<T>();
LocalTensor<T> logF = buf[offLogF].ReinterpretCast<T>();
LocalTensor<T> pVec = buf[offPVec].ReinterpretCast<T>();
LocalTensor<T> mxF = buf[offVec].ReinterpretCast<T>();
LocalTensor<T> sumF = mxF[N1];
LocalTensor<T> tmpF = mxF[2u * N1];
LocalTensor<T> qiF = buf[offQiF].ReinterpretCast<T>();
LocalTensor<T> kiF = buf[offKiF].ReinterpretCast<T>();
LocalTensor<T> simM = buf[offSimF].ReinterpretCast<T>();
LocalTensor<T> coefM = buf[offDSF].ReinterpretCast<T>();
LocalTensor<T> dST = buf[offDS1F].ReinterpretCast<T>();
LocalTensor<T> coefL1 = buf[offCoefL1F].ReinterpretCast<T>();
LocalTensor<T> wTile = buf[offWTileF].ReinterpretCast<T>();
LocalTensor<T> onesV = buf[offOnesF].ReinterpretCast<T>();
LocalTensor<T> dlogL1 = buf[offTmpJTF].ReinterpretCast<T>();
LocalTensor<T> iTile = buf[offTmp2F].ReinterpretCast<T>();
LocalTensor<T> lossAcc = buf[offAccF].ReinterpretCast<T>();
LocalTensor<T> prodAll = (kProdAll ? buf[offProdAll] : buf[offProdF]).ReinterpretCast<T>();
LocalTensor<T> dqF = buf[offDqF].ReinterpretCast<T>();
LocalTensor<T> dkF = buf[offDkF].ReinterpretCast<T>();
LocalTensor<uint8_t> work = buf[offWork].ReinterpretCast<uint8_t>();
LocalTensor<T> small = buf[offSmall].ReinterpretCast<T>();
LocalTensor<T> iVec = small;
LocalTensor<T> yVec = small[2u * JTP];
LocalTensor<T> dlog = small[3u * JTP];
LocalTensor<T> lp = small[4u * JTP];
LocalTensor<T> ly = small[5u * JTP];
LocalTensor<T> pmax = small[6u * JTP];
LocalTensor<T> ymax = small[7u * JTP];
LocalTensor<T> wRow = small[8u * JTP];
LocalTensor<T> mxRep = small[9u * JTP];
LocalTensor<T> zRep = small[10u * JTP];
LocalTensor<T> keepV = buf[offKeepF].ReinterpretCast<T>();
LocalTensor<T> compV = keepV[JTP];
LocalTensor<T> tailV = keepV[2u * JTP];
LocalTensor<uint8_t> mask = buf[offMask].ReinterpretCast<uint8_t>();
LocalTensor<OUT_T> outStage = buf[offOut].ReinterpretCast<OUT_T>();
if (core == 0u) {
lossGmRef.SetValue(0, 0.0F);
DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lossGmRef);
}
SyncAll<true>();
constexpr int32_t EV_MTE2V_K1 = 5;
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, IN_T(0));
DataCopyPad(qIn, queryGm[static_cast<uint64_t>(core) * n1D],
DataCopyExtParams(1, n1D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(kIn, keyGm[static_cast<uint64_t>(core) * NJ * n1D],
DataCopyExtParams(1, (NJ / 2u) * n1D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(qiIn, queryIndexGm[static_cast<uint64_t>(core) * NG * D],
DataCopyExtParams(1, NG * D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(kiIn, keyIndexGm[static_cast<uint64_t>(core) * njD],
DataCopyExtParams(1, njD * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(wIn, weights16Gm[static_cast<uint64_t>(core) * NG],
DataCopyExtParams(1, NG * sizeof(IN_T), 0, 0, 0), pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
DataCopyPad(kIn[(NJ / 2u) * n1D],
keyGm[static_cast<uint64_t>(core) * NJ * n1D + (NJ / 2u) * n1D],
DataCopyExtParams(1, (NJ / 2u) * n1D * sizeof(IN_T), 0, 0, 0), pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V_K1);
{
LocalTensor<uint8_t> lb0 = buf;
Duplicate(lb0[offAccF].ReinterpretCast<T>(), static_cast<T>(0.0f), kJTP);
LocalTensor<T> kb = lb0[offKeepF].ReinterpretCast<T>();
LocalTensor<T> keepV = kb;
LocalTensor<T> compV = kb[kJTP];
Duplicate(keepV, static_cast<T>(0.0f), kJTP);
Adds(keepV, keepV, static_cast<T>(1.0f), kNJ);
Duplicate(compV, static_cast<T>(1.0f), kJTP);
Adds(compV, compV, static_cast<T>(-1.0f), kNJ);
PipeBarrier<PIPE_V>();
}
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
Cast(qF, qIn, RoundMode::CAST_NONE, n1D);
Cast(qiF, qiIn, RoundMode::CAST_NONE, NG * D);
Cast(kiF, kiIn, RoundMode::CAST_NONE, njD);
Duplicate(wRow, static_cast<T>(0.0f), JTP);
Cast(wRow, wIn, RoundMode::CAST_NONE, NG);
Duplicate(onesV, static_cast<T>(1.0f), JTP);
PipeBarrier<PIPE_V>();
Mul(wTile, wRow, onesV, static_cast<uint64_t>(JTP), static_cast<uint8_t>(JTP),
BinaryRepeatParams(1, 1, 1, 1, 0, 0));
Duplicate(mxF, static_cast<T>(-3.0e38f), N1);
Duplicate(sumF, static_cast<T>(0.0f), N1);
uint32_t shRowOne[2] = {N1, D};
if constexpr (!WIDTH64) {
Cast(prodAll, kIn, RoundMode::CAST_NONE, (NJ / 2u) * n1D);
}
for (uint32_t j = 0; j < NJ / 2u; ++j) {
if constexpr (WIDTH64) {
Cast(kRowF, kIn[j * n1D], RoundMode::CAST_NONE, n1D);
Mul(kRowF, kRowF, qF, n1D);
if (kProdAll) {
DataCopy(prodAll[j * n1D], kRowF, n1D);
} else {
ReduceSum<T, Pattern::Reduce::AR, true>(logF[j * N1], kRowF, work,
shRowOne, true);
}
} else {
Mul(prodAll[j * n1D], prodAll[j * n1D], qF, n1D);
}
}
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V_K1);
if constexpr (!WIDTH64) {
Cast(prodAll[(NJ / 2u) * n1D], kIn[(NJ / 2u) * n1D],
RoundMode::CAST_NONE, (NJ / 2u) * n1D);
}
for (uint32_t j = NJ / 2u; j < NJ; ++j) {
if constexpr (WIDTH64) {
Cast(kRowF, kIn[j * n1D], RoundMode::CAST_NONE, n1D);
Mul(kRowF, kRowF, qF, n1D);
if (kProdAll) {
DataCopy(prodAll[j * n1D], kRowF, n1D);
} else {
ReduceSum<T, Pattern::Reduce::AR, true>(logF[j * N1], kRowF, work,
shRowOne, true);
}
} else {
Mul(prodAll[j * n1D], prodAll[j * n1D], qF, n1D);
}
}
if (kProdAll) {
uint32_t shRowAll[2] = {NJ * N1, D};
ReduceSum<T, Pattern::Reduce::AR, true>(logF, prodAll, work, shRowAll, true);
}
Muls(logF, logF, scale_, NJ * N1);
for (uint32_t j = 0; j < NJ; ++j) {
Max(mxF, mxF, logF[j * N1], N1);
}
{
const uint64_t maskRow = static_cast<uint64_t>(N1);
const uint8_t repNJ = static_cast<uint8_t>(NJ);
const uint8_t rowBlk = static_cast<uint8_t>(N1 / 8u);
Sub(logF, logF, mxF, maskRow, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
Exp(logF, logF, NJ * N1);
for (uint32_t j = 0; j < NJ; ++j) {
Add(sumF, sumF, logF[j * N1], N1);
}
Div(logF, logF, sumF, maskRow, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
}
Duplicate(pVec, static_cast<T>(0.0f), JTP);
uint32_t shAll[2] = {NJ, N1};
ReduceSum<T, Pattern::Reduce::AR, true>(pVec, logF, work, shAll, true);
Muls(pVec, pVec, static_cast<T>(1.0f / 32.0f), JTP);
Duplicate(simM, static_cast<T>(0.0f), kMaskElems);
PipeBarrier<PIPE_V>();
uint32_t shK[2] = {NG, D};
if (kProdAll) {
if (NG != JTP) {
Duplicate(prodAll, static_cast<T>(0.0f), NJ * JTP * D);
PipeBarrier<PIPE_V>();
}
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodAll[(j * JTP) * D + off], kiF[j * D + off], qiF[off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8), 0,
static_cast<uint8_t>(D / 8)));
}
}
uint32_t shKAll[2] = {NJ * JTP, D};
ReduceSum<T, Pattern::Reduce::AR, true>(simM, prodAll, work, shKAll, true);
} else {
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodF[off], kiF[j * D + off], qiF[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8), 0,
static_cast<uint8_t>(D / 8)));
}
ReduceSum<T, Pattern::Reduce::AR, true>(simM[j * JTP], prodF, work, shK, true);
PipeBarrier<PIPE_V>();
}
}
Maxs(simM, simM, static_cast<T>(0.0f), kMaskElems);
PipeBarrier<PIPE_V>();
Mul(dST, simM, wTile, kMaskElems);
PipeBarrier<PIPE_V>();
uint32_t shI0[2] = {NJ, JTP};
Duplicate(iVec, static_cast<T>(0.0f), JTP);
PipeBarrier<PIPE_V>();
ReduceSum<T, Pattern::Reduce::AR, true>(iVec, dST, work, shI0, true);
PipeBarrier<PIPE_V>();
uint32_t sh1[2] = {1u, NJ};
ReduceMax<T, Pattern::Reduce::AR, true>(tmpF, iVec, work, sh1, true);
PipeBarrier<PIPE_V>();
Brcb(mxRep, tmpF, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Sub(iVec, iVec, mxRep, JTP);
Muls(tailV, compV, static_cast<T>(3.0e38f), JTP);
Sub(iVec, iVec, tailV, JTP);
Exp(iVec, iVec, JTP);
ReduceSum<T, Pattern::Reduce::AR, true>(tmpF, iVec, work, sh1, true);
PipeBarrier<PIPE_V>();
Brcb(zRep, tmpF, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Div(yVec, iVec, zRep, JTP);
Maxs(pmax, pVec, static_cast<T>(LOSS_EPS), JTP);
Maxs(ymax, yVec, static_cast<T>(LOSS_EPS), JTP);
Log(lp, pmax, JTP);
Log(ly, ymax, JTP);
Sub(iTile, lp, ly, JTP);
Mul(iTile, iTile, pmax, JTP);
Add(lossAcc, lossAcc, iTile, JTP);
PipeBarrier<PIPE_V>();
Sub(dlog, yVec, pVec, JTP);
PipeBarrier<PIPE_V>();
Brcb(dlogL1, dlog, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Mul(coefM, dlogL1, wTile, kMaskElems);
CompareScalar(mask, simM, static_cast<T>(0.0f), CMPMODE::GT, kMaskElems);
PipeBarrier<PIPE_V>();
Select(coefM, mask, coefM, static_cast<T>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
kMaskElems);
PipeBarrier<PIPE_V>();
Brcb(coefL1, coefM, static_cast<uint8_t>(NJ), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
Mul(dST, simM, dlogL1, kMaskElems);
PipeBarrier<PIPE_V>();
Select(dST, mask, dST, static_cast<T>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
kMaskElems);
PipeBarrier<PIPE_V>();
uint32_t shDW[2] = {NJ, JTP};
ReduceSum<T, Pattern::Reduce::RA, true>(wRow, dST, work, shDW, true);
PipeBarrier<PIPE_V>();
Duplicate(dqF, static_cast<T>(0.0f), NG * D);
PipeBarrier<PIPE_V>();
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
MulAddDst(dqF[off], coefL1[j * 64u], kiF[j * D + off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D / 8), 1, 0));
}
PipeBarrier<PIPE_V>();
}
uint32_t shDK[2] = {NG, D};
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodF[off], coefL1[j * 64u], qiF[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D / 8), 1,
static_cast<uint8_t>(D / 8)));
}
ReduceSum<T, Pattern::Reduce::RA, true>(dkF[j * D], prodF, work, shDK, true);
PipeBarrier<PIPE_V>();
}
LocalTensor<OUT_T> outQ = outStage;
LocalTensor<OUT_T> outK = outStage[NG * D];
LocalTensor<OUT_T> outW = outStage[NG * D + njD];
Cast(outQ, dqF, RoundMode::CAST_RINT, NG * D);
Cast(outK, dkF, RoundMode::CAST_RINT, njD);
Cast(outW, wRow, RoundMode::CAST_RINT, NG);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyPad(dQueryIndexGm[static_cast<uint64_t>(core) * NG * D], outQ,
DataCopyExtParams(1, NG * D * sizeof(OUT_T), 0, 0, 0));
DataCopyPad(dKeyIndexGm[static_cast<uint64_t>(core) * njD], outK,
DataCopyExtParams(1, njD * sizeof(OUT_T), 0, 0, 0));
DataCopyPad(dWeights16Gm[static_cast<uint64_t>(core) * NG], outW,
DataCopyExtParams(1, NG * sizeof(OUT_T), 0, 0, 0));
LocalTensor<T> ls = SmallBuf();
{
LocalTensor<uint8_t> lb = buf;
LocalTensor<T> acc = lb[offAccF].ReinterpretCast<T>();
LocalTensor<uint8_t> wk = lb[offWork].ReinterpretCast<uint8_t>();
uint32_t shAcc[2] = {1u, kJTP};
ReduceSum<T, Pattern::Reduce::AR, true>(ls, acc, wk, shAcc, true);
PipeBarrier<PIPE_V>();
}
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
SetAtomicAdd<float>();
DataCopyExtParams cpl(1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
DataCopyPad(lossGmRef, ls, cpl);
SetAtomicNone();
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
private:
GlobalTensor<IN_T> queryGm, keyGm, queryIndexGm, keyIndexGm;
GlobalTensor<IN_T> weights16Gm;
GlobalTensor<OUT_T> dQueryIndexGm, dKeyIndexGm;
GlobalTensor<IN_T> dWeights16Gm;
GlobalTensor<T> lossGmRef;
float scale_ = 0.0f;
static constexpr int32_t EV_MTE2V = 0;
static constexpr int32_t EV_VMTE3 = 1;
static constexpr int32_t EV_MTE3V = 6;
};
template <typename DTQ, bool WIDTH64>
class KernelLeanPublicDynamic {
public:
using T = float;
using IN_T = DTQ;
using OUT_T = DTQ;
static constexpr uint32_t kN1 = 32u;
static constexpr uint32_t kNJ = 4u;
static constexpr uint32_t kNG = 8u;
static constexpr uint32_t kD = WIDTH64 ? 64u : 128u;
static constexpr uint32_t kJTP = 8u;
static constexpr uint32_t kMaskElems = 64u;
static constexpr uint32_t kDHalf = kD / 64u;
static constexpr bool kProdAll = (kNJ <= 4u);
static constexpr uint32_t Align512(uint32_t x) { return (x + 511u) / 512u * 512u; }
static constexpr uint32_t Align32(uint32_t x) { return (x + 31u) / 32u * 32u; }
static constexpr uint32_t kInBytes = sizeof(IN_T);
static constexpr uint32_t kOutBytes = sizeof(OUT_T);
static constexpr uint32_t kF32Bytes = sizeof(T);
static constexpr uint32_t offQIn = 0u;
static constexpr uint32_t offKIn = offQIn + Align512(kN1 * kD * kInBytes);
static constexpr uint32_t offQiIn = offKIn + Align512(kNJ * kN1 * kD * kInBytes);
static constexpr uint32_t offKiIn = offQiIn + Align512(kNG * kD * kInBytes);
static constexpr uint32_t offWIn = offKiIn + Align512(kNJ * kD * kInBytes);
static constexpr uint32_t offQF = offWIn + Align512(kNG * kInBytes);
static constexpr uint32_t offKRowF = offQF + Align512(kN1 * kD * kF32Bytes);
static constexpr uint32_t offProdF = offKRowF + Align512(kN1 * kD * kF32Bytes);
static constexpr uint32_t offProdAll = offProdF + Align512(kN1 * kD * kF32Bytes);
static constexpr uint32_t offLogF = offProdAll + Align512(kNJ * kN1 * kD * kF32Bytes);
static constexpr uint32_t offPVec = offLogF + Align512(kNJ * kN1 * kF32Bytes);
static constexpr uint32_t offVec = offPVec + Align512(kJTP * kF32Bytes);
static constexpr uint32_t offQiF = offVec + Align512(3u * kN1 * kF32Bytes);
static constexpr uint32_t offKiF = offQiF + Align512(kNG * kD * kF32Bytes);
static constexpr uint32_t offSimF = offKiF + Align512(kNJ * kD * kF32Bytes);
static constexpr uint32_t offDSF = offSimF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offDS1F = offDSF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offCoefL1F = offDS1F + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offWTileF = offCoefL1F + Align512(kJTP * kMaskElems * kF32Bytes);
static constexpr uint32_t offOnesF = offWTileF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offTmpJTF = offOnesF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offTmp2F = offTmpJTF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offKeepF = offTmp2F + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offAccF = offKeepF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offDqF = offAccF + Align512(kMaskElems * kF32Bytes);
static constexpr uint32_t offDkF = offDqF + Align512(kNG * kD * kF32Bytes);
static constexpr uint32_t offWork = offDkF + Align512(kNJ * kD * kF32Bytes);
static constexpr uint32_t offSmall = offWork + Align512(4096u);
static constexpr uint32_t offMask = offSmall + Align512(16u * kJTP * kF32Bytes);
static constexpr uint32_t offOut = offMask + Align512(512u);
static constexpr uint32_t kArenaBytes =
offOut + Align512((kNG + kNJ + 16u) * kD * kOutBytes);
__aicore__ inline KernelLeanPublicDynamic() {}
__aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
GM_ADDR weights, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex,
GM_ADDR dWeights, GM_ADDR loss, GM_ADDR workspace,
const DenseLightningIndexerGradKlLossTilingData *td) {
t = td;
blockIdx_ = GetBlockIdx();
blockNum_ = GetBlockNum();
queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
queryIndexGm.SetGlobalBuffer((__gm__ IN_T *)queryIndex);
keyIndexGm.SetGlobalBuffer((__gm__ IN_T *)keyIndex);
weights16Gm.SetGlobalBuffer((__gm__ IN_T *)weights);
dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
dWeights16Gm.SetGlobalBuffer((__gm__ IN_T *)dWeights);
lossGmRef.SetGlobalBuffer((__gm__ T *)loss);
pipe.InitBuffer(leanBuf_, Align32(kArenaBytes));
pipe.InitBuffer(smallBuf, 256u);
if (blockNum_ > 1u) {
__gm__ uint8_t *ws = (__gm__ uint8_t *)workspace;
const uint64_t scatterBytes = static_cast<uint64_t>(t->bSize) * t->s2Size *
t->dQueryIndex * sizeof(T);
const uint64_t syncOffset = t->wsOffsetScatter + ((scatterBytes + 511u) / 512u) * 512u;
syncGm.SetGlobalBuffer((__gm__ int32_t *)(ws + syncOffset));
pipe.InitBuffer(syncBuf, blockNum_ * 32u);
}
}
__aicore__ inline void Process() { RunLeanProcess(); }
private:
__aicore__ inline void RunLeanProcess() {
const uint32_t totalTasks = t->totalTasks;
lossSum_ = 0.0f;
const uint32_t taskBegin = blockIdx_ * t->tasksPerCore;
uint32_t taskEnd = taskBegin + t->tasksPerCore;
if (taskEnd > totalTasks) taskEnd = totalTasks;
if constexpr (WIDTH64) {
if (blockNum_ > 1u) { InitSoftSync(); }
}
if (blockNum_ > 1u && blockIdx_ == 0u) {
lossGmRef.SetValue(0, 0.0F);
DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lossGmRef);
}
if (blockNum_ > 1u) {
if constexpr (WIDTH64) {
SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_);
} else {
SyncAll<true>();
}
}
{
LocalTensor<uint8_t> lb0 = leanBuf_.Get<uint8_t>();
Duplicate(lb0[offAccF].ReinterpretCast<T>(), static_cast<T>(0.0f), kJTP);
LocalTensor<T> kb = lb0[offKeepF].ReinterpretCast<T>();
LocalTensor<T> keepV = kb;
LocalTensor<T> compV = kb[kJTP];
Duplicate(keepV, static_cast<T>(0.0f), kJTP);
Adds(keepV, keepV, static_cast<T>(1.0f), kNJ);
Duplicate(compV, static_cast<T>(1.0f), kJTP);
Adds(compV, compV, static_cast<T>(-1.0f), kNJ);
PipeBarrier<PIPE_V>();
}
for (uint32_t task = taskBegin; task < taskEnd; ++task) {
lossSum_ += RunLeanTask(task / t->s1Size, task % t->s1Size);
}
if constexpr (WIDTH64) {
if (blockNum_ > 1u) { SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_); }
}
LocalTensor<T> ls = smallBuf.Get<T>();
{
LocalTensor<uint8_t> lb = leanBuf_.Get<uint8_t>();
LocalTensor<T> acc = lb[offAccF].ReinterpretCast<T>();
LocalTensor<uint8_t> wk = lb[offWork].ReinterpretCast<uint8_t>();
uint32_t shAcc[2] = {1u, kJTP};
ReduceSum<T, Pattern::Reduce::AR, true>(ls, acc, wk, shAcc, true);
PipeBarrier<PIPE_V>();
}
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
if (blockNum_ > 1u) { SetAtomicAdd<float>(); }
DataCopyExtParams cpl(1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
DataCopyPad(lossGmRef, ls, cpl);
if (blockNum_ > 1u) { SetAtomicNone(); }
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
__aicore__ inline float RunLeanTask(uint32_t b, uint32_t s1) {
const uint32_t N1 = kN1, NJ = kNJ, NG = kNG, D = kD;
const uint32_t n1D = N1 * D, njD = NJ * D, JTP = kJTP;
const uint32_t dHalf = kDHalf;
LocalTensor<uint8_t> buf = leanBuf_.Get<uint8_t>();
LocalTensor<IN_T> qIn = buf[offQIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> kIn = buf[offKIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> qiIn = buf[offQiIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> kiIn = buf[offKiIn].ReinterpretCast<IN_T>();
LocalTensor<IN_T> wIn = buf[offWIn].ReinterpretCast<IN_T>();
LocalTensor<T> qF = buf[offQF].ReinterpretCast<T>();
LocalTensor<T> kRowF = buf[offKRowF].ReinterpretCast<T>();
LocalTensor<T> prodF = buf[offProdF].ReinterpretCast<T>();
LocalTensor<T> logF = buf[offLogF].ReinterpretCast<T>();
LocalTensor<T> pVec = buf[offPVec].ReinterpretCast<T>();
LocalTensor<T> mxF = buf[offVec].ReinterpretCast<T>();
LocalTensor<T> sumF = mxF[N1];
LocalTensor<T> tmpF = mxF[2u * N1];
LocalTensor<T> qiF = buf[offQiF].ReinterpretCast<T>();
LocalTensor<T> kiF = buf[offKiF].ReinterpretCast<T>();
LocalTensor<T> simM = buf[offSimF].ReinterpretCast<T>();
LocalTensor<T> coefM = buf[offDSF].ReinterpretCast<T>();
LocalTensor<T> dST = buf[offDS1F].ReinterpretCast<T>();
LocalTensor<T> coefL1 = buf[offCoefL1F].ReinterpretCast<T>();
LocalTensor<T> wTile = buf[offWTileF].ReinterpretCast<T>();
LocalTensor<T> onesV = buf[offOnesF].ReinterpretCast<T>();
LocalTensor<T> dlogL1 = buf[offTmpJTF].ReinterpretCast<T>();
LocalTensor<T> iTile = buf[offTmp2F].ReinterpretCast<T>();
LocalTensor<T> lossAcc = buf[offAccF].ReinterpretCast<T>();
LocalTensor<T> prodAll = (kProdAll ? buf[offProdAll] : buf[offProdF]).ReinterpretCast<T>();
LocalTensor<T> dqF = buf[offDqF].ReinterpretCast<T>();
LocalTensor<T> dkF = buf[offDkF].ReinterpretCast<T>();
LocalTensor<uint8_t> work = buf[offWork].ReinterpretCast<uint8_t>();
LocalTensor<T> small = buf[offSmall].ReinterpretCast<T>();
LocalTensor<T> iVec = small;
LocalTensor<T> yVec = small[2u * JTP];
LocalTensor<T> dlog = small[3u * JTP];
LocalTensor<T> lp = small[4u * JTP];
LocalTensor<T> ly = small[5u * JTP];
LocalTensor<T> pmax = small[6u * JTP];
LocalTensor<T> ymax = small[7u * JTP];
LocalTensor<T> wRow = small[8u * JTP];
LocalTensor<T> mxRep = small[9u * JTP];
LocalTensor<T> zRep = small[10u * JTP];
LocalTensor<T> keepV = buf[offKeepF].ReinterpretCast<T>();
LocalTensor<T> compV = keepV[JTP];
LocalTensor<T> tailV = keepV[2u * JTP];
LocalTensor<uint8_t> mask = buf[offMask].ReinterpretCast<uint8_t>();
LocalTensor<OUT_T> outStage = buf[offOut].ReinterpretCast<OUT_T>();
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, IN_T(0));
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qIn, queryGm[(static_cast<uint64_t>(b) * t->s1Size + s1) * n1D],
DataCopyExtParams(1, n1D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(kIn, keyGm[static_cast<uint64_t>(b) * NJ * n1D],
DataCopyExtParams(1, NJ * n1D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(qiIn, queryIndexGm[(static_cast<uint64_t>(b) * t->s1Size + s1) * NG * D],
DataCopyExtParams(1, NG * D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(kiIn, keyIndexGm[static_cast<uint64_t>(b) * njD],
DataCopyExtParams(1, njD * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(wIn, weights16Gm[(static_cast<uint64_t>(b) * t->s1Size + s1) * NG],
DataCopyExtParams(1, NG * sizeof(IN_T), 0, 0, 0), pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
Cast(qF, qIn, RoundMode::CAST_NONE, n1D);
Cast(qiF, qiIn, RoundMode::CAST_NONE, NG * D);
Cast(kiF, kiIn, RoundMode::CAST_NONE, njD);
Duplicate(wRow, static_cast<T>(0.0f), JTP);
Cast(wRow, wIn, RoundMode::CAST_NONE, NG);
Duplicate(onesV, static_cast<T>(1.0f), JTP);
PipeBarrier<PIPE_V>();
Mul(wTile, wRow, onesV, static_cast<uint64_t>(JTP), static_cast<uint8_t>(JTP),
BinaryRepeatParams(1, 1, 1, 1, 0, 0));
Duplicate(mxF, static_cast<T>(-3.0e38f), N1);
Duplicate(sumF, static_cast<T>(0.0f), N1);
uint32_t shRowOne[2] = {N1, D};
if constexpr (!WIDTH64) {
Cast(prodAll, kIn, RoundMode::CAST_NONE, NJ * n1D);
}
for (uint32_t j = 0; j < NJ; ++j) {
if constexpr (WIDTH64) {
Cast(kRowF, kIn[j * n1D], RoundMode::CAST_NONE, n1D);
Mul(kRowF, kRowF, qF, n1D);
if (kProdAll) {
DataCopy(prodAll[j * n1D], kRowF, n1D);
} else {
ReduceSum<T, Pattern::Reduce::AR, true>(logF[j * N1], kRowF, work,
shRowOne, true);
}
} else {
Mul(prodAll[j * n1D], prodAll[j * n1D], qF, n1D);
}
}
if (kProdAll) {
uint32_t shRowAll[2] = {NJ * N1, D};
ReduceSum<T, Pattern::Reduce::AR, true>(logF, prodAll, work, shRowAll, true);
}
Muls(logF, logF, t->scaleValue, NJ * N1);
for (uint32_t j = 0; j < NJ; ++j) {
Max(mxF, mxF, logF[j * N1], N1);
}
{
const uint64_t maskRow = static_cast<uint64_t>(N1);
const uint8_t repNJ = static_cast<uint8_t>(NJ);
const uint8_t rowBlk = static_cast<uint8_t>(N1 / 8u);
Sub(logF, logF, mxF, maskRow, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
Exp(logF, logF, NJ * N1);
for (uint32_t j = 0; j < NJ; ++j) {
Add(sumF, sumF, logF[j * N1], N1);
}
Div(logF, logF, sumF, maskRow, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
}
Duplicate(pVec, static_cast<T>(0.0f), JTP);
uint32_t shAll[2] = {NJ, N1};
ReduceSum<T, Pattern::Reduce::AR, true>(pVec, logF, work, shAll, true);
Muls(pVec, pVec, t->invN1, JTP);
Duplicate(simM, static_cast<T>(0.0f), kMaskElems);
PipeBarrier<PIPE_V>();
uint32_t shK[2] = {NG, D};
if (kProdAll) {
if (NG != JTP) {
Duplicate(prodAll, static_cast<T>(0.0f), NJ * JTP * D);
PipeBarrier<PIPE_V>();
}
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodAll[(j * JTP) * D + off], kiF[j * D + off], qiF[off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8), 0,
static_cast<uint8_t>(D / 8)));
}
}
uint32_t shKAll[2] = {NJ * JTP, D};
ReduceSum<T, Pattern::Reduce::AR, true>(simM, prodAll, work, shKAll, true);
} else {
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodF[off], kiF[j * D + off], qiF[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8), 0,
static_cast<uint8_t>(D / 8)));
}
ReduceSum<T, Pattern::Reduce::AR, true>(simM[j * JTP], prodF, work, shK, true);
PipeBarrier<PIPE_V>();
}
}
Maxs(simM, simM, static_cast<T>(0.0f), kMaskElems);
PipeBarrier<PIPE_V>();
Mul(dST, simM, wTile, kMaskElems);
PipeBarrier<PIPE_V>();
uint32_t shI0[2] = {NJ, JTP};
Duplicate(iVec, static_cast<T>(0.0f), JTP);
PipeBarrier<PIPE_V>();
ReduceSum<T, Pattern::Reduce::AR, true>(iVec, dST, work, shI0, true);
PipeBarrier<PIPE_V>();
uint32_t sh1[2] = {1u, NJ};
ReduceMax<T, Pattern::Reduce::AR, true>(tmpF, iVec, work, sh1, true);
PipeBarrier<PIPE_V>();
Brcb(mxRep, tmpF, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Sub(iVec, iVec, mxRep, JTP);
Muls(tailV, compV, static_cast<T>(3.0e38f), JTP);
Sub(iVec, iVec, tailV, JTP);
Exp(iVec, iVec, JTP);
ReduceSum<T, Pattern::Reduce::AR, true>(tmpF, iVec, work, sh1, true);
PipeBarrier<PIPE_V>();
Brcb(zRep, tmpF, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Div(yVec, iVec, zRep, JTP);
Maxs(pmax, pVec, static_cast<T>(LOSS_EPS), JTP);
Maxs(ymax, yVec, static_cast<T>(LOSS_EPS), JTP);
Log(lp, pmax, JTP);
Log(ly, ymax, JTP);
Sub(iTile, lp, ly, JTP);
Mul(iTile, iTile, pmax, JTP);
Add(lossAcc, lossAcc, iTile, JTP);
PipeBarrier<PIPE_V>();
Sub(dlog, yVec, pVec, JTP);
PipeBarrier<PIPE_V>();
Brcb(dlogL1, dlog, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Mul(coefM, dlogL1, wTile, kMaskElems);
CompareScalar(mask, simM, static_cast<T>(0.0f), CMPMODE::GT, kMaskElems);
PipeBarrier<PIPE_V>();
Select(coefM, mask, coefM, static_cast<T>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
kMaskElems);
PipeBarrier<PIPE_V>();
Brcb(coefL1, coefM, static_cast<uint8_t>(NJ), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
Mul(dST, simM, dlogL1, kMaskElems);
PipeBarrier<PIPE_V>();
Select(dST, mask, dST, static_cast<T>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
kMaskElems);
PipeBarrier<PIPE_V>();
uint32_t shDW[2] = {NJ, JTP};
ReduceSum<T, Pattern::Reduce::RA, true>(wRow, dST, work, shDW, true);
PipeBarrier<PIPE_V>();
Duplicate(dqF, static_cast<T>(0.0f), NG * D);
PipeBarrier<PIPE_V>();
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
MulAddDst(dqF[off], coefL1[j * 64u], kiF[j * D + off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D / 8), 1, 0));
}
PipeBarrier<PIPE_V>();
}
uint32_t shDK[2] = {NG, D};
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodF[off], coefL1[j * 64u], qiF[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D / 8), 1,
static_cast<uint8_t>(D / 8)));
}
ReduceSum<T, Pattern::Reduce::RA, true>(dkF[j * D], prodF, work, shDK, true);
PipeBarrier<PIPE_V>();
}
LocalTensor<OUT_T> outQ = outStage;
LocalTensor<OUT_T> outK = outStage[NG * D];
LocalTensor<OUT_T> outW = outStage[NG * D + njD];
Cast(outQ, dqF, RoundMode::CAST_RINT, NG * D);
Cast(outK, dkF, RoundMode::CAST_RINT, njD);
Cast(outW, wRow, RoundMode::CAST_RINT, NG);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyPad(dQueryIndexGm[(static_cast<uint64_t>(b) * t->s1Size + s1) * NG * D], outQ,
DataCopyExtParams(1, NG * D * sizeof(OUT_T), 0, 0, 0));
DataCopyPad(dKeyIndexGm[static_cast<uint64_t>(b) * njD], outK,
DataCopyExtParams(1, njD * sizeof(OUT_T), 0, 0, 0));
DataCopyPad(dWeights16Gm[(static_cast<uint64_t>(b) * t->s1Size + s1) * NG], outW,
DataCopyExtParams(1, NG * sizeof(OUT_T), 0, 0, 0));
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
return 0.0f;
}
__aicore__ inline void InitSoftSync() {
auto local = syncBuf.Get<int32_t>();
const uint32_t count = blockNum_ * 8u;
Duplicate(local, 0, count);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, count * sizeof(int32_t), 0, 0, 0);
DataCopyPad(syncGm, local, cp);
SetFlag<HardEvent::MTE3_S>(1); WaitFlag<HardEvent::MTE3_S>(1);
}
const DenseLightningIndexerGradKlLossTilingData *t = nullptr;
TPipe pipe;
TBuf<TPosition::VECCALC> leanBuf_;
TBuf<TPosition::VECCALC> smallBuf;
TBuf<TPosition::VECCALC> syncBuf;
GlobalTensor<IN_T> queryGm, keyGm, queryIndexGm, keyIndexGm;
GlobalTensor<IN_T> weights16Gm;
GlobalTensor<OUT_T> dQueryIndexGm, dKeyIndexGm;
GlobalTensor<IN_T> dWeights16Gm;
GlobalTensor<T> lossGmRef;
GlobalTensor<int32_t> syncGm;
float lossSum_ = 0.0f;
uint32_t blockIdx_ = 0;
uint32_t blockNum_ = 1;
static constexpr int32_t EV_MTE2V = 0;
static constexpr int32_t EV_VMTE3 = 1;
static constexpr int32_t EV_VMTE2 = 5;
static constexpr int32_t EV_MTE3V = 6;
};
template <typename DTQ, bool FUSE_BIG, bool WIDE_FUSE, bool WIDTH64>
class KernelDenseLightningIndexerGradKlLoss {
public:
static constexpr int32_t JT = WIDE_FUSE ? 16 : 8;
static constexpr uint32_t kBatchWidth = WIDTH64 ? 64u : 128u;
static constexpr uint32_t kKeyBatchRows = 4u;
static constexpr uint32_t kKeyBatchHeads = 32u;
using T = float;
using IN_T = DTQ;
using OUT_T = DTQ;
__aicore__ inline KernelDenseLightningIndexerGradKlLoss() {}
__aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
GM_ADDR weights, GM_ADDR sparseIndices, GM_ADDR softmaxMax,
GM_ADDR softmaxSum, GM_ADDR queryRope, GM_ADDR keyRope,
GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKey,
GM_ADDR dQueryIndex, GM_ADDR dKeyIndex, GM_ADDR dWeights, GM_ADDR loss,
GM_ADDR workspace, const DenseLightningIndexerGradKlLossTilingData *td) {
t = td;
if (LeanDomain()) { LeanInit(query, key, queryIndex, keyIndex, weights, dQueryIndex,
dKeyIndex, dWeights, loss, workspace); return; }
if (t->reserved[0] == 1u) {
HeadSplitInit(query, key, queryIndex, keyIndex, weights, dQueryIndex, dKeyIndex,
dWeights, loss, workspace);
return;
}
queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
queryIndexGm.SetGlobalBuffer((__gm__ IN_T *)queryIndex);
keyIndexGm.SetGlobalBuffer((__gm__ IN_T *)keyIndex);
sparseIdxGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
softmaxMaxGm.SetGlobalBuffer((__gm__ T *)softmaxMax);
softmaxSumGm.SetGlobalBuffer((__gm__ T *)softmaxSum);
if (t->hasRope != 0u) {
queryRopeGm.SetGlobalBuffer((__gm__ IN_T *)queryRope);
keyRopeGm.SetGlobalBuffer((__gm__ IN_T *)keyRope);
}
if (t->useSeqLens != 0u) {
if (t->seqLensIsI64 != 0u) {
seqQ64Gm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQuery);
seqK64Gm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsKey);
} else {
seqQ32Gm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
seqK32Gm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKey);
}
}
dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
dWeights16Gm.SetGlobalBuffer((__gm__ IN_T *)dWeights);
dWeightsF32Gm.SetGlobalBuffer((__gm__ float *)dWeights);
weights16Gm.SetGlobalBuffer((__gm__ IN_T *)weights);
weightsF32Gm.SetGlobalBuffer((__gm__ float *)weights);
lossGmRef.SetGlobalBuffer((__gm__ T *)loss);
__gm__ uint8_t *ws = (__gm__ uint8_t *)workspace;
pGm.SetGlobalBuffer((__gm__ T *)(ws + t->wsOffsetP),
static_cast<uint64_t>(t->totalTasks) * t->kSize);
directKey_ = (t->totalTasks <= 1u) && (t->s2Size >= 1u) && (t->s2Size <= 8u) &&
(t->dQuery == t->dQueryIndex) && (t->hasRope == 0u) &&
(t->nidx1Size <= 64u) &&
(t->n1Size == 32u || t->n1Size == 64u || t->n1Size == 128u);
iGm.SetGlobalBuffer((__gm__ T *)(ws + t->wsOffsetI),
static_cast<uint64_t>(t->totalTasks) * t->kSize);
scatterGm.SetGlobalBuffer((__gm__ T *)(ws + t->wsOffsetScatter),
static_cast<uint64_t>(t->bSize) * t->s2Size * t->dQueryIndex);
blockIdx_ = GetBlockIdx();
blockNum_ = GetBlockNum();
if (t->reserved[0] == 1u) {
headSplit_ = true;
headParts_ = blockNum_ / t->totalTasks;
splitTask_ = blockIdx_ / headParts_;
const uint32_t part = blockIdx_ % headParts_;
mainH0_ = part * (t->n1Size / headParts_);
mainH1_ = mainH0_ + (t->n1Size / headParts_);
idxH0_ = part * (t->nidx1Size / headParts_);
idxH1_ = idxH0_ + (t->nidx1Size / headParts_);
pPartGm_.SetGlobalBuffer((__gm__ T *)(ws + t->reserved[1]), blockNum_ * 16u);
iPartGm_.SetGlobalBuffer((__gm__ T *)(ws + t->reserved[2]), blockNum_ * 16u);
dkPartGm_.SetGlobalBuffer((__gm__ T *)(ws + t->reserved[3]),
static_cast<uint64_t>(blockNum_) * t->s2Size * t->dQueryIndex);
}
if (blockNum_ > 1u) {
const uint64_t scatterBytes = static_cast<uint64_t>(t->bSize) * t->s2Size *
t->dQueryIndex * sizeof(T);
const uint64_t syncOffset = t->wsOffsetScatter +
((scatterBytes + 511u) / 512u) * 512u;
syncGm.SetGlobalBuffer((__gm__ int32_t *)(ws + syncOffset));
pipe.InitBuffer(syncBuf, blockNum_ * 32u);
}
const uint32_t DQ = t->dQuery, DQI = t->dQueryIndex, DR = t->dRope;
const uint32_t NIDX1 = t->nidx1Size;
const uint32_t mainRows =
(t->s2Size <= 16u &&
(t->n1Size == 32u || t->n1Size == 64u || t->n1Size == 128u) &&
DQ == kBatchWidth && t->hasRope == 0u) ?
(WIDE_FUSE ? 32u :
((sizeof(IN_T) == 2u && t->n1Size >= 64u) ? 64u : 32u)) : JT;
pipe.InitBuffer(idxBuf, Align32(JT * sizeof(int32_t)));
pipe.InitBuffer(kTileBuf, Align32(mainRows * DQ * sizeof(T)));
pipe.InitBuffer(krTileBuf, Align32(JT * DR * sizeof(T)));
pipe.InitBuffer(kiTileBuf, Align32(JT * DQI * sizeof(T)));
pipe.InitBuffer(qgBuf, Align32(((DQ > DQI) ? DQ : DQI) * sizeof(T)));
pipe.InitBuffer(qrgBuf, Align32(DR * sizeof(T)));
pipe.InitBuffer(qidxgBuf, Align32(DQI * sizeof(T)));
pipe.InitBuffer(repBuf, Align32(mainRows * DQ * sizeof(T)));
pipe.InitBuffer(repBuf2, Align32(JT * DQI * sizeof(T)));
pipe.InitBuffer(vecBuf, Align32(10 * JT * sizeof(T)));
pipe.InitBuffer(s2dBuf, Align32(GTILE * JT * sizeof(T)));
pipe.InitBuffer(dIrepBuf, Align32(GTILE * JT * sizeof(T)));
pipe.InitBuffer(dWtBuf, Align32(GTILE * JT * sizeof(T)));
pipe.InitBuffer(dWaccBuf, Align32(NIDX1_CAP * sizeof(T)));
pipe.InitBuffer(dKTileBuf, Align32(JT * DQI * sizeof(T)));
pipe.InitBuffer(accDqBuf, Align32(static_cast<uint32_t>(GTILE) * DQI * sizeof(T)));
pipe.InitBuffer(inTmpBuf, Align32(DQ * sizeof(IN_T)));
pipe.InitBuffer(inKBuf, Align32(mainRows * DQ * sizeof(IN_T)));
if (sizeof(IN_T) == 2u && t->n1Size <= kKeyBatchHeads) {
pipe.InitBuffer(kInBatchBuf,
Align32(kKeyBatchRows * kKeyBatchHeads * DQ * sizeof(IN_T)));
} else {
pipe.InitBuffer(kInBatchBuf, 32u);
}
pipe.InitBuffer(inKiBuf, Align32(JT * DQI * sizeof(IN_T)));
pipe.InitBuffer(inKrBuf, Align32(JT * DR * sizeof(IN_T)));
const uint32_t inQElems = (mainRows * DQ > GTILE * DQI) ? mainRows * DQ : GTILE * DQI;
pipe.InitBuffer(inQBuf, Align32(inQElems * sizeof(IN_T)));
pipe.InitBuffer(inQrBuf, Align32(JT * DR * sizeof(IN_T)));
pipe.InitBuffer(wRowBuf, Align32(NIDX1_CAP * sizeof(T)));
pipe.InitBuffer(smRowBuf, Align32(N1_CAP * sizeof(T)));
pipe.InitBuffer(ssRowBuf, Align32(N1_CAP * sizeof(T)));
cacheMain_ = (t->s2Size <= MAIN_CACHE_COLS && t->n1Size <= N1_CAP);
batchMain_ = (t->s2Size <= 16u &&
(t->n1Size == 32u || t->n1Size == 64u || t->n1Size == 128u) &&
t->dQuery == kBatchWidth && t->hasRope == 0u);
if (cacheMain_) {
pipe.InitBuffer(mainProbBuf, Align32(N1_CAP * MAIN_CACHE_COLS * sizeof(T)));
}
if (batchMain_) {
pipe.InitBuffer(mainLogitBuf, Align32(16u * t->n1Size * sizeof(T)));
pipe.InitBuffer(mainPBuf, Align32(16u * sizeof(T)));
}
cacheS_ = (t->s2Size <= JT && t->nidx1Size <= NIDX1_CAP);
batchIndex_ = (t->s2Size <= 128u && t->dQueryIndex == 128u &&
t->nidx1Size >= 8u && t->nidx1Size <= NIDX1_CAP &&
(t->nidx1Size % 8u) == 0u);
if (cacheS_) {
pipe.InitBuffer(sCacheBuf, Align32(NIDX1_CAP * JT * sizeof(T)));
}
pipe.InitBuffer(wInBuf, Align32(NIDX1_CAP * sizeof(IN_T)));
pipe.InitBuffer(chunkBuf, Align32(CHUNK * sizeof(T)));
pipe.InitBuffer(workBuf, 16 * 1024);
pipe.InitBuffer(smallBuf, 1024);
pipe.InitBuffer(seqBuf, 128);
pipe.InitBuffer(dWresBuf, Align32(GTILE * sizeof(T)));
pipe.InitBuffer(dS2dBuf, Align32(GTILE * JT * sizeof(T)));
pipe.InitBuffer(outQBuf, Align32(static_cast<uint32_t>(GTILE) * DQI * sizeof(OUT_T)));
pipe.InitBuffer(outWBuf, Align32(NIDX1_CAP * sizeof(T)));
pipe.InitBuffer(outKBuf, Align32(((t->totalTasks <= 1u && t->s2Size <= 8u) ?
8u : 1u) * DQI * sizeof(OUT_T)));
pipe.InitBuffer(qiAllBuf, Align32(static_cast<uint32_t>(GTILE) * DQI * sizeof(T)));
{
int64_t kMax = static_cast<int64_t>(t->kSize);
if (kMax < 1) { kMax = 1; }
resElems_ = ((kMax + JT - 1) / JT) * JT;
useResident_ = (resElems_ <= static_cast<int64_t>(RESIDENT_CAP));
if (useResident_) {
pipe.InitBuffer(pResBuf, Align32(static_cast<uint32_t>(resElems_ * sizeof(T))));
pipe.InitBuffer(iResBuf, Align32(static_cast<uint32_t>(resElems_ * sizeof(T))));
}
}
}
__aicore__ inline void Process() {
if (lean_) { RunLeanProcess(); return; }
const bool directKey = directKey_;
if (blockNum_ > 1u) {
InitSoftSync();
SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_);
}
if (headSplit_) {
lossSum_ = 0.0f;
if (blockIdx_ == 0u) {
lossGmRef.SetValue(0, 0.0F);
DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lossGmRef);
}
if (blockNum_ > 1u) { SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_); }
DoTaskSplitHead(splitTask_);
if (blockNum_ > 1u) SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_);
if (blockIdx_ == 0u) {
WriteDkFromParts();
}
if ((blockIdx_ % headParts_) == 0u) {
LocalTensor<T> ls = smallBuf.Get<T>();
ls.SetValue(0, static_cast<T>(lossSum_));
SetFlag<HardEvent::S_MTE3>(EV_SMTE3);
WaitFlag<HardEvent::S_MTE3>(EV_SMTE3);
if (blockNum_ > 1u) { SetAtomicAdd<float>(); }
DataCopyExtParams cpl(1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
DataCopyPad(lossGmRef, ls, cpl);
if (blockNum_ > 1u) { SetAtomicNone(); }
SetFlag<HardEvent::MTE3_S>(1);
WaitFlag<HardEvent::MTE3_S>(1);
}
return;
}
if (blockIdx_ == 0 && blockNum_ > 1u) {
lossGmRef.SetValue(0, 0.0F);
DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lossGmRef);
}
if (!directKey) ZeroScatter();
if (blockNum_ > 1u) SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_);
{
const uint32_t totalTasks = t->totalTasks;
lossSum_ = 0.0f;
const uint32_t taskBegin = blockIdx_ * t->tasksPerCore;
uint32_t taskEnd = taskBegin + t->tasksPerCore;
if (taskEnd > totalTasks) taskEnd = totalTasks;
for (uint32_t task = taskBegin; task < taskEnd; ++task) {
DoTask(task);
}
if (blockNum_ > 1u) SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_);
{
LocalTensor<T> ls = smallBuf.Get<T>();
ls.SetValue(0, static_cast<T>(lossSum_));
SetFlag<HardEvent::S_MTE3>(EV_SMTE3);
WaitFlag<HardEvent::S_MTE3>(EV_SMTE3);
if (blockNum_ > 1u) { SetAtomicAdd<float>(); }
DataCopyExtParams cpl(1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
DataCopyPad(lossGmRef, ls, cpl);
if (blockNum_ > 1u) { SetAtomicNone(); }
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
if (!directKey) WriteDKeyIndex();
}
private:
__aicore__ inline void InitSoftSync() {
auto local = syncBuf.Get<int32_t>();
const uint32_t count = blockNum_ * 8u;
Duplicate(local, 0, count);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, count * sizeof(int32_t), 0, 0, 0);
DataCopyPad(syncGm, local, cp);
SetFlag<HardEvent::MTE3_S>(1); WaitFlag<HardEvent::MTE3_S>(1);
}
static __aicore__ inline uint32_t Align32(uint32_t x) { return (x + 31u) / 32u * 32u; }
static constexpr uint32_t LEAN_JT = 8u;
static constexpr uint32_t LEAN_MASK_ELEMS = 64u;
__aicore__ inline bool LeanDomain() const {
return sizeof(IN_T) == 2u && t->totalTasks >= 1u && t->totalTasks <= 2u &&
t->coreNum >= 1u && t->coreNum <= 2u &&
t->s2Size >= 1u && t->s2Size <= LEAN_JT && t->kSize == t->s2Size &&
t->s1Size == 1u &&
t->n1Size == 32u && (t->dQuery == 64u || t->dQuery == 128u) &&
t->dQueryIndex == t->dQuery && t->hasRope == 0u &&
t->nidx1Size >= 1u && t->nidx1Size <= 8u;
}
__aicore__ inline uint32_t LeanTake(uint64_t bytes) {
const uint32_t at = leanOff_;
leanOff_ += static_cast<uint32_t>((bytes + 511u) / 512u * 512u);
return at;
}
__aicore__ inline void LeanInit(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
GM_ADDR weights, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex,
GM_ADDR dWeights, GM_ADDR loss, GM_ADDR workspace) {
blockIdx_ = GetBlockIdx();
blockNum_ = GetBlockNum();
queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
queryIndexGm.SetGlobalBuffer((__gm__ IN_T *)queryIndex);
keyIndexGm.SetGlobalBuffer((__gm__ IN_T *)keyIndex);
weights16Gm.SetGlobalBuffer((__gm__ IN_T *)weights);
dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
dWeights16Gm.SetGlobalBuffer((__gm__ IN_T *)dWeights);
lossGmRef.SetGlobalBuffer((__gm__ T *)loss);
const uint32_t D = t->dQuery, N1 = t->n1Size, NJ = t->s2Size, NG = t->nidx1Size;
leanOff_ = 0;
lQIn_ = LeanTake(static_cast<uint64_t>(N1) * D * sizeof(IN_T));
lKIn_ = LeanTake(static_cast<uint64_t>(NJ) * N1 * D * sizeof(IN_T));
lQiIn_ = LeanTake(static_cast<uint64_t>(NG) * D * sizeof(IN_T));
lKiIn_ = LeanTake(static_cast<uint64_t>(NJ) * D * sizeof(IN_T));
lWIn_ = LeanTake(static_cast<uint64_t>(NG) * sizeof(IN_T));
lQF_ = LeanTake(static_cast<uint64_t>(N1) * D * sizeof(T));
lKRowF_ = LeanTake(static_cast<uint64_t>(N1) * D * sizeof(T));
lProdF_ = LeanTake(static_cast<uint64_t>(N1) * D * sizeof(T));
lProdAll_ = (NJ <= 4u) ? LeanTake(static_cast<uint64_t>(NJ) * N1 * D * sizeof(T)) : 0u;
lLogF_ = LeanTake(static_cast<uint64_t>(NJ) * N1 * sizeof(T));
lPVec_ = LeanTake(LEAN_JT * sizeof(T));
lVec_ = LeanTake(3u * N1 * sizeof(T));
lQiF_ = LeanTake(static_cast<uint64_t>(NG) * D * sizeof(T));
lKiF_ = LeanTake(static_cast<uint64_t>(NJ) * D * sizeof(T));
lSimF_ = LeanTake(LEAN_MASK_ELEMS * sizeof(T));
lDSF_ = LeanTake(LEAN_MASK_ELEMS * sizeof(T));
lDS1F_ = LeanTake(LEAN_MASK_ELEMS * sizeof(T));
lCoefL1F_ = LeanTake(static_cast<uint64_t>(LEAN_JT) * LEAN_MASK_ELEMS * sizeof(T));
lWTileF_ = LeanTake(LEAN_MASK_ELEMS * sizeof(T));
lOnesF_ = LeanTake(LEAN_MASK_ELEMS * sizeof(T));
lTmpJTF_ = LeanTake(LEAN_MASK_ELEMS * sizeof(T));
lTmp2F_ = LeanTake(LEAN_MASK_ELEMS * sizeof(T));
lKeepF_ = LeanTake(LEAN_MASK_ELEMS * sizeof(T));
lAccF_ = LeanTake(LEAN_MASK_ELEMS * sizeof(T));
lDqF_ = LeanTake(static_cast<uint64_t>(NG) * D * sizeof(T));
lDkF_ = LeanTake(static_cast<uint64_t>(NJ) * D * sizeof(T));
lWork_ = LeanTake(4096u);
lSmall_ = LeanTake(16u * LEAN_JT * sizeof(T));
lMask_ = LeanTake(512u);
lOut_ = LeanTake(static_cast<uint64_t>(NG + NJ + 16u) * D * sizeof(OUT_T));
pipe.InitBuffer(leanBuf_, Align32(leanOff_));
pipe.InitBuffer(smallBuf, 256u);
if (blockNum_ > 1u) {
__gm__ uint8_t *ws = (__gm__ uint8_t *)workspace;
const uint64_t scatterBytes = static_cast<uint64_t>(t->bSize) * t->s2Size *
t->dQueryIndex * sizeof(T);
const uint64_t syncOffset = t->wsOffsetScatter + ((scatterBytes + 511u) / 512u) * 512u;
syncGm.SetGlobalBuffer((__gm__ int32_t *)(ws + syncOffset));
pipe.InitBuffer(syncBuf, blockNum_ * 32u);
}
lean_ = true;
}
__aicore__ inline void RunLeanProcess() {
const uint32_t totalTasks = t->totalTasks;
lossSum_ = 0.0f;
const uint32_t taskBegin = blockIdx_ * t->tasksPerCore;
uint32_t taskEnd = taskBegin + t->tasksPerCore;
if (taskEnd > totalTasks) taskEnd = totalTasks;
if (blockNum_ > 1u) { InitSoftSync(); }
if (blockNum_ > 1u && blockIdx_ == 0u) {
lossGmRef.SetValue(0, 0.0F);
DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lossGmRef);
}
if (blockNum_ > 1u) { SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_); }
{
LocalTensor<uint8_t> lb0 = leanBuf_.Get<uint8_t>();
Duplicate(lb0[lAccF_].ReinterpretCast<T>(), static_cast<T>(0.0f), LEAN_JT);
LocalTensor<T> kb = lb0[lKeepF_].ReinterpretCast<T>();
LocalTensor<T> keepV = kb;
LocalTensor<T> compV = kb[LEAN_JT];
Duplicate(keepV, static_cast<T>(0.0f), LEAN_JT);
Adds(keepV, keepV, static_cast<T>(1.0f), t->s2Size);
Duplicate(compV, static_cast<T>(1.0f), LEAN_JT);
Adds(compV, compV, static_cast<T>(-1.0f), t->s2Size);
PipeBarrier<PIPE_V>();
}
for (uint32_t task = taskBegin; task < taskEnd; ++task) {
lossSum_ += RunLeanTask(task / t->s1Size, task % t->s1Size);
}
if (blockNum_ > 1u) { SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_); }
LocalTensor<T> ls = smallBuf.Get<T>();
{
LocalTensor<uint8_t> lb = leanBuf_.Get<uint8_t>();
LocalTensor<T> acc = lb[lAccF_].ReinterpretCast<T>();
LocalTensor<uint8_t> wk = lb[lWork_].ReinterpretCast<uint8_t>();
uint32_t shAcc[2] = {1u, LEAN_JT};
ReduceSum<T, Pattern::Reduce::AR, true>(ls, acc, wk, shAcc, true);
PipeBarrier<PIPE_V>();
}
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
if (blockNum_ > 1u) { SetAtomicAdd<float>(); }
DataCopyExtParams cpl(1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
DataCopyPad(lossGmRef, ls, cpl);
if (blockNum_ > 1u) { SetAtomicNone(); }
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
__aicore__ inline float RunLeanTask(uint32_t b, uint32_t s1) {
const uint32_t D = t->dQuery, N1 = t->n1Size, NJ = t->s2Size, NG = t->nidx1Size;
const uint32_t n1D = N1 * D, njD = NJ * D, JTP = LEAN_JT;
const uint32_t dHalf = D / 64u;
LocalTensor<uint8_t> buf = leanBuf_.Get<uint8_t>();
LocalTensor<IN_T> qIn = buf[lQIn_].ReinterpretCast<IN_T>();
LocalTensor<IN_T> kIn = buf[lKIn_].ReinterpretCast<IN_T>();
LocalTensor<IN_T> qiIn = buf[lQiIn_].ReinterpretCast<IN_T>();
LocalTensor<IN_T> kiIn = buf[lKiIn_].ReinterpretCast<IN_T>();
LocalTensor<IN_T> wIn = buf[lWIn_].ReinterpretCast<IN_T>();
LocalTensor<T> qF = buf[lQF_].ReinterpretCast<T>();
LocalTensor<T> kRowF = buf[lKRowF_].ReinterpretCast<T>();
LocalTensor<T> prodF = buf[lProdF_].ReinterpretCast<T>();
LocalTensor<T> logF = buf[lLogF_].ReinterpretCast<T>();
LocalTensor<T> pVec = buf[lPVec_].ReinterpretCast<T>();
LocalTensor<T> mxF = buf[lVec_].ReinterpretCast<T>();
LocalTensor<T> sumF = mxF[N1];
LocalTensor<T> tmpF = mxF[2u * N1];
LocalTensor<T> qiF = buf[lQiF_].ReinterpretCast<T>();
LocalTensor<T> kiF = buf[lKiF_].ReinterpretCast<T>();
LocalTensor<T> simM = buf[lSimF_].ReinterpretCast<T>();
LocalTensor<T> coefM = buf[lDSF_].ReinterpretCast<T>();
LocalTensor<T> dST = buf[lDS1F_].ReinterpretCast<T>();
LocalTensor<T> coefL1 = buf[lCoefL1F_].ReinterpretCast<T>();
LocalTensor<T> wTile = buf[lWTileF_].ReinterpretCast<T>();
LocalTensor<T> onesV = buf[lOnesF_].ReinterpretCast<T>();
LocalTensor<T> dlogL1 = buf[lTmpJTF_].ReinterpretCast<T>();
LocalTensor<T> iTile = buf[lTmp2F_].ReinterpretCast<T>();
LocalTensor<T> lossAcc = buf[lAccF_].ReinterpretCast<T>();
LocalTensor<T> prodAll = (lProdAll_ != 0u) ? buf[lProdAll_].ReinterpretCast<T>() : prodF;
LocalTensor<T> dqF = buf[lDqF_].ReinterpretCast<T>();
LocalTensor<T> dkF = buf[lDkF_].ReinterpretCast<T>();
LocalTensor<uint8_t> work = buf[lWork_].ReinterpretCast<uint8_t>();
LocalTensor<T> small = buf[lSmall_].ReinterpretCast<T>();
LocalTensor<T> iVec = small;
LocalTensor<T> yVec = small[2u * JTP];
LocalTensor<T> dlog = small[3u * JTP];
LocalTensor<T> lp = small[4u * JTP];
LocalTensor<T> ly = small[5u * JTP];
LocalTensor<T> pmax = small[6u * JTP];
LocalTensor<T> ymax = small[7u * JTP];
LocalTensor<T> wRow = small[8u * JTP];
LocalTensor<T> mxRep = small[9u * JTP];
LocalTensor<T> zRep = small[10u * JTP];
LocalTensor<T> keepV = buf[lKeepF_].ReinterpretCast<T>();
LocalTensor<T> compV = keepV[JTP];
LocalTensor<T> tailV = keepV[2u * JTP];
LocalTensor<uint8_t> mask = buf[lMask_].ReinterpretCast<uint8_t>();
LocalTensor<OUT_T> outStage = buf[lOut_].ReinterpretCast<OUT_T>();
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, IN_T(0));
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qIn, queryGm[(static_cast<uint64_t>(b) * t->s1Size + s1) * n1D],
DataCopyExtParams(1, n1D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(kIn, keyGm[static_cast<uint64_t>(b) * NJ * n1D],
DataCopyExtParams(1, NJ * n1D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(qiIn, queryIndexGm[(static_cast<uint64_t>(b) * t->s1Size + s1) * NG * D],
DataCopyExtParams(1, NG * D * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(kiIn, keyIndexGm[static_cast<uint64_t>(b) * njD],
DataCopyExtParams(1, njD * sizeof(IN_T), 0, 0, 0), pad);
DataCopyPad(wIn, weights16Gm[(static_cast<uint64_t>(b) * t->s1Size + s1) * NG],
DataCopyExtParams(1, NG * sizeof(IN_T), 0, 0, 0), pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
Cast(qF, qIn, RoundMode::CAST_NONE, n1D);
Cast(qiF, qiIn, RoundMode::CAST_NONE, NG * D);
Cast(kiF, kiIn, RoundMode::CAST_NONE, njD);
Duplicate(wRow, static_cast<T>(0.0f), JTP);
Cast(wRow, wIn, RoundMode::CAST_NONE, NG);
Duplicate(onesV, static_cast<T>(1.0f), JTP);
PipeBarrier<PIPE_V>();
Mul(wTile, wRow, onesV, static_cast<uint64_t>(JTP), static_cast<uint8_t>(JTP),
BinaryRepeatParams(1, 1, 1, 1, 0, 0));
Duplicate(mxF, static_cast<T>(-3.0e38f), N1);
Duplicate(sumF, static_cast<T>(0.0f), N1);
uint32_t shRowOne[2] = {N1, D};
for (uint32_t j = 0; j < NJ; ++j) {
Cast(kRowF, kIn[j * n1D], RoundMode::CAST_NONE, n1D);
Mul(kRowF, kRowF, qF, n1D);
if (lProdAll_ != 0u) {
DataCopy(prodAll[j * n1D], kRowF, n1D);
} else {
ReduceSum<T, Pattern::Reduce::AR, true>(logF[j * N1], kRowF, work,
shRowOne, true);
}
}
if (lProdAll_ != 0u) {
uint32_t shRowAll[2] = {NJ * N1, D};
ReduceSum<T, Pattern::Reduce::AR, true>(logF, prodAll, work, shRowAll, true);
}
for (uint32_t j = 0; j < NJ; ++j) {
Muls(logF[j * N1], logF[j * N1], t->scaleValue, N1);
Max(mxF, mxF, logF[j * N1], N1);
}
for (uint32_t j = 0; j < NJ; ++j) {
Sub(logF[j * N1], logF[j * N1], mxF, N1);
Exp(logF[j * N1], logF[j * N1], N1);
Add(sumF, sumF, logF[j * N1], N1);
}
for (uint32_t j = 0; j < NJ; ++j) {
Div(logF[j * N1], logF[j * N1], sumF, N1);
}
Duplicate(pVec, static_cast<T>(0.0f), JTP);
uint32_t shAll[2] = {NJ, N1};
ReduceSum<T, Pattern::Reduce::AR, true>(pVec, logF, work, shAll, true);
Muls(pVec, pVec, t->invN1, JTP);
Duplicate(simM, static_cast<T>(0.0f), LEAN_MASK_ELEMS);
PipeBarrier<PIPE_V>();
uint32_t shK[2] = {NG, D};
if (lProdAll_ != 0u) {
if (NG != JTP) {
Duplicate(prodAll, static_cast<T>(0.0f), NJ * JTP * D);
PipeBarrier<PIPE_V>();
}
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodAll[(j * JTP) * D + off], kiF[j * D + off], qiF[off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8), 0,
static_cast<uint8_t>(D / 8)));
}
}
uint32_t shKAll[2] = {NJ * JTP, D};
ReduceSum<T, Pattern::Reduce::AR, true>(simM, prodAll, work, shKAll, true);
} else {
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodF[off], kiF[j * D + off], qiF[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8), 0,
static_cast<uint8_t>(D / 8)));
}
ReduceSum<T, Pattern::Reduce::AR, true>(simM[j * JTP], prodF, work, shK, true);
PipeBarrier<PIPE_V>();
}
}
Maxs(simM, simM, static_cast<T>(0.0f), LEAN_MASK_ELEMS);
PipeBarrier<PIPE_V>();
Mul(dST, simM, wTile, LEAN_MASK_ELEMS);
PipeBarrier<PIPE_V>();
uint32_t shI0[2] = {NJ, JTP};
Duplicate(iVec, static_cast<T>(0.0f), JTP);
PipeBarrier<PIPE_V>();
ReduceSum<T, Pattern::Reduce::AR, true>(iVec, dST, work, shI0, true);
PipeBarrier<PIPE_V>();
uint32_t sh1[2] = {1u, NJ};
ReduceMax<T, Pattern::Reduce::AR, true>(tmpF, iVec, work, sh1, true);
PipeBarrier<PIPE_V>();
Brcb(mxRep, tmpF, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Sub(iVec, iVec, mxRep, JTP);
Muls(tailV, compV, static_cast<T>(3.0e38f), JTP);
Sub(iVec, iVec, tailV, JTP);
Exp(iVec, iVec, JTP);
ReduceSum<T, Pattern::Reduce::AR, true>(tmpF, iVec, work, sh1, true);
PipeBarrier<PIPE_V>();
Brcb(zRep, tmpF, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Div(yVec, iVec, zRep, JTP);
Maxs(pmax, pVec, static_cast<T>(LOSS_EPS), JTP);
Maxs(ymax, yVec, static_cast<T>(LOSS_EPS), JTP);
Log(lp, pmax, JTP);
Log(ly, ymax, JTP);
Sub(iTile, lp, ly, JTP);
Mul(iTile, iTile, pmax, JTP);
Add(lossAcc, lossAcc, iTile, JTP);
PipeBarrier<PIPE_V>();
Sub(dlog, yVec, pVec, JTP);
PipeBarrier<PIPE_V>();
Brcb(dlogL1, dlog, 1, BrcbRepeatParams(1, 1));
PipeBarrier<PIPE_V>();
Mul(coefM, dlogL1, wTile, LEAN_MASK_ELEMS);
CompareScalar(mask, simM, static_cast<T>(0.0f), CMPMODE::GT, LEAN_MASK_ELEMS);
PipeBarrier<PIPE_V>();
Select(coefM, mask, coefM, static_cast<T>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
LEAN_MASK_ELEMS);
PipeBarrier<PIPE_V>();
Brcb(coefL1, coefM, static_cast<uint8_t>(NJ), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
Mul(dST, simM, dlogL1, LEAN_MASK_ELEMS);
PipeBarrier<PIPE_V>();
Select(dST, mask, dST, static_cast<T>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE,
LEAN_MASK_ELEMS);
PipeBarrier<PIPE_V>();
uint32_t shDW[2] = {NJ, JTP};
ReduceSum<T, Pattern::Reduce::RA, true>(wRow, dST, work, shDW, true);
PipeBarrier<PIPE_V>();
Duplicate(dqF, static_cast<T>(0.0f), NG * D);
PipeBarrier<PIPE_V>();
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
MulAddDst(dqF[off], coefL1[j * 64u], kiF[j * D + off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D / 8), 1, 0));
}
PipeBarrier<PIPE_V>();
}
uint32_t shDK[2] = {NG, D};
for (uint32_t j = 0; j < NJ; ++j) {
for (uint32_t h = 0; h < dHalf; ++h) {
const uint32_t off = h * 64u;
Mul(prodF[off], coefL1[j * 64u], qiF[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(NG),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D / 8), 1,
static_cast<uint8_t>(D / 8)));
}
ReduceSum<T, Pattern::Reduce::RA, true>(dkF[j * D], prodF, work, shDK, true);
PipeBarrier<PIPE_V>();
}
LocalTensor<OUT_T> outQ = outStage;
LocalTensor<OUT_T> outK = outStage[NG * D];
LocalTensor<OUT_T> outW = outStage[NG * D + njD];
Cast(outQ, dqF, RoundMode::CAST_RINT, NG * D);
Cast(outK, dkF, RoundMode::CAST_RINT, njD);
Cast(outW, wRow, RoundMode::CAST_RINT, NG);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyPad(dQueryIndexGm[(static_cast<uint64_t>(b) * t->s1Size + s1) * NG * D], outQ,
DataCopyExtParams(1, NG * D * sizeof(OUT_T), 0, 0, 0));
DataCopyPad(dKeyIndexGm[static_cast<uint64_t>(b) * njD], outK,
DataCopyExtParams(1, njD * sizeof(OUT_T), 0, 0, 0));
DataCopyPad(dWeights16Gm[(static_cast<uint64_t>(b) * t->s1Size + s1) * NG], outW,
DataCopyExtParams(1, NG * sizeof(OUT_T), 0, 0, 0));
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
return 0.0f;
}
__aicore__ inline void LoadRowF32(LocalTensor<T> dst, const GlobalTensor<IN_T> &gm, uint64_t row,
int32_t d) {
LocalTensor<IN_T> tmp = inTmpBuf.Get<IN_T>();
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyExtParams cp(1, static_cast<uint32_t>(d * sizeof(IN_T)), 0, 0, 0);
DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
DataCopyPad(tmp, gm[row * static_cast<uint64_t>(d)], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(dst, tmp, d);
PipeBarrier<PIPE_ALL>();
} else {
Cast(dst, tmp, RoundMode::CAST_NONE, d);
}
PipeBarrier<PIPE_V>();
}
__aicore__ inline void ChunkToScatter(uint64_t off) {
LocalTensor<T> z = chunkBuf.Get<T>();
Duplicate(z, static_cast<T>(0.0), CHUNK);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, static_cast<uint32_t>(CHUNK * sizeof(T)), 0, 0, 0);
DataCopyPad(scatterGm[off], z, cp);
}
__aicore__ inline void ChunkToDKey(uint64_t off) {
LocalTensor<OUT_T> z = chunkBuf.Get<OUT_T>();
Duplicate(z, static_cast<OUT_T>(0), CHUNK);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, static_cast<uint32_t>(CHUNK * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dKeyIndexGm[off], z, cp);
}
__aicore__ inline void ZeroScatter() {
const uint64_t total = static_cast<uint64_t>(t->bSize) * t->s2Size * t->dQueryIndex;
LocalTensor<T> z = chunkBuf.Get<T>();
Duplicate(z, static_cast<T>(0.0), CHUNK);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
const uint64_t stride = static_cast<uint64_t>(blockNum_) * CHUNK;
for (uint64_t off = static_cast<uint64_t>(blockIdx_) * CHUNK; off < total; off += stride) {
uint32_t n = static_cast<uint32_t>((total - off < CHUNK) ? (total - off) : CHUNK);
DataCopyExtParams cp(1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0);
DataCopyPad(scatterGm[off], z, cp);
}
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
__aicore__ inline void MaskSparseTile(const LocalTensor<T> &tile, int32_t nj, bool target) {
LocalTensor<int32_t> idx = idxBuf.Get<int32_t>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
for (int32_t j = 0; j < nj; ++j) {
const int64_t r = static_cast<int64_t>(idx.GetValue(j));
if (r >= 0 && r < curVisible_) { continue; }
tile.SetValue(j, target ? static_cast<T>(0.0f) : static_cast<T>(-3.0e38f));
}
SetFlag<HardEvent::S_V>(EV_SV);
WaitFlag<HardEvent::S_V>(EV_SV);
}
__aicore__ inline void DoTask(uint32_t task) {
const uint32_t b = task / t->s1Size;
const uint32_t s1 = task % t->s1Size;
const int64_t kReal = static_cast<int64_t>(t->s2Size);
if (kReal <= 0) {
ZeroTaskRow(b, s1);
return;
}
LoadWeightsRow(b, s1);
ComputeMainSoftmaxStats(b, s1);
Duplicate(dWaccBuf.Get<T>(), static_cast<T>(0.0), NIDX1_CAP);
PipeBarrier<PIPE_V>();
if constexpr (FUSE_BIG) {
lossSum_ += RunTaskFusedBigHeads(b, s1);
} else if constexpr (WIDE_FUSE) {
if (t->nidx1Size <= GTILE) {
lossSum_ += RunTaskFusedSmall<128>(b, s1);
} else {
lossSum_ += RunTaskFusedBigHeads(b, s1);
}
} else {
if (batchMain_ && t->s2Size <= JT && t->nidx1Size <= GTILE &&
t->dQueryIndex == kBatchWidth) {
lossSum_ += RunTaskFusedSmall<static_cast<int32_t>(kBatchWidth)>(b, s1);
} else {
lossSum_ += RunTask(b, s1, kReal);
}
}
WriteDWeights(b, s1, dWaccBuf.Get<T>());
}
template <int32_t DW>
__aicore__ inline float RunTaskFusedSmall(uint32_t b, uint32_t s1) {
const int32_t nj = static_cast<int32_t>(t->s2Size);
const int32_t ng = static_cast<int32_t>(t->nidx1Size);
constexpr int32_t D = DW;
LocalTensor<T> qi = qiAllBuf.Get<T>();
LocalTensor<T> ki = kiTileBuf.Get<T>();
LocalTensor<T> scores = s2dBuf.Get<T>();
LocalTensor<T> dScores = dS2dBuf.Get<T>();
LocalTensor<T> dq = accDqBuf.Get<T>();
LocalTensor<T> dk = dKTileBuf.Get<T>();
LocalTensor<T> rep = repBuf2.Get<T>();
LocalTensor<T> p = vecBuf.Get<T>();
LocalTensor<T> ii = vecBuf.Get<T>()[JT];
LocalTensor<T> y = vecBuf.Get<T>()[2 * JT];
LocalTensor<T> tmp = vecBuf.Get<T>()[4 * JT];
LocalTensor<T> ss = vecBuf.Get<T>()[5 * JT];
LocalTensor<T> lp = vecBuf.Get<T>()[6 * JT];
LocalTensor<T> ly = vecBuf.Get<T>()[7 * JT];
LocalTensor<T> pmax = vecBuf.Get<T>()[8 * JT];
LocalTensor<T> one = smallBuf.Get<T>();
LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
LocalTensor<IN_T> qiIn = inQBuf.Get<IN_T>();
LocalTensor<IN_T> kiIn = inKiBuf.Get<IN_T>();
const uint64_t qiBase =
(static_cast<uint64_t>(b) * t->s1Size + s1) * t->nidx1Size * D;
const uint64_t kiBase = static_cast<uint64_t>(b) * t->s2Size * D;
DataCopyExtParams cpQ(1, ng * D * sizeof(IN_T), 0, 0, 0);
DataCopyExtParams cpK(1, nj * D * sizeof(IN_T), 0, 0, 0);
DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qiIn, queryIndexGm[qiBase], cpQ, pad);
DataCopyPad(kiIn, keyIndexGm[kiBase], cpK, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(qi, qiIn, ng * D);
DataCopy(ki, kiIn, nj * D);
} else {
Cast(qi, qiIn, RoundMode::CAST_NONE, ng * D);
Cast(ki, kiIn, RoundMode::CAST_NONE, nj * D);
}
PipeBarrier<PIPE_V>();
Duplicate(ii, static_cast<T>(0.0f), JT);
DataCopy(p, mainPBuf.Get<T>(), JT);
Duplicate(ss, static_cast<T>(0.0f), JT);
PipeBarrier<PIPE_V>();
for (int32_t g = 0; g < ng; ++g) {
for (int32_t h = 0; h < D / 64; ++h) {
const int32_t off = h * 64;
Mul(rep[off], ki[off], qi[g * D + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(nj),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8),
static_cast<uint8_t>(D / 8), 0));
}
PipeBarrier<PIPE_V>();
uint32_t shape[2] = {static_cast<uint32_t>(nj), static_cast<uint32_t>(D)};
ReduceSum<T, Pattern::Reduce::AR, true>(ss, rep, work, shape, true);
PipeBarrier<PIPE_V>();
Maxs(ss, ss, static_cast<T>(0.0f), JT);
PipeBarrier<PIPE_V>();
DataCopy(scores[g * JT], ss, JT);
Muls(tmp, ss, weightsAt(g), JT);
PipeBarrier<PIPE_V>();
Add(ii, ii, tmp, JT);
PipeBarrier<PIPE_V>();
}
const float psum = SumVec(p, nj);
uint32_t shapeI[2] = {1u, static_cast<uint32_t>(nj)};
ReduceMax<T, Pattern::Reduce::AR, !WIDE_FUSE>(one, ii, work, shapeI, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
const float mx = one.GetValue(0);
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
Adds(ii, ii, -mx, JT); PipeBarrier<PIPE_V>();
Exp(ii, ii, JT); PipeBarrier<PIPE_V>();
const float z = SumVec(ii, nj);
const float invZ = z > 0.0f ? 1.0f / z : 0.0f;
const float invP = psum > 0.0f ? 1.0f / psum : 0.0f;
Muls(y, ii, invZ, JT);
Muls(p, p, invP, JT);
PipeBarrier<PIPE_V>();
Maxs(pmax, p, static_cast<T>(LOSS_EPS), JT);
Maxs(ly, y, static_cast<T>(LOSS_EPS), JT);
PipeBarrier<PIPE_V>();
Log(lp, pmax, JT); Log(ly, ly, JT);
PipeBarrier<PIPE_V>();
Sub(tmp, lp, ly, JT); PipeBarrier<PIPE_V>();
Mul(tmp, tmp, pmax, JT); PipeBarrier<PIPE_V>();
const float lossAcc = SumVec(tmp, nj);
Sub(ii, y, p, JT); PipeBarrier<PIPE_V>();
Duplicate(dq, static_cast<T>(0.0f), ng * D);
Duplicate(dk, static_cast<T>(0.0f), nj * D);
PipeBarrier<PIPE_V>();
for (int32_t g = 0; g < ng; ++g) {
Muls(dScores[g * JT], ii, weightsAt(g), JT);
PipeBarrier<PIPE_V>();
}
Mul(dWtBuf.Get<T>(), scores, ii, static_cast<uint64_t>(nj), static_cast<uint8_t>(ng),
BinaryRepeatParams(1, 1, 1, 1, 1, 0));
PipeBarrier<PIPE_V>();
{
uint32_t dWshape[2] = {static_cast<uint32_t>(ng), static_cast<uint32_t>(nj)};
ReduceSum<T, Pattern::Reduce::AR, true>(dWaccBuf.Get<T>(), dWtBuf.Get<T>(), work,
dWshape, true);
}
PipeBarrier<PIPE_V>();
LocalTensor<uint8_t> mask = smallBuf.Get<uint8_t>()[512];
CompareScalar(mask, scores, static_cast<T>(0.0f), CMPMODE::GT, ng * JT);
PipeBarrier<PIPE_V>();
Select(dScores, mask, dScores, static_cast<T>(0.0f),
SELMODE::VSEL_TENSOR_SCALAR_MODE, ng * JT);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
for (int32_t g = 0; g < ng; ++g) {
for (int32_t j = 0; j < nj; ++j) {
const float c = dScores.GetValue(g * JT + j);
if (c == 0.0f) continue;
Axpy(dq[g * D], ki[j * D], c, D);
PipeBarrier<PIPE_V>();
Axpy(dk[j * D], qi[g * D], c, D);
PipeBarrier<PIPE_V>();
}
}
LocalTensor<OUT_T> outQ = outQBuf.Get<OUT_T>();
if constexpr (IsSameType<OUT_T, T>::value) DataCopy(outQ, dq, ng * D);
else Cast(outQ, dq, RoundMode::CAST_RINT, ng * D);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpOut(1, ng * D * sizeof(OUT_T), 0, 0, 0);
DataCopyPad(dQueryIndexGm[qiBase], outQ, cpOut);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
if (directKey_) {
WriteDirectDKey(dk, nj);
} else {
LocalTensor<int32_t> idx = idxBuf.Get<int32_t>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
for (int32_t j = 0; j < nj; ++j) idx.SetValue(j, j);
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
ScatterAddTile(b, s1, 0, nj, dk);
}
return lossAcc;
}
__aicore__ inline float RunTaskFusedBigHeads(uint32_t b, uint32_t s1) {
constexpr int32_t D = 128;
constexpr int32_t KCH = 16;
const int32_t nj = static_cast<int32_t>(t->s2Size);
const int32_t ng = static_cast<int32_t>(t->nidx1Size);
LocalTensor<T> scratch = repBuf.Get<T>();
LocalTensor<T> prod = scratch;
LocalTensor<T> coef = scratch[KCH * D];
LocalTensor<T> wRep = coef[nj * ng];
LocalTensor<T> deltaRep = wRep[nj * ng];
LocalTensor<T> cbrTmp = deltaRep[nj * ng];
LocalTensor<T> cbr = cbrTmp[KCH * 8];
LocalTensor<T> sim = sCacheBuf.Get<T>();
LocalTensor<T> qi = qiAllBuf.Get<T>();
LocalTensor<T> ki = kiTileBuf.Get<T>();
LocalTensor<T> dq = accDqBuf.Get<T>();
LocalTensor<T> dk = dKTileBuf.Get<T>();
LocalTensor<T> p = vecBuf.Get<T>();
LocalTensor<T> ii = vecBuf.Get<T>()[JT];
LocalTensor<T> y = vecBuf.Get<T>()[2 * JT];
LocalTensor<T> tmp = vecBuf.Get<T>()[4 * JT];
LocalTensor<T> lp = vecBuf.Get<T>()[6 * JT];
LocalTensor<T> ly = vecBuf.Get<T>()[7 * JT];
LocalTensor<T> pmax = vecBuf.Get<T>()[8 * JT];
LocalTensor<T> one = smallBuf.Get<T>();
LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
LocalTensor<uint8_t> maskBits = smallBuf.Get<uint8_t>()[512];
LocalTensor<IN_T> qiIn = inQBuf.Get<IN_T>();
LocalTensor<IN_T> kiIn = inKiBuf.Get<IN_T>();
LocalTensor<T> wRow = wRowBuf.Get<T>();
const uint64_t qiBase = (static_cast<uint64_t>(b) * t->s1Size + s1) * t->nidx1Size * D;
const uint64_t kiBase = static_cast<uint64_t>(b) * t->s2Size * D;
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
{
DataCopyExtParams cpK(1, static_cast<uint32_t>(nj * D * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kiIn, keyIndexGm[kiBase], cpK, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(ki, kiIn, nj * D);
else Cast(ki, kiIn, RoundMode::CAST_NONE, nj * D);
PipeBarrier<PIPE_V>();
}
for (int32_t g0 = 0; g0 < ng; g0 += KCH) {
const int32_t gt = ((ng - g0) < KCH) ? (ng - g0) : KCH;
DataCopyExtParams cpQ(1, static_cast<uint32_t>(gt * D * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qiIn, queryIndexGm[qiBase + static_cast<uint64_t>(g0) * D], cpQ, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qi, qiIn, gt * D);
else Cast(qi, qiIn, RoundMode::CAST_NONE, gt * D);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
for (int32_t h = 0; h < D / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], qi[off], ki[j * D + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8),
static_cast<uint8_t>(D / 8), 0));
}
PipeBarrier<PIPE_V>();
uint32_t shD[2] = {static_cast<uint32_t>(gt), static_cast<uint32_t>(D)};
ReduceSum<T, Pattern::Reduce::AR, true>(sim[j * ng + g0], prod, work, shD, true);
PipeBarrier<PIPE_V>();
}
}
Maxs(sim, sim, static_cast<T>(0.0f), nj * ng);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
DataCopy(wRep[j * ng], wRow, ng);
}
DataCopy(p, mainPBuf.Get<T>(), JT);
PipeBarrier<PIPE_V>();
Mul(prod, sim, wRep, nj * ng);
PipeBarrier<PIPE_V>();
{
uint32_t shI[2] = {static_cast<uint32_t>(nj), static_cast<uint32_t>(ng)};
ReduceSum<T, Pattern::Reduce::AR, true>(ii, prod, work, shI, true);
PipeBarrier<PIPE_V>();
uint32_t shMax[2] = {1u, static_cast<uint32_t>(nj)};
ReduceMax<T, Pattern::Reduce::AR, !WIDE_FUSE>(one, ii, work, shMax, true);
}
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
const float mx = one.GetValue(0);
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
Adds(ii, ii, -mx, JT);
PipeBarrier<PIPE_V>();
Exp(ii, ii, JT);
PipeBarrier<PIPE_V>();
const float z = SumVec(ii, nj);
const float psum = SumVec(p, nj);
const float invZ = z > 0.0f ? 1.0f / z : 0.0f;
const float invP = psum > 0.0f ? 1.0f / psum : 0.0f;
Muls(y, ii, invZ, JT);
Muls(p, p, invP, JT);
PipeBarrier<PIPE_V>();
Maxs(pmax, p, static_cast<T>(LOSS_EPS), JT);
Maxs(ly, y, static_cast<T>(LOSS_EPS), JT);
PipeBarrier<PIPE_V>();
Log(lp, pmax, JT); Log(ly, ly, JT);
PipeBarrier<PIPE_V>();
Sub(tmp, lp, ly, JT); PipeBarrier<PIPE_V>();
Mul(tmp, tmp, pmax, JT); PipeBarrier<PIPE_V>();
const float lossAcc = SumVec(tmp, nj);
Sub(ii, y, p, JT);
PipeBarrier<PIPE_V>();
CompareScalar(maskBits, sim, static_cast<T>(0.0f), CMPMODE::GT, nj * ng);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
for (int32_t j = 0; j < nj; ++j) {
Duplicate(deltaRep[j * ng], ii.GetValue(j), ng);
}
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
Mul(coef, wRep, deltaRep, nj * ng);
PipeBarrier<PIPE_V>();
Select(coef, maskBits, coef, static_cast<T>(0.0f),
SELMODE::VSEL_TENSOR_SCALAR_MODE, nj * ng);
PipeBarrier<PIPE_V>();
Mul(prod, sim, deltaRep, nj * ng);
PipeBarrier<PIPE_V>();
{
int32_t r = nj;
while (r > 1) {
const int32_t half = r / 2;
Add(prod[0], prod[0], prod[half * ng], static_cast<uint64_t>(ng),
static_cast<uint8_t>(half),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(ng / 8),
static_cast<uint8_t>(ng / 8), static_cast<uint8_t>(ng / 8)));
PipeBarrier<PIPE_V>();
r -= half;
}
}
DataCopy(dWaccBuf.Get<T>(), prod, ng);
PipeBarrier<PIPE_V>();
Duplicate(dk, static_cast<T>(0.0f), nj * D);
PipeBarrier<PIPE_V>();
for (int32_t g0 = 0; g0 < ng; g0 += KCH) {
const int32_t gt = ((ng - g0) < KCH) ? (ng - g0) : KCH;
Duplicate(dq, static_cast<T>(0.0f), gt * D);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
Brcb(cbrTmp, coef[j * ng + g0], static_cast<uint8_t>(gt / 8),
BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
Brcb(cbr, cbrTmp, static_cast<uint8_t>(gt), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
for (int32_t h = 0; h < D / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], cbr, ki[j * D + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8),
static_cast<uint8_t>(64 / 8), 0));
PipeBarrier<PIPE_V>();
Add(dq[off], dq[off], prod[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8),
static_cast<uint8_t>(D / 8), static_cast<uint8_t>(D / 8)));
PipeBarrier<PIPE_V>();
}
}
LocalTensor<OUT_T> outQ = outQBuf.Get<OUT_T>();
if constexpr (IsSameType<OUT_T, T>::value) DataCopy(outQ, dq, gt * D);
else Cast(outQ, dq, RoundMode::CAST_RINT, gt * D);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpOut(1, static_cast<uint32_t>(gt * D * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dQueryIndexGm[qiBase + static_cast<uint64_t>(g0) * D], outQ, cpOut);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
for (int32_t g0 = 0; g0 < ng; g0 += KCH) {
const int32_t gt = ((ng - g0) < KCH) ? (ng - g0) : KCH;
DataCopyExtParams cpQ(1, static_cast<uint32_t>(gt * D * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qiIn, queryIndexGm[qiBase + static_cast<uint64_t>(g0) * D], cpQ, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qi, qiIn, gt * D);
else Cast(qi, qiIn, RoundMode::CAST_NONE, gt * D);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
Brcb(cbrTmp, coef[j * ng + g0], static_cast<uint8_t>(gt / 8),
BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
Brcb(cbr, cbrTmp, static_cast<uint8_t>(gt), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
for (int32_t h = 0; h < D / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], cbr, qi[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(D / 8)));
PipeBarrier<PIPE_V>();
int32_t r = gt;
while (r > 1) {
const int32_t half = r / 2;
Add(prod[off], prod[off], prod[off + half * 64], static_cast<uint64_t>(64u),
static_cast<uint8_t>(half),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8)));
PipeBarrier<PIPE_V>();
r -= half;
}
Add(dk[j * D + off], dk[j * D + off], prod[off], static_cast<uint64_t>(64u), 1,
BinaryRepeatParams(1, 1, 1, 1, 1, 1));
PipeBarrier<PIPE_V>();
}
}
}
if (t->totalTasks <= 1u && t->s2Size >= 5u && t->s2Size <= 8u &&
t->hasRope == 0u) {
WriteDirectDKey(dk, nj);
} else {
LocalTensor<int32_t> idx = idxBuf.Get<int32_t>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
for (int32_t j = 0; j < nj; ++j) idx.SetValue(j, j);
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
ScatterAddTile(b, s1, 0, nj, dk);
}
return lossAcc;
}
__aicore__ inline uint32_t HeadTake(uint64_t bytes) {
const uint32_t at = headOff_;
headOff_ += static_cast<uint32_t>((bytes + 511u) / 512u * 512u);
return at;
}
template <typename ET>
__aicore__ inline LocalTensor<ET> HeadSlice(uint32_t off) {
LocalTensor<uint8_t> buf = headBuf_.Get<uint8_t>();
return buf[off].ReinterpretCast<ET>();
}
__aicore__ inline void HeadSplitInit(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex,
GM_ADDR keyIndex, GM_ADDR weights, GM_ADDR dQueryIndex,
GM_ADDR dKeyIndex, GM_ADDR dWeights, GM_ADDR loss,
GM_ADDR workspace) {
blockIdx_ = GetBlockIdx();
blockNum_ = GetBlockNum();
queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
queryIndexGm.SetGlobalBuffer((__gm__ IN_T *)queryIndex);
keyIndexGm.SetGlobalBuffer((__gm__ IN_T *)keyIndex);
if (t->weightsFp32 != 0u) {
weightsF32Gm.SetGlobalBuffer((__gm__ float *)weights);
dWeightsF32Gm.SetGlobalBuffer((__gm__ float *)dWeights);
} else {
weights16Gm.SetGlobalBuffer((__gm__ IN_T *)weights);
dWeights16Gm.SetGlobalBuffer((__gm__ IN_T *)dWeights);
}
dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
lossGmRef.SetGlobalBuffer((__gm__ T *)loss);
headSplit_ = true;
headParts_ = blockNum_ / t->totalTasks;
splitTask_ = blockIdx_ / headParts_;
const uint32_t part = blockIdx_ % headParts_;
mainH0_ = part * (t->n1Size / headParts_);
mainH1_ = mainH0_ + (t->n1Size / headParts_);
idxH0_ = part * (t->nidx1Size / headParts_);
idxH1_ = idxH0_ + (t->nidx1Size / headParts_);
__gm__ uint8_t *ws = (__gm__ uint8_t *)workspace;
pPartGm_.SetGlobalBuffer((__gm__ T *)(ws + t->reserved[1]), blockNum_ * 16u);
iPartGm_.SetGlobalBuffer((__gm__ T *)(ws + t->reserved[2]), blockNum_ * 16u);
dkPartGm_.SetGlobalBuffer((__gm__ T *)(ws + t->reserved[3]),
static_cast<uint64_t>(blockNum_) * t->s2Size * t->dQueryIndex);
const uint64_t scatterBytes = static_cast<uint64_t>(t->bSize) * t->s2Size *
t->dQueryIndex * sizeof(T);
const uint64_t syncOffset = t->wsOffsetScatter + ((scatterBytes + 511u) / 512u) * 512u;
syncGm.SetGlobalBuffer((__gm__ int32_t *)(ws + syncOffset));
constexpr uint32_t DMAX = 128u;
constexpr uint32_t NJMAX = 8u;
constexpr uint32_t MTILE = 32u;
constexpr uint32_t KCH = 16u;
headOff_ = 0;
hLogits_ = HeadTake(NJMAX * N1_CAP * sizeof(T));
hP_ = HeadTake(16u * sizeof(T));
hRep_ = HeadTake(4096u * sizeof(T));
hKTile_ = HeadTake(4096u * sizeof(T));
hDot_ = HeadTake(MTILE * sizeof(T));
hVec_ = HeadTake(10u * JT * sizeof(T));
hMx_ = HeadTake(N1_CAP * sizeof(T));
hSum_ = HeadTake(N1_CAP * sizeof(T));
hQi_ = HeadTake(KCH * DMAX * sizeof(T));
hKi_ = HeadTake(NJMAX * DMAX * sizeof(T));
hDq_ = HeadTake(KCH * DMAX * sizeof(T));
hDk_ = HeadTake(NJMAX * DMAX * sizeof(T));
hWRow_ = HeadTake(NIDX1_CAP * sizeof(T));
hWIn_ = HeadTake(NIDX1_CAP * sizeof(IN_T));
hOutW_ = HeadTake(NIDX1_CAP * sizeof(T));
hOutQ_ = HeadTake(static_cast<uint32_t>(GTILE) * DMAX * sizeof(OUT_T));
hAcc_ = HeadTake(NJMAX * DMAX * sizeof(T));
hPart_ = HeadTake(NJMAX * DMAX * sizeof(T));
hSim_ = HeadTake(512u * sizeof(T));
hWork_ = HeadTake(16u * 1024u);
hQIn_ = HeadTake(MTILE * DMAX * sizeof(IN_T));
hKIn_ = HeadTake(MTILE * DMAX * sizeof(IN_T));
hKiIn_ = HeadTake(NJMAX * DMAX * sizeof(IN_T));
pipe.InitBuffer(headBuf_, Align32(headOff_));
pipe.InitBuffer(smallBuf, 1024u);
pipe.InitBuffer(syncBuf, blockNum_ * 32u);
}
__aicore__ inline void ComputeMainBatchedRange(uint32_t b, uint32_t s1, uint32_t hBegin,
uint32_t hEnd) {
constexpr uint32_t dQ = 128u;
const uint32_t nL = hEnd - hBegin;
if (nL == 0u) { return; }
LocalTensor<T> logits = HeadSlice<T>(hLogits_);
LocalTensor<T> p = HeadSlice<T>(hP_);
LocalTensor<T> qTile = HeadSlice<T>(hRep_);
LocalTensor<T> kTile = HeadSlice<T>(hKTile_);
LocalTensor<T> dot = HeadSlice<T>(hDot_);
LocalTensor<T> mx = HeadSlice<T>(hMx_);
LocalTensor<T> sum = HeadSlice<T>(hSum_);
LocalTensor<T> one = smallBuf.Get<T>();
LocalTensor<uint8_t> work = HeadSlice<uint8_t>(hWork_);
LocalTensor<IN_T> qIn = HeadSlice<IN_T>(hQIn_);
LocalTensor<IN_T> kIn = HeadSlice<IN_T>(hKIn_);
Duplicate(p, static_cast<T>(0.0f), 16);
PipeBarrier<PIPE_V>();
const uint32_t headTile = (nL < 32u) ? nL : 32u;
const DataCopyExtParams cp(1, static_cast<uint32_t>(headTile * dQ * sizeof(IN_T)), 0, 0, 0);
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
for (uint32_t g0 = hBegin; g0 < hEnd; g0 += headTile) {
const uint32_t gt = ((hEnd - g0) < headTile) ? (hEnd - g0) : headTile;
DataCopyExtParams cpg(1, static_cast<uint32_t>(gt * dQ * sizeof(IN_T)), 0, 0, 0);
const uint64_t qOff = ((static_cast<uint64_t>(b) * t->s1Size + s1) * t->n1Size + g0) * dQ;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qIn, queryGm[qOff], cpg, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qTile, qIn, gt * dQ);
else Cast(qTile, qIn, RoundMode::CAST_NONE, gt * dQ);
PipeBarrier<PIPE_V>();
for (uint32_t j = 0; j < t->s2Size; ++j) {
const uint64_t kOff = ((static_cast<uint64_t>(b) * t->s2Size + j) * t->n1Size + g0) * dQ;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kIn, keyGm[kOff], cpg, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(kTile, kIn, gt * dQ);
else Cast(kTile, kIn, RoundMode::CAST_NONE, gt * dQ);
PipeBarrier<PIPE_V>();
Mul(kTile, kTile, qTile, gt * dQ);
PipeBarrier<PIPE_V>();
uint32_t shape[2] = {gt, dQ};
ReduceSum<T, Pattern::Reduce::AR, true>(dot, kTile, work, shape, true);
PipeBarrier<PIPE_V>();
Muls(dot, dot, t->scaleValue, gt);
PipeBarrier<PIPE_V>();
DataCopy(logits[j * nL + (g0 - hBegin)], dot, gt);
PipeBarrier<PIPE_V>();
}
}
(void)cp;
Duplicate(mx, static_cast<T>(-3.0e38f), nL);
Duplicate(sum, static_cast<T>(0.0f), nL);
PipeBarrier<PIPE_V>();
for (uint32_t j = 0; j < t->s2Size; ++j) {
Max(mx, mx, logits[j * nL], nL);
PipeBarrier<PIPE_V>();
}
for (uint32_t j = 0; j < t->s2Size; ++j) {
LocalTensor<T> row = logits[j * nL];
Sub(row, row, mx, nL);
PipeBarrier<PIPE_V>();
Exp(row, row, nL);
PipeBarrier<PIPE_V>();
Add(sum, sum, row, nL);
PipeBarrier<PIPE_V>();
}
uint32_t rowShape[2] = {1u, nL};
for (uint32_t j = 0; j < t->s2Size; ++j) {
LocalTensor<T> row = logits[j * nL];
Div(row, row, sum, nL);
PipeBarrier<PIPE_V>();
ReduceSum<T, Pattern::Reduce::AR, true>(one, row, work, rowShape, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
p.SetValue(j, one.GetValue(0));
SetFlag<HardEvent::S_V>(EV_SV);
WaitFlag<HardEvent::S_V>(EV_SV);
}
PipeBarrier<PIPE_V>();
}
__aicore__ inline void DoTaskSplitHead(uint32_t task) {
constexpr int32_t D = 128;
constexpr int32_t KCH = 16;
const uint32_t b = task / t->s1Size;
const uint32_t s1 = task % t->s1Size;
const int32_t nj = static_cast<int32_t>(t->s2Size);
const int32_t ngL = static_cast<int32_t>(idxH1_ - idxH0_);
const uint32_t K = blockNum_;
LocalTensor<T> scratch = HeadSlice<T>(hRep_);
LocalTensor<T> prod = scratch;
LocalTensor<T> coef = scratch[KCH * D];
LocalTensor<T> wRep = coef[256];
LocalTensor<T> deltaRep = wRep[256];
LocalTensor<T> cbrTmp = deltaRep[256];
LocalTensor<T> cbr = cbrTmp[128];
LocalTensor<T> sim = HeadSlice<T>(hSim_);
LocalTensor<T> qi = HeadSlice<T>(hQi_);
LocalTensor<T> ki = HeadSlice<T>(hKi_);
LocalTensor<T> dq = HeadSlice<T>(hDq_);
LocalTensor<T> dk = HeadSlice<T>(hDk_);
LocalTensor<T> vec = HeadSlice<T>(hVec_);
LocalTensor<T> p = vec;
LocalTensor<T> ii = vec[JT];
LocalTensor<T> y = vec[2 * JT];
LocalTensor<T> tmp = vec[4 * JT];
LocalTensor<T> lp = vec[6 * JT];
LocalTensor<T> ly = vec[7 * JT];
LocalTensor<T> pmax = vec[8 * JT];
LocalTensor<T> one = smallBuf.Get<T>();
LocalTensor<uint8_t> work = HeadSlice<uint8_t>(hWork_);
LocalTensor<uint8_t> maskBits = smallBuf.Get<uint8_t>()[512];
LocalTensor<IN_T> qiIn = HeadSlice<IN_T>(hQIn_);
LocalTensor<IN_T> kiIn = HeadSlice<IN_T>(hKiIn_);
LocalTensor<T> wRow = HeadSlice<T>(hWRow_);
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
const uint64_t qiBase = (static_cast<uint64_t>(b) * t->s1Size + s1) * t->nidx1Size * D;
const uint64_t kiBase = static_cast<uint64_t>(b) * t->s2Size * D;
LoadSplitWeightsRow(b, s1);
ComputeMainBatchedRange(b, s1, mainH0_, mainH1_);
{
LocalTensor<T> pLocal = HeadSlice<T>(hP_);
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, static_cast<uint32_t>(nj * sizeof(T)), 0, 0, 0);
DataCopyPad(pPartGm_[blockIdx_ * 16u], pLocal, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
{
DataCopyExtParams cpK(1, static_cast<uint32_t>(nj * D * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kiIn, keyIndexGm[kiBase], cpK, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(ki, kiIn, nj * D);
else Cast(ki, kiIn, RoundMode::CAST_NONE, nj * D);
PipeBarrier<PIPE_V>();
}
for (int32_t j = 0; j < nj; ++j) {
DataCopy(wRep[j * ngL], wRow[idxH0_], ngL);
}
PipeBarrier<PIPE_V>();
for (int32_t g0 = 0; g0 < ngL; g0 += KCH) {
const int32_t gt = ((ngL - g0) < KCH) ? (ngL - g0) : KCH;
DataCopyExtParams cpQ(1, static_cast<uint32_t>(gt * D * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qiIn, queryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D], cpQ, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qi, qiIn, gt * D);
else Cast(qi, qiIn, RoundMode::CAST_NONE, gt * D);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
for (int32_t h = 0; h < D / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], qi[off], ki[j * D + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8),
static_cast<uint8_t>(D / 8), 0));
}
PipeBarrier<PIPE_V>();
uint32_t shD[2] = {static_cast<uint32_t>(gt), static_cast<uint32_t>(D)};
ReduceSum<T, Pattern::Reduce::AR, true>(sim[j * ngL + g0], prod, work, shD, true);
PipeBarrier<PIPE_V>();
}
}
Maxs(sim, sim, static_cast<T>(0.0f), nj * ngL);
PipeBarrier<PIPE_V>();
Mul(prod, sim, wRep, nj * ngL);
PipeBarrier<PIPE_V>();
{
uint32_t shI[2] = {static_cast<uint32_t>(nj), static_cast<uint32_t>(ngL)};
ReduceSum<T, Pattern::Reduce::AR, true>(ii, prod, work, shI, true);
}
PipeBarrier<PIPE_V>();
{
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, static_cast<uint32_t>(nj * sizeof(T)), 0, 0, 0);
DataCopyPad(iPartGm_[blockIdx_ * 16u], ii, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
if (K > 1u) { SyncAll(syncGm, syncBuf.Get<int32_t>(), blockNum_); }
{
LocalTensor<T> pp = prod;
LocalTensor<T> ip = prod[64];
const uint32_t slot0 = splitTask_ * headParts_;
DataCopyExtParams cp(1, static_cast<uint32_t>(headParts_ * 16u * sizeof(T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(pp, pPartGm_[slot0 * 16u], cp, DataCopyPadExtParams<T>(false, 0, 0, 0));
DataCopyPad(ip, iPartGm_[slot0 * 16u], cp, DataCopyPadExtParams<T>(false, 0, 0, 0));
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
for (uint32_t c = 1u; c < headParts_; ++c) {
Add(pp, pp, pp[c * 16u], static_cast<uint64_t>(16u), 1,
BinaryRepeatParams(1, 1, 1, 2, 2, 2));
Add(ip, ip, ip[c * 16u], static_cast<uint64_t>(16u), 1,
BinaryRepeatParams(1, 1, 1, 2, 2, 2));
PipeBarrier<PIPE_V>();
}
Muls(p, pp, t->invN1, JT);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
ii.SetValue(j, ip.GetValue(j));
}
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
}
PipeBarrier<PIPE_V>();
{
uint32_t shMax[2] = {1u, static_cast<uint32_t>(nj)};
ReduceMax<T, Pattern::Reduce::AR, !WIDE_FUSE>(one, ii, work, shMax, true);
}
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
const float mx = one.GetValue(0);
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
Adds(ii, ii, -mx, JT);
PipeBarrier<PIPE_V>();
Exp(ii, ii, JT);
PipeBarrier<PIPE_V>();
const float z = SumVecSplit(ii, nj);
const float psum = SumVecSplit(p, nj);
const float invZ = z > 0.0f ? 1.0f / z : 0.0f;
const float invP = psum > 0.0f ? 1.0f / psum : 0.0f;
Muls(y, ii, invZ, JT);
Muls(p, p, invP, JT);
PipeBarrier<PIPE_V>();
Maxs(pmax, p, static_cast<T>(LOSS_EPS), JT);
Maxs(ly, y, static_cast<T>(LOSS_EPS), JT);
PipeBarrier<PIPE_V>();
Log(lp, pmax, JT); Log(ly, ly, JT);
PipeBarrier<PIPE_V>();
Sub(tmp, lp, ly, JT); PipeBarrier<PIPE_V>();
Mul(tmp, tmp, pmax, JT); PipeBarrier<PIPE_V>();
lossSum_ += SumVecSplit(tmp, nj);
Sub(ii, y, p, JT);
PipeBarrier<PIPE_V>();
const int32_t maskCount = ((nj * ngL + 63) / 64) * 64;
CompareScalar(maskBits, sim, static_cast<T>(0.0f), CMPMODE::GT, maskCount);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
for (int32_t j = 0; j < nj; ++j) {
Duplicate(deltaRep[j * ngL], ii.GetValue(j), ngL);
}
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
Mul(coef, wRep, deltaRep, nj * ngL);
PipeBarrier<PIPE_V>();
Select(coef, maskBits, coef, static_cast<T>(0.0f),
SELMODE::VSEL_TENSOR_SCALAR_MODE, maskCount);
PipeBarrier<PIPE_V>();
{
Mul(prod, sim, deltaRep, nj * ngL);
PipeBarrier<PIPE_V>();
int32_t r = nj;
while (r > 1) {
const int32_t half = r / 2;
Add(prod[0], prod[0], prod[(r - half) * ngL], static_cast<uint64_t>(ngL),
static_cast<uint8_t>(half),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(ngL / 8),
static_cast<uint8_t>(ngL / 8), static_cast<uint8_t>(ngL / 8)));
PipeBarrier<PIPE_V>();
r -= half;
}
const uint64_t wOff = (static_cast<uint64_t>(b) * t->s1Size + s1) * t->nidx1Size + idxH0_;
if (t->weightsFp32 != 0u) {
LocalTensor<float> w32 = HeadSlice<T>(hOutW_);
DataCopy(w32, prod, ngL);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpf(1, static_cast<uint32_t>(ngL * sizeof(float)), 0, 0, 0);
DataCopyPad(dWeightsF32Gm[wOff], w32, cpf);
} else {
LocalTensor<OUT_T> outW = HeadSlice<OUT_T>(hOutW_);
if constexpr (IsSameType<OUT_T, T>::value) DataCopy(outW, prod, ngL);
else Cast(outW, prod, RoundMode::CAST_RINT, ngL);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpw(1, static_cast<uint32_t>(ngL * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dWeights16Gm[wOff], outW, cpw);
}
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
for (int32_t g0 = 0; g0 < ngL; g0 += KCH) {
const int32_t gt = ((ngL - g0) < KCH) ? (ngL - g0) : KCH;
Duplicate(dq, static_cast<T>(0.0f), gt * D);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
Brcb(cbrTmp, coef[j * ngL + g0], static_cast<uint8_t>(gt / 8), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
Brcb(cbr, cbrTmp, static_cast<uint8_t>(gt), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
for (int32_t h = 0; h < D / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], cbr, ki[j * D + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8),
static_cast<uint8_t>(64 / 8), 0));
PipeBarrier<PIPE_V>();
Add(dq[off], dq[off], prod[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D / 8),
static_cast<uint8_t>(D / 8), static_cast<uint8_t>(D / 8)));
PipeBarrier<PIPE_V>();
}
}
LocalTensor<OUT_T> outQ = HeadSlice<OUT_T>(hOutQ_);
if constexpr (IsSameType<OUT_T, T>::value) DataCopy(outQ, dq, gt * D);
else Cast(outQ, dq, RoundMode::CAST_RINT, gt * D);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpOut(1, static_cast<uint32_t>(gt * D * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dQueryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D], outQ, cpOut);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
Duplicate(dk, static_cast<T>(0.0f), nj * D);
PipeBarrier<PIPE_V>();
for (int32_t g0 = 0; g0 < ngL; g0 += KCH) {
const int32_t gt = ((ngL - g0) < KCH) ? (ngL - g0) : KCH;
DataCopyExtParams cpQ(1, static_cast<uint32_t>(gt * D * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qiIn, queryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D], cpQ, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qi, qiIn, gt * D);
else Cast(qi, qiIn, RoundMode::CAST_NONE, gt * D);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
Brcb(cbrTmp, coef[j * ngL + g0], static_cast<uint8_t>(gt / 8), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
Brcb(cbr, cbrTmp, static_cast<uint8_t>(gt), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
for (int32_t h = 0; h < D / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], cbr, qi[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(D / 8)));
PipeBarrier<PIPE_V>();
int32_t r = gt;
while (r > 1) {
const int32_t half = r / 2;
Add(prod[off], prod[off], prod[off + half * 64], static_cast<uint64_t>(64u),
static_cast<uint8_t>(half),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8)));
PipeBarrier<PIPE_V>();
r -= half;
}
Add(dk[j * D + off], dk[j * D + off], prod[off], static_cast<uint64_t>(64u), 1,
BinaryRepeatParams(1, 1, 1, 1, 1, 1));
PipeBarrier<PIPE_V>();
}
}
}
{
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpd(1, static_cast<uint32_t>(nj * D * sizeof(T)), 0, 0, 0);
const uint64_t dkSlot = static_cast<uint64_t>(blockIdx_) * nj * D;
DataCopyPad(dkPartGm_[dkSlot], dk, cpd);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
__aicore__ inline void WriteDkFromParts() {
const int32_t nj = static_cast<int32_t>(t->s2Size);
const int32_t D = static_cast<int32_t>(t->dQueryIndex);
const uint32_t partsPerBatch = t->s1Size * headParts_;
LocalTensor<T> acc = HeadSlice<T>(hAcc_);
LocalTensor<T> part = HeadSlice<T>(hPart_);
LocalTensor<OUT_T> out = HeadSlice<OUT_T>(hOutQ_);
DataCopyExtParams cp(1, static_cast<uint32_t>(nj * D * sizeof(T)), 0, 0, 0);
DataCopyExtParams cpo(1, static_cast<uint32_t>(nj * D * sizeof(OUT_T)), 0, 0, 0);
const DataCopyPadExtParams<T> pad(false, 0, 0, 0);
for (uint32_t b = 0; b < t->bSize; ++b) {
const uint64_t slot0 = static_cast<uint64_t>(b) * partsPerBatch;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(acc, dkPartGm_[slot0 * static_cast<uint64_t>(nj * D)], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
for (uint32_t c = 1u; c < partsPerBatch; ++c) {
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(part, dkPartGm_[(slot0 + c) * static_cast<uint64_t>(nj * D)], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
Add(acc, acc, part, nj * D);
PipeBarrier<PIPE_V>();
}
if constexpr (IsSameType<OUT_T, T>::value) DataCopy(out, acc, nj * D);
else Cast(out, acc, RoundMode::CAST_RINT, nj * D);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyPad(dKeyIndexGm[static_cast<uint64_t>(b) * nj * D], out, cpo);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
__aicore__ inline void WriteDirectDKey(LocalTensor<T> dk, int32_t nj) {
const uint32_t count = static_cast<uint32_t>(nj) * t->dQueryIndex;
LocalTensor<OUT_T> out = outKBuf.Get<OUT_T>();
if constexpr (IsSameType<OUT_T, T>::value) DataCopy(out, dk, count);
else Cast(out, dk, RoundMode::CAST_RINT, count);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, count * sizeof(OUT_T), 0, 0, 0);
DataCopyPad(dKeyIndexGm, out, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
__aicore__ inline void LoadSeqLens(uint32_t b) {
const uint64_t idx = static_cast<uint64_t>(b);
int64_t vq = 0, vk = 0;
if (t->seqLensIsI64 != 0u) {
LocalTensor<int64_t> s8 = seqBuf.Get<int64_t>();
DataCopyExtParams cp(1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0);
DataCopyPadExtParams<int64_t> pad(false, 0, 0, 0);
DataCopyPad(s8, seqQ64Gm[idx], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
vq = s8.GetValue(0);
DataCopyPad(s8, seqK64Gm[idx], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
vk = s8.GetValue(0);
} else {
LocalTensor<int32_t> s4 = seqBuf.Get<int32_t>();
DataCopyExtParams cp(1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0);
DataCopyPadExtParams<int32_t> pad(false, 0, 0, 0);
DataCopyPad(s4, seqQ32Gm[idx], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
vq = static_cast<int64_t>(s4.GetValue(0));
DataCopyPad(s4, seqK32Gm[idx], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
vk = static_cast<int64_t>(s4.GetValue(0));
}
if (vq < 0) { vq = 0; }
if (vk < 0) { vk = 0; }
if (vq > static_cast<int64_t>(t->s1Size)) { vq = static_cast<int64_t>(t->s1Size); }
if (vk > static_cast<int64_t>(t->s2Size)) { vk = static_cast<int64_t>(t->s2Size); }
cachedActS1_ = static_cast<uint32_t>(vq);
cachedActS2_ = static_cast<uint32_t>(vk);
cachedSeqB_ = b;
seqCacheValid_ = true;
}
__aicore__ inline void ZeroTaskRow(uint32_t b, uint32_t s1) {
LocalTensor<T> zw = dWaccBuf.Get<T>();
Duplicate(zw, static_cast<T>(0.0), NIDX1_CAP);
PipeBarrier<PIPE_V>();
WriteDWeights(b, s1, zw);
const int32_t NIDX = static_cast<int32_t>(t->nidx1Size);
const int32_t DQII = static_cast<int32_t>(t->dQueryIndex);
const uint64_t qiBaseRow =
(static_cast<uint64_t>(b) * t->s1Size + s1) * static_cast<uint64_t>(t->nidx1Size);
LocalTensor<T> accDq = accDqBuf.Get<T>();
LocalTensor<OUT_T> outC = outQBuf.Get<OUT_T>();
for (int32_t g0 = 0; g0 < NIDX; g0 += GTILE) {
const int32_t gt = ((NIDX - g0) < GTILE) ? (NIDX - g0) : GTILE;
Duplicate(accDq, static_cast<T>(0.0), gt * DQII);
PipeBarrier<PIPE_V>();
if constexpr (IsSameType<OUT_T, T>::value) {
DataCopy(outC, accDq, gt * DQII);
} else {
Cast(outC, accDq, RoundMode::CAST_RINT, gt * DQII);
}
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
const uint64_t off =
(qiBaseRow + static_cast<uint64_t>(g0)) * static_cast<uint64_t>(DQII);
DataCopyExtParams cp(1, static_cast<uint32_t>(gt * DQII * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dQueryIndexGm[off], outC, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
__aicore__ inline float RunTask(uint32_t b, uint32_t s1, int64_t kReal) {
const uint32_t K = t->kSize;
const uint64_t pBase = static_cast<uint64_t>(b) * t->s1Size * K + static_cast<uint64_t>(s1) * K;
const int64_t ntile = (kReal + JT - 1) / JT;
float lossAcc = 0.0f;
LocalTensor<T> psumVec = vecBuf.Get<T>()[9 * JT];
Duplicate(psumVec, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
if (useResident_) {
LocalTensor<T> pr0 = pResBuf.Get<T>();
LocalTensor<T> ir0 = iResBuf.Get<T>();
Duplicate(pr0, static_cast<T>(0.0), static_cast<int32_t>(resElems_));
PipeBarrier<PIPE_V>();
Duplicate(ir0, static_cast<T>(0.0), static_cast<int32_t>(resElems_));
PipeBarrier<PIPE_V>();
}
for (int64_t jt = 0; jt < ntile; ++jt) {
const int32_t nj = static_cast<int32_t>((kReal - jt * JT < JT) ? (kReal - jt * JT) : JT);
GatherTile(b, s1, jt, nj, false, true);
LocalTensor<T> pTile = useResident_ ? pResBuf.Get<T>()[jt * JT] : vecBuf.Get<T>();
LocalTensor<T> iTile = useResident_ ? iResBuf.Get<T>()[jt * JT] : vecBuf.Get<T>()[JT];
LocalTensor<T> dotVec = vecBuf.Get<T>()[2 * JT];
LocalTensor<T> rpVec = vecBuf.Get<T>()[3 * JT];
LocalTensor<T> tmpVec = vecBuf.Get<T>()[4 * JT];
LocalTensor<T> sVec = vecBuf.Get<T>()[5 * JT];
Duplicate(pTile, static_cast<T>(0.0), JT);
Duplicate(iTile, static_cast<T>(0.0), JT);
Duplicate(rpVec, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
LocalTensor<T> kTile = kTileBuf.Get<T>();
LocalTensor<T> krTile = krTileBuf.Get<T>();
LocalTensor<T> kiTile = kiTileBuf.Get<T>();
LocalTensor<T> rep = repBuf.Get<T>();
LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
if (batchMain_) {
DataCopy(pTile, mainPBuf.Get<T>()[jt * JT], JT);
PipeBarrier<PIPE_V>();
} else if (cacheMain_) {
LocalTensor<T> cached = mainProbBuf.Get<T>();
for (uint32_t g = 0; g < t->n1Size; ++g) {
Add(pTile, pTile, cached[g * MAIN_CACHE_COLS + jt * JT], JT);
PipeBarrier<PIPE_V>();
}
} else for (uint32_t g = 0; g < t->n1Size; ++g) {
LoadMainKeyTile(b, jt, nj, g);
const uint64_t qOff =
(static_cast<uint64_t>(b) * t->s1Size + s1) * t->n1Size * t->dQuery +
static_cast<uint64_t>(g) * t->dQuery;
RepFromGm(rep, queryGm, qOff, nj, t->dQuery, inQBuf);
Mul(rep, rep, kTile, nj * t->dQuery);
PipeBarrier<PIPE_V>();
Duplicate(dotVec, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
uint32_t rshape[2] = {static_cast<uint32_t>(nj), t->dQuery};
ReduceSum<T, Pattern::Reduce::AR, true>(dotVec, rep, work, rshape, true);
PipeBarrier<PIPE_V>();
if (t->hasRope != 0u) {
const uint64_t qrOff =
(static_cast<uint64_t>(b) * t->s1Size + s1) * t->n1Size * t->dRope +
static_cast<uint64_t>(g) * t->dRope;
RepFromGmChunk(rep, queryRopeGm, qrOff, nj, t->dRope, t->dRope, inQrBuf);
Mul(rep, rep, krTile, nj * t->dRope);
PipeBarrier<PIPE_V>();
Duplicate(rpVec, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
uint32_t rshapeR[2] = {static_cast<uint32_t>(nj), t->dRope};
ReduceSum<T, Pattern::Reduce::AR, true>(rpVec, rep, work, rshapeR, true);
PipeBarrier<PIPE_V>();
Add(dotVec, dotVec, rpVec, JT);
PipeBarrier<PIPE_V>();
}
const float mx = smRowBuf.Get<T>().GetValue(g);
const float sm = ssRowBuf.Get<T>().GetValue(g);
const float invN = (sm != 0.0f) ? (1.0f / sm) * t->invN1 : 0.0f;
Muls(dotVec, dotVec, t->scaleValue, JT);
PipeBarrier<PIPE_V>();
Adds(dotVec, dotVec, -mx, JT);
PipeBarrier<PIPE_V>();
Exp(dotVec, dotVec, JT);
PipeBarrier<PIPE_V>();
Muls(dotVec, dotVec, invN, JT);
PipeBarrier<PIPE_V>();
Add(pTile, pTile, dotVec, JT);
PipeBarrier<PIPE_V>();
}
Add(psumVec, psumVec, pTile, nj);
PipeBarrier<PIPE_V>();
LocalTensor<T> rep2 = repBuf2.Get<T>();
if (batchIndex_) {
LocalTensor<T> qRows = qiAllBuf.Get<T>();
LocalTensor<T> dot8 = vecBuf.Get<T>()[5 * JT];
LocalTensor<T> one = smallBuf.Get<T>();
LocalTensor<IN_T> qIn = inQBuf.Get<IN_T>();
for (uint32_t g0 = 0; g0 < t->nidx1Size; g0 += 8u) {
const uint64_t qOff =
((static_cast<uint64_t>(b) * t->s1Size + s1) * t->nidx1Size + g0) * 128u;
DataCopyExtParams cpq(1, 8u * 128u * sizeof(IN_T), 0, 0, 0);
DataCopyPadExtParams<IN_T> padq(false, 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qIn, queryIndexGm[qOff], cpq, padq);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(qRows, qIn, 8 * 128);
} else {
Cast(qRows, qIn, RoundMode::CAST_NONE, 8 * 128);
}
PipeBarrier<PIPE_V>();
if (cacheS_) {
Duplicate(sCacheBuf.Get<T>()[g0 * JT], static_cast<T>(0.0), 8 * JT);
PipeBarrier<PIPE_V>();
}
for (int32_t j = 0; j < nj; ++j) {
RepRow(rep2, kiTile[j * 128], 8, 128);
Mul(rep2, rep2, qRows, 8 * 128);
PipeBarrier<PIPE_V>();
uint32_t rowShape[2] = {8u, 128u};
ReduceSum<T, Pattern::Reduce::AR, true>(dot8, rep2, work, rowShape, true);
PipeBarrier<PIPE_V>();
Maxs(dot8, dot8, static_cast<T>(0.0), 8);
PipeBarrier<PIPE_V>();
if (cacheS_) {
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
for (uint32_t gg = 0; gg < 8u; ++gg) {
sCacheBuf.Get<T>().SetValue((g0 + gg) * JT + j, dot8.GetValue(gg));
}
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
}
Mul(dot8, dot8, wRowBuf.Get<T>()[g0], 8);
PipeBarrier<PIPE_V>();
uint32_t sumShape[2] = {1u, 8u};
ReduceSum<T, Pattern::Reduce::AR, true>(one, dot8, work, sumShape, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
iTile.SetValue(j, iTile.GetValue(j) + one.GetValue(0));
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
}
}
} else for (uint32_t g = 0; g < t->nidx1Size; ++g) {
const uint64_t qiOff =
(static_cast<uint64_t>(b) * t->s1Size + s1) * t->nidx1Size * t->dQueryIndex +
static_cast<uint64_t>(g) * t->dQueryIndex;
RepFromGm(rep2, queryIndexGm, qiOff, nj, t->dQueryIndex, inQBuf);
Mul(rep2, rep2, kiTile, nj * t->dQueryIndex);
PipeBarrier<PIPE_V>();
Duplicate(sVec, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
Duplicate(tmpVec, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
uint32_t rshape2[2] = {static_cast<uint32_t>(nj), t->dQueryIndex};
ReduceSum<T, Pattern::Reduce::AR, true>(sVec, rep2, work, rshape2, true);
PipeBarrier<PIPE_V>();
Maxs(sVec, sVec, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
if (cacheS_) {
DataCopy(sCacheBuf.Get<T>()[g * JT], sVec, JT);
PipeBarrier<PIPE_ALL>();
}
const float wg = weightsAt(g);
Muls(tmpVec, sVec, wg, JT);
PipeBarrier<PIPE_V>();
Add(iTile, iTile, tmpVec, JT);
PipeBarrier<PIPE_V>();
}
if (!useResident_) {
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, static_cast<uint32_t>(nj * sizeof(T)), 0, 0, 0);
DataCopyPad(pGm[pBase + jt * JT], pTile, cp);
DataCopyPad(iGm[pBase + jt * JT], iTile, cp);
}
}
if (!useResident_) {
SetFlag<HardEvent::MTE3_MTE2>(EV_MTE3MTE2);
WaitFlag<HardEvent::MTE3_MTE2>(EV_MTE3MTE2);
}
float m = -3.0e38f, Z = 0.0f;
if (useResident_) {
LocalTensor<T> ir = iResBuf.Get<T>();
LocalTensor<T> acc = chunkBuf.Get<T>();
const int32_t kr = static_cast<int32_t>(kReal);
Duplicate(acc, static_cast<T>(-3.0e38f), JT);
PipeBarrier<PIPE_V>();
for (int32_t c = 0; c < kr; c += JT) {
int32_t n = kr - c;
if (n > JT) { n = JT; }
Max(acc, acc, ir[c], n);
}
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
float mm = acc.GetValue(0);
for (int32_t t = 1; t < JT; ++t) {
const float v = acc.GetValue(t);
if (v > mm) { mm = v; }
}
m = mm;
PipeBarrier<PIPE_V>();
Adds(ir, ir, static_cast<T>(-m), kr);
PipeBarrier<PIPE_V>();
Exp(ir, ir, kr);
PipeBarrier<PIPE_V>();
Duplicate(acc, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
for (int32_t c = 0; c < kr; c += JT) {
int32_t n = kr - c;
if (n > JT) { n = JT; }
Add(acc, acc, ir[c], n);
}
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
float zz = acc.GetValue(0);
for (int32_t t = 1; t < JT; ++t) { zz += acc.GetValue(t); }
Z = zz;
PipeBarrier<PIPE_V>();
} else {
LocalTensor<T> ch = chunkBuf.Get<T>();
LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
LocalTensor<T> one = smallBuf.Get<T>();
for (int64_t c = 0; c < kReal; c += CHUNK) {
int32_t n = static_cast<int32_t>((kReal - c < CHUNK) ? (kReal - c) : CHUNK);
Duplicate(ch, static_cast<T>(-3.0e38f), CHUNK);
PipeBarrier<PIPE_V>();
LoadChunk(ch, iGm[pBase + c], n);
uint32_t sh[2] = {1, static_cast<uint32_t>(CHUNK)};
ReduceMax<T, Pattern::Reduce::AR, true>(one, ch, work, sh, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
float tm = one.GetValue(0);
if (tm > m) { m = tm; }
PipeBarrier<PIPE_V>();
}
for (int64_t c = 0; c < kReal; c += CHUNK) {
int32_t n = static_cast<int32_t>((kReal - c < CHUNK) ? (kReal - c) : CHUNK);
Duplicate(ch, static_cast<T>(-3.0e38f), CHUNK);
PipeBarrier<PIPE_V>();
LoadChunk(ch, iGm[pBase + c], n);
Adds(ch, ch, -m, CHUNK);
PipeBarrier<PIPE_V>();
Exp(ch, ch, CHUNK);
PipeBarrier<PIPE_V>();
uint32_t sh[2] = {1, static_cast<uint32_t>(CHUNK)};
ReduceSum<T, Pattern::Reduce::AR, true>(one, ch, work, sh, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
Z += one.GetValue(0);
PipeBarrier<PIPE_V>();
}
}
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
float psum = 0.0f;
for (int32_t q = 0; q < JT; ++q) { psum += psumVec.GetValue(q); }
const T invPsum = (psum > 0.0f) ? static_cast<T>(1.0f / psum) : static_cast<T>(0.0f);
const float invZ = (Z > 0.0f) ? (1.0f / Z) : 0.0f;
for (int64_t jt = 0; jt < ntile; ++jt) {
const int32_t nj = static_cast<int32_t>((kReal - jt * JT < JT) ? (kReal - jt * JT) : JT);
LocalTensor<T> pTile = useResident_ ? pResBuf.Get<T>()[jt * JT] : vecBuf.Get<T>();
LocalTensor<T> iTile = useResident_ ? iResBuf.Get<T>()[jt * JT] : vecBuf.Get<T>()[JT];
LocalTensor<T> yVec = vecBuf.Get<T>()[2 * JT];
LocalTensor<T> dIVec = vecBuf.Get<T>()[3 * JT];
LocalTensor<T> tmpVec = vecBuf.Get<T>()[4 * JT];
LocalTensor<T> lp = vecBuf.Get<T>()[6 * JT];
LocalTensor<T> ly = vecBuf.Get<T>()[7 * JT];
LocalTensor<T> pmax = vecBuf.Get<T>()[8 * JT];
if (useResident_) {
Muls(yVec, iTile, static_cast<T>(invZ), JT);
PipeBarrier<PIPE_V>();
} else {
Duplicate(pTile, static_cast<T>(0.0), JT);
Duplicate(iTile, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
LoadChunk(pTile, pGm[pBase + jt * JT], nj);
LoadChunk(iTile, iGm[pBase + jt * JT], nj);
Adds(yVec, iTile, -m, JT);
PipeBarrier<PIPE_V>();
Exp(yVec, yVec, JT);
PipeBarrier<PIPE_V>();
Muls(yVec, yVec, invZ, JT);
PipeBarrier<PIPE_V>();
}
Muls(pTile, pTile, invPsum, JT);
PipeBarrier<PIPE_V>();
Maxs(pmax, pTile, static_cast<T>(LOSS_EPS), JT);
Maxs(ly, yVec, static_cast<T>(LOSS_EPS), JT);
PipeBarrier<PIPE_V>();
Log(lp, pmax, JT);
Log(ly, ly, JT);
PipeBarrier<PIPE_V>();
Sub(tmpVec, lp, ly, JT);
PipeBarrier<PIPE_V>();
Mul(tmpVec, tmpVec, pmax, JT);
PipeBarrier<PIPE_V>();
lossAcc += SumVec(tmpVec, nj);
if (useResident_) {
Sub(iTile, yVec, pTile, JT);
PipeBarrier<PIPE_V>();
} else {
Sub(dIVec, yVec, pTile, JT);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpd(1, static_cast<uint32_t>(nj * sizeof(T)), 0, 0, 0);
DataCopyPad(pGm[pBase + jt * JT], dIVec, cpd);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
if (!useResident_) {
SetFlag<HardEvent::MTE3_MTE2>(EV_MTE3MTE2);
WaitFlag<HardEvent::MTE3_MTE2>(EV_MTE3MTE2);
}
{
const int32_t NIDX = static_cast<int32_t>(t->nidx1Size);
const int32_t DQII = static_cast<int32_t>(t->dQueryIndex);
const uint64_t qiBaseRow =
(static_cast<uint64_t>(b) * t->s1Size + s1) * static_cast<uint64_t>(t->nidx1Size);
LocalTensor<T> kiTile = kiTileBuf.Get<T>();
LocalTensor<T> rep2 = repBuf2.Get<T>();
LocalTensor<T> s2d = s2dBuf.Get<T>();
LocalTensor<T> dIrep = dIrepBuf.Get<T>();
LocalTensor<T> dWt = dWtBuf.Get<T>();
LocalTensor<T> dWres = dWresBuf.Get<T>();
LocalTensor<T> dS2d = dS2dBuf.Get<T>();
LocalTensor<T> dKTile = dKTileBuf.Get<T>();
LocalTensor<T> accDq = accDqBuf.Get<T>();
LocalTensor<T> qiAll = qiAllBuf.Get<T>();
LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
LocalTensor<uint8_t> mask = smallBuf.Get<uint8_t>()[512];
LocalTensor<OUT_T> outC = outQBuf.Get<OUT_T>();
for (int32_t g0 = 0; g0 < NIDX; g0 += GTILE) {
const int32_t gt = ((NIDX - g0) < GTILE) ? (NIDX - g0) : GTILE;
{
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
LocalTensor<IN_T> inq = inQBuf.Get<IN_T>();
DataCopyExtParams cpq(1, static_cast<uint32_t>(DQII * sizeof(IN_T)), 0, 0, 0);
DataCopyPadExtParams<IN_T> padq(false, 0, 0, 0);
for (int32_t gg = 0; gg < gt; ++gg) {
DataCopyPad(inq[gg * DQII],
queryIndexGm[(qiBaseRow + static_cast<uint64_t>(g0 + gg)) *
static_cast<uint64_t>(DQII)],
cpq, padq);
}
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
for (int32_t gg = 0; gg < gt; ++gg) {
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(qiAll[gg * DQII], inq[gg * DQII], DQII);
PipeBarrier<PIPE_ALL>();
} else {
Cast(qiAll[gg * DQII], inq[gg * DQII], RoundMode::CAST_NONE, DQII);
}
}
PipeBarrier<PIPE_V>();
}
Duplicate(accDq, static_cast<T>(0.0), gt * DQII);
PipeBarrier<PIPE_V>();
for (int64_t jt = 0; jt < ntile; ++jt) {
const int32_t nj =
static_cast<int32_t>((kReal - jt * JT < JT) ? (kReal - jt * JT) : JT);
GatherTile(b, s1, jt, nj, false, true);
LocalTensor<T> dITile =
useResident_ ? iResBuf.Get<T>()[jt * JT] : vecBuf.Get<T>()[3 * JT];
LocalTensor<T> sVec = vecBuf.Get<T>()[5 * JT];
if (!useResident_) {
Duplicate(dITile, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
LoadChunk(dITile, pGm[pBase + jt * JT], nj);
}
for (int32_t gg = 0; gg < gt; ++gg) {
if (cacheS_) {
DataCopy(s2d[gg * JT], sCacheBuf.Get<T>()[(g0 + gg) * JT], JT);
PipeBarrier<PIPE_ALL>();
continue;
}
const uint64_t qiOff =
(qiBaseRow + static_cast<uint64_t>(g0 + gg)) *
static_cast<uint64_t>(DQII);
RepFromGm(rep2, queryIndexGm, qiOff, nj, static_cast<uint32_t>(DQII), inQBuf);
Mul(rep2, rep2, kiTile, nj * DQII);
PipeBarrier<PIPE_V>();
Duplicate(sVec, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
uint32_t rshape2[2] = {static_cast<uint32_t>(nj),
static_cast<uint32_t>(DQII)};
ReduceSum<T, Pattern::Reduce::AR, true>(sVec, rep2, work, rshape2, true);
PipeBarrier<PIPE_V>();
Maxs(sVec, sVec, static_cast<T>(0.0), JT);
PipeBarrier<PIPE_V>();
DataCopy(s2d[gg * JT], sVec, JT);
PipeBarrier<PIPE_V>();
}
for (int32_t gg = 0; gg < gt; ++gg) {
DataCopy(dIrep[gg * JT], dITile, JT);
}
PipeBarrier<PIPE_V>();
Mul(dWt, s2d, dIrep, gt * JT);
PipeBarrier<PIPE_V>();
uint32_t sh3[2] = {static_cast<uint32_t>(gt), static_cast<uint32_t>(JT)};
ReduceSum<T, Pattern::Reduce::AR, true>(dWres, dWt, work, sh3, true);
PipeBarrier<PIPE_V>();
Add(dWaccBuf.Get<T>()[g0], dWaccBuf.Get<T>()[g0], dWres, gt);
PipeBarrier<PIPE_V>();
for (int32_t gg = 0; gg < gt; ++gg) {
Muls(dS2d[gg * JT], dITile,
static_cast<T>(weightsAt(static_cast<uint32_t>(g0 + gg))), JT);
}
PipeBarrier<PIPE_V>();
CompareScalar(mask, s2d, static_cast<T>(0.0), CMPMODE::GT, gt * JT);
PipeBarrier<PIPE_V>();
Select(dS2d, mask, dS2d, static_cast<T>(0.0),
SELMODE::VSEL_TENSOR_SCALAR_MODE, gt * JT);
PipeBarrier<PIPE_V>();
Duplicate(dKTile, static_cast<T>(0.0), JT * DQII);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
for (int32_t gg = 0; gg < gt; ++gg) {
LocalTensor<T> qiG = qiAll[gg * DQII];
for (int32_t j = 0; j < nj; ++j) {
float c = dS2d.GetValue(gg * JT + j);
if (c == 0.0f) { continue; }
Axpy(accDq[gg * DQII], kiTile[j * DQII], c, DQII);
PipeBarrier<PIPE_V>();
Axpy(dKTile[j * DQII], qiG, c, DQII);
PipeBarrier<PIPE_V>();
}
}
ScatterAddTile(b, s1, jt, nj, dKTile);
}
if constexpr (IsSameType<OUT_T, T>::value) {
DataCopy(outC, accDq, gt * DQII);
PipeBarrier<PIPE_ALL>();
} else {
Cast(outC, accDq, RoundMode::CAST_RINT, gt * DQII);
}
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
uint64_t off = (qiBaseRow + static_cast<uint64_t>(g0)) * static_cast<uint64_t>(DQII);
DataCopyExtParams cp(1, static_cast<uint32_t>(gt * DQII * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dQueryIndexGm[off], outC, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
return lossAcc;
}
__aicore__ inline void LoadChunk(LocalTensor<T> dst, const GlobalTensor<T> &src, int32_t n) {
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyExtParams cp(1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0);
DataCopyPadExtParams<T> pad(false, 0, 0, 0);
DataCopyPad(dst, src, cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
}
__aicore__ inline float SumVec(const LocalTensor<T> &v, int32_t n = JT) {
LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
LocalTensor<T> one = smallBuf.Get<T>();
uint32_t sh[2] = {1, static_cast<uint32_t>(n)};
ReduceSum<T, Pattern::Reduce::AR, !WIDE_FUSE>(one, v, work, sh, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
return one.GetValue(0);
}
__aicore__ inline void RepRow(const LocalTensor<T> &dst, const LocalTensor<T> &src, int32_t nj,
uint32_t d) {
for (int32_t r = 0; r < nj; ++r) {
DataCopy(dst[r * d], src, d);
}
PipeBarrier<PIPE_V>();
}
template <typename GT, typename BUF>
__aicore__ inline void RepFromGm(const LocalTensor<T> &dst, const GT &gm, uint64_t off, int32_t nj,
uint32_t d, BUF &scratchBuf) {
LocalTensor<IN_T> in = scratchBuf.template Get<IN_T>();
LocalTensor<T> one = qgBuf.Get<T>();
DataCopyExtParams cp(1, static_cast<uint32_t>(d * sizeof(IN_T)), 0, 0, 0);
DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
if constexpr (IsSameType<IN_T, T>::value) DataCopyPad(one, gm[off], cp, pad);
else DataCopyPad(in, gm[off], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (!IsSameType<IN_T, T>::value)
Cast(one, in, RoundMode::CAST_NONE, d);
PipeBarrier<PIPE_V>();
RepRow(dst, one, nj, d);
}
template <typename DST, typename GT>
__aicore__ inline void RepOneChunk(const DST &dst, int32_t dstOff, const GT &gm, uint64_t off,
uint32_t w, uint32_t chunk) {
DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
if ((chunk == 0u) || (chunk >= w)) {
DataCopyExtParams cp1(1, static_cast<uint32_t>(w * sizeof(IN_T)), 0, 0, 0);
DataCopyPad(dst[dstOff], gm[off], cp1, pad);
} else {
for (uint32_t c = 0; c < w; c += chunk) {
const uint32_t n = ((w - c) < chunk) ? (w - c) : chunk;
DataCopyExtParams cp1(1, static_cast<uint32_t>(n * sizeof(IN_T)), 0, 0, 0);
DataCopyPad(dst[dstOff + static_cast<int32_t>(c)], gm[off + c], cp1, pad);
}
}
}
template <typename GT, typename BUF>
__aicore__ inline void RepFromGmChunk(const LocalTensor<T> &dst, const GT &gm, uint64_t off,
int32_t nj, uint32_t w, uint32_t chunk, BUF &scratchBuf) {
LocalTensor<IN_T> in = scratchBuf.template Get<IN_T>();
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
for (int32_t r = 0; r < nj; ++r) {
RepOneChunk(in, r * static_cast<int32_t>(w), gm, off, w, chunk);
}
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
for (int32_t r = 0; r < nj; ++r) {
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(dst[r * static_cast<int32_t>(w)], in[r * static_cast<int32_t>(w)], w);
} else {
Cast(dst[r * static_cast<int32_t>(w)], in[r * static_cast<int32_t>(w)],
RoundMode::CAST_NONE, w);
}
}
PipeBarrier<PIPE_V>();
}
__aicore__ inline void LoadMainKeyTile(uint32_t b, int64_t jt, int32_t nj, uint32_t head) {
LocalTensor<IN_T> inK = inKBuf.Get<IN_T>();
LocalTensor<T> dst = kTileBuf.Get<T>();
DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
DataCopyExtParams cp(1, static_cast<uint32_t>(t->dQuery * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
for (int32_t j = 0; j < nj; ++j) {
const uint64_t row = static_cast<uint64_t>(b) * t->s2Size +
static_cast<uint64_t>(jt * JT + j);
const uint64_t off = (row * t->n1Size + head) * t->dQuery;
if constexpr (IsSameType<IN_T, T>::value)
DataCopyPad(dst[j * t->dQuery], keyGm[off], cp, pad);
else
DataCopyPad(inK[j * t->dQuery], keyGm[off], cp, pad);
}
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (!IsSameType<IN_T, T>::value) {
for (int32_t j = 0; j < nj; ++j) {
Cast(dst[j * t->dQuery], inK[j * t->dQuery], RoundMode::CAST_NONE, t->dQuery);
}
}
PipeBarrier<PIPE_V>();
}
template <int32_t DW>
__aicore__ inline void ComputeMainBatched(uint32_t b, uint32_t s1) {
const uint32_t nHeads = t->n1Size;
constexpr uint32_t dQ = static_cast<uint32_t>(DW);
LocalTensor<T> logits = mainLogitBuf.Get<T>();
LocalTensor<T> p = mainPBuf.Get<T>();
LocalTensor<T> qTile = repBuf.Get<T>();
LocalTensor<T> kTile = kTileBuf.Get<T>();
LocalTensor<T> dot = vecBuf.Get<T>();
LocalTensor<T> mx = smRowBuf.Get<T>();
LocalTensor<T> sum = ssRowBuf.Get<T>();
LocalTensor<T> one = smallBuf.Get<T>();
LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
LocalTensor<IN_T> qIn = inQBuf.Get<IN_T>();
LocalTensor<IN_T> kIn = inKBuf.Get<IN_T>();
Duplicate(p, static_cast<T>(0.0f), 16);
PipeBarrier<PIPE_V>();
const uint32_t headTile = WIDE_FUSE ? 32u :
((sizeof(IN_T) == 2u && nHeads >= 64u) ? 64u : 32u);
const DataCopyExtParams cp(1, static_cast<uint32_t>(headTile * dQ * sizeof(IN_T)), 0, 0, 0);
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
const bool batchKeyLoad = (sizeof(IN_T) == 2u) && headTile == kKeyBatchHeads &&
t->s2Size <= kKeyBatchRows;
for (uint32_t g0 = 0; g0 < nHeads; g0 += headTile) {
const uint64_t qOff = ((static_cast<uint64_t>(b) * t->s1Size + s1) * nHeads + g0) * dQ;
if (batchKeyLoad) {
LocalTensor<IN_T> kInB = kInBatchBuf.Get<IN_T>();
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qIn, queryGm[qOff], cp, pad);
for (uint32_t j = 0; j < t->s2Size; ++j) {
const uint64_t kbOff = ((static_cast<uint64_t>(b) * t->s2Size + j) * nHeads + g0) * dQ;
DataCopyPad(kInB[j * headTile * dQ], keyGm[kbOff], cp, pad);
}
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qTile, qIn, headTile * dQ);
else Cast(qTile, qIn, RoundMode::CAST_NONE, headTile * dQ);
for (uint32_t j = 0; j < t->s2Size; ++j) {
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(kTile, kInB[j * headTile * dQ], headTile * dQ);
} else {
Cast(kTile, kInB[j * headTile * dQ], RoundMode::CAST_NONE, headTile * dQ);
}
Mul(kTile, kTile, qTile, headTile * dQ);
uint32_t shape[2] = {headTile, dQ};
ReduceSum<T, Pattern::Reduce::AR, true>(dot, kTile, work, shape, true);
Muls(dot, dot, t->scaleValue, headTile);
DataCopy(logits[j * nHeads + g0], dot, headTile);
}
PipeBarrier<PIPE_V>();
continue;
}
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qIn, queryGm[qOff], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qTile, qIn, headTile * dQ);
else Cast(qTile, qIn, RoundMode::CAST_NONE, headTile * dQ);
PipeBarrier<PIPE_V>();
for (uint32_t j = 0; j < t->s2Size; ++j) {
const uint64_t kOff = ((static_cast<uint64_t>(b) * t->s2Size + j) * nHeads + g0) * dQ;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kIn, keyGm[kOff], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(kTile, kIn, headTile * dQ);
else Cast(kTile, kIn, RoundMode::CAST_NONE, headTile * dQ);
PipeBarrier<PIPE_V>();
Mul(kTile, kTile, qTile, headTile * dQ);
PipeBarrier<PIPE_V>();
uint32_t shape[2] = {headTile, dQ};
ReduceSum<T, Pattern::Reduce::AR, true>(dot, kTile, work, shape, true);
PipeBarrier<PIPE_V>();
Muls(dot, dot, t->scaleValue, headTile);
PipeBarrier<PIPE_V>();
DataCopy(logits[j * nHeads + g0], dot, headTile);
PipeBarrier<PIPE_V>();
}
}
Duplicate(mx, static_cast<T>(-3.0e38f), nHeads);
Duplicate(sum, static_cast<T>(0.0f), nHeads);
PipeBarrier<PIPE_V>();
for (uint32_t j = 0; j < t->s2Size; ++j) {
Max(mx, mx, logits[j * nHeads], nHeads);
PipeBarrier<PIPE_V>();
}
for (uint32_t j = 0; j < t->s2Size; ++j) {
LocalTensor<T> row = logits[j * nHeads];
Sub(row, row, mx, nHeads);
PipeBarrier<PIPE_V>();
Exp(row, row, nHeads);
PipeBarrier<PIPE_V>();
Add(sum, sum, row, nHeads);
PipeBarrier<PIPE_V>();
}
uint32_t rowShape[2] = {1u, nHeads};
for (uint32_t j = 0; j < t->s2Size; ++j) {
LocalTensor<T> row = logits[j * nHeads];
Div(row, row, sum, nHeads);
PipeBarrier<PIPE_V>();
ReduceSum<T, Pattern::Reduce::AR, true>(one, row, work, rowShape, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
p.SetValue(j, one.GetValue(0) * t->invN1);
SetFlag<HardEvent::S_V>(EV_SV);
WaitFlag<HardEvent::S_V>(EV_SV);
}
}
__aicore__ inline void ComputeMainSoftmaxStats(uint32_t b, uint32_t s1) {
if (batchMain_) {
ComputeMainBatched<static_cast<int32_t>(kBatchWidth)>(b, s1);
return;
}
LocalTensor<T> mxRow = smRowBuf.Get<T>();
LocalTensor<T> sumRow = ssRowBuf.Get<T>();
LocalTensor<T> rep = repBuf.Get<T>();
LocalTensor<T> dot = vecBuf.Get<T>()[2 * JT];
LocalTensor<T> one = smallBuf.Get<T>();
LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
const int64_t ntile = (static_cast<int64_t>(t->s2Size) + JT - 1) / JT;
for (uint32_t g = 0; g < t->n1Size; ++g) {
float rowMax = -3.0e38f;
float rowSum = 0.0f;
for (int64_t jt = 0; jt < ntile; ++jt) {
const int32_t nj = static_cast<int32_t>((t->s2Size - jt * JT < JT) ?
(t->s2Size - jt * JT) : JT);
LoadMainKeyTile(b, jt, nj, g);
const uint64_t qOff = ((static_cast<uint64_t>(b) * t->s1Size + s1) *
t->n1Size + g) * t->dQuery;
RepFromGm(rep, queryGm, qOff, nj, t->dQuery, inQBuf);
Mul(rep, rep, kTileBuf.Get<T>(), nj * t->dQuery);
PipeBarrier<PIPE_V>();
uint32_t shape[2] = {static_cast<uint32_t>(nj), t->dQuery};
ReduceSum<T, Pattern::Reduce::AR, true>(dot, rep, work, shape, true);
PipeBarrier<PIPE_V>();
Muls(dot, dot, t->scaleValue, nj);
PipeBarrier<PIPE_V>();
if (cacheMain_ && ntile > 1) {
DataCopy(mainProbBuf.Get<T>()[g * MAIN_CACHE_COLS + jt * JT], dot, JT);
PipeBarrier<PIPE_V>();
}
uint32_t rowShape[2] = {1, static_cast<uint32_t>(nj)};
ReduceMax<T, Pattern::Reduce::AR, true>(one, dot, work, rowShape, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
const float v = one.GetValue(0); if (v > rowMax) rowMax = v;
if (ntile == 1) {
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
Adds(dot, dot, -rowMax, nj); PipeBarrier<PIPE_V>();
Exp(dot, dot, nj); PipeBarrier<PIPE_V>();
ReduceSum<T, Pattern::Reduce::AR, true>(one, dot, work, rowShape, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
rowSum = one.GetValue(0);
const float invN = (rowSum != 0.0f) ? (1.0f / rowSum) * t->invN1 : 0.0f;
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
Muls(dot, dot, invN, nj);
PipeBarrier<PIPE_V>();
LocalTensor<T> dst = mainProbBuf.Get<T>()[g * MAIN_CACHE_COLS];
Duplicate(dst, static_cast<T>(0.0f), JT);
PipeBarrier<PIPE_V>();
Add(dst, dst, dot, nj);
PipeBarrier<PIPE_V>();
}
}
for (int64_t jt = 0; ntile > 1 && jt < ntile; ++jt) {
const int32_t nj = static_cast<int32_t>((t->s2Size - jt * JT < JT) ?
(t->s2Size - jt * JT) : JT);
if (cacheMain_) {
DataCopy(dot, mainProbBuf.Get<T>()[g * MAIN_CACHE_COLS + jt * JT], JT);
PipeBarrier<PIPE_V>();
} else {
LoadMainKeyTile(b, jt, nj, g);
const uint64_t qOff = ((static_cast<uint64_t>(b) * t->s1Size + s1) *
t->n1Size + g) * t->dQuery;
RepFromGm(rep, queryGm, qOff, nj, t->dQuery, inQBuf);
Mul(rep, rep, kTileBuf.Get<T>(), nj * t->dQuery);
PipeBarrier<PIPE_V>();
uint32_t shape[2] = {static_cast<uint32_t>(nj), t->dQuery};
ReduceSum<T, Pattern::Reduce::AR, true>(dot, rep, work, shape, true);
PipeBarrier<PIPE_V>();
Muls(dot, dot, t->scaleValue, nj); PipeBarrier<PIPE_V>();
}
Adds(dot, dot, -rowMax, nj); PipeBarrier<PIPE_V>();
Exp(dot, dot, nj); PipeBarrier<PIPE_V>();
if (cacheMain_) {
LocalTensor<T> dst = mainProbBuf.Get<T>()[g * MAIN_CACHE_COLS + jt * JT];
Duplicate(dst, static_cast<T>(0.0f), JT);
PipeBarrier<PIPE_V>();
Add(dst, dst, dot, nj);
PipeBarrier<PIPE_V>();
}
uint32_t rowShape[2] = {1, static_cast<uint32_t>(nj)};
ReduceSum<T, Pattern::Reduce::AR, true>(one, dot, work, rowShape, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
rowSum += one.GetValue(0);
}
if (cacheMain_ && ntile > 1) {
const float invN = (rowSum != 0.0f) ? (1.0f / rowSum) * t->invN1 : 0.0f;
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
for (int64_t jt = 0; jt < ntile; ++jt) {
const int32_t nj = static_cast<int32_t>((t->s2Size - jt * JT < JT) ?
(t->s2Size - jt * JT) : JT);
LocalTensor<T> dst = mainProbBuf.Get<T>()[g * MAIN_CACHE_COLS + jt * JT];
Muls(dst, dst, invN, nj);
}
PipeBarrier<PIPE_V>();
}
mxRow.SetValue(g, rowMax);
sumRow.SetValue(g, rowSum);
}
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
}
__aicore__ inline void GatherTile(uint32_t b, uint32_t s1, int64_t jt, int32_t nj, bool needK,
bool needKi) {
LocalTensor<int32_t> idx = idxBuf.Get<int32_t>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
for (int32_t j = 0; j < nj; ++j) idx.SetValue(j, static_cast<int32_t>(jt * JT + j));
SetFlag<HardEvent::S_V>(EV_SV);
WaitFlag<HardEvent::S_V>(EV_SV);
LocalTensor<IN_T> inK = inKBuf.Get<IN_T>();
LocalTensor<IN_T> inKi = inKiBuf.Get<IN_T>();
LocalTensor<IN_T> inKr = inKrBuf.Get<IN_T>();
DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyExtParams cpq(1, static_cast<uint32_t>(t->dQuery * sizeof(IN_T)), 0, 0, 0);
DataCopyExtParams cpi2(1, static_cast<uint32_t>(t->dQueryIndex * sizeof(IN_T)), 0, 0, 0);
DataCopyExtParams cpr(1, static_cast<uint32_t>(t->dRope * sizeof(IN_T)), 0, 0, 0);
for (int32_t j = 0; j < nj; ++j) {
int64_t r = static_cast<int64_t>(idx.GetValue(j));
if (r < 0 || r >= static_cast<int64_t>(t->s2Size)) { r = 0; }
uint64_t rowBase = static_cast<uint64_t>(b) * t->s2Size + static_cast<uint64_t>(r);
if (needK) {
DataCopyPad(inK[j * t->dQuery], keyGm[rowBase * t->n1Size * t->dQuery], cpq, pad);
if (t->hasRope != 0u) {
RepOneChunk(inKr, j * static_cast<int32_t>(t->dRope), keyRopeGm,
rowBase * static_cast<uint64_t>(t->dRope), t->dRope, t->dRope);
}
}
if (needKi) {
DataCopyPad(inKi[j * t->dQueryIndex], keyIndexGm[rowBase * t->dQueryIndex], cpi2, pad);
}
}
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
LocalTensor<T> kTile = kTileBuf.Get<T>();
LocalTensor<T> krTile = krTileBuf.Get<T>();
LocalTensor<T> kiTile = kiTileBuf.Get<T>();
if (t->hasRope != 0u) {
Duplicate(krTile, static_cast<T>(0.0), nj * static_cast<int32_t>(t->dRope));
PipeBarrier<PIPE_V>();
}
for (int32_t j = 0; j < nj; ++j) {
if (needK) {
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(kTile[j * t->dQuery], inK[j * t->dQuery], t->dQuery);
PipeBarrier<PIPE_ALL>();
} else {
Cast(kTile[j * t->dQuery], inK[j * t->dQuery], RoundMode::CAST_NONE, t->dQuery);
}
if (t->hasRope != 0u) {
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(krTile[j * t->dRope], inKr[j * t->dRope], t->dRope);
PipeBarrier<PIPE_ALL>();
} else {
Cast(krTile[j * t->dRope], inKr[j * t->dRope], RoundMode::CAST_NONE, t->dRope);
}
}
}
if (needKi) {
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(kiTile[j * t->dQueryIndex], inKi[j * t->dQueryIndex], t->dQueryIndex);
PipeBarrier<PIPE_ALL>();
} else {
Cast(kiTile[j * t->dQueryIndex], inKi[j * t->dQueryIndex], RoundMode::CAST_NONE,
t->dQueryIndex);
}
}
}
PipeBarrier<PIPE_V>();
}
__aicore__ inline void ScatterAddTile(uint32_t b, uint32_t s1, int64_t jt, int32_t nj,
const LocalTensor<T> &dKTile) {
LocalTensor<int32_t> idx = idxBuf.Get<int32_t>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
SetAtomicAdd<float>();
for (int32_t j = 0; j < nj; ++j) {
int64_t r = static_cast<int64_t>(idx.GetValue(j));
if (r < 0 || r >= static_cast<int64_t>(t->s2Size)) { continue; }
uint64_t off = (static_cast<uint64_t>(b) * t->s2Size + static_cast<uint64_t>(r)) *
t->dQueryIndex;
DataCopyExtParams cp(1, static_cast<uint32_t>(t->dQueryIndex * sizeof(T)), 0, 0, 0);
DataCopyPad(scatterGm[off], dKTile[j * t->dQueryIndex], cp);
}
SetAtomicNone();
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
__aicore__ inline void WriteDWeights(uint32_t b, uint32_t s1, const LocalTensor<T> &dWacc) {
LocalTensor<T> tmp = outWBuf.Get<T>();
if (t->weightsFp32 != 0u) {
LocalTensor<float> w32 = tmp;
DataCopy(w32, dWacc, t->nidx1Size);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
uint64_t off = (static_cast<uint64_t>(b) * t->s1Size + s1) * t->nidx1Size;
DataCopyExtParams cp(1, static_cast<uint32_t>(t->nidx1Size * sizeof(float)), 0, 0, 0);
DataCopyPad(dWeightsF32Gm[off], w32, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
} else {
LocalTensor<OUT_T> w16 = tmp.ReinterpretCast<OUT_T>();
Cast(w16, dWacc, RoundMode::CAST_RINT, t->nidx1Size);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
uint64_t off = (static_cast<uint64_t>(b) * t->s1Size + s1) * t->nidx1Size;
DataCopyExtParams cp(1, static_cast<uint32_t>(t->nidx1Size * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dWeights16Gm[off], w16, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
private:
__aicore__ inline void WriteDKeyIndex() {
SetFlag<HardEvent::MTE3_MTE2>(EV_MTE3MTE2);
WaitFlag<HardEvent::MTE3_MTE2>(EV_MTE3MTE2);
const uint64_t rows = static_cast<uint64_t>(t->bSize) * t->s2Size;
LocalTensor<T> src = chunkBuf.Get<T>();
for (uint64_t row = blockIdx_; row < rows; row += blockNum_) {
uint64_t off = row * t->dQueryIndex;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
LoadChunk(src, scatterGm[off], static_cast<int32_t>(t->dQueryIndex));
LocalTensor<OUT_T> outC = outKBuf.Get<OUT_T>();
if constexpr (IsSameType<OUT_T, T>::value) {
DataCopy(outC, src, t->dQueryIndex);
PipeBarrier<PIPE_ALL>();
} else {
Cast(outC, src, RoundMode::CAST_RINT, t->dQueryIndex);
}
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3);
WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, static_cast<uint32_t>(t->dQueryIndex * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dKeyIndexGm[off], outC, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V);
WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
__aicore__ inline void LoadSoftmaxRows(uint32_t b, uint32_t s1) {
LocalTensor<T> mx = smRowBuf.Get<T>();
LocalTensor<T> ss = ssRowBuf.Get<T>();
if (t->n1Size == 0u) { return; }
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
const uint64_t off = (static_cast<uint64_t>(b) * t->s1Size + s1) * t->n1Size;
DataCopyExtParams cp(1, static_cast<uint32_t>(t->n1Size * sizeof(T)), 0, 0, 0);
DataCopyPadExtParams<T> pad(false, 0, 0, 0);
DataCopyPad(mx, softmaxMaxGm[off], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
DataCopyPad(ss, softmaxSumGm[off], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
}
__aicore__ inline void LoadWeightsRow(uint32_t b, uint32_t s1) {
LocalTensor<T> wr = wRowBuf.Get<T>();
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
uint64_t off = (static_cast<uint64_t>(b) * t->s1Size + s1) * t->nidx1Size;
DataCopyExtParams cp(1, static_cast<uint32_t>(t->nidx1Size * sizeof(IN_T)), 0, 0, 0);
DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
if (t->weightsFp32 != 0u) {
DataCopyExtParams cp32(1, static_cast<uint32_t>(t->nidx1Size * sizeof(T)), 0, 0, 0);
DataCopyPadExtParams<T> pad32(false, 0, 0, 0);
DataCopyPad(wr, weightsF32Gm[off], cp32, pad32);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
} else {
LocalTensor<IN_T> wi = wInBuf.Get<IN_T>();
DataCopyPad(wi, weights16Gm[off], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(wr, wi, t->nidx1Size);
PipeBarrier<PIPE_ALL>();
} else {
Cast(wr, wi, RoundMode::CAST_NONE, t->nidx1Size);
}
PipeBarrier<PIPE_V>();
}
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
}
__aicore__ inline float SumVecSplit(const LocalTensor<T> &v, int32_t n = JT) {
LocalTensor<uint8_t> work = HeadSlice<uint8_t>(hWork_);
LocalTensor<T> one = smallBuf.Get<T>();
uint32_t sh[2] = {1, static_cast<uint32_t>(n)};
ReduceSum<T, Pattern::Reduce::AR, !WIDE_FUSE>(one, v, work, sh, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
return one.GetValue(0);
}
__aicore__ inline void LoadSplitWeightsRow(uint32_t b, uint32_t s1) {
LocalTensor<T> wr = HeadSlice<T>(hWRow_);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
uint64_t off = (static_cast<uint64_t>(b) * t->s1Size + s1) * t->nidx1Size;
DataCopyExtParams cp(1, static_cast<uint32_t>(t->nidx1Size * sizeof(IN_T)), 0, 0, 0);
DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
if (t->weightsFp32 != 0u) {
DataCopyExtParams cp32(1, static_cast<uint32_t>(t->nidx1Size * sizeof(T)), 0, 0, 0);
DataCopyPadExtParams<T> pad32(false, 0, 0, 0);
DataCopyPad(wr, weightsF32Gm[off], cp32, pad32);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
} else {
LocalTensor<IN_T> wi = HeadSlice<IN_T>(hWIn_);
DataCopyPad(wi, weights16Gm[off], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) {
DataCopy(wr, wi, t->nidx1Size);
PipeBarrier<PIPE_ALL>();
} else {
Cast(wr, wi, RoundMode::CAST_NONE, t->nidx1Size);
}
PipeBarrier<PIPE_V>();
}
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
}
__aicore__ inline float weightsAt(uint32_t g) { return wRowBuf.Get<T>().GetValue(g); }
static constexpr int32_t EV_MTE2V = 0;
static constexpr int32_t EV_VMTE3 = 1;
static constexpr int32_t EV_VS = 2;
static constexpr int32_t EV_SMTE3 = 3;
static constexpr int32_t EV_MTE3MTE2 = 4;
static constexpr int32_t EV_VMTE2 = 5;
static constexpr int32_t EV_MTE3V = 6;
static constexpr int32_t EV_SV = 7;
const DenseLightningIndexerGradKlLossTilingData *t = nullptr;
TPipe pipe;
TBuf<TPosition::VECCALC> idxBuf, kTileBuf, krTileBuf, kiTileBuf, qgBuf, qrgBuf, qidxgBuf;
TBuf<TPosition::VECCALC> repBuf, repBuf2, vecBuf, s2dBuf, dIrepBuf, dWtBuf, dWaccBuf;
TBuf<TPosition::VECCALC> dKTileBuf, accDqBuf, inTmpBuf, chunkBuf, workBuf, smallBuf;
TBuf<TPosition::VECCALC> inKBuf, inKiBuf, inKrBuf, inQBuf, inQrBuf, kInBatchBuf;
TBuf<TPosition::VECCALC> leanBuf_;
bool lean_ = false;
uint32_t leanOff_ = 0;
TBuf<TPosition::VECCALC> headBuf_;
uint32_t headOff_ = 0;
uint32_t hLogits_ = 0, hP_ = 0, hRep_ = 0, hKTile_ = 0, hDot_ = 0, hVec_ = 0;
uint32_t hMx_ = 0, hSum_ = 0, hQi_ = 0, hKi_ = 0, hDq_ = 0, hDk_ = 0;
uint32_t hWRow_ = 0, hWIn_ = 0, hOutW_ = 0, hOutQ_ = 0, hAcc_ = 0, hPart_ = 0;
uint32_t hSim_ = 0, hWork_ = 0, hQIn_ = 0, hKIn_ = 0, hKiIn_ = 0;
uint32_t lQIn_ = 0, lKIn_ = 0, lQiIn_ = 0, lKiIn_ = 0, lWIn_ = 0, lQF_ = 0, lKRowF_ = 0;
uint32_t lProdF_ = 0, lLogF_ = 0, lPVec_ = 0, lVec_ = 0, lQiF_ = 0, lKiF_ = 0, lSimF_ = 0;
uint32_t lDSF_ = 0, lDqF_ = 0, lDkF_ = 0, lWork_ = 0, lSmall_ = 0, lMask_ = 0, lOut_ = 0;
uint32_t lDS1F_ = 0, lCoefL1F_ = 0, lWTileF_ = 0, lOnesF_ = 0, lTmpJTF_ = 0, lTmp2F_ = 0;
uint32_t lAccF_ = 0, lProdAll_ = 0, lKeepF_ = 0;
float wS_[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
TBuf<TPosition::VECCALC> wRowBuf, wInBuf, smRowBuf, ssRowBuf;
TBuf<TPosition::VECCALC> mainProbBuf, mainLogitBuf, mainPBuf;
TBuf<TPosition::VECCALC> sCacheBuf;
TBuf<TPosition::VECCALC> dWresBuf, dS2dBuf, outQBuf, outWBuf, outKBuf;
TBuf<TPosition::VECCALC> qiAllBuf;
TBuf<TPosition::VECCALC> syncBuf;
TBuf<TPosition::VECCALC> pResBuf, iResBuf;
TBuf<TPosition::VECCALC> seqBuf;
GlobalTensor<IN_T> queryGm, keyGm, queryIndexGm, keyIndexGm, queryRopeGm, keyRopeGm;
GlobalTensor<int32_t> seqQ32Gm, seqK32Gm;
GlobalTensor<int64_t> seqQ64Gm, seqK64Gm;
GlobalTensor<int32_t> sparseIdxGm;
GlobalTensor<T> softmaxMaxGm, softmaxSumGm;
GlobalTensor<OUT_T> dQueryIndexGm, dKeyIndexGm;
GlobalTensor<T> pGm, iGm, scatterGm;
GlobalTensor<int32_t> syncGm;
GlobalTensor<T> lossGmRef;
GlobalTensor<IN_T> dWeights16Gm;
GlobalTensor<float> dWeightsF32Gm;
GlobalTensor<IN_T> weights16Gm;
GlobalTensor<float> weightsF32Gm;
float lossSum_ = 0.0f;
bool useResident_ = false;
bool cacheMain_ = false;
bool batchMain_ = false;
bool headSplit_ = false;
uint32_t headParts_ = 1u, splitTask_ = 0u;
uint32_t mainH0_ = 0, mainH1_ = 0, idxH0_ = 0, idxH1_ = 0;
GlobalTensor<T> pPartGm_, iPartGm_, dkPartGm_;
bool batchIndex_ = false;
bool cacheS_ = false;
int64_t resElems_ = 0;
int64_t curVisible_ = 0;
uint32_t blockIdx_ = 0;
uint32_t blockNum_ = 1;
bool directKey_ = false;
bool seqCacheValid_ = false;
uint32_t cachedSeqB_ = 0;
uint32_t cachedActS1_ = 0;
uint32_t cachedActS2_ = 0;
};
template <typename DTQ, bool FUSE_BIG>
class KernelHeadPublic {
public:
using T = float;
using IN_T = DTQ;
using OUT_T = DTQ;
static constexpr bool kFp32 = IsSameType<IN_T, T>::value;
static constexpr int32_t S1 = kFp32 ? 1 : (FUSE_BIG ? 1 : 2);
static constexpr int32_t NJ = kFp32 ? 4 : 8;
static constexpr int32_t N1 = kFp32 ? 64 : (FUSE_BIG ? 128 : 64);
static constexpr int32_t NG = kFp32 ? 32 : (FUSE_BIG ? 64 : 16);
static constexpr int32_t HP = kFp32 ? 4 : (FUSE_BIG ? 8 : 2);
static constexpr int32_t NL = N1 / HP;
static constexpr int32_t NGL = NG / HP;
static constexpr int32_t D = 128;
static constexpr uint32_t kB = 1u;
static constexpr uint32_t kTotalTasks = static_cast<uint32_t>(S1);
static constexpr uint32_t kBlockNum = kTotalTasks * static_cast<uint32_t>(HP);
static constexpr int32_t KCH = 16;
static constexpr int32_t MTILE = 32;
static constexpr int32_t JT = 8;
static constexpr uint32_t kKeyBatch = 4u;
static constexpr uint32_t kInQElems = static_cast<uint32_t>(MTILE) * D;
static constexpr uint32_t kQidElems = static_cast<uint32_t>(KCH) * D;
static constexpr uint32_t kKiElems = static_cast<uint32_t>(NJ) * D;
static constexpr uint32_t kLogitsElems = static_cast<uint32_t>(NJ) * static_cast<uint32_t>(NL);
static constexpr uint32_t kRepElems = 4096u;
static constexpr uint32_t kKeyRows = kKeyBatch * static_cast<uint32_t>(NL);
static constexpr uint32_t kKTileElems = kKeyRows * D;
static constexpr uint32_t kKInElems = kKeyRows * D;
static constexpr uint32_t kDotElems = kKeyBatch * static_cast<uint32_t>(MTILE);
static constexpr uint32_t kVecElems = 10u * static_cast<uint32_t>(JT);
static constexpr uint32_t kSimElems = 512u;
static constexpr uint32_t kWorkBytes = 16u * 1024u;
static constexpr uint32_t kOutQElems = 16u * D;
static constexpr uint32_t kDkParts = static_cast<uint32_t>(S1) * static_cast<uint32_t>(HP);
static_assert(kBlockNum == 4u || kBlockNum == 8u, "公开 head 小类只有 4/8 核两种");
static_assert(kDkParts == 4u || kDkParts == 8u, "公开 head 小类 dk 部分和只有 4/8 份");
static_assert(kB == 1u, "公开 head 小类 bSize 恒为 1");
static_assert(S1 == 1 || S1 == 2, "公开 head 小类只有 S1=1/2 两种任务数");
static_assert(NL <= MTILE, "主头片单次行数不得超过 MTILE");
static_assert(static_cast<uint32_t>(NJ) % kKeyBatch == 0u, "key 成组宽度须整除 NJ");
static_assert(kOutQElems >= kKiElems, "outQ 必须兼顾 NJ*D 的 dk 写出");
static_assert(kQidElems >= static_cast<uint32_t>(KCH) * D, "qi/dq 至少 16*D");
static_assert(kRepElems >= 2048u + 3u * 256u + 128u + 1024u, "rep 内部布局容量");
static_assert(kDotElems >= kKeyRows, "dot 必须容纳一批 key 的 gt 行归约输出");
static_assert(kKTileElems >= kKiElems, "kTile 兼顾主分 tile 与收尾 dk 汇总");
static_assert(static_cast<uint32_t>(NGL) <= static_cast<uint32_t>(KCH),
"公开家族本核 indexer 头片须为唯一 gt 块");
static_assert(kKTileElems >=
2u * static_cast<uint32_t>(NJ) * static_cast<uint32_t>(NGL) * 64u,
"kTile 必须容纳 dk 批式 prodAll[2][NJ][NGL][64]");
static_assert(kKTileElems >= static_cast<uint32_t>(NJ) * static_cast<uint32_t>(NGL) *
static_cast<uint32_t>(D),
"kTile 必须容纳 indexer 批式 prodAll[NJ][NGL][D]");
__aicore__ inline KernelHeadPublic() {}
__aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
GM_ADDR weights, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex,
GM_ADDR dWeights, GM_ADDR loss, GM_ADDR workspace,
const DenseLightningIndexerGradKlLossTilingData *td) {
t = td;
blockIdx_ = GetBlockIdx();
blockNum_ = GetBlockNum();
queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
queryIndexGm.SetGlobalBuffer((__gm__ IN_T *)queryIndex);
keyIndexGm.SetGlobalBuffer((__gm__ IN_T *)keyIndex);
weightsGm.SetGlobalBuffer((__gm__ IN_T *)weights);
dWeightsGm.SetGlobalBuffer((__gm__ IN_T *)dWeights);
dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
lossGmRef.SetGlobalBuffer((__gm__ T *)loss);
headParts_ = blockNum_ / kTotalTasks;
splitTask_ = blockIdx_ / headParts_;
const uint32_t part = blockIdx_ % headParts_;
mainH0_ = part * static_cast<uint32_t>(NL);
mainH1_ = mainH0_ + static_cast<uint32_t>(NL);
idxH0_ = part * static_cast<uint32_t>(NGL);
idxH1_ = idxH0_ + static_cast<uint32_t>(NGL);
__gm__ uint8_t *ws = (__gm__ uint8_t *)workspace;
pPartGm_.SetGlobalBuffer((__gm__ T *)(ws + t->reserved[1]), blockNum_ * 16u);
iPartGm_.SetGlobalBuffer((__gm__ T *)(ws + t->reserved[2]), blockNum_ * 16u);
dkPartGm_.SetGlobalBuffer((__gm__ T *)(ws + t->reserved[3]),
static_cast<uint64_t>(blockNum_) * NJ * D);
const uint64_t scatterBytes = static_cast<uint64_t>(kB) * NJ * D * sizeof(T);
const uint64_t syncOffset = t->wsOffsetScatter + ((scatterBytes + 511u) / 512u) * 512u;
syncGm.SetGlobalBuffer((__gm__ int32_t *)(ws + syncOffset));
headOff_ = 0;
hLogits_ = HeadTake(kLogitsElems * sizeof(T));
hP_ = HeadTake(16u * sizeof(T));
hRep_ = HeadTake(kRepElems * sizeof(T));
hKTile_ = HeadTake(kKTileElems * sizeof(T));
hDot_ = HeadTake(kDotElems * sizeof(T));
hVec_ = HeadTake(kVecElems * sizeof(T));
hMx_ = HeadTake(static_cast<uint32_t>(NL) * sizeof(T));
hSum_ = HeadTake(static_cast<uint32_t>(NL) * sizeof(T));
hQi_ = HeadTake(kQidElems * sizeof(T));
hKi_ = HeadTake(kKiElems * sizeof(T));
hDq_ = HeadTake(kQidElems * sizeof(T));
hDk_ = HeadTake(kKiElems * sizeof(T));
hWRow_ = HeadTake(static_cast<uint32_t>(NG) * sizeof(T));
hWIn_ = HeadTake(static_cast<uint32_t>(NG) * sizeof(IN_T));
hOutW_ = HeadTake(static_cast<uint32_t>(NG) * sizeof(T));
hOutQ_ = HeadTake(kOutQElems * sizeof(OUT_T));
hAcc_ = HeadTake(kKiElems * sizeof(T));
hPart_ = HeadTake(kKiElems * sizeof(T));
hSim_ = HeadTake(kSimElems * sizeof(T));
hWork_ = HeadTake(kWorkBytes);
hQIn_ = HeadTake(kInQElems * sizeof(IN_T));
hKIn_ = HeadTake(kKInElems * sizeof(IN_T));
hKiIn_ = HeadTake(kKiElems * sizeof(IN_T));
pipe.InitBuffer(headBuf_, Align32(headOff_));
pipe.InitBuffer(smallBuf, 1024u);
pipe.InitBuffer(syncBuf, blockNum_ * 32u);
}
__aicore__ inline void Process() {
lossSum_ = 0.0f;
if constexpr (S1 > 1) {
if (blockIdx_ == 0u) {
lossGmRef.SetValue(0, 0.0F);
DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lossGmRef);
}
if (blockNum_ > 1u) { SyncAll<true>(); }
}
DoTaskSplitHead(splitTask_);
if (blockNum_ > 1u) SyncAll<true>();
if (blockIdx_ == 0u) {
WriteDkFromParts();
}
if ((blockIdx_ % headParts_) == 0u) {
LocalTensor<T> ls = smallBuf.Get<T>();
ls.SetValue(0, static_cast<T>(lossSum_));
SetFlag<HardEvent::S_MTE3>(EV_SMTE3);
WaitFlag<HardEvent::S_MTE3>(EV_SMTE3);
if constexpr (S1 > 1) { if (blockNum_ > 1u) { SetAtomicAdd<float>(); } }
DataCopyExtParams cpl(1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
DataCopyPad(lossGmRef, ls, cpl);
if constexpr (S1 > 1) { if (blockNum_ > 1u) { SetAtomicNone(); } }
SetFlag<HardEvent::MTE3_S>(1);
WaitFlag<HardEvent::MTE3_S>(1);
}
}
private:
__aicore__ inline void InitSoftSync() {
auto local = syncBuf.Get<int32_t>();
const uint32_t count = blockNum_ * 8u;
Duplicate(local, 0, count);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, count * sizeof(int32_t), 0, 0, 0);
DataCopyPad(syncGm, local, cp);
SetFlag<HardEvent::MTE3_S>(1); WaitFlag<HardEvent::MTE3_S>(1);
}
static __aicore__ inline uint32_t Align32(uint32_t x) { return (x + 31u) / 32u * 32u; }
__aicore__ inline uint32_t HeadTake(uint64_t bytes) {
const uint32_t at = headOff_;
headOff_ += static_cast<uint32_t>((bytes + 511u) / 512u * 512u);
return at;
}
template <typename ET>
__aicore__ inline LocalTensor<ET> HeadSlice(uint32_t off) {
LocalTensor<uint8_t> buf = headBuf_.Get<uint8_t>();
return buf[off].ReinterpretCast<ET>();
}
__aicore__ inline void ReduceSumArHeadD128(const LocalTensor<T>& dst, const LocalTensor<T>& src,
const LocalTensor<T>& scratch, uint32_t repeat) {
static_assert(D == 128, "E58 fixed AR helper assumes D=128");
static_assert(8u * (kKeyBatch * static_cast<uint32_t>(NL)) <= kWorkBytes / sizeof(T),
"E58 scratch (hWork_ 16 KiB) must hold the largest R rows of 8 floats");
constexpr uint8_t kBlk = 1u;
constexpr uint8_t kRowBlk = static_cast<uint8_t>(D / B32_DATA_NUM_PER_BLOCK);
constexpr int32_t kOutBlk = 1;
constexpr int32_t kMidBlk = B32_DATA_NUM_PER_REPEAT / B32_DATA_NUM_PER_BLOCK;
const uint8_t rep = static_cast<uint8_t>(repeat);
SetMaskNorm();
SetVectorMask<T, MaskMode::NORMAL>(B32_DATA_NUM_PER_REPEAT);
Add<T, false>(src, src, src[B32_DATA_NUM_PER_REPEAT], MASK_PLACEHOLDER, rep,
BinaryRepeatParams(kBlk, kBlk, kBlk, kRowBlk, kRowBlk, kRowBlk));
PipeBarrier<PIPE_V>();
ResetMask();
SetMaskCount();
SetVectorMask<T, MaskMode::COUNTER>(static_cast<int32_t>(repeat) * B32_DATA_NUM_PER_REPEAT);
BlockReduceSum<T, false>(scratch, src, 1, MASK_PLACEHOLDER, kOutBlk, kBlk,
static_cast<int32_t>(kRowBlk));
PipeBarrier<PIPE_V>();
SetVectorMask<T, MaskMode::COUNTER>(static_cast<int32_t>(repeat) * B32_DATA_NUM_PER_BLOCK);
BlockReduceSum<T, false>(dst, scratch, 1, MASK_PLACEHOLDER, kOutBlk, kBlk, kMidBlk);
PipeBarrier<PIPE_V>();
SetMaskNorm();
ResetMask();
}
__aicore__ inline void ComputeMainBatchedRange(uint32_t b, uint32_t s1, uint32_t hBegin,
uint32_t hEnd) {
constexpr uint32_t dQ = 128u;
const uint32_t nL = hEnd - hBegin;
if (nL == 0u) { return; }
LocalTensor<T> logits = HeadSlice<T>(hLogits_);
LocalTensor<T> p = HeadSlice<T>(hP_);
LocalTensor<T> qTile = HeadSlice<T>(hRep_);
LocalTensor<T> kTile = HeadSlice<T>(hKTile_);
LocalTensor<T> dot = HeadSlice<T>(hDot_);
LocalTensor<T> mx = HeadSlice<T>(hMx_);
LocalTensor<T> sum = HeadSlice<T>(hSum_);
LocalTensor<uint8_t> work = HeadSlice<uint8_t>(hWork_);
LocalTensor<IN_T> qIn = HeadSlice<IN_T>(hQIn_);
LocalTensor<IN_T> kIn = HeadSlice<IN_T>(hKIn_);
Duplicate(p, static_cast<T>(0.0f), 16);
PipeBarrier<PIPE_V>();
const uint32_t headTile = (nL < 32u) ? nL : 32u;
const DataCopyExtParams cp(1, static_cast<uint32_t>(headTile * dQ * sizeof(IN_T)), 0, 0, 0);
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
for (uint32_t g0 = hBegin; g0 < hEnd; g0 += headTile) {
const uint32_t gt = ((hEnd - g0) < headTile) ? (hEnd - g0) : headTile;
DataCopyExtParams cpg(1, static_cast<uint32_t>(gt * dQ * sizeof(IN_T)), 0, 0, 0);
const uint64_t qOff = ((static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1) * N1 + g0) * dQ;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
if constexpr (kFp32) {
DataCopyPad(qTile, queryGm[qOff], cpg, pad);
} else {
DataCopyPad(qIn, queryGm[qOff], cpg, pad);
}
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (!kFp32) {
Cast(qTile, qIn, RoundMode::CAST_NONE, gt * dQ);
}
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && !FUSE_BIG) {
static_assert(static_cast<uint32_t>(NJ) == 2u * kKeyBatch,
"公开 A 族 main key 组恰为两组");
const uint64_t kOff0 = (static_cast<uint64_t>(b) * static_cast<uint32_t>(NJ) * N1
+ g0) * dQ;
DataCopyExtParams cpk0(static_cast<uint16_t>(kKeyBatch),
static_cast<uint32_t>(gt * dQ * sizeof(IN_T)),
static_cast<uint32_t>((static_cast<uint32_t>(N1) - gt) * dQ * sizeof(IN_T)),
0u, 0u);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kIn, keyGm[kOff0], cpk0, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
}
for (uint32_t j0 = 0u; j0 < static_cast<uint32_t>(NJ); j0 += kKeyBatch) {
const uint32_t jc = ((static_cast<uint32_t>(NJ) - j0) < kKeyBatch)
? (static_cast<uint32_t>(NJ) - j0) : kKeyBatch;
const uint64_t kOff = ((static_cast<uint64_t>(b) * static_cast<uint32_t>(NJ) + j0)
* N1 + g0) * dQ;
DataCopyExtParams cpk(static_cast<uint16_t>(jc),
static_cast<uint32_t>(gt * dQ * sizeof(IN_T)),
static_cast<uint32_t>((static_cast<uint32_t>(N1) - gt) * dQ * sizeof(IN_T)),
0u, 0u);
if constexpr (!kFp32 && !FUSE_BIG) {
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
} else if constexpr (kFp32) {
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kTile, keyGm[kOff], cpk, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
} else {
if (j0 == 0u) {
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kIn, keyGm[kOff], cpk, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
} else {
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
static_assert(static_cast<uint32_t>(NJ) == 2u * kKeyBatch,
"E78 P7 main key 组恰为两组");
static_assert(NGL <= KCH, "E78 P7 qi 首段是唯一 gt 块");
LocalTensor<IN_T> kiPf = HeadSlice<IN_T>(hKiIn_);
LocalTensor<IN_T> qiPf = HeadSlice<IN_T>(hQIn_);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyExtParams cpK78(1, static_cast<uint32_t>(NJ * D * sizeof(IN_T)), 0, 0, 0);
DataCopyPad(kiPf,
keyIndexGm[static_cast<uint64_t>(b) * static_cast<uint32_t>(NJ) * D],
cpK78, pad);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyExtParams cpQ78(1, static_cast<uint32_t>(NGL * D * sizeof(IN_T)), 0, 0, 0);
DataCopyPad(qiPf,
queryIndexGm[(static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1)
* NG * D + static_cast<uint64_t>(idxH0_) * D],
cpQ78, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
}
}
if constexpr (!kFp32) {
Cast(kTile, kIn, RoundMode::CAST_NONE, jc * gt * dQ);
}
if constexpr (!kFp32 && !FUSE_BIG) {
if (j0 + kKeyBatch < static_cast<uint32_t>(NJ)) {
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kIn, keyGm[kOff + static_cast<uint64_t>(kKeyBatch) * N1 * dQ],
cpk, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
}
} else if constexpr (!kFp32 && FUSE_BIG) {
if (j0 + kKeyBatch < static_cast<uint32_t>(NJ)) {
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kIn, keyGm[kOff + static_cast<uint64_t>(kKeyBatch) * N1 * dQ],
cpk, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
}
}
if constexpr (kFp32 || !FUSE_BIG) {
for (uint32_t jj = 0u; jj < jc; ++jj) {
Mul(kTile[jj * gt * dQ], kTile[jj * gt * dQ], qTile, gt * dQ);
}
PipeBarrier<PIPE_V>();
} else {
for (uint32_t jj = 0u; jj < jc; ++jj) {
Mul(kTile[jj * gt * dQ], kTile[jj * gt * dQ], qTile, gt * dQ);
PipeBarrier<PIPE_V>();
}
}
LocalTensor<T> arScratch = HeadSlice<T>(hWork_);
if constexpr (kFp32) {
ReduceSumArHeadD128(logits, kTile, arScratch, jc * gt);
} else {
ReduceSumArHeadD128(dot, kTile, arScratch, jc * gt);
}
PipeBarrier<PIPE_V>();
if constexpr (kFp32) {
Muls(logits, logits, t->scaleValue, jc * gt);
} else {
Muls(dot, dot, t->scaleValue, jc * gt);
}
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && !FUSE_BIG) {
DataCopy(logits[j0 * nL], dot, jc * nL);
PipeBarrier<PIPE_V>();
} else if constexpr (!kFp32 && FUSE_BIG) {
for (uint32_t jj = 0u; jj < jc; ++jj) {
DataCopy(logits[(j0 + jj) * nL + (g0 - hBegin)], dot[jj * gt], gt);
PipeBarrier<PIPE_V>();
}
}
}
}
(void)cp;
Duplicate(mx, static_cast<T>(-3.0e38f), nL);
Duplicate(sum, static_cast<T>(0.0f), nL);
PipeBarrier<PIPE_V>();
LocalTensor<T> tree = HeadSlice<T>(hWork_);
const uint32_t half = static_cast<uint32_t>(NJ) / 2u;
Max(tree, logits, logits[half * nL], half * nL);
PipeBarrier<PIPE_V>();
for (uint32_t rows = half; rows > 1u; rows >>= 1u) {
const uint32_t hf = rows >> 1u;
Max(tree, tree, tree[hf * nL], hf * nL);
PipeBarrier<PIPE_V>();
}
Max(mx, mx, tree, nL);
PipeBarrier<PIPE_V>();
const uint64_t maskNL = static_cast<uint64_t>(nL);
const uint8_t repNJ = static_cast<uint8_t>(NJ);
const uint8_t rowBlk = static_cast<uint8_t>(nL / 8u);
Sub(logits, logits, mx, maskNL, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
PipeBarrier<PIPE_V>();
Exp(logits, logits, static_cast<int32_t>(NJ) * static_cast<int32_t>(nL));
PipeBarrier<PIPE_V>();
Add(tree, logits, logits[half * nL], half * nL);
PipeBarrier<PIPE_V>();
for (uint32_t rows = half; rows > 1u; rows >>= 1u) {
const uint32_t hf = rows >> 1u;
Add(tree, tree, tree[hf * nL], hf * nL);
PipeBarrier<PIPE_V>();
}
Add(sum, sum, tree, nL);
PipeBarrier<PIPE_V>();
Div(logits, logits, sum, maskNL, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
PipeBarrier<PIPE_V>();
uint32_t shP[2] = {static_cast<uint32_t>(NJ), static_cast<uint32_t>(nL)};
ReduceSum<T, Pattern::Reduce::AR, true>(p, logits, work, shP, true);
PipeBarrier<PIPE_V>();
}
__aicore__ inline void DoTaskSplitHead(uint32_t task) {
constexpr int32_t D_ = D;
constexpr int32_t KCH_ = KCH;
const uint32_t b = task / static_cast<uint32_t>(S1);
const uint32_t s1 = task % static_cast<uint32_t>(S1);
const int32_t nj = NJ;
const int32_t ngL = NGL;
const uint32_t K = blockNum_;
LocalTensor<T> scratch = HeadSlice<T>(hRep_);
LocalTensor<T> prod = scratch;
LocalTensor<T> coef = scratch[KCH_ * D_];
LocalTensor<T> wRep = coef[256];
LocalTensor<T> deltaRep = wRep[256];
LocalTensor<T> cbrTmp = deltaRep[256];
LocalTensor<T> cbr = cbrTmp[128];
LocalTensor<T> sim = HeadSlice<T>(hSim_);
LocalTensor<T> qi = HeadSlice<T>(hQi_);
LocalTensor<T> ki = HeadSlice<T>(hKi_);
LocalTensor<T> dq = HeadSlice<T>(hDq_);
LocalTensor<T> dk = HeadSlice<T>(hDk_);
LocalTensor<T> vec = HeadSlice<T>(hVec_);
LocalTensor<T> p = vec;
LocalTensor<T> ii = vec[JT];
LocalTensor<T> y = vec[2 * JT];
LocalTensor<T> tmp = vec[4 * JT];
LocalTensor<T> lp = vec[6 * JT];
LocalTensor<T> ly = vec[7 * JT];
LocalTensor<T> pmax = vec[8 * JT];
LocalTensor<T> one = smallBuf.Get<T>();
LocalTensor<uint8_t> work = HeadSlice<uint8_t>(hWork_);
LocalTensor<uint8_t> maskBits = smallBuf.Get<uint8_t>()[512];
LocalTensor<IN_T> qiIn = HeadSlice<IN_T>(hQIn_);
LocalTensor<IN_T> kiIn = HeadSlice<IN_T>(hKiIn_);
LocalTensor<T> wRow = HeadSlice<T>(hWRow_);
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
const uint64_t qiBase = (static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1) * NG * D_;
const uint64_t kiBase = static_cast<uint64_t>(b) * NJ * D_;
LoadSplitWeightsRow(b, s1);
ComputeMainBatchedRange(b, s1, mainH0_, mainH1_);
{
LocalTensor<T> pLocal = HeadSlice<T>(hP_);
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, static_cast<uint32_t>(nj * sizeof(T)), 0, 0, 0);
DataCopyPad(pPartGm_[blockIdx_ * 16u], pLocal, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
{
if constexpr (!kFp32 && FUSE_BIG) {
static_assert(static_cast<uint32_t>(NJ) == 2u * kKeyBatch,
"E78 P7 main key 预取覆盖全部 NJ 行");
(void)kiBase;
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
LocalTensor<IN_T> wiPf = HeadSlice<IN_T>(hWIn_);
LocalTensor<T> wrPf = HeadSlice<T>(hWRow_);
Cast(wrPf, wiPf, RoundMode::CAST_NONE, NG);
PipeBarrier<PIPE_V>();
} else {
DataCopyExtParams cpK(1, static_cast<uint32_t>(nj * D_ * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
if constexpr (kFp32) {
DataCopyPad(ki, keyIndexGm[kiBase], cpK, pad);
} else {
DataCopyPad(kiIn, keyIndexGm[kiBase], cpK, pad);
}
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
}
if constexpr (!kFp32) {
Cast(ki, kiIn, RoundMode::CAST_NONE, nj * D_);
}
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && FUSE_BIG) {
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
}
}
for (int32_t j = 0; j < nj; ++j) {
DataCopy(wRep[j * ngL], wRow[idxH0_], ngL);
}
PipeBarrier<PIPE_V>();
if constexpr (!kFp32) {
{
LocalTensor<T> prodAll = HeadSlice<T>(hKTile_);
for (int32_t g0 = 0; g0 < ngL; g0 += KCH_) {
const int32_t gt = ((ngL - g0) < KCH_) ? (ngL - g0) : KCH_;
if constexpr (!(!kFp32 && FUSE_BIG)) {
DataCopyExtParams cpQ(1, static_cast<uint32_t>(gt * D_ * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qiIn, queryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D_], cpQ, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
}
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qi, qiIn, gt * D_);
else Cast(qi, qiIn, RoundMode::CAST_NONE, gt * D_);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prodAll[j * ngL * D_ + off], qi[off], ki[j * D_ + off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(D_ / 8), 0));
}
PipeBarrier<PIPE_V>();
}
}
LocalTensor<T> arScratch = HeadSlice<T>(hWork_);
ReduceSumArHeadD128(sim, prodAll, arScratch, static_cast<uint32_t>(nj * ngL));
PipeBarrier<PIPE_V>();
}
} else {
for (int32_t g0 = 0; g0 < ngL; g0 += KCH_) {
const int32_t gt = ((ngL - g0) < KCH_) ? (ngL - g0) : KCH_;
DataCopyExtParams cpQ(1, static_cast<uint32_t>(gt * D_ * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qi, queryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D_], cpQ, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], qi[off], ki[j * D_ + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(D_ / 8), 0));
}
PipeBarrier<PIPE_V>();
LocalTensor<T> arScratch = HeadSlice<T>(hWork_);
ReduceSumArHeadD128(sim[j * ngL + g0], prod, arScratch,
static_cast<uint32_t>(gt));
PipeBarrier<PIPE_V>();
}
}
}
Maxs(sim, sim, static_cast<T>(0.0f), nj * ngL);
PipeBarrier<PIPE_V>();
Mul(prod, sim, wRep, nj * ngL);
PipeBarrier<PIPE_V>();
{
uint32_t shI[2] = {static_cast<uint32_t>(nj), static_cast<uint32_t>(ngL)};
ReduceSum<T, Pattern::Reduce::AR, true>(ii, prod, work, shI, true);
}
PipeBarrier<PIPE_V>();
{
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, static_cast<uint32_t>(nj * sizeof(T)), 0, 0, 0);
DataCopyPad(iPartGm_[blockIdx_ * 16u], ii, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
if (K > 1u) { SyncAll<true>(); }
{
constexpr uint32_t kIPOffset = (!kFp32 && FUSE_BIG) ? static_cast<uint32_t>(HP) * 16u : 64u;
static_assert(kIPOffset >= static_cast<uint32_t>(HP) * 16u,
"p 与 I 的部分和槽位不得重叠");
LocalTensor<T> pp = prod;
LocalTensor<T> ip = prod[kIPOffset];
const uint32_t slot0 = splitTask_ * headParts_;
DataCopyExtParams cp(1, static_cast<uint32_t>(headParts_ * 16u * sizeof(T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(pp, pPartGm_[slot0 * 16u], cp, DataCopyPadExtParams<T>(false, 0, 0, 0));
DataCopyPad(ip, iPartGm_[slot0 * 16u], cp, DataCopyPadExtParams<T>(false, 0, 0, 0));
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && FUSE_BIG) {
static_assert(!(FUSE_BIG && !kFp32) ||
(HP == 8 && NJ == 8 && JT == 8 &&
(static_cast<uint32_t>(HP) / 2u) * 16u == 64u),
"E66 P7 gather 折半树按 8 份/8 有效 lane/首层单 repeat 64 float 展开");
uint32_t r = static_cast<uint32_t>(HP);
while (r > 1u) {
const uint32_t half = r / 2u;
const uint64_t count = static_cast<uint64_t>(half) * 16u;
const int32_t hiOff = static_cast<int32_t>(half * 16u);
Add(pp, pp, pp[hiOff], count, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
Add(ip, ip, ip[hiOff], count, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
PipeBarrier<PIPE_V>();
r = half;
}
} else {
for (uint32_t c = 1u; c < headParts_; ++c) {
Add(pp, pp, pp[c * 16u], static_cast<uint64_t>(16u), 1,
BinaryRepeatParams(1, 1, 1, 2, 2, 2));
Add(ip, ip, ip[c * 16u], static_cast<uint64_t>(16u), 1,
BinaryRepeatParams(1, 1, 1, 2, 2, 2));
PipeBarrier<PIPE_V>();
}
}
Muls(p, pp, t->invN1, JT);
PipeBarrier<PIPE_V>();
Copy(ii, ip, static_cast<uint64_t>(NJ), static_cast<uint8_t>(1),
CopyRepeatParams(1, 1, 0, 0));
PipeBarrier<PIPE_V>();
}
PipeBarrier<PIPE_V>();
{
uint32_t shMax[2] = {1u, static_cast<uint32_t>(nj)};
ReduceMax<T, Pattern::Reduce::AR, true>(one, ii, work, shMax, true);
}
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
const float mx = one.GetValue(0);
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
Adds(ii, ii, -mx, JT);
PipeBarrier<PIPE_V>();
Exp(ii, ii, JT);
PipeBarrier<PIPE_V>();
const float z = SumVecSplit(ii, nj);
const float psum = SumVecSplit(p, nj);
const float invZ = z > 0.0f ? 1.0f / z : 0.0f;
const float invP = psum > 0.0f ? 1.0f / psum : 0.0f;
Muls(y, ii, invZ, JT);
Muls(p, p, invP, JT);
PipeBarrier<PIPE_V>();
Maxs(pmax, p, static_cast<T>(LOSS_EPS), JT);
Maxs(ly, y, static_cast<T>(LOSS_EPS), JT);
PipeBarrier<PIPE_V>();
Log(lp, pmax, JT); Log(ly, ly, JT);
PipeBarrier<PIPE_V>();
Sub(tmp, lp, ly, JT); PipeBarrier<PIPE_V>();
Mul(tmp, tmp, pmax, JT); PipeBarrier<PIPE_V>();
lossSum_ += SumVecSplit(tmp, nj);
Sub(ii, y, p, JT);
PipeBarrier<PIPE_V>();
const int32_t maskCount = ((nj * ngL + 63) / 64) * 64;
CompareScalar(maskBits, sim, static_cast<T>(0.0f), CMPMODE::GT, maskCount);
PipeBarrier<PIPE_V>();
static_assert(NGL == 8 || NGL == 16, "公开 head 本核头片列数只有 8/16");
static_assert(!FUSE_BIG || NGL == 8, "E62 P6/P7 新路径本核列数恒为 8");
if constexpr (FUSE_BIG) {
Brcb(deltaRep, ii, 1, BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
} else {
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
for (int32_t j = 0; j < nj; ++j) {
Duplicate(deltaRep[j * ngL], ii.GetValue(j), ngL);
}
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
}
Mul(coef, wRep, deltaRep, nj * ngL);
PipeBarrier<PIPE_V>();
Select(coef, maskBits, coef, static_cast<T>(0.0f),
SELMODE::VSEL_TENSOR_SCALAR_MODE, maskCount);
PipeBarrier<PIPE_V>();
{
Mul(prod, sim, deltaRep, nj * ngL);
PipeBarrier<PIPE_V>();
int32_t r = nj;
while (r > 1) {
const int32_t half = r / 2;
Add(prod[0], prod[0], prod[(r - half) * ngL], static_cast<uint64_t>(ngL),
static_cast<uint8_t>(half),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(ngL / 8),
static_cast<uint8_t>(ngL / 8), static_cast<uint8_t>(ngL / 8)));
PipeBarrier<PIPE_V>();
r -= half;
}
const uint64_t wOff = (static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1) * NG + idxH0_;
if constexpr (kFp32) {
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpf(1, static_cast<uint32_t>(ngL * sizeof(float)), 0, 0, 0);
DataCopyPad(dWeightsGm[wOff], prod, cpf);
} else {
LocalTensor<OUT_T> outW = HeadSlice<OUT_T>(hOutW_);
Cast(outW, prod, RoundMode::CAST_RINT, ngL);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpw(1, static_cast<uint32_t>(ngL * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dWeightsGm[wOff], outW, cpw);
}
if constexpr (!(!kFp32 && FUSE_BIG)) {
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
for (int32_t g0 = 0; g0 < ngL; g0 += KCH_) {
const int32_t gt = ((ngL - g0) < KCH_) ? (ngL - g0) : KCH_;
Duplicate(dq, static_cast<T>(0.0f), gt * D_);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
Brcb(cbrTmp, coef[j * ngL + g0], static_cast<uint8_t>(gt / 8), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && FUSE_BIG) {
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], cbrTmp, ki[j * D_ + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(1), 0));
PipeBarrier<PIPE_V>();
Add(dq[off], dq[off], prod[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(D_ / 8), static_cast<uint8_t>(D_ / 8)));
PipeBarrier<PIPE_V>();
}
} else {
Brcb(cbr, cbrTmp, static_cast<uint8_t>(gt), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], cbr, ki[j * D_ + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(64 / 8), 0));
PipeBarrier<PIPE_V>();
Add(dq[off], dq[off], prod[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(D_ / 8), static_cast<uint8_t>(D_ / 8)));
PipeBarrier<PIPE_V>();
}
}
}
if constexpr (IsSameType<OUT_T, T>::value) {
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpOut(1, static_cast<uint32_t>(gt * D_ * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dQueryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D_], dq, cpOut);
} else {
LocalTensor<OUT_T> outQ = HeadSlice<OUT_T>(hOutQ_);
Cast(outQ, dq, RoundMode::CAST_RINT, gt * D_);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpOut(1, static_cast<uint32_t>(gt * D_ * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dQueryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D_], outQ, cpOut);
}
if constexpr (!(!kFp32 && FUSE_BIG)) {
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
if constexpr (!kFp32) {
Duplicate(dk, static_cast<T>(0.0f), nj * D_);
PipeBarrier<PIPE_V>();
}
if constexpr ((kFp32 || !FUSE_BIG) || (!kFp32 && FUSE_BIG)) {
LocalTensor<T> prodAll = HeadSlice<T>(hKTile_);
const int32_t gt = ngL;
const uint64_t hStride = static_cast<uint64_t>(nj) * static_cast<uint64_t>(ngL) * 64ull;
if constexpr (S1 > 1) {
DataCopyExtParams cpQ(1, static_cast<uint32_t>(gt * D_ * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qiIn, queryIndexGm[qiBase + static_cast<uint64_t>(idxH0_) * D_], cpQ, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qi, qiIn, gt * D_);
else Cast(qi, qiIn, RoundMode::CAST_NONE, gt * D_);
}
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
Brcb(cbrTmp, coef[j * ngL], static_cast<uint8_t>(gt / 8), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && FUSE_BIG) {
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prodAll[static_cast<int32_t>(hStride) * h + j * ngL * 64], cbrTmp, qi[off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(1),
static_cast<uint8_t>(D_ / 8)));
}
} else {
Brcb(cbr, cbrTmp, static_cast<uint8_t>(gt), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prodAll[static_cast<int32_t>(hStride) * h + j * ngL * 64], cbr, qi[off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(D_ / 8)));
}
}
PipeBarrier<PIPE_V>();
}
for (int32_t h = 0; h < D_ / 64; ++h) {
LocalTensor<T> base = prodAll[static_cast<int32_t>(hStride) * h];
int32_t r = gt;
while (r > 1) {
const int32_t half = r / 2;
for (int32_t g0 = 0; g0 < half; ++g0) {
Add(base[g0 * 64], base[g0 * 64], base[(g0 + half) * 64],
static_cast<uint64_t>(64u), static_cast<uint8_t>(nj),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(ngL * 64 / 8),
static_cast<uint8_t>(ngL * 64 / 8),
static_cast<uint8_t>(ngL * 64 / 8)));
}
PipeBarrier<PIPE_V>();
r -= half;
}
Adds(dk[h * 64], base, static_cast<T>(0.0f), static_cast<uint64_t>(64u),
static_cast<uint8_t>(nj),
UnaryRepeatParams(1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(ngL * 64 / 8)));
PipeBarrier<PIPE_V>();
}
} else {
for (int32_t g0 = 0; g0 < ngL; g0 += KCH_) {
const int32_t gt = ((ngL - g0) < KCH_) ? (ngL - g0) : KCH_;
if constexpr (S1 > 1) {
DataCopyExtParams cpQ(1, static_cast<uint32_t>(gt * D_ * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qiIn, queryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D_], cpQ, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qi, qiIn, gt * D_);
else Cast(qi, qiIn, RoundMode::CAST_NONE, gt * D_);
}
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
Brcb(cbrTmp, coef[j * ngL + g0], static_cast<uint8_t>(gt / 8), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
Brcb(cbr, cbrTmp, static_cast<uint8_t>(gt), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], cbr, qi[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(D_ / 8)));
PipeBarrier<PIPE_V>();
int32_t r = gt;
while (r > 1) {
const int32_t half = r / 2;
Add(prod[off], prod[off], prod[off + half * 64], static_cast<uint64_t>(64u),
static_cast<uint8_t>(half),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8)));
PipeBarrier<PIPE_V>();
r -= half;
}
Add(dk[j * D_ + off], dk[j * D_ + off], prod[off], static_cast<uint64_t>(64u), 1,
BinaryRepeatParams(1, 1, 1, 1, 1, 1));
PipeBarrier<PIPE_V>();
}
}
}
}
{
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpd(1, static_cast<uint32_t>(nj * D_ * sizeof(T)), 0, 0, 0);
const uint64_t dkSlot = static_cast<uint64_t>(blockIdx_) * nj * D_;
DataCopyPad(dkPartGm_[dkSlot], dk, cpd);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
__aicore__ inline void WriteDkFromParts() {
if constexpr (!kFp32) {
WriteDkFromPartsBatch();
} else {
WriteDkFromPartsSerial();
}
}
__aicore__ inline void WriteDkFromPartsBatch() {
constexpr int32_t D_ = D;
const int32_t nj = NJ;
const uint32_t partsPerBatch = static_cast<uint32_t>(S1) * headParts_;
static_assert(static_cast<uint32_t>(S1) * (kBlockNum / kTotalTasks) == kDkParts,
"公开 head 小类每 batch dk 部分和份数须与编译常量一致");
static_assert(kKTileElems >= kDkParts * kKiElems,
"kTile 必须容纳 dk 部分和 4/8 份");
LocalTensor<T> acc = HeadSlice<T>(hAcc_);
LocalTensor<T> partsAll = HeadSlice<T>(hKTile_);
LocalTensor<OUT_T> out = HeadSlice<OUT_T>(hOutQ_);
DataCopyExtParams cpAll(1, static_cast<uint32_t>(partsPerBatch * nj * D_ * sizeof(T)),
0, 0, 0);
DataCopyExtParams cpo(1, static_cast<uint32_t>(nj * D_ * sizeof(OUT_T)), 0, 0, 0);
const DataCopyPadExtParams<T> pad(false, 0, 0, 0);
for (uint32_t b = 0; b < kB; ++b) {
const uint64_t slot0 = static_cast<uint64_t>(b) * partsPerBatch;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(partsAll, dkPartGm_[slot0 * static_cast<uint64_t>(nj * D_)], cpAll, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && FUSE_BIG) {
static_assert(!(FUSE_BIG && !kFp32) ||
(HP == 8 && kDkParts == 8u && (kKiElems % 64u) == 0u &&
(static_cast<uint32_t>(HP) / 2u) * kKiElems / 64u <= 255u),
"E66 P7 dk 折半树按 8 份/64 元素掩码/首层 repeat<=255 展开");
uint32_t r = static_cast<uint32_t>(HP);
while (r > 1u) {
const uint32_t half = r / 2u;
const uint32_t count = half * static_cast<uint32_t>(nj * D_);
const int32_t hiOff = static_cast<int32_t>(count);
Add(partsAll, partsAll, partsAll[hiOff], static_cast<uint64_t>(64u),
static_cast<uint8_t>(count / 64u), BinaryRepeatParams(1, 1, 1, 8, 8, 8));
PipeBarrier<PIPE_V>();
r = half;
}
DataCopy(acc, partsAll, nj * D_);
PipeBarrier<PIPE_V>();
} else {
DataCopy(acc, partsAll, nj * D_);
PipeBarrier<PIPE_V>();
for (uint32_t c = 1u; c < partsPerBatch; ++c) {
Add(acc, acc, partsAll[c * nj * D_], nj * D_);
PipeBarrier<PIPE_V>();
}
}
if constexpr (IsSameType<OUT_T, T>::value) DataCopy(out, acc, nj * D_);
else Cast(out, acc, RoundMode::CAST_RINT, nj * D_);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyPad(dKeyIndexGm[static_cast<uint64_t>(b) * nj * D_], out, cpo);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
__aicore__ inline void WriteDkFromPartsSerial() {
constexpr int32_t D_ = D;
const int32_t nj = NJ;
const uint32_t partsPerBatch = static_cast<uint32_t>(S1) * headParts_;
LocalTensor<T> acc = HeadSlice<T>(hAcc_);
LocalTensor<T> part = HeadSlice<T>(hPart_);
DataCopyExtParams cp(1, static_cast<uint32_t>(nj * D_ * sizeof(T)), 0, 0, 0);
DataCopyExtParams cpo(1, static_cast<uint32_t>(nj * D_ * sizeof(OUT_T)), 0, 0, 0);
const DataCopyPadExtParams<T> pad(false, 0, 0, 0);
for (uint32_t b = 0; b < kB; ++b) {
const uint64_t slot0 = static_cast<uint64_t>(b) * partsPerBatch;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(acc, dkPartGm_[slot0 * static_cast<uint64_t>(nj * D_)], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
for (uint32_t c = 1u; c < partsPerBatch; ++c) {
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(part, dkPartGm_[(slot0 + c) * static_cast<uint64_t>(nj * D_)], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
Add(acc, acc, part, nj * D_);
PipeBarrier<PIPE_V>();
}
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyPad(dKeyIndexGm[static_cast<uint64_t>(b) * nj * D_], acc, cpo);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
__aicore__ inline float SumVecSplit(const LocalTensor<T> &v, int32_t n) {
LocalTensor<uint8_t> work = HeadSlice<uint8_t>(hWork_);
LocalTensor<T> one = smallBuf.Get<T>();
uint32_t sh[2] = {1, static_cast<uint32_t>(n)};
ReduceSum<T, Pattern::Reduce::AR, true>(one, v, work, sh, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
return one.GetValue(0);
}
__aicore__ inline void LoadSplitWeightsRow(uint32_t b, uint32_t s1) {
LocalTensor<T> wr = HeadSlice<T>(hWRow_);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
const uint64_t off = (static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1) * NG;
if constexpr (kFp32) {
DataCopyExtParams cp32(1, static_cast<uint32_t>(NG * sizeof(T)), 0, 0, 0);
DataCopyPadExtParams<T> pad32(false, 0, 0, 0);
DataCopyPad(wr, weightsGm[off], cp32, pad32);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
} else {
DataCopyExtParams cp(1, static_cast<uint32_t>(NG * sizeof(IN_T)), 0, 0, 0);
DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
LocalTensor<IN_T> wi = HeadSlice<IN_T>(hWIn_);
DataCopyPad(wi, weightsGm[off], cp, pad);
if constexpr (!FUSE_BIG) {
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
Cast(wr, wi, RoundMode::CAST_NONE, NG);
PipeBarrier<PIPE_V>();
} else {
(void)wr;
}
}
if constexpr (kFp32 || !FUSE_BIG) {
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
}
}
static constexpr int32_t EV_MTE2V = 0;
static constexpr int32_t EV_VMTE3 = 1;
static constexpr int32_t EV_VS = 2;
static constexpr int32_t EV_SMTE3 = 3;
static constexpr int32_t EV_VMTE2 = 5;
static constexpr int32_t EV_MTE3V = 6;
static constexpr int32_t EV_SV = 7;
const DenseLightningIndexerGradKlLossTilingData *t = nullptr;
TPipe pipe;
TBuf<TPosition::VECCALC> headBuf_;
TBuf<TPosition::VECCALC> smallBuf;
TBuf<TPosition::VECCALC> syncBuf;
uint32_t headOff_ = 0;
uint32_t hLogits_ = 0, hP_ = 0, hRep_ = 0, hKTile_ = 0, hDot_ = 0, hVec_ = 0;
uint32_t hMx_ = 0, hSum_ = 0, hQi_ = 0, hKi_ = 0, hDq_ = 0, hDk_ = 0;
uint32_t hWRow_ = 0, hWIn_ = 0, hOutW_ = 0, hOutQ_ = 0, hAcc_ = 0, hPart_ = 0;
uint32_t hSim_ = 0, hWork_ = 0, hQIn_ = 0, hKIn_ = 0, hKiIn_ = 0;
float lossSum_ = 0.0f;
uint32_t blockIdx_ = 0;
uint32_t blockNum_ = 1;
uint32_t headParts_ = 1u, splitTask_ = 0u;
uint32_t mainH0_ = 0, mainH1_ = 0, idxH0_ = 0, idxH1_ = 0;
GlobalTensor<IN_T> queryGm, keyGm, queryIndexGm, keyIndexGm, weightsGm, dWeightsGm;
GlobalTensor<OUT_T> dQueryIndexGm, dKeyIndexGm;
GlobalTensor<T> lossGmRef;
GlobalTensor<T> pPartGm_, iPartGm_, dkPartGm_;
GlobalTensor<int32_t> syncGm;
};
template <typename DTQ, bool FUSE_BIG>
class KernelHeadRaw {
public:
using T = float;
using IN_T = DTQ;
using OUT_T = DTQ;
static constexpr bool kFp32 = IsSameType<IN_T, T>::value;
static constexpr int32_t S1 = kFp32 ? 1 : (FUSE_BIG ? 1 : 2);
static constexpr int32_t NJ = kFp32 ? 4 : 8;
static constexpr int32_t N1 = kFp32 ? 64 : (FUSE_BIG ? 128 : 64);
static constexpr int32_t NG = kFp32 ? 32 : (FUSE_BIG ? 64 : 16);
static constexpr int32_t HP = kFp32 ? 4 : (FUSE_BIG ? 8 : 2);
static constexpr int32_t NL = N1 / HP;
static constexpr int32_t NGL = NG / HP;
static constexpr int32_t D = 128;
static constexpr uint32_t kB = 1u;
static constexpr uint32_t kTotalTasks = static_cast<uint32_t>(S1);
static constexpr uint32_t kBlockNum = kTotalTasks * static_cast<uint32_t>(HP);
static constexpr uint32_t kHeadParts = static_cast<uint32_t>(HP);
static constexpr int32_t KCH = 16;
static constexpr int32_t MTILE = 32;
static constexpr int32_t JT = 8;
static constexpr uint32_t kKeyBatch = 4u;
static constexpr uint32_t kInQElems = static_cast<uint32_t>(MTILE) * D;
static constexpr uint32_t kQidElems = static_cast<uint32_t>(KCH) * D;
static constexpr uint32_t kKiElems = static_cast<uint32_t>(NJ) * D;
static constexpr uint32_t kLogitsElems = static_cast<uint32_t>(NJ) * static_cast<uint32_t>(NL);
static constexpr uint32_t kRepElems = 4096u;
static constexpr uint32_t kKeyRows = kKeyBatch * static_cast<uint32_t>(NL);
static constexpr uint32_t kKTileElems = kKeyRows * D;
static constexpr uint32_t kKInElems = kKeyRows * D;
static constexpr uint32_t kKIn2Bytes =
(kFp32 || FUSE_BIG) ? static_cast<uint32_t>(kKInElems * sizeof(IN_T)) : 0u;
static constexpr uint32_t kPfKiBytes = kKiElems * sizeof(IN_T);
static constexpr uint32_t kPfQiBytes =
static_cast<uint32_t>(NGL) * static_cast<uint32_t>(D) * sizeof(IN_T);
static_assert(kPfKiBytes <= kKiElems * sizeof(T) &&
kPfQiBytes <= kQidElems * sizeof(T),
"P6 prefetched ki/qi must fit the hKi_/hQi_ slices");
static constexpr uint32_t kDotElems = kKeyBatch * static_cast<uint32_t>(MTILE);
static constexpr uint32_t kVecElems = 10u * static_cast<uint32_t>(JT);
static constexpr uint32_t kSimElems = 512u;
static constexpr uint32_t kWorkBytes = 16u * 1024u;
static constexpr uint32_t kOutQElems = 16u * D;
static constexpr uint32_t kSlabBytes = 32768u;
static constexpr uint32_t kSlabQiOff = 0u;
static constexpr uint32_t kSlabKiOff = 8192u;
static constexpr uint32_t kSlabDqOff = 12288u;
static constexpr uint32_t kSlabDkOff = 20480u;
static constexpr uint32_t kSlabOutQOff = 24576u;
static_assert(kSlabQiOff + kQidElems * sizeof(T) <= kSlabKiOff, "P5 slab qi/ki disjoint");
static_assert(kSlabKiOff + kKiElems * sizeof(T) <= kSlabDqOff, "P5 slab ki/dq disjoint");
static_assert(kSlabDqOff + kQidElems * sizeof(T) <= kSlabDkOff, "P5 slab dq/dk disjoint");
static_assert(kSlabDkOff + kKiElems * sizeof(T) <= kSlabOutQOff, "P5 slab dk/outQ disjoint");
static_assert(kSlabOutQOff + kOutQElems * sizeof(OUT_T) <= kSlabBytes, "P5 slab outQ fits");
static_assert(kSlabQiOff % 512u == 0u && kSlabKiOff % 512u == 0u && kSlabDqOff % 512u == 0u &&
kSlabDkOff % 512u == 0u && kSlabOutQOff % 512u == 0u,
"P5 slab aliases must stay 512B aligned");
static constexpr uint32_t kDkParts = static_cast<uint32_t>(S1) * static_cast<uint32_t>(HP);
static_assert(kBlockNum == 4u || kBlockNum == 8u, "公开 head 小类只有 4/8 核两种");
static_assert(kDkParts == 4u || kDkParts == 8u, "公开 head 小类 dk 部分和只有 4/8 份");
static_assert(kB == 1u, "公开 head 小类 bSize 恒为 1");
static_assert(S1 == 1 || S1 == 2, "公开 head 小类只有 S1=1/2 两种任务数");
static_assert(NL <= MTILE, "主头片单次行数不得超过 MTILE");
static_assert(static_cast<uint32_t>(NJ) % kKeyBatch == 0u, "key 成组宽度须整除 NJ");
static_assert(kOutQElems >= kKiElems, "outQ 必须兼顾 NJ*D 的 dk 写出");
static_assert(kQidElems >= static_cast<uint32_t>(KCH) * D, "qi/dq 至少 16*D");
static_assert(kRepElems >= 2048u + 3u * 256u + 128u + 1024u, "rep 内部布局容量");
static_assert(kDotElems >= kKeyRows, "dot 必须容纳一批 key 的 gt 行归约输出");
static_assert(kKTileElems >= kKiElems, "kTile 兼顾主分 tile 与收尾 dk 汇总");
static_assert(static_cast<uint32_t>(NGL) <= static_cast<uint32_t>(KCH),
"公开家族本核 indexer 头片须为唯一 gt 块");
static_assert(kKTileElems >=
2u * static_cast<uint32_t>(NJ) * static_cast<uint32_t>(NGL) * 64u,
"kTile 必须容纳 dk 批式 prodAll[2][NJ][NGL][64]");
static_assert(kKTileElems >= static_cast<uint32_t>(NJ) * static_cast<uint32_t>(NGL) *
static_cast<uint32_t>(D),
"kTile 必须容纳 indexer 批式 prodAll[NJ][NGL][D]");
__aicore__ inline KernelHeadRaw() {}
__aicore__ inline void ReadTilingAndBindWorkspace() {
GlobalTensor<float> scaleGm, invN1Gm;
scaleGm.SetGlobalBuffer((__gm__ float *)(tilingRaw_ + kScaleValueOff));
invN1Gm.SetGlobalBuffer((__gm__ float *)(tilingRaw_ + kInvN1Off));
scale_ = scaleGm.GetValue(0);
invN1_ = invN1Gm.GetValue(0);
GlobalTensor<uint32_t> reservedGm;
reservedGm.SetGlobalBuffer((__gm__ uint32_t *)(tilingRaw_ + kReservedOff));
reserved1_ = reservedGm.GetValue(1u);
reserved2_ = reservedGm.GetValue(2u);
reserved3_ = reservedGm.GetValue(3u);
__gm__ uint8_t *ws = (__gm__ uint8_t *)wsRaw_;
pPartGm_.SetGlobalBuffer((__gm__ T *)(ws + reserved1_), blockNum_ * 16u);
iPartGm_.SetGlobalBuffer((__gm__ T *)(ws + reserved2_), blockNum_ * 16u);
dkPartGm_.SetGlobalBuffer((__gm__ T *)(ws + reserved3_),
static_cast<uint64_t>(blockNum_) * NJ * D);
}
__aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
GM_ADDR weights, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex,
GM_ADDR dWeights, GM_ADDR loss, GM_ADDR workspace,
GM_ADDR tiling) {
blockIdx_ = GetBlockIdx();
blockNum_ = kBlockNum;
tilingRaw_ = tiling;
queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
queryIndexGm.SetGlobalBuffer((__gm__ IN_T *)queryIndex);
keyIndexGm.SetGlobalBuffer((__gm__ IN_T *)keyIndex);
weightsGm.SetGlobalBuffer((__gm__ IN_T *)weights);
dWeightsGm.SetGlobalBuffer((__gm__ IN_T *)dWeights);
dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
lossGmRef.SetGlobalBuffer((__gm__ T *)loss);
headParts_ = kHeadParts;
splitTask_ = blockIdx_ / headParts_;
const uint32_t part = blockIdx_ % headParts_;
mainH0_ = part * static_cast<uint32_t>(NL);
mainH1_ = mainH0_ + static_cast<uint32_t>(NL);
idxH0_ = part * static_cast<uint32_t>(NGL);
idxH1_ = idxH0_ + static_cast<uint32_t>(NGL);
__gm__ uint8_t *ws = (__gm__ uint8_t *)workspace;
(void)ws;
wsRaw_ = workspace;
headOff_ = 0;
hLogits_ = HeadTake(kLogitsElems * sizeof(T));
hP_ = HeadTake(16u * sizeof(T));
hRep_ = HeadTake(kRepElems * sizeof(T));
hKTile_ = HeadTake(kKTileElems * sizeof(T));
hDot_ = HeadTake(kDotElems * sizeof(T));
hVec_ = HeadTake(kVecElems * sizeof(T));
hMx_ = HeadTake(static_cast<uint32_t>(NL) * sizeof(T));
hSum_ = HeadTake(static_cast<uint32_t>(NL) * sizeof(T));
if constexpr (kFp32 || FUSE_BIG) {
hQi_ = HeadTake(kQidElems * sizeof(T));
hKi_ = HeadTake(kKiElems * sizeof(T));
hDq_ = HeadTake(kQidElems * sizeof(T));
hDk_ = HeadTake(kKiElems * sizeof(T));
}
hWRow_ = HeadTake(static_cast<uint32_t>(NG) * sizeof(T));
hWIn_ = HeadTake(static_cast<uint32_t>(NG) * sizeof(IN_T));
hOutW_ = HeadTake(static_cast<uint32_t>(NG) * sizeof(T));
if constexpr (kFp32 || FUSE_BIG) {
hOutQ_ = HeadTake(kOutQElems * sizeof(OUT_T));
hAcc_ = HeadTake(kKiElems * sizeof(T));
hPart_ = HeadTake(kKiElems * sizeof(T));
}
hSim_ = HeadTake(kSimElems * sizeof(T));
hWork_ = HeadTake(kWorkBytes);
hQIn_ = HeadTake(kInQElems * sizeof(IN_T));
hKIn_ = HeadTake(kKInElems * sizeof(IN_T));
if constexpr (kFp32 || FUSE_BIG) {
hKIn1_ = HeadTake(kKIn2Bytes);
}
hKiIn_ = HeadTake(kKiElems * sizeof(IN_T));
if constexpr (!kFp32 && !FUSE_BIG) {
hSlab_ = HeadTake(kSlabBytes);
hQi_ = hSlab_ + kSlabQiOff;
hKi_ = hSlab_ + kSlabKiOff;
hDq_ = hSlab_ + kSlabDqOff;
hDk_ = hSlab_ + kSlabDkOff;
hOutQ_ = hSlab_ + kSlabOutQOff;
hAcc_ = hSlab_ + kSlabKiOff;
hPart_ = hSlab_ + kSlabDkOff;
}
smallOff_ = Align32(headOff_);
arenaBytes_ = smallOff_;
}
__aicore__ inline void Process() {
lossSum_ = 0.0f;
DoTaskSplitHead(splitTask_);
if (blockNum_ > 1u) SyncAll<true>();
if (blockIdx_ == 0u) {
WriteDkFromParts();
}
if ((blockIdx_ % headParts_) == 0u) {
LocalTensor<T> ls = SmallBuf();
ls.SetValue(0, static_cast<T>(lossSum_));
SetFlag<HardEvent::S_MTE3>(EV_SMTE3);
WaitFlag<HardEvent::S_MTE3>(EV_SMTE3);
if constexpr (S1 > 1) { if (blockNum_ > 1u) { SetAtomicAdd<float>(); } }
DataCopyExtParams cpl(1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0);
DataCopyPad(lossGmRef, ls, cpl);
if constexpr (S1 > 1) { if (blockNum_ > 1u) { SetAtomicNone(); } }
SetFlag<HardEvent::MTE3_S>(1);
WaitFlag<HardEvent::MTE3_S>(1);
}
}
private:
static __aicore__ inline uint32_t Align32(uint32_t x) { return (x + 31u) / 32u * 32u; }
__aicore__ inline uint32_t HeadTake(uint64_t bytes) {
const uint32_t at = headOff_;
headOff_ += static_cast<uint32_t>((bytes + 511u) / 512u * 512u);
return at;
}
__aicore__ inline LocalTensor<uint8_t> UbArena() const {
return LocalTensor<uint8_t>(TPosition::VECCALC, 0u, arenaBytes_);
}
__aicore__ inline LocalTensor<T> SmallBuf() const {
return LocalTensor<T>(TPosition::VECCALC, smallOff_, kSmallBytes / sizeof(T));
}
__aicore__ inline LocalTensor<uint8_t> SmallBufU8() const {
return LocalTensor<uint8_t>(TPosition::VECCALC, smallOff_, kSmallBytes);
}
template <typename ET>
__aicore__ inline LocalTensor<ET> HeadSlice(uint32_t off) {
LocalTensor<uint8_t> buf = UbArena();
return buf[off].ReinterpretCast<ET>();
}
__aicore__ inline void ReduceSumArHeadD128(const LocalTensor<T>& dst, const LocalTensor<T>& src,
const LocalTensor<T>& scratch, uint32_t repeat) {
static_assert(D == 128, "E58 fixed AR helper assumes D=128");
static_assert(8u * (kKeyBatch * static_cast<uint32_t>(NL)) <= kWorkBytes / sizeof(T),
"E58 scratch (hWork_ 16 KiB) must hold the largest R rows of 8 floats");
constexpr uint8_t kBlk = 1u;
constexpr uint8_t kRowBlk = static_cast<uint8_t>(D / B32_DATA_NUM_PER_BLOCK);
constexpr int32_t kOutBlk = 1;
constexpr int32_t kMidBlk = B32_DATA_NUM_PER_REPEAT / B32_DATA_NUM_PER_BLOCK;
const uint8_t rep = static_cast<uint8_t>(repeat);
SetMaskNorm();
SetVectorMask<T, MaskMode::NORMAL>(B32_DATA_NUM_PER_REPEAT);
Add<T, false>(src, src, src[B32_DATA_NUM_PER_REPEAT], MASK_PLACEHOLDER, rep,
BinaryRepeatParams(kBlk, kBlk, kBlk, kRowBlk, kRowBlk, kRowBlk));
PipeBarrier<PIPE_V>();
ResetMask();
SetMaskCount();
SetVectorMask<T, MaskMode::COUNTER>(static_cast<int32_t>(repeat) * B32_DATA_NUM_PER_REPEAT);
BlockReduceSum<T, false>(scratch, src, 1, MASK_PLACEHOLDER, kOutBlk, kBlk,
static_cast<int32_t>(kRowBlk));
PipeBarrier<PIPE_V>();
SetVectorMask<T, MaskMode::COUNTER>(static_cast<int32_t>(repeat) * B32_DATA_NUM_PER_BLOCK);
BlockReduceSum<T, false>(dst, scratch, 1, MASK_PLACEHOLDER, kOutBlk, kBlk, kMidBlk);
PipeBarrier<PIPE_V>();
SetMaskNorm();
ResetMask();
}
__aicore__ inline void ComputeMainBatchedRange(uint32_t b, uint32_t s1, uint32_t hBegin,
uint32_t hEnd) {
constexpr uint32_t dQ = 128u;
const uint32_t nL = hEnd - hBegin;
if (nL == 0u) { return; }
LocalTensor<T> logits = HeadSlice<T>(hLogits_);
LocalTensor<T> p = HeadSlice<T>(hP_);
LocalTensor<T> qTile = HeadSlice<T>(hRep_);
LocalTensor<T> kTile = HeadSlice<T>(hKTile_);
LocalTensor<T> dot = HeadSlice<T>(hDot_);
LocalTensor<T> mx = HeadSlice<T>(hMx_);
LocalTensor<T> sum = HeadSlice<T>(hSum_);
LocalTensor<uint8_t> work = HeadSlice<uint8_t>(hWork_);
LocalTensor<IN_T> qIn = HeadSlice<IN_T>(hQIn_);
LocalTensor<IN_T> kIn = HeadSlice<IN_T>(hKIn_);
Duplicate(p, static_cast<T>(0.0f), 16);
PipeBarrier<PIPE_V>();
const uint32_t headTile = (nL < 32u) ? nL : 32u;
const DataCopyExtParams cp(1, static_cast<uint32_t>(headTile * dQ * sizeof(IN_T)), 0, 0, 0);
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
for (uint32_t g0 = hBegin; g0 < hEnd; g0 += headTile) {
const uint32_t gt = ((hEnd - g0) < headTile) ? (hEnd - g0) : headTile;
DataCopyExtParams cpg(1, static_cast<uint32_t>(gt * dQ * sizeof(IN_T)), 0, 0, 0);
const uint64_t qOff = ((static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1) * N1 + g0) * dQ;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
if constexpr (kFp32) {
DataCopyPad(qTile, queryGm[qOff], cpg, pad);
} else {
DataCopyPad(qIn, queryGm[qOff], cpg, pad);
}
if constexpr (!kFp32 && !FUSE_BIG) {
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
}
if constexpr (kFp32) {
static_assert(static_cast<uint32_t>(NJ) == kKeyBatch, "E91 P6 main key group is exactly one");
const uint64_t kOff91 =
(static_cast<uint64_t>(b) * static_cast<uint32_t>(NJ) * N1 + g0) * dQ;
constexpr uint32_t kSubRows = kKeyBatch / 2u;
constexpr uint32_t kSubElems = kSubRows * static_cast<uint32_t>(NL) * dQ;
DataCopyExtParams cpk91(static_cast<uint16_t>(kSubRows),
static_cast<uint32_t>(gt * dQ * sizeof(IN_T)),
static_cast<uint32_t>((static_cast<uint32_t>(N1) - gt) * dQ * sizeof(IN_T)),
0u, 0u);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kTile, keyGm[kOff91], cpk91, pad);
DataCopyPad(kTile[static_cast<int32_t>(kSubElems)],
keyGm[kOff91 + static_cast<uint64_t>(kSubRows) * N1 * dQ],
cpk91, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
ReadTilingAndBindWorkspace();
} else if constexpr (FUSE_BIG) {
static_assert(static_cast<uint32_t>(NJ) == 2u * kKeyBatch,
"E91 P7 main key groups are exactly two");
static_assert(static_cast<uint32_t>(NL) <= static_cast<uint32_t>(MTILE),
"E91 P7 prefetch window assumes a single gt block (headTile=min(nL,32)=NL)");
const uint64_t kOff91 =
(static_cast<uint64_t>(b) * static_cast<uint32_t>(NJ) * N1 + g0) * dQ;
DataCopyExtParams cpk91(static_cast<uint16_t>(kKeyBatch),
static_cast<uint32_t>(gt * dQ * sizeof(IN_T)),
static_cast<uint32_t>((static_cast<uint32_t>(N1) - gt) * dQ * sizeof(IN_T)),
0u, 0u);
LocalTensor<IN_T> kIn1Pf = HeadSlice<IN_T>(hKIn1_);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kIn, keyGm[kOff91], cpk91, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
DataCopyPad(kIn1Pf, keyGm[kOff91 + static_cast<uint64_t>(kKeyBatch) * N1 * dQ],
cpk91, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V1);
ReadTilingAndBindWorkspace();
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
}
if constexpr (!kFp32) {
Cast(qTile, qIn, RoundMode::CAST_NONE, gt * dQ);
}
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && !FUSE_BIG) {
static_assert(static_cast<uint32_t>(NJ) == 2u * kKeyBatch,
"公开 A 族 main key 组恰为两组");
const uint64_t kOff0 = (static_cast<uint64_t>(b) * static_cast<uint32_t>(NJ) * N1
+ g0) * dQ;
DataCopyExtParams cpk0(static_cast<uint16_t>(kKeyBatch),
static_cast<uint32_t>(gt * dQ * sizeof(IN_T)),
static_cast<uint32_t>((static_cast<uint32_t>(N1) - gt) * dQ * sizeof(IN_T)),
0u, 0u);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kIn, keyGm[kOff0], cpk0, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
ReadTilingAndBindWorkspace();
}
for (uint32_t j0 = 0u; j0 < static_cast<uint32_t>(NJ); j0 += kKeyBatch) {
const uint32_t jc = ((static_cast<uint32_t>(NJ) - j0) < kKeyBatch)
? (static_cast<uint32_t>(NJ) - j0) : kKeyBatch;
const uint64_t kOff = ((static_cast<uint64_t>(b) * static_cast<uint32_t>(NJ) + j0)
* N1 + g0) * dQ;
DataCopyExtParams cpk(static_cast<uint16_t>(jc),
static_cast<uint32_t>(gt * dQ * sizeof(IN_T)),
static_cast<uint32_t>((static_cast<uint32_t>(N1) - gt) * dQ * sizeof(IN_T)),
0u, 0u);
if constexpr (!kFp32 && !FUSE_BIG) {
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
} else if constexpr (kFp32) {
(void)kOff;
(void)cpk;
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
} else {
if (j0 == 0u) {
} else {
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V1);
static_assert(static_cast<uint32_t>(NJ) == 2u * kKeyBatch,
"E78 P7 main key 组恰为两组");
static_assert(NGL <= KCH, "E78 P7 qi 首段是唯一 gt 块");
LocalTensor<IN_T> kiPf = HeadSlice<IN_T>(hKiIn_);
LocalTensor<IN_T> qiPf = HeadSlice<IN_T>(hQIn_);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyExtParams cpK78(1, static_cast<uint32_t>(NJ * D * sizeof(IN_T)), 0, 0, 0);
DataCopyPad(kiPf,
keyIndexGm[static_cast<uint64_t>(b) * static_cast<uint32_t>(NJ) * D],
cpK78, pad);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyExtParams cpQ78(1, static_cast<uint32_t>(NGL * D * sizeof(IN_T)), 0, 0, 0);
DataCopyPad(qiPf,
queryIndexGm[(static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1)
* NG * D + static_cast<uint64_t>(idxH0_) * D],
cpQ78, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
}
}
if constexpr (!kFp32) {
if constexpr (FUSE_BIG) {
LocalTensor<IN_T> kInG = (j0 == 0u) ? kIn : HeadSlice<IN_T>(hKIn1_);
Cast(kTile, kInG, RoundMode::CAST_NONE, jc * gt * dQ);
} else {
Cast(kTile, kIn, RoundMode::CAST_NONE, jc * gt * dQ);
}
}
if constexpr (!kFp32 && !FUSE_BIG) {
if (j0 + kKeyBatch < static_cast<uint32_t>(NJ)) {
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(kIn, keyGm[kOff + static_cast<uint64_t>(kKeyBatch) * N1 * dQ],
cpk, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
const uint64_t kiPrefOff =
static_cast<uint64_t>(b) * static_cast<uint32_t>(NJ) * dQ;
const uint64_t qiPrefOff =
(static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1) *
static_cast<uint32_t>(NG) * dQ +
static_cast<uint64_t>(idxH0_) * dQ;
DataCopyExtParams cpkPrefKi(
1, static_cast<uint32_t>(static_cast<uint32_t>(NJ) * dQ * sizeof(IN_T)),
0, 0, 0);
DataCopyPad(HeadSlice<IN_T>(hKiIn_), keyIndexGm[kiPrefOff], cpkPrefKi, pad);
DataCopyExtParams cpkPrefQi(
1, static_cast<uint32_t>(static_cast<uint32_t>(NGL) * dQ * sizeof(IN_T)),
0, 0, 0);
DataCopyPad(HeadSlice<IN_T>(hQIn_), queryIndexGm[qiPrefOff], cpkPrefQi, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V1);
}
} else if constexpr (!kFp32 && FUSE_BIG) {
(void)kOff;
(void)cpk;
}
if constexpr (kFp32) {
{
static_assert(NGL <= KCH, "P6 first qi block is the only gt block");
static_assert(static_cast<uint32_t>(NJ) == kKeyBatch, "P6 ki transfer shape check");
LocalTensor<IN_T> kiPfC = HeadSlice<IN_T>(hKi_);
LocalTensor<IN_T> qiPfC = HeadSlice<IN_T>(hQi_);
DataCopyExtParams cpKiC(1, static_cast<uint32_t>(NJ * D * sizeof(IN_T)), 0, 0, 0);
DataCopyPad(kiPfC,
keyIndexGm[static_cast<uint64_t>(b) * static_cast<uint32_t>(NJ) * D],
cpKiC, pad);
DataCopyExtParams cpQiC(1, static_cast<uint32_t>(NGL * D * sizeof(IN_T)), 0, 0, 0);
DataCopyPad(qiPfC,
queryIndexGm[(static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1)
* NG * D + static_cast<uint64_t>(idxH0_) * D],
cpQiC, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V1);
}
for (uint32_t jj = 0u; jj < jc; ++jj) {
Mul(kTile[jj * gt * dQ], kTile[jj * gt * dQ], qTile, gt * dQ);
}
PipeBarrier<PIPE_V>();
} else if constexpr (!FUSE_BIG) {
for (uint32_t jj = 0u; jj < jc; ++jj) {
Mul(kTile[jj * gt * dQ], kTile[jj * gt * dQ], qTile, gt * dQ);
}
PipeBarrier<PIPE_V>();
} else {
for (uint32_t jj = 0u; jj < jc; ++jj) {
Mul(kTile[jj * gt * dQ], kTile[jj * gt * dQ], qTile, gt * dQ);
PipeBarrier<PIPE_V>();
}
}
LocalTensor<T> arScratch = HeadSlice<T>(hWork_);
if constexpr (kFp32) {
ReduceSumArHeadD128(logits, kTile, arScratch, jc * gt);
} else {
ReduceSumArHeadD128(dot, kTile, arScratch, jc * gt);
}
PipeBarrier<PIPE_V>();
if constexpr (kFp32) {
Muls(logits, logits, scale_, jc * gt);
} else {
Muls(dot, dot, scale_, jc * gt);
}
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && !FUSE_BIG) {
DataCopy(logits[j0 * nL], dot, jc * nL);
PipeBarrier<PIPE_V>();
} else if constexpr (!kFp32 && FUSE_BIG) {
for (uint32_t jj = 0u; jj < jc; ++jj) {
DataCopy(logits[(j0 + jj) * nL + (g0 - hBegin)], dot[jj * gt], gt);
PipeBarrier<PIPE_V>();
}
}
}
}
(void)cp;
Duplicate(mx, static_cast<T>(-3.0e38f), nL);
Duplicate(sum, static_cast<T>(0.0f), nL);
PipeBarrier<PIPE_V>();
LocalTensor<T> tree = HeadSlice<T>(hWork_);
const uint32_t half = static_cast<uint32_t>(NJ) / 2u;
Max(tree, logits, logits[half * nL], half * nL);
PipeBarrier<PIPE_V>();
for (uint32_t rows = half; rows > 1u; rows >>= 1u) {
const uint32_t hf = rows >> 1u;
Max(tree, tree, tree[hf * nL], hf * nL);
PipeBarrier<PIPE_V>();
}
Max(mx, mx, tree, nL);
PipeBarrier<PIPE_V>();
const uint64_t maskNL = static_cast<uint64_t>(nL);
const uint8_t repNJ = static_cast<uint8_t>(NJ);
const uint8_t rowBlk = static_cast<uint8_t>(nL / 8u);
Sub(logits, logits, mx, maskNL, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
PipeBarrier<PIPE_V>();
Exp(logits, logits, static_cast<int32_t>(NJ) * static_cast<int32_t>(nL));
PipeBarrier<PIPE_V>();
Add(tree, logits, logits[half * nL], half * nL);
PipeBarrier<PIPE_V>();
for (uint32_t rows = half; rows > 1u; rows >>= 1u) {
const uint32_t hf = rows >> 1u;
Add(tree, tree, tree[hf * nL], hf * nL);
PipeBarrier<PIPE_V>();
}
Add(sum, sum, tree, nL);
PipeBarrier<PIPE_V>();
Div(logits, logits, sum, maskNL, repNJ, BinaryRepeatParams(1, 1, 1, rowBlk, rowBlk, 0));
PipeBarrier<PIPE_V>();
uint32_t shP[2] = {static_cast<uint32_t>(NJ), static_cast<uint32_t>(nL)};
ReduceSum<T, Pattern::Reduce::AR, true>(p, logits, work, shP, true);
PipeBarrier<PIPE_V>();
}
__aicore__ inline void DoTaskSplitHead(uint32_t task) {
constexpr int32_t D_ = D;
constexpr int32_t KCH_ = KCH;
const uint32_t b = task / static_cast<uint32_t>(S1);
const uint32_t s1 = task % static_cast<uint32_t>(S1);
const int32_t nj = NJ;
const int32_t ngL = NGL;
const uint32_t K = blockNum_;
LocalTensor<T> scratch = HeadSlice<T>(hRep_);
LocalTensor<T> prod = scratch;
LocalTensor<T> coef = scratch[KCH_ * D_];
LocalTensor<T> wRep = coef[256];
LocalTensor<T> deltaRep = wRep[256];
LocalTensor<T> cbrTmp = deltaRep[256];
LocalTensor<T> cbr = cbrTmp[128];
LocalTensor<T> sim = HeadSlice<T>(hSim_);
LocalTensor<T> qi = HeadSlice<T>(hQi_);
LocalTensor<T> ki = HeadSlice<T>(hKi_);
LocalTensor<T> dq = HeadSlice<T>(hDq_);
LocalTensor<T> dk = HeadSlice<T>(hDk_);
LocalTensor<T> vec = HeadSlice<T>(hVec_);
LocalTensor<T> p = vec;
LocalTensor<T> ii = vec[JT];
LocalTensor<T> y = vec[2 * JT];
LocalTensor<T> tmp = vec[4 * JT];
LocalTensor<T> lp = vec[6 * JT];
LocalTensor<T> ly = vec[7 * JT];
LocalTensor<T> pmax = vec[8 * JT];
LocalTensor<T> one = SmallBuf();
LocalTensor<uint8_t> work = HeadSlice<uint8_t>(hWork_);
LocalTensor<uint8_t> maskBits = SmallBufU8()[512];
LocalTensor<IN_T> qiIn = HeadSlice<IN_T>(hQIn_);
LocalTensor<IN_T> kiIn = HeadSlice<IN_T>(hKiIn_);
LocalTensor<T> wRow = HeadSlice<T>(hWRow_);
const DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
const uint64_t qiBase = (static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1) * NG * D_;
const uint64_t kiBase = static_cast<uint64_t>(b) * NJ * D_;
LoadSplitWeightsRow(b, s1);
ComputeMainBatchedRange(b, s1, mainH0_, mainH1_);
{
LocalTensor<T> pLocal = HeadSlice<T>(hP_);
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, static_cast<uint32_t>(nj * sizeof(T)), 0, 0, 0);
DataCopyPad(pPartGm_[blockIdx_ * 16u], pLocal, cp);
}
{
if constexpr (!kFp32 && FUSE_BIG) {
static_assert(static_cast<uint32_t>(NJ) == 2u * kKeyBatch,
"E78 P7 main key 预取覆盖全部 NJ 行");
(void)kiBase;
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
LocalTensor<IN_T> wiPf = HeadSlice<IN_T>(hWIn_);
LocalTensor<T> wrPf = HeadSlice<T>(hWRow_);
Cast(wrPf, wiPf, RoundMode::CAST_NONE, NG);
PipeBarrier<PIPE_V>();
} else if constexpr (kFp32) {
(void)kiBase;
(void)pad;
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V1);
} else {
(void)kiBase;
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V1);
}
if constexpr (!kFp32) {
Cast(ki, kiIn, RoundMode::CAST_NONE, nj * D_);
}
PipeBarrier<PIPE_V>();
if constexpr (kFp32) {
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
}
if constexpr (!kFp32 && FUSE_BIG) {
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
}
}
for (int32_t j = 0; j < nj; ++j) {
DataCopy(wRep[j * ngL], wRow[idxH0_], ngL);
}
PipeBarrier<PIPE_V>();
if constexpr (!kFp32) {
{
LocalTensor<T> prodAll = HeadSlice<T>(hKTile_);
for (int32_t g0 = 0; g0 < ngL; g0 += KCH_) {
const int32_t gt = ((ngL - g0) < KCH_) ? (ngL - g0) : KCH_;
if constexpr (kFp32) {
DataCopyExtParams cpQ(1, static_cast<uint32_t>(gt * D_ * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qiIn, queryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D_], cpQ, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
} else if constexpr (FUSE_BIG) {
} else {
(void)qiBase;
(void)idxH0_;
}
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qi, qiIn, gt * D_);
else Cast(qi, qiIn, RoundMode::CAST_NONE, gt * D_);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prodAll[j * ngL * D_ + off], qi[off], ki[j * D_ + off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(D_ / 8), 0));
}
PipeBarrier<PIPE_V>();
}
}
LocalTensor<T> arScratch = HeadSlice<T>(hWork_);
ReduceSumArHeadD128(sim, prodAll, arScratch, static_cast<uint32_t>(nj * ngL));
PipeBarrier<PIPE_V>();
}
} else {
static_assert(NGL <= KCH, "P6 first qi block is the only gt block");
(void)qiBase;
for (int32_t g0 = 0; g0 < ngL; g0 += KCH_) {
const int32_t gt = ((ngL - g0) < KCH_) ? (ngL - g0) : KCH_;
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], qi[off], ki[j * D_ + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(D_ / 8), 0));
}
PipeBarrier<PIPE_V>();
LocalTensor<T> arScratch = HeadSlice<T>(hWork_);
ReduceSumArHeadD128(sim[j * ngL + g0], prod, arScratch,
static_cast<uint32_t>(gt));
PipeBarrier<PIPE_V>();
}
}
}
Maxs(sim, sim, static_cast<T>(0.0f), nj * ngL);
PipeBarrier<PIPE_V>();
Mul(prod, sim, wRep, nj * ngL);
PipeBarrier<PIPE_V>();
{
uint32_t shI[2] = {static_cast<uint32_t>(nj), static_cast<uint32_t>(ngL)};
ReduceSum<T, Pattern::Reduce::AR, true>(ii, prod, work, shI, true);
}
PipeBarrier<PIPE_V>();
{
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cp(1, static_cast<uint32_t>(nj * sizeof(T)), 0, 0, 0);
DataCopyPad(iPartGm_[blockIdx_ * 16u], ii, cp);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
if constexpr (S1 > 1) {
if (blockIdx_ == 0u) {
lossGmRef.SetValue(0, 0.0F);
DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lossGmRef);
}
}
if (K > 1u) { SyncAll<true>(); }
{
constexpr uint32_t kIPOffset = (!kFp32 && FUSE_BIG) ? static_cast<uint32_t>(HP) * 16u : 64u;
static_assert(kIPOffset >= static_cast<uint32_t>(HP) * 16u,
"p 与 I 的部分和槽位不得重叠");
LocalTensor<T> pp = prod;
LocalTensor<T> ip = prod[kIPOffset];
const uint32_t slot0 = splitTask_ * headParts_;
DataCopyExtParams cp(1, static_cast<uint32_t>(headParts_ * 16u * sizeof(T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(pp, pPartGm_[slot0 * 16u], cp, DataCopyPadExtParams<T>(false, 0, 0, 0));
DataCopyPad(ip, iPartGm_[slot0 * 16u], cp, DataCopyPadExtParams<T>(false, 0, 0, 0));
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && FUSE_BIG) {
static_assert(!(FUSE_BIG && !kFp32) ||
(HP == 8 && NJ == 8 && JT == 8 &&
(static_cast<uint32_t>(HP) / 2u) * 16u == 64u),
"E66 P7 gather 折半树按 8 份/8 有效 lane/首层单 repeat 64 float 展开");
uint32_t r = static_cast<uint32_t>(HP);
while (r > 1u) {
const uint32_t half = r / 2u;
const uint64_t count = static_cast<uint64_t>(half) * 16u;
const int32_t hiOff = static_cast<int32_t>(half * 16u);
Add(pp, pp, pp[hiOff], count, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
Add(ip, ip, ip[hiOff], count, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
PipeBarrier<PIPE_V>();
r = half;
}
} else {
for (uint32_t c = 1u; c < headParts_; ++c) {
Add(pp, pp, pp[c * 16u], static_cast<uint64_t>(16u), 1,
BinaryRepeatParams(1, 1, 1, 2, 2, 2));
Add(ip, ip, ip[c * 16u], static_cast<uint64_t>(16u), 1,
BinaryRepeatParams(1, 1, 1, 2, 2, 2));
PipeBarrier<PIPE_V>();
}
}
Muls(p, pp, invN1_, JT);
PipeBarrier<PIPE_V>();
Copy(ii, ip, static_cast<uint64_t>(NJ), static_cast<uint8_t>(1),
CopyRepeatParams(1, 1, 0, 0));
PipeBarrier<PIPE_V>();
}
PipeBarrier<PIPE_V>();
{
uint32_t shMax[2] = {1u, static_cast<uint32_t>(nj)};
ReduceMax<T, Pattern::Reduce::AR, true>(one, ii, work, shMax, true);
}
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
const float mx = one.GetValue(0);
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
Adds(ii, ii, -mx, JT);
PipeBarrier<PIPE_V>();
Exp(ii, ii, JT);
PipeBarrier<PIPE_V>();
const float z = SumVecSplit(ii, nj);
const float psum = SumVecSplit(p, nj);
const float invZ = z > 0.0f ? 1.0f / z : 0.0f;
const float invP = psum > 0.0f ? 1.0f / psum : 0.0f;
Muls(y, ii, invZ, JT);
Muls(p, p, invP, JT);
PipeBarrier<PIPE_V>();
Maxs(pmax, p, static_cast<T>(LOSS_EPS), JT);
Maxs(ly, y, static_cast<T>(LOSS_EPS), JT);
PipeBarrier<PIPE_V>();
Log(lp, pmax, JT); Log(ly, ly, JT);
PipeBarrier<PIPE_V>();
Sub(tmp, lp, ly, JT); PipeBarrier<PIPE_V>();
Mul(tmp, tmp, pmax, JT); PipeBarrier<PIPE_V>();
lossSum_ += SumVecSplit(tmp, nj);
Sub(ii, y, p, JT);
PipeBarrier<PIPE_V>();
const int32_t maskCount = ((nj * ngL + 63) / 64) * 64;
CompareScalar(maskBits, sim, static_cast<T>(0.0f), CMPMODE::GT, maskCount);
PipeBarrier<PIPE_V>();
static_assert(NGL == 8 || NGL == 16, "公开 head 本核头片列数只有 8/16");
static_assert(!FUSE_BIG || NGL == 8, "E62 P6/P7 新路径本核列数恒为 8");
if constexpr (FUSE_BIG) {
Brcb(deltaRep, ii, 1, BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
} else {
SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
for (int32_t j = 0; j < nj; ++j) {
Duplicate(deltaRep[j * ngL], ii.GetValue(j), ngL);
}
SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
}
Mul(coef, wRep, deltaRep, nj * ngL);
PipeBarrier<PIPE_V>();
Select(coef, maskBits, coef, static_cast<T>(0.0f),
SELMODE::VSEL_TENSOR_SCALAR_MODE, maskCount);
PipeBarrier<PIPE_V>();
{
Mul(prod, sim, deltaRep, nj * ngL);
PipeBarrier<PIPE_V>();
int32_t r = nj;
while (r > 1) {
const int32_t half = r / 2;
Add(prod[0], prod[0], prod[(r - half) * ngL], static_cast<uint64_t>(ngL),
static_cast<uint8_t>(half),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(ngL / 8),
static_cast<uint8_t>(ngL / 8), static_cast<uint8_t>(ngL / 8)));
PipeBarrier<PIPE_V>();
r -= half;
}
const uint64_t wOff = (static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1) * NG + idxH0_;
if constexpr (kFp32) {
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpf(1, static_cast<uint32_t>(ngL * sizeof(float)), 0, 0, 0);
DataCopyPad(dWeightsGm[wOff], prod, cpf);
} else {
LocalTensor<OUT_T> outW = HeadSlice<OUT_T>(hOutW_);
Cast(outW, prod, RoundMode::CAST_RINT, ngL);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpw(1, static_cast<uint32_t>(ngL * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dWeightsGm[wOff], outW, cpw);
}
if constexpr (!(!kFp32 && FUSE_BIG)) {
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
for (int32_t g0 = 0; g0 < ngL; g0 += KCH_) {
const int32_t gt = ((ngL - g0) < KCH_) ? (ngL - g0) : KCH_;
Duplicate(dq, static_cast<T>(0.0f), gt * D_);
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
Brcb(cbrTmp, coef[j * ngL + g0], static_cast<uint8_t>(gt / 8), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && FUSE_BIG) {
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], cbrTmp, ki[j * D_ + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(1), 0));
PipeBarrier<PIPE_V>();
Add(dq[off], dq[off], prod[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(D_ / 8), static_cast<uint8_t>(D_ / 8)));
PipeBarrier<PIPE_V>();
}
} else {
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], cbrTmp, ki[j * D_ + off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(1), 0));
PipeBarrier<PIPE_V>();
Add(dq[off], dq[off], prod[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(D_ / 8), static_cast<uint8_t>(D_ / 8)));
PipeBarrier<PIPE_V>();
}
}
}
if constexpr (IsSameType<OUT_T, T>::value) {
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpOut(1, static_cast<uint32_t>(gt * D_ * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dQueryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D_], dq, cpOut);
} else {
LocalTensor<OUT_T> outQ = HeadSlice<OUT_T>(hOutQ_);
Cast(outQ, dq, RoundMode::CAST_RINT, gt * D_);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpOut(1, static_cast<uint32_t>(gt * D_ * sizeof(OUT_T)), 0, 0, 0);
DataCopyPad(dQueryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D_], outQ, cpOut);
}
if constexpr (!(!kFp32 && FUSE_BIG)) {
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
if constexpr (!kFp32) {
Duplicate(dk, static_cast<T>(0.0f), nj * D_);
PipeBarrier<PIPE_V>();
}
if constexpr ((kFp32 || !FUSE_BIG) || (!kFp32 && FUSE_BIG)) {
LocalTensor<T> prodAll = HeadSlice<T>(hKTile_);
const int32_t gt = ngL;
const uint64_t hStride = static_cast<uint64_t>(nj) * static_cast<uint64_t>(ngL) * 64ull;
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
Brcb(cbrTmp, coef[j * ngL], static_cast<uint8_t>(gt / 8), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && FUSE_BIG) {
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prodAll[static_cast<int32_t>(hStride) * h + j * ngL * 64], cbrTmp, qi[off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(1),
static_cast<uint8_t>(D_ / 8)));
}
} else {
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prodAll[static_cast<int32_t>(hStride) * h + j * ngL * 64], cbrTmp, qi[off],
static_cast<uint64_t>(64u), static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 0, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(1),
static_cast<uint8_t>(D_ / 8)));
}
}
PipeBarrier<PIPE_V>();
}
for (int32_t h = 0; h < D_ / 64; ++h) {
LocalTensor<T> base = prodAll[static_cast<int32_t>(hStride) * h];
int32_t r = gt;
while (r > 1) {
const int32_t half = r / 2;
for (int32_t g0 = 0; g0 < half; ++g0) {
Add(base[g0 * 64], base[g0 * 64], base[(g0 + half) * 64],
static_cast<uint64_t>(64u), static_cast<uint8_t>(nj),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(ngL * 64 / 8),
static_cast<uint8_t>(ngL * 64 / 8),
static_cast<uint8_t>(ngL * 64 / 8)));
}
PipeBarrier<PIPE_V>();
r -= half;
}
Adds(dk[h * 64], base, static_cast<T>(0.0f), static_cast<uint64_t>(64u),
static_cast<uint8_t>(nj),
UnaryRepeatParams(1, 1, static_cast<uint8_t>(D_ / 8),
static_cast<uint8_t>(ngL * 64 / 8)));
PipeBarrier<PIPE_V>();
}
} else {
for (int32_t g0 = 0; g0 < ngL; g0 += KCH_) {
const int32_t gt = ((ngL - g0) < KCH_) ? (ngL - g0) : KCH_;
if constexpr (S1 > 1) {
DataCopyExtParams cpQ(1, static_cast<uint32_t>(gt * D_ * sizeof(IN_T)), 0, 0, 0);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(qiIn, queryIndexGm[qiBase + static_cast<uint64_t>(idxH0_ + g0) * D_], cpQ, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
if constexpr (IsSameType<IN_T, T>::value) DataCopy(qi, qiIn, gt * D_);
else Cast(qi, qiIn, RoundMode::CAST_NONE, gt * D_);
}
PipeBarrier<PIPE_V>();
for (int32_t j = 0; j < nj; ++j) {
Brcb(cbrTmp, coef[j * ngL + g0], static_cast<uint8_t>(gt / 8), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
Brcb(cbr, cbrTmp, static_cast<uint8_t>(gt), BrcbRepeatParams(1, 8));
PipeBarrier<PIPE_V>();
for (int32_t h = 0; h < D_ / 64; ++h) {
const int32_t off = h * 64;
Mul(prod[off], cbr, qi[off], static_cast<uint64_t>(64u),
static_cast<uint8_t>(gt),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(D_ / 8)));
PipeBarrier<PIPE_V>();
int32_t r = gt;
while (r > 1) {
const int32_t half = r / 2;
Add(prod[off], prod[off], prod[off + half * 64], static_cast<uint64_t>(64u),
static_cast<uint8_t>(half),
BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8),
static_cast<uint8_t>(64 / 8)));
PipeBarrier<PIPE_V>();
r -= half;
}
Add(dk[j * D_ + off], dk[j * D_ + off], prod[off], static_cast<uint64_t>(64u), 1,
BinaryRepeatParams(1, 1, 1, 1, 1, 1));
PipeBarrier<PIPE_V>();
}
}
}
}
{
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyExtParams cpd(1, static_cast<uint32_t>(nj * D_ * sizeof(T)), 0, 0, 0);
const uint64_t dkSlot = static_cast<uint64_t>(blockIdx_) * nj * D_;
DataCopyPad(dkPartGm_[dkSlot], dk, cpd);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
__aicore__ inline void WriteDkFromParts() {
WriteDkFromPartsBatch();
}
__aicore__ inline void WriteDkFromPartsBatch() {
constexpr int32_t D_ = D;
const int32_t nj = NJ;
const uint32_t partsPerBatch = static_cast<uint32_t>(S1) * headParts_;
static_assert(static_cast<uint32_t>(S1) * (kBlockNum / kTotalTasks) == kDkParts,
"公开 head 小类每 batch dk 部分和份数须与编译常量一致");
static_assert(kKTileElems >= kDkParts * kKiElems,
"kTile 必须容纳 dk 部分和 4/8 份");
LocalTensor<T> acc = HeadSlice<T>(hAcc_);
LocalTensor<T> partsAll = HeadSlice<T>(hKTile_);
(void)acc;
LocalTensor<OUT_T> out = HeadSlice<OUT_T>(hOutQ_);
DataCopyExtParams cpAll(1, static_cast<uint32_t>(partsPerBatch * nj * D_ * sizeof(T)),
0, 0, 0);
DataCopyExtParams cpo(1, static_cast<uint32_t>(nj * D_ * sizeof(OUT_T)), 0, 0, 0);
const DataCopyPadExtParams<T> pad(false, 0, 0, 0);
for (uint32_t b = 0; b < kB; ++b) {
const uint64_t slot0 = static_cast<uint64_t>(b) * partsPerBatch;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(partsAll, dkPartGm_[slot0 * static_cast<uint64_t>(nj * D_)], cpAll, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
if constexpr (!kFp32 && FUSE_BIG) {
static_assert(!(FUSE_BIG && !kFp32) ||
(HP == 8 && kDkParts == 8u && (kKiElems % 64u) == 0u &&
(static_cast<uint32_t>(HP) / 2u) * kKiElems / 64u <= 255u),
"E66 P7 dk 折半树按 8 份/64 元素掩码/首层 repeat<=255 展开");
uint32_t r = static_cast<uint32_t>(HP);
while (r > 1u) {
const uint32_t half = r / 2u;
const uint32_t count = half * static_cast<uint32_t>(nj * D_);
const int32_t hiOff = static_cast<int32_t>(count);
Add(partsAll, partsAll, partsAll[hiOff], static_cast<uint64_t>(64u),
static_cast<uint8_t>(count / 64u), BinaryRepeatParams(1, 1, 1, 8, 8, 8));
PipeBarrier<PIPE_V>();
r = half;
}
} else {
for (uint32_t c = 1u; c < partsPerBatch; ++c) {
Add(partsAll, partsAll, partsAll[c * nj * D_], nj * D_);
PipeBarrier<PIPE_V>();
}
}
if constexpr (IsSameType<OUT_T, T>::value) DataCopy(out, partsAll, nj * D_);
else Cast(out, partsAll, RoundMode::CAST_RINT, nj * D_);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyPad(dKeyIndexGm[static_cast<uint64_t>(b) * nj * D_], out, cpo);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
__aicore__ inline void WriteDkFromPartsSerial() {
constexpr int32_t D_ = D;
const int32_t nj = NJ;
const uint32_t partsPerBatch = static_cast<uint32_t>(S1) * headParts_;
LocalTensor<T> acc = HeadSlice<T>(hAcc_);
LocalTensor<T> part = HeadSlice<T>(hPart_);
DataCopyExtParams cp(1, static_cast<uint32_t>(nj * D_ * sizeof(T)), 0, 0, 0);
DataCopyExtParams cpo(1, static_cast<uint32_t>(nj * D_ * sizeof(OUT_T)), 0, 0, 0);
const DataCopyPadExtParams<T> pad(false, 0, 0, 0);
for (uint32_t b = 0; b < kB; ++b) {
const uint64_t slot0 = static_cast<uint64_t>(b) * partsPerBatch;
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(acc, dkPartGm_[slot0 * static_cast<uint64_t>(nj * D_)], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
for (uint32_t c = 1u; c < partsPerBatch; ++c) {
SetFlag<HardEvent::V_MTE2>(EV_VMTE2); WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
DataCopyPad(part, dkPartGm_[(slot0 + c) * static_cast<uint64_t>(nj * D_)], cp, pad);
SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
PipeBarrier<PIPE_V>();
Add(acc, acc, part, nj * D_);
PipeBarrier<PIPE_V>();
}
SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
DataCopyPad(dKeyIndexGm[static_cast<uint64_t>(b) * nj * D_], acc, cpo);
SetFlag<HardEvent::MTE3_V>(EV_MTE3V); WaitFlag<HardEvent::MTE3_V>(EV_MTE3V);
}
}
__aicore__ inline float SumVecSplit(const LocalTensor<T> &v, int32_t n) {
LocalTensor<uint8_t> work = HeadSlice<uint8_t>(hWork_);
LocalTensor<T> one = SmallBuf();
uint32_t sh[2] = {1, static_cast<uint32_t>(n)};
ReduceSum<T, Pattern::Reduce::AR, true>(one, v, work, sh, true);
PipeBarrier<PIPE_V>();
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
return one.GetValue(0);
}
__aicore__ inline void LoadSplitWeightsRow(uint32_t b, uint32_t s1) {
LocalTensor<T> wr = HeadSlice<T>(hWRow_);
SetFlag<HardEvent::V_MTE2>(EV_VMTE2);
WaitFlag<HardEvent::V_MTE2>(EV_VMTE2);
const uint64_t off = (static_cast<uint64_t>(b) * static_cast<uint32_t>(S1) + s1) * NG;
if constexpr (kFp32) {
DataCopyExtParams cp32(1, static_cast<uint32_t>(NG * sizeof(T)), 0, 0, 0);
DataCopyPadExtParams<T> pad32(false, 0, 0, 0);
DataCopyPad(wr, weightsGm[off], cp32, pad32);
(void)wr;
} else {
DataCopyExtParams cp(1, static_cast<uint32_t>(NG * sizeof(IN_T)), 0, 0, 0);
DataCopyPadExtParams<IN_T> pad(false, 0, 0, 0);
LocalTensor<IN_T> wi = HeadSlice<IN_T>(hWIn_);
DataCopyPad(wi, weightsGm[off], cp, pad);
if constexpr (!FUSE_BIG) {
SetFlag<HardEvent::MTE2_V>(EV_MTE2V);
WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
Cast(wr, wi, RoundMode::CAST_NONE, NG);
PipeBarrier<PIPE_V>();
} else {
(void)wr;
}
}
if constexpr (!kFp32 && !FUSE_BIG) {
SetFlag<HardEvent::V_S>(EV_VS);
WaitFlag<HardEvent::V_S>(EV_VS);
}
}
static constexpr int32_t EV_MTE2V = 0;
static constexpr int32_t EV_VMTE3 = 1;
static constexpr int32_t EV_VS = 2;
static constexpr int32_t EV_SMTE3 = 3;
static constexpr int32_t EV_MTE2V1 = 4;
static constexpr int32_t EV_VMTE2 = 5;
static constexpr int32_t EV_MTE3V = 6;
static constexpr int32_t EV_SV = 7;
static constexpr uint32_t kScaleValueOff = static_cast<uint32_t>(
__builtin_offsetof(DenseLightningIndexerGradKlLossTilingData, scaleValue));
static constexpr uint32_t kInvN1Off = static_cast<uint32_t>(
__builtin_offsetof(DenseLightningIndexerGradKlLossTilingData, invN1));
static constexpr uint32_t kReservedOff = static_cast<uint32_t>(
__builtin_offsetof(DenseLightningIndexerGradKlLossTilingData, reserved));
static_assert(kScaleValueOff + sizeof(float) <= sizeof(DenseLightningIndexerGradKlLossTilingData),
"E83: scaleValue offset must stay inside the frozen 128B tiling ABI");
static_assert(kInvN1Off + sizeof(float) <= sizeof(DenseLightningIndexerGradKlLossTilingData),
"E83: invN1 offset must stay inside the frozen 128B tiling ABI");
static_assert(kReservedOff + 4u * sizeof(uint32_t) <= sizeof(DenseLightningIndexerGradKlLossTilingData),
"E83: reserved[1..3] offset must stay inside the frozen 128B tiling ABI");
static_assert(kScaleValueOff % sizeof(float) == 0u && kInvN1Off % sizeof(float) == 0u &&
kReservedOff % sizeof(uint32_t) == 0u,
"E83: tiling scalar offsets must be naturally aligned");
static constexpr uint32_t kSmallBytes = 1024u;
static constexpr uint32_t RawAlign512(uint64_t x) {
return static_cast<uint32_t>((x + 511u) / 512u * 512u);
}
static constexpr uint32_t RawArenaEnd() {
uint64_t off = 0;
off += RawAlign512(kLogitsElems * sizeof(T));
off += RawAlign512(16u * sizeof(T));
off += RawAlign512(kRepElems * sizeof(T));
off += RawAlign512(kKTileElems * sizeof(T));
off += RawAlign512(kDotElems * sizeof(T));
off += RawAlign512(kVecElems * sizeof(T));
off += RawAlign512(static_cast<uint32_t>(NL) * sizeof(T));
off += RawAlign512(static_cast<uint32_t>(NL) * sizeof(T));
if constexpr (kFp32 || FUSE_BIG) {
off += RawAlign512(kQidElems * sizeof(T));
off += RawAlign512(kKiElems * sizeof(T));
off += RawAlign512(kQidElems * sizeof(T));
off += RawAlign512(kKiElems * sizeof(T));
}
off += RawAlign512(static_cast<uint32_t>(NG) * sizeof(T));
off += RawAlign512(static_cast<uint32_t>(NG) * sizeof(IN_T));
off += RawAlign512(static_cast<uint32_t>(NG) * sizeof(T));
if constexpr (kFp32 || FUSE_BIG) {
off += RawAlign512(kOutQElems * sizeof(OUT_T));
off += RawAlign512(kKiElems * sizeof(T));
off += RawAlign512(kKiElems * sizeof(T));
}
off += RawAlign512(kSimElems * sizeof(T));
off += RawAlign512(kWorkBytes);
off += RawAlign512(kInQElems * sizeof(IN_T));
off += RawAlign512(kKInElems * sizeof(IN_T));
off += RawAlign512(kKIn2Bytes);
off += RawAlign512(kKiElems * sizeof(IN_T));
if constexpr (!kFp32 && !FUSE_BIG) {
off += RawAlign512(kSlabBytes);
}
return static_cast<uint32_t>(off);
}
static_assert(RawArenaEnd() + kSmallBytes <= 192u * 1024u,
"E83 raw head UB layout exceeds 910B 192KiB");
uint32_t arenaBytes_ = 0;
uint32_t smallOff_ = 0;
float scale_ = 0.0f;
float invN1_ = 0.0f;
uint32_t reserved1_ = 0, reserved2_ = 0, reserved3_ = 0;
uint32_t headOff_ = 0;
uint32_t hLogits_ = 0, hP_ = 0, hRep_ = 0, hKTile_ = 0, hDot_ = 0, hVec_ = 0;
uint32_t hMx_ = 0, hSum_ = 0, hQi_ = 0, hKi_ = 0, hDq_ = 0, hDk_ = 0;
uint32_t hWRow_ = 0, hWIn_ = 0, hOutW_ = 0, hOutQ_ = 0, hAcc_ = 0, hPart_ = 0;
uint32_t hSim_ = 0, hWork_ = 0, hQIn_ = 0, hKIn_ = 0, hKIn1_ = 0, hKiIn_ = 0, hSlab_ = 0;
float lossSum_ = 0.0f;
uint32_t blockIdx_ = 0;
uint32_t blockNum_ = 1;
uint32_t headParts_ = 1u, splitTask_ = 0u;
uint32_t mainH0_ = 0, mainH1_ = 0, idxH0_ = 0, idxH1_ = 0;
GlobalTensor<IN_T> queryGm, keyGm, queryIndexGm, keyIndexGm, weightsGm, dWeightsGm;
GlobalTensor<OUT_T> dQueryIndexGm, dKeyIndexGm;
GlobalTensor<T> lossGmRef;
GlobalTensor<T> pPartGm_, iPartGm_, dkPartGm_;
GM_ADDR tilingRaw_ = nullptr;
GM_ADDR wsRaw_ = nullptr;
};
template <typename DT_QUERY, bool FUSE_BIG, bool WIDE_FUSE, bool WIDTH64, bool LEAN_PUBLIC, bool HEAD_PUBLIC>
__global__ __aicore__ void dense_lightning_indexer_grad_kl_loss(
GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex, GM_ADDR weights,
GM_ADDR dQueryIndex, GM_ADDR dKeyIndex, GM_ADDR dWeights, GM_ADDR loss,
GM_ADDR workspace, GM_ADDR tiling) {
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
REGISTER_TILING_DEFAULT(DenseLightningIndexerGradKlLossTilingData);
static_assert(!(LEAN_PUBLIC && HEAD_PUBLIC), "LEAN_PUBLIC 与 HEAD_PUBLIC 互斥");
if constexpr (LEAN_PUBLIC) {
if constexpr (!WIDE_FUSE) {
if constexpr (kLeanRawB1) {
KernelLeanRawB1<half, WIDTH64> op;
op.Init(query, key, queryIndex, keyIndex, weights, dQueryIndex, dKeyIndex, dWeights,
loss, tiling);
op.Process();
} else {
GET_TILING_DATA_WITH_STRUCT(DenseLightningIndexerGradKlLossTilingData, tiling_data, tiling);
KernelLeanPublic<half, WIDTH64, false> op;
op.Init(query, key, queryIndex, keyIndex, weights, dQueryIndex, dKeyIndex, dWeights, loss,
GetUserWorkspace(workspace), &tiling_data);
op.Process();
}
} else {
if constexpr (!WIDTH64 && kLeanRawB2) {
KernelLeanRawB2<half> op;
op.Init(query, key, queryIndex, keyIndex, weights, dQueryIndex, dKeyIndex,
dWeights, loss, tiling);
op.Process();
} else {
GET_TILING_DATA_WITH_STRUCT(DenseLightningIndexerGradKlLossTilingData, tiling_data, tiling);
KernelLeanPublicDynamic<half, WIDTH64> op;
op.Init(query, key, queryIndex, keyIndex, weights, dQueryIndex, dKeyIndex, dWeights, loss,
GetUserWorkspace(workspace), &tiling_data);
op.Process();
}
}
} else if constexpr (HEAD_PUBLIC) {
static_assert(!(IsSameType<DT_QUERY, float>::value && !FUSE_BIG),
"HEAD_PUBLIC fp32 只对应家族 B");
static_assert(!WIDE_FUSE && !WIDTH64, "HEAD_PUBLIC 只在 128 宽短上下文家族");
constexpr bool kHeadRawThis =
IsSameType<DT_QUERY, float>::value ? kHeadRawP6 : (FUSE_BIG ? kHeadRawP7 : kHeadRawP5);
if constexpr (kHeadRawThis) {
KernelHeadRaw<DT_QUERY, FUSE_BIG> op;
op.Init(query, key, queryIndex, keyIndex, weights, dQueryIndex, dKeyIndex, dWeights, loss,
GetUserWorkspace(workspace), tiling);
op.Process();
} else {
GET_TILING_DATA_WITH_STRUCT(DenseLightningIndexerGradKlLossTilingData, tiling_data, tiling);
KernelHeadPublic<DT_QUERY, FUSE_BIG> op;
op.Init(query, key, queryIndex, keyIndex, weights, dQueryIndex, dKeyIndex, dWeights, loss,
GetUserWorkspace(workspace), &tiling_data);
op.Process();
}
} else {
GET_TILING_DATA_WITH_STRUCT(DenseLightningIndexerGradKlLossTilingData, tiling_data, tiling);
KernelDenseLightningIndexerGradKlLoss<DT_QUERY, FUSE_BIG, WIDE_FUSE, WIDTH64> op;
op.Init(query, key, queryIndex, keyIndex, weights, query, query, query, query,
query, query, query, dQueryIndex, dKeyIndex, dWeights, loss,
GetUserWorkspace(workspace), &tiling_data);
op.Process();
}
}
