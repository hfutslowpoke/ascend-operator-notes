#!/bin/bash

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

if [ ! -f "${_ASCEND_INSTALL_PATH}/bin/setenv.bash" ]; then
    echo "ERROR: can not find ${_ASCEND_INSTALL_PATH}/bin/setenv.bash"
    exit 1
fi

source "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"
export DDK_PATH=${_ASCEND_INSTALL_PATH}
export NPU_HOST_LIB=${_ASCEND_INSTALL_PATH}/lib64

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"
CASE="single_case"
SEED=${ACLNN_FLOAT32_SEED:-2025}

rm -rf "$HOME/ascend/log/"*
rm -f "${SCRIPT_DIR}"/input/*.bin 2>/dev/null || true
rm -f "${SCRIPT_DIR}"/output/*.bin 2>/dev/null || true

cd "${SCRIPT_DIR}"
python3 gen_data.py --seed "${SEED}" --case "${CASE}"
if [ $? -ne 0 ]; then
    echo "ERROR: generate input data failed!"
    exit 1
fi
echo "INFO: generate input data success!"

set -e
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cmake -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

(
    cd "${BUILD_DIR}"
    ./execute_segment_min_csr_float32 0 "${CASE}" "${SCRIPT_DIR}/input" "${SCRIPT_DIR}/output"
)

python3 verify_result.py \
    "output/output_y_${CASE}.bin" \
    "output/golden_${CASE}.bin"

if [ $? -eq 0 ]; then
    echo ""
    echo "#####################################"
    echo "INFO: SegmentMinCsr precision PASS!"
    echo "#####################################"
    echo ""
fi
