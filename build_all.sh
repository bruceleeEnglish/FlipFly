#!/bin/bash

# Exit immediately if any command fails
set -e

# Record the current shell directory
# Run this script from the project root
CURRENT_DIR=$(pwd)

# Check that the current directory is the project root
if [ ! -f "${CURRENT_DIR}/CMakeLists.txt" ]; then
  echo "错误：当前目录不是项目根目录，没有找到 CMakeLists.txt"
  echo "请先 cd 到仓库根目录（CMakeLists.txt 所在目录）再执行本脚本"
  exit 1
fi

# Check that the local Eigen copy exists
if [ ! -d "${CURRENT_DIR}/third_party/eigen/Eigen" ]; then
  echo "错误：没有找到 Eigen 头文件目录"
  echo "请先下载 Eigen："
  echo "  mkdir -p third_party"
  echo "  cd third_party"
  echo "  git clone --branch 3.4.0 --depth 1 https://gitlab.com/libeigen/eigen.git"
  exit 1
fi

# Create the temporary directory under the current directory
TMP_DIR="${CURRENT_DIR}/tmp"

# Create the build directory under the current directory
BUILD_DIR="${CURRENT_DIR}/build"

# If build/ already exists, ask whether to delete the old build cache
if [ -d "$BUILD_DIR" ]; then
  echo "检测到已有编译目录: $BUILD_DIR"
  read -p "是否删除旧的 build 目录并重新编译？[y/N]: " CLEAN_BUILD

  case "$CLEAN_BUILD" in
    y|Y|yes|YES)
      echo "正在删除旧的编译目录: $BUILD_DIR"
      rm -rf "$BUILD_DIR"
      ;;
    *)
      echo "保留已有 build 目录，进行增量编译"
      ;;
  esac
fi

# Create the temporary directory
mkdir -p "$TMP_DIR"

# Create the build directory
mkdir -p "$BUILD_DIR"

# Enter the build directory
cd "$BUILD_DIR"

# Run cmake on the CMakeLists.txt of the parent directory
TMPDIR="$TMP_DIR" cmake ..

# Build the listed targets
TMPDIR="$TMP_DIR" make \
  hnsw_index \
  hnsw_search \
  hnsw_sample \
  hnsw_bruteforce \
  generate_ground_truth \
  check_result \
  -j4

echo "编译完成"
echo "当前目录: $CURRENT_DIR"
echo "编译目录: $BUILD_DIR"
echo "临时目录: $TMP_DIR"
echo "Eigen目录: ${CURRENT_DIR}/third_party/eigen"