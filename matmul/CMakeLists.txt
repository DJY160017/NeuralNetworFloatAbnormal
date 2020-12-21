cmake_minimum_required(VERSION 3.10)
project(matmul)

set(CMAKE_CXX_STANDARD 11)

# system header files
set(INC_DIR /usr/local/include)
# system lib
set(LINK_DIR /usr/local/lib)

#add_executable(tensorflow_c___demo main.cpp)
set(SOURCE_FILES iRRAM_test.cpp)

#设置版本为1.15
set(TENSORFLOW_VERSION tensorflow1.15)

#设置为未编译版本
set(TENSORFLOW_TYPE common)

##引入tensotflow编译结果的根目录
set(TENSORFLOW_ROOT_DIR /home/byron/Documents/workspace/tensorflow)

#引入tensotflow编译结果的根目录
#set(TENSORFLOW_ROOT_DIR /home/byron/Desktop/tensorflow_improved)

# 设置源文件的引用目录
include_directories(
        ${TENSORFLOW_ROOT_DIR}
        ${TENSORFLOW_ROOT_DIR}/bazel-genfiles
        ${TENSORFLOW_ROOT_DIR}/bazel-out
        ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/gen/proto
        ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/gen/host_obj
        ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/gen/protobuf/include
        ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/downloads/nsync/public
        ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/downloads/eigen
        ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/downloads/absl
        ${INC_DIR}
)

#lib
link_directories(
        ${TENSORFLOW_ROOT_DIR}/bazel-bin/tensorflow
        ${LINK_DIR}
)

# 设置程序启动入口，后期可能指定某个包
add_executable(matmul ${SOURCE_FILES})

# 设置需要链接的tensorflow库
target_link_libraries(matmul
        mpfr
        gmp
        iRRAM
        ${TENSORFLOW_ROOT_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so
        ${TENSORFLOW_ROOT_DIR}/bazel-bin/tensorflow/libtensorflow_framework.so.2
        )
