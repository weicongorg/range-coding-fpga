
cmake_minimum_required (VERSION 3.0)
set(CMAKE_CXX_COMPILER "icpx")

project(seqarc-oneapi CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


# Intel supported FPGA cards
set(S10_PAC_BOARD_NAME "intel_s10sx_pac:pac_s10")
set(AGX_PAC_BOARD_NAME "de10_agilex:B2E2_8GBx4")
set(FPGA_DEVICE_NAME ${AGX_PAC_BOARD_NAME})
if (DEFINED FPGA_BOARD)
   set(FPGA_DEVICE_NAME ${FPGA_BOARD})
endif()
message(STATUS "Using board: ${FPGA_BOARD}")

# Compile flags
option(IGN_FAIL "Ignore timing failure" OFF)
option(HIGH_EFF "High effort compilation" OFF)

if (DEFINED USER_FLAG)
   MESSAGE(STATUS "User-added flag: ${USER_FLAG}")
endif()
set(COMPILE_SEED "")
if (DEFINED SEED)
   set(COMPILE_SEED "-Xsseed=${SEED}")
   MESSAGE(STATUS "Compile hardware with seed: ${SEED}")
endif()
set(EMULATOR_COMPILE_FLAGS "-fsycl -fintelfpga -g -O0 -qactypes ${USER_FLAG} \
-DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS    "-pthread -fsycl -fintelfpga -qactypes ")
set(HARDWARE_COMPILE_FLAGS "-fsycl -fintelfpga -qactypes")
set(HARDWARE_LINK_FLAGS    "-fsycl -fintelfpga -qactypes \
 -Xshardware -Xsopt-arg=-nocaching -Xsno-interleaving=DDR \
-Xsbsp-flow=flat -Xsboard=${FPGA_DEVICE_NAME} ${USER_FLAG} ${COMPILE_SEED}")
if(IGN_FAIL)
    string(APPEND HARDWARE_LINK_FLAGS " -Xstiming-failure-mode=ignore")
endif()
if(HIGH_EFF)
    string(APPEND HARDWARE_LINK_FLAGS "  -Xshigh-effort")
endif()
# -Xsglobal-ring -Xsforce-single-store-ring 
# -Xssysinteg-arg=--ring-max-requests-per-lsu 
# -Xssysinteg-arg=128

include_directories("${CMAKE_SOURCE_DIR}/include/")
include_directories("${CMAKE_SOURCE_DIR}/src/")

link_libraries(pthread)


macro(add_haredware_execuable HW_NAME)
    add_executable(${HW_NAME} EXCLUDE_FROM_ALL ${ARGN})
    set_target_properties(${HW_NAME} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
    set_target_properties(${HW_NAME} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})
endmacro()

macro(add_emulation_execuable EMU_NAME)
    add_executable(${EMU_NAME} EXCLUDE_FROM_ALL ${ARGN})
    set_target_properties(${EMU_NAME} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
    set_target_properties(${EMU_NAME} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})
endmacro()

macro(add_report RPT_NAME)
    add_executable(${RPT_NAME} EXCLUDE_FROM_ALL ${ARGN}) 
    set_target_properties(${RPT_NAME} PROPERTIES COMPILE_FLAGS "-DFPGA_REPORT ${HARDWARE_COMPILE_FLAGS}")
    set_target_properties(${RPT_NAME} PROPERTIES LINK_FLAGS "-fsycl-link ${HARDWARE_LINK_FLAGS}")
endmacro()

macro(add_fpga_target_set SET_NAME)
    add_emulation_execuable("${SET_NAME}.emu"   ${ARGN})
    add_report("${SET_NAME}.report"   ${ARGN})
    add_haredware_execuable("${SET_NAME}.fpga"   ${ARGN})
endmacro()


add_fpga_target_set(decoder  ${CMAKE_SOURCE_DIR}/src/main.cpp )


