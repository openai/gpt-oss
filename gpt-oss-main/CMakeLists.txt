cmake_minimum_required(VERSION 3.26)
project(gpt_oss LANGUAGES C CXX)

# If not defined externally, auto-detect
if(NOT DEFINED GPTOSS_BUILD_METAL)
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS "Apple Silicon detected → enabling GPTOSS_BUILD_METAL")
    set(GPTOSS_BUILD_METAL ON)
  else()
    message(STATUS "Non-Apple Silicon → disabling GPTOSS_BUILD_METAL")
    set(GPTOSS_BUILD_METAL OFF)
  endif()
else()
  message(STATUS "GPTOSS_BUILD_METAL manually set to: ${GPTOSS_BUILD_METAL}")
endif()

# Now declare it as a cache variable (respects user-provided value)
set(GPTOSS_BUILD_METAL "${GPTOSS_BUILD_METAL}" CACHE BOOL "Enable Metal backend")

if(GPTOSS_BUILD_METAL)
  enable_language(OBJC)
  add_subdirectory(gpt_oss/metal)
endif()
