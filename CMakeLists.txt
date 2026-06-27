cmake_minimum_required(VERSION 3.16)

project(libtorchedclips LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(libtorchedclips SHARED
  libtorchedclips_global.h
  libtorchedclips.cpp
  libtorchedclips.h
)

target_compile_definitions(libtorchedclips PRIVATE LIBTORCHEDCLIPS_LIBRARY)
