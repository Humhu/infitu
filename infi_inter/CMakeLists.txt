cmake_minimum_required(VERSION 2.8.3)
project(infi_inter)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++0x")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

catkin_python_setup()

catkin_package( CATKIN_DEPENDS  rospy
                                argus_utils
                                paraset
                                dynamic_reconfigure
)
