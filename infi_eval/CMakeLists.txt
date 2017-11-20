cmake_minimum_required(VERSION 2.8.3)
project(infi_eval)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++0x")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS 
                    roscpp
                    argus_utils
                    rosbag_lib
                    infi_msgs
                    fieldtrack
)

catkin_package( CATKIN_DEPENDS  roscpp
                                argus_utils
                                rosbag_lib
                                fieldtrack
                                infi_msgs
                                fieldtrack
)

include_directories( ${catkin_INCLUDE_DIRS} )

add_executable( bag_replayer nodes/bag_replayer.cpp )
add_dependencies( bag_replayer ${catkin_EXPORTED_TARGETS} )
target_link_libraries( bag_replayer ${catkin_LIBRARIES} )

add_executable( multi_bag_replayer nodes/multi_bag_replayer.cpp )
add_dependencies( multi_bag_replayer ${catkin_EXPORTED_TARGETS} )
target_link_libraries( multi_bag_replayer ${catkin_LIBRARIES} )

install( TARGETS        bag_replayer 
                        multi_bag_replayer

         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
