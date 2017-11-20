cmake_minimum_required(VERSION 2.8.3)
project(confeval)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++0x")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS 
                    roscpp
                    message_generation
                    argus_utils
                    std_msgs
                    rosbag
                    fieldtrack
)

add_service_files( FILES
                   StartEvaluation.srv
                   StartSetup.srv
                   StartTeardown.srv
                   SetRecording.srv
)

generate_messages()

# Use Boost for most utilities, threading
find_package(Boost REQUIRED COMPONENTS random)

catkin_package( CATKIN_DEPENDS  roscpp
                                message_runtime
                                argus_utils
                                std_msgs
                                rosbag
                                fieldtrack

                DEPENDS         Boost
                                system_lib
)

add_executable( bag_replayer nodes/bag_replayer.cpp )
target_link_libraries( bag_replayer ${catkin_LIBRARIES} )

add_executable( multi_bag_replayer nodes/multi_bag_replayer.cpp )
target_link_libraries( multi_bag_replayer ${catkin_LIBRARIES} )

install( TARGETS        bag_replayer 
                        multi_bag_replayer

         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
