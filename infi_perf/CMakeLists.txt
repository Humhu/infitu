cmake_minimum_required(VERSION 2.8.3)
project(infi_perf)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++0x")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS 
                    roscpp
                    argus_utils
                    infi_msgs
)

catkin_package( CATKIN_DEPENDS  roscpp
                                argus_utils
                                infi_msgs
)

include_directories( ${catkin_INCLUDE_DIRS} )

add_executable( ape_reward_node nodes/ape_reward.cpp )
add_dependencies( ape_reward_node ${catkin_EXPORTED_TARGETS} )
target_link_libraries( ape_reward_node ${catkin_LIBRARIES} )

add_executable( sse_reward_node nodes/sse_reward.cpp )
add_dependencies( sse_reward_node ${catkin_EXPORTED_TARGETS} )
target_link_libraries( sse_reward_node ${catkin_LIBRARIES} )

add_executable( oll_reward_node nodes/obs_log_likelihood_reward.cpp )
add_dependencies( oll_reward_node ${catkin_EXPORTED_TARGETS} )
target_link_libraries( oll_reward_node ${catkin_LIBRARIES} )

add_executable( reward_accumulator_node nodes/reward_accumulator.cpp )
add_dependencies( reward_accumulator_node ${catkin_EXPORTED_TARGETS} )
target_link_libraries( reward_accumulator_node ${catkin_LIBRARIES} )

install( TARGETS        ape_reward_node 
                        sse_reward_node 
                        oll_reward_node 
                        reward_accumulator_node 

         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark executables and/or libraries for installation
install( DIRECTORY           include/${PROJECT_NAME}/
         DESTINATION         ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
         FILES_MATCHING PATTERN "*.h *.hpp"
)
