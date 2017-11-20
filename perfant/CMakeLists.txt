cmake_minimum_required(VERSION 2.8.3)
project(pquant)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++0x")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS 
                    roscpp
                    argus_utils
)

# Use Boost for most utilities, threading
find_package(Boost REQUIRED COMPONENTS random)

catkin_package( CATKIN_DEPENDS  roscpp
                                argus_utils

                DEPENDS         Boost
                                system_lib
)

add_executable( ape_reward_node nodes/rewards/ape_reward.cpp )
target_link_libraries( ape_reward_node ${catkin_LIBRARIES} )

add_executable( sse_reward_node nodes/rewards/sse_reward.cpp )
target_link_libraries( sse_reward_node ${catkin_LIBRARIES} )

add_executable( oll_reward_node nodes/rewards/obs_log_likelihood_reward.cpp )
target_link_libraries( oll_reward_node ${catkin_LIBRARIES} )

add_executable( reward_accumulator_node nodes/rewards/reward_accumulator.cpp )
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
