cmake_minimum_required(VERSION 2.8.3)
project(infitu)

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

add_executable( belief_state_broadcaster nodes/belief_state_broadcaster.cpp )
target_link_libraries( belief_state_broadcaster ${catkin_LIBRARIES} )


install( TARGETS        belief_state_broadcaster

         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark executables and/or libraries for installation
install( DIRECTORY           include/${PROJECT_NAME}/
         DESTINATION         ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
         FILES_MATCHING PATTERN "*.h *.hpp"
)
