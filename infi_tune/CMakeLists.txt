cmake_minimum_required(VERSION 2.8.3)
project(infi_tune)
find_package(catkin REQUIRED)

catkin_package()

catkin_python_setup()

# install
file(GLOB launch_files launch/*.launch)
install(FILES ${launch_files}
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
