cmake_minimum_required(VERSION 2.8.3)
project(infi_msgs)

find_package(catkin REQUIRED COMPONENTS
                    message_generation
                    geometry_msgs
                    std_msgs
                    sensor_msgs
)

## Generate service files
add_message_files( FILES
                   EpisodeBreak.msg
                   RewardStamped.msg
)

add_service_files( FILES
                   GetParameters.srv
                   SetParameters.srv
                   StartEvaluation.srv
                   StartSetup.srv
                   StartTeardown.srv
                   SetRecording.srv 
)

generate_messages( DEPENDENCIES 
                   std_msgs
                   sensor_msgs
)

catkin_package( CATKIN_DEPENDS message_runtime 
                               std_msgs 
                               sensor_msgs
)
