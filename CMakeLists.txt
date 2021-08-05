cmake_minimum_required(VERSION 3.0.0)
project(
    netflix
    VERSION 0.1.0
    LANGUAGES C
)
set(CMAKE_C_STANDARD 11)

include(CTest)
enable_testing()

set(
    SOURCE_FILES
    dataset.c
    main.c
    model.c
    utils.c
)

add_executable(
    ${PROJECT_NAME}
    ${SOURCE_FILES}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

cmake_minimum_required(VERSION 3.0.0)
