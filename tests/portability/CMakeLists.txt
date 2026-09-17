cmake_minimum_required(VERSION 3.22)
project(InferFluxTemplatePortability LANGUAGES CXX)

# Exercise the same renderer tests on MSVC without pulling in model/GPU dependencies.
set(INFERFLUX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
add_executable(template_portability_tests
  "${INFERFLUX_ROOT}/model/chat_template_renderer.cpp"
  "${INFERFLUX_ROOT}/tests/unit/test_chat_template_renderer.cpp"
  "${INFERFLUX_ROOT}/external/catch2/catch_amalgamated.cpp")
target_compile_features(template_portability_tests PRIVATE cxx_std_17)
target_include_directories(template_portability_tests PRIVATE
  "${INFERFLUX_ROOT}" "${INFERFLUX_ROOT}/external")
enable_testing()
add_test(NAME TemplatePortability COMMAND template_portability_tests)
