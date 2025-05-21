project('PArUco', 'cpp', version: '0.9.0', default_options: ['warning_level=3', 'cpp_std=c++20'])

dependencies = [
    dependency('opencv4'),
    dependency('tbb')
]

libs = both_libraries(
    'paruco', 'paruco.cpp',
    dependencies: dependencies,
    install: true,
    version: meson.project_version()
)

executable(
    'paruco-demo',
    'paruco-demo.cpp',
    link_with: libs.get_shared_lib(),
    dependencies: dependencies + [dependency('genicvbridge')]
)

executable(
    'paruco-bench',
    'paruco-bench.cpp',
    link_with: libs.get_shared_lib(),
    dependencies: dependencies + [dependency('benchmark')]
)

install_headers('paruco.hpp')
