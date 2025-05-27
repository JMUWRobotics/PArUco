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

genicvbridge = dependency('genicvbridge', required: false, not_found_message: 'genicvbridge not available. not building paruco-demo')
if genicvbridge.found()
    executable(
        'paruco-demo',
        'paruco-demo.cpp',
        link_with: libs.get_shared_lib(),
        dependencies: dependencies + genicvbridge
    )
endif

googlebench = dependency('benchmark', required: false, not_found_message: 'googlebenchmark library not found. not building paruco-bench')
if googlebench.found()
    executable(
        'paruco-bench',
        'paruco-bench.cpp',
        link_with: libs.get_shared_lib(),
        dependencies: dependencies + googlebench
    )
endif

install_headers('paruco.hpp')
