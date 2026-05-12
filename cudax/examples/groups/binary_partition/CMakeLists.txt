cudax_add_example(example_target "binary_partition.cu")
target_compile_definitions(${example_target} PRIVATE _CUDAX_GROUP)
