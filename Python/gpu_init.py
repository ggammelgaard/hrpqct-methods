def startup_limit_gpu(gpu_mem_fraction=None):
    import tensorflow as tf
    if gpu_mem_fraction is None:
        gpu_options = tf.GPUOptions(allow_growth=True)
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    return sess


if __name__ == '__main__':
    startup_limit_gpu()
