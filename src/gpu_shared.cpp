#pragma once

#include "gpu_shared.h"

template <typename _type>
gpu_shared_ptr<_type> new_gpu_( int buffer ) {
    void *src = nullptr;
    int old_gpu_index;
    cudaGetDevice( &old_gpu_index );

    //spdlog::set_pattern("*** [%H:%M:%S %z] [thread %t] %v ***");
    cudaError_t ret = cudaMalloc( ( void ** )&src, buffer * sizeof( _type ) );
    SPDLOG_TRACE( "cudaMalloc std : {} {:p} {} : {} {} ", old_gpu_index, src, cudaGetErrorName( ret ), cudaGetErrorString( ret ), buffer * sizeof( _type ) );
    if ( ret != cudaError::cudaSuccess ) {
        SPDLOG_ERROR( "cudaMalloc error1 : {} : {}", cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
        return gpu_shared_ptr<_type>();
    }
    if ( src != nullptr ) {
        return gpu_shared_ptr<_type>( reinterpret_cast<_type *>( src ), cudaDeleter<_type>{} );
    }
    return gpu_shared_ptr<_type>();
}

template <typename _Ty>
class cudaHostDeleter {
public:
    cudaHostDeleter() {
    }
    void operator()( _Ty *val ) {
        int old_gpu_index = -1;
        cudaGetDevice( &old_gpu_index );

        //spdlog::set_pattern("*** [%H:%M:%S %z] [thread %t] %v ***");
        cudaError_t ret = cudaFreeHost( val );
        SPDLOG_TRACE( "cudaFree std : {} {:p} {} : {}", old_gpu_index, ( void * )val, cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
        if ( ret != cudaError::cudaSuccess ) {
            SPDLOG_ERROR( "cudaFree error2 : {} : {}", cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
        }
    }
};

template <typename _type>
std::shared_ptr<_type> new_cpu_( int buffer ) {
    void *src = nullptr;
    int old_gpu_index;
    cudaGetDevice( &old_gpu_index );

    //spdlog::set_pattern("*** [%H:%M:%S %z] [thread %t] %v ***");
    cudaError_t ret = cudaMallocHost( ( void ** )&src, buffer * sizeof( _type ) );
    SPDLOG_TRACE( "cudaMalloc std : {} {:p} {} : {} {} ", old_gpu_index, src, cudaGetErrorName( ret ), cudaGetErrorString( ret ), buffer * sizeof( _type ) );
    if ( ret != cudaError::cudaSuccess ) {
        SPDLOG_ERROR( "cudaMalloc error1 : {} : {}", cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
        return std::shared_ptr<_type>();
    }
    if ( src != nullptr ) {
        return std::shared_ptr<_type>( reinterpret_cast<_type *>( src ), cudaHostDeleter<_type>{} );
    }
    return std::shared_ptr<_type>();
}

#if 1
gpu_shared_ptr<unsigned char> new_gpu_uint8( int buffer ) {
    return new_gpu_<unsigned char>( buffer );
}

gpu_shared_ptr<float> new_gpu_float( int buffer ) {
    return new_gpu_<float>( buffer );
}

std::shared_ptr<unsigned char> new_cpu_uint8( int buffer ) {
    return new_cpu_<unsigned char>( buffer );
}
#else

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>

#include <iostream>
template <typename T>
class GPUBufferPool {
public:
    tbb::concurrent_unordered_map<size_t, tbb::concurrent_queue<T *> > bufferPools;
    tbb::concurrent_unordered_map<size_t, int> buffera;
    tbb::concurrent_unordered_map<size_t, int> bufferf;
    virtual ~GPUBufferPool() {
        // free all
        for ( auto it = bufferPools.begin(); it != bufferPools.end(); it++ ) {
            auto &q = it->second;
            T *value = nullptr;
            while ( q.try_pop( value ) ) {
                cudaError_t ret = cudaFree( value );
                // SPDLOG_TRACE( "cudaFree std :{:p} {} : {}", ( void * )val, cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
                if ( ret != cudaError::cudaSuccess ) {
                    // SPDLOG_ERROR( "cudaFree error2 : {} : {}", cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
                }
            }
        }
    }
    void Free( size_t size, T *value ) {
        bufferf[ size ]++;
        if ( !value )
            return;
        auto &pool = bufferPools[ size ];
        if ( pool.unsafe_size() > 100 ) {
            cudaError_t ret = cudaFree( value );
            // SPDLOG_TRACE( "cudaFree std :{:p} {} : {}", ( void * )val, cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
            if ( ret != cudaError::cudaSuccess ) {
                // SPDLOG_ERROR( "cudaFree error2 : {} : {}", cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
            }
        } else {
            pool.push( value );
        }
    }
    gpu_shared_ptr<T> Alloc( size_t size, int id = 0 ) {
        auto &pool = bufferPools[ size ];
        buffera[ size ]++;
        if ( buffera[ size ] % 10 == 0 ) {
            std::cout << "galloc : " << id << " " << size << " " << buffera[ size ] - bufferf[ size ] << " " << buffera[ size ] << " " << bufferf[ size ] << std::endl;
        }
        T *ptr = nullptr;
        if ( !pool.try_pop( ptr ) ) {
            cudaError_t ret = cudaMalloc( ( void ** )&ptr, size * sizeof( T ) );
            // SPDLOG_TRACE( "cudaMalloc std : {} {:p} {} : {} {} ", old_gpu_index, src, cudaGetErrorName( ret ), cudaGetErrorString( ret ), buffer * sizeof( _type ) );
            if ( ret != cudaError::cudaSuccess ) {
                // SPDLOG_ERROR( "cudaMalloc error1 : {} : {}", cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
                ptr = nullptr;
            }
        }
        if ( ptr != nullptr ) {
            return gpu_shared_ptr<T>( ptr, [this, size]( T *ptr ) { this->Free( size, ptr ); } );
        }
        return gpu_shared_ptr<T>();
    }
};

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/avassert.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

class CPUBufferPool {
public:
    tbb::concurrent_unordered_map<size_t, std::shared_ptr<AVBufferPool> > bufferPools;

    std::shared_ptr<unsigned char> Alloc( size_t size ) {
        std::shared_ptr<AVBufferPool> pool = bufferPools[ size ];
        if ( !pool ) {
            pool = bufferPools[ size ] = std::shared_ptr<AVBufferPool>( av_buffer_pool_init( size, nullptr ), []( AVBufferPool *pool ) {if(pool)av_buffer_pool_uninit(&pool); } );
        }
        if ( pool ) {
            AVBufferRef *buffer = av_buffer_pool_get( pool.get() );
            if ( buffer ) {
                return std::shared_ptr<unsigned char>( buffer->data, [buffer]( unsigned char *ptr ) mutable {if (buffer)av_buffer_unref(&buffer); } );
            }
        }
        return std::shared_ptr<unsigned char>();
    }
};

GPUBufferPool<unsigned char> gpuPoolUint8[ 8 ];
GPUBufferPool<float> gpuPoolFloat[ 8 ];
CPUBufferPool cpuPool;

gpu_shared_ptr<unsigned char> new_gpu_uint8( int buffer ) {
    int old_gpu_index = -1;
    cudaError_t ret = cudaGetDevice( &old_gpu_index );
    if ( ret != cudaError::cudaSuccess ) {
        return gpu_shared_ptr<unsigned char>();
    }
    return gpuPoolUint8[ old_gpu_index ].Alloc( buffer, old_gpu_index );
}

gpu_shared_ptr<float> new_gpu_float( int buffer ) {
    int old_gpu_index = -1;
    cudaError_t ret = cudaGetDevice( &old_gpu_index );
    if ( ret != cudaError::cudaSuccess ) {
        return gpu_shared_ptr<float>();
    }
    return gpuPoolFloat[ old_gpu_index ].Alloc( buffer, old_gpu_index );
}

std::shared_ptr<unsigned char> new_cpu_uint8( int buffer ) {
    return cpuPool.Alloc( buffer );
}

// gpu_shared_ptr<unsigned char> new_gpu_uint8( int buffer ) {
//     int old_gpu_index = -1;
//     cudaError_t ret = cudaGetDevice( &old_gpu_index );
//     if ( ret != cudaError::cudaSuccess ) {
//         return gpu_shared_ptr<unsigned char>();
//     }
//     return gpuPoolUint8[ old_gpu_index ].Alloc( buffer, old_gpu_index );
// }

// gpu_shared_ptr<float> new_gpu_float( int buffer ) {
//     int old_gpu_index = -1;
//     cudaError_t ret = cudaGetDevice( &old_gpu_index );
//     if ( ret != cudaError::cudaSuccess ) {
//         return gpu_shared_ptr<float>();
//     }
//     return gpuPoolFloat[ old_gpu_index ].Alloc( buffer, old_gpu_index );
// }

// std::shared_ptr<unsigned char> new_cpu_uint8( int buffer ) {
//     return cpuPool.Alloc( buffer );
// }
#endif
