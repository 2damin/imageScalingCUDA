#pragma once
#include <memory>
#include <spdlog/spdlog.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename _Ty>
class cudaDeleter {
public:
    cudaDeleter() {
    }
    void operator()( _Ty *val ) {
        int old_gpu_index = -1;
        cudaGetDevice( &old_gpu_index );

        //spdlog::set_pattern("*** [%H:%M:%S %z] [thread %t] %v ***");
        cudaError_t ret = cudaFree( val );
        SPDLOG_TRACE( "cudaFree std :{:p} {} : {}", ( void * )val, cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
        if ( ret != cudaError::cudaSuccess ) {
            SPDLOG_ERROR( "cudaFree error2 : {} : {}", cudaGetErrorName( ret ), cudaGetErrorString( ret ) );
        }
    }
};

template <typename _Ty>
class gpu_shared_ptr : public std::shared_ptr<_Ty>, public std::enable_shared_from_this<gpu_shared_ptr<_Ty>> {
public:
    gpu_shared_ptr() {
    }
    template <typename _Tz>
    gpu_shared_ptr( _Ty *ptr, _Tz del ) : std::shared_ptr<_Ty>( ptr, del ) {
    }
    void operator=( const std::shared_ptr<_Ty> & ) = delete;
};

gpu_shared_ptr<unsigned char> new_gpu_uint8( int buffer );
gpu_shared_ptr<float> new_gpu_float( int buffer );

std::shared_ptr<unsigned char> new_cpu_uint8( int buffer );
