#pragma once    

#include <array>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <math.h>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846

#define WINDOW_WIDTH 500
#define WINDOW_HEIGHT WINDOW_WIDTH

// WIDTH x HEIGHT
using FrameBuffer =  uint8_t*;
using DepthBuffer = float*;

struct Mat4 {
    float4 rows[4];

    __host__ __device__
    Mat4() {
            rows[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
            rows[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
            rows[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
            rows[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
};

typedef float4 Vec4;
typedef float3 Vec3;
typedef uchar3 Color;

struct minZUnary{
    __host__ __device__
    float operator()(const Vec4& v) const { return v.z; }
};

struct minZBinary {
    __host__ __device__
    float operator()(const float z1, const float z2) const { return fminf(z1, z2); }
};

namespace utilities{
    struct boundingBox {
        float minX, maxX, minY, maxY;
    };

    struct viewVolume {
        float N, F, L, R, T, B;
    };

    struct camera 
    {
        Vec3 base;
        float3 angles;
        utilities::viewVolume viewVolume;
    };

    struct object 
    {
        std::vector<Vec4> vertices;
        std::vector<int> faces;
        std::vector<Color> colors;
        float scale;
        Vec3 base;
        float3 angles;
    };

    struct light {
        Vec3 base;
        float3 angles;
    };

    void loadObject(const char *filename, object &obj);

    void initCamera(utilities::camera &cam, const utilities::viewVolume &viewVol);

    void initLight(utilities::light &light);
}

__host__ __device__ Vec4 vectorMultiply(const Mat4 a, const Vec4 b);

__host__ __device__ Mat4 matrixMultiply(const Mat4 a, const Mat4 b);

__host__ __device__ Vec4 canonicalToFullCoords(Vec4 homoCoord3d, utilities::viewVolume viewVolume);

__host__ __device__ Vec4 cartToCanvasCoords(Vec4 homoCoord3d);

// __device__ Vec4 canvasToCartCoords(Vec4 homoCoord3d);

__host__ __device__ Vec4 translateCoord(Vec4 homoCoord3d, float3 translate);

__host__ __device__ Vec4 rotateCoord(Vec4 homoCoord3d, float3 rotate);

__host__ __device__ Vec4 scaleCoord(Vec4 homoCoord3d, float scale);

__host__ __device__ void moveCamera(utilities::camera* camera, float3 translate, float3 rotate);

__host__ __device__ Vec4 modelToWorld(Vec4 homoCoord3d, float scale, float3 angles, Vec3 base);

__host__ __device__ Vec4 worldToCamera(Vec4 homoCoord3d, float3 angles, Vec3 base);

__host__ __device__ Vec4 orthographicProjectCoord(Vec4 homoCoord3d, utilities::viewVolume viewVolume);

__host__ __device__ Vec4 perspectiveProjectCoord(Vec4 homoCoord3d, utilities::viewVolume viewVolume);

__host__ __device__ float2 getBarycentric(Vec4* triangle, float2 point);

__host__ __device__ bool isValidBarycentric(float2 barycentric);

__host__ __device__ float getDepth(Vec4* triangle, float2 barycentric);

__host__ __device__ Color getColor(float2 barycentric, Color* colors);

__host__ __device__ utilities::boundingBox getBoundingBox(Vec4* triangle);

__host__ __device__ void updateAngles(float3* angles, float3 rotate);

__global__ void transformVerticesToCamKernel(
     Vec4* d_transformedVertices, int numVertices, float3 d_objAngles, Vec3 d_objBase,
    utilities::camera d_camera, float scale );

__global__ void transformCamToCanvasKernel(
    Vec4* d_transformedVertices, int numVertices, float3 offsetVector, bool perspective,
    utilities::viewVolume d_viewVol);

__global__ void rasterizeKernel(
    const Vec4* d_transformedVertices, // canvas coordinates 
    const int* d_faces,               
    const Color* d_face_colors,      
    uint8_t* d_frameBuffer,            
    float* d_depthBuffer,              
    int numFaces,
    Color bgColor
); 