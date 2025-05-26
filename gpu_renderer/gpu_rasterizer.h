#include <cuda_runtime.h>
#include <math.h>
#include "gpu_utilities.h"


__device__ Vec4 vectorMultiply(const Mat4 a, const Vec4 b)
{
    Vec4 resultVector;
    resultVector.x = a.rows[0].x * b.x + a.rows[0].y * b.y + a.rows[0].z * b.z + a.rows[0].w * b.w;
    resultVector.y = a.rows[1].x * b.x + a.rows[1].y * b.y + a.rows[1].z * b.z + a.rows[1].w * b.w;
    resultVector.z = a.rows[2].x * b.x + a.rows[2].y * b.y + a.rows[2].z * b.z + a.rows[2].w * b.w;
    resultVector.w = a.rows[3].x * b.x + a.rows[3].y * b.y + a.rows[3].z * b.z + a.rows[3].w * b.w;
    return resultVector;
}

__device__ Mat4 matrixMultiply(const Mat4 a, const Mat4 b)
{
    Mat4 resultMat= Mat4();
    for (int i = 0; i < 4; ++i) {
        resultMat.rows[i].x = a.rows[i].x * b.rows[0].x + a.rows[i].y * b.rows[1].x + a.rows[i].z * b.rows[2].x + a.rows[i].w * b.rows[3].x;
        resultMat.rows[i].y = a.rows[i].x * b.rows[0].y + a.rows[i].y * b.rows[1].y + a.rows[i].z * b.rows[2].y + a.rows[i].w * b.rows[3].y;
        resultMat.rows[i].z = a.rows[i].x * b.rows[0].z + a.rows[i].y * b.rows[1].z + a.rows[i].z * b.rows[2].z + a.rows[i].w * b.rows[3].z;
        resultMat.rows[i].w = a.rows[i].x * b.rows[0].w + a.rows[i].y * b.rows[1].w + a.rows[i].z * b.rows[2].w + a.rows[i].w * b.rows[3].w;
    }
    return resultMat;
}

__device__ Vec4 canonicalToFullCoords(Vec4 homoCoord3d, utilities::viewVolume viewVolume) {
    const float L = viewVolume.L;
    const float R = viewVolume.R;
    const float T = viewVolume.T;
    const float B = viewVolume.B;

    Mat4 mat;
    mat.rows[0] = make_float4((R-L)/2, 0, 0, (L+R)/2);
    mat.rows[1] = make_float4(0, (T-B)/2, 0, (T+B)/2);
    mat.rows[2] = make_float4(0, 0, 1, 0);
    mat.rows[3] = make_float4(0, 0, 0, 1);

    return vectorMultiply(mat, homoCoord3d);
}

__device__ Vec4 cartToCanvasCoords(Vec4 homoCoord3d)
{
    Mat4 mat1;
    mat1.rows[0] = make_float4(1, 0, 0, 0);
    mat1.rows[1] = make_float4(0, -1, 0, 0), 
    mat1.rows[2] = make_float4(0, 0, 1, 0), 
    mat1.rows[3] = make_float4(0, 0, 0, 1);

    Mat4 mat2;
    mat2.rows[0] = make_float4(1, 0, 0, WINDOW_WIDTH/2);
    mat2.rows[1] = make_float4(0, 1, 0, WINDOW_HEIGHT/2);
    mat2.rows[2] = make_float4(0, 0, 1, 0);
    mat2.rows[3] = make_float4(0, 0, 0, 1);
    return vectorMultiply(mat2, vectorMultiply(mat1, homoCoord3d));
}

// __device__ Vec4 canvasToCartCoords(Vec4 homoCoord3d) {
//     Mat4 mat1;
//     mat1.rows[0] = make_float4(1, 0, 0, 0);
//     mat1.rows[1] = make_float4(0, -1, 0, 0), 
//     mat1.rows[2] = make_float4(0, 0, 1, 0), 
//     mat1.rows[3] = make_float4(0, 0, 0, 1);

//     Mat4 mat2;
//     mat2.rows[0] = make_float4(1, 0, 0, -WINDOW_WIDTH/2);
//     mat2.rows[1] = make_float4(0, 1, 0, WINDOW_HEIGHT/2);
//     mat2.rows[2] = make_float4(0, 0, 1, 0);
//     mat2.rows[3] = make_float4(0, 0, 0, 1);

//     return vectorMultiply(mat2, vectorMultiply(mat1, homoCoord3d));
// }

__device__ Vec4 translateCoord(Vec4 homoCoord3d, float3 translate) {
    float dx = translate.x;
    float dy = translate.y;
    float dz = translate.z;

    Mat4 mat;
    mat.rows[0] = make_float4(1, 0, 0, dx);
    mat.rows[1] = make_float4(0, 1, 0, dy);
    mat.rows[2] = make_float4(0, 0, 1, dz);
    mat.rows[3] = make_float4(0, 0, 0, 1);

    return vectorMultiply(mat, homoCoord3d);
}

__device__ Vec4 rotateCoord(Vec4 homoCoord3d, float3 rotate) {
    float rx = rotate.x;
    float ry = rotate.y;
    float rz = rotate.z;
    Mat4 rotateMatX;
    rotateMatX.rows[0] = make_float4(1, 0, 0, 0);
    rotateMatX.rows[1] = make_float4(0, cos(rx), -sin(rx), 0); 
    rotateMatX.rows[2] = make_float4(0, sin(rx), cos(rx), 0); 
    rotateMatX.rows[3] = make_float4(0, 0, 0, 1);

    Mat4 rotateMatY;
    rotateMatY.rows[0] = make_float4(cos(ry), 0, sin(ry), 0);
    rotateMatY.rows[1] = make_float4(0, 1, 0, 0); 
    rotateMatY.rows[2] = make_float4(-sin(ry), 0, cos(ry), 0);
    rotateMatY.rows[3] = make_float4(0, 0, 0, 1);

    Mat4 rotateMatZ;
    rotateMatZ.rows[0] = make_float4(cos(rz), -sin(rz), 0, 0); 
    rotateMatZ.rows[1] = make_float4(sin(rz), cos(rz), 0, 0); 
    rotateMatZ.rows[2] = make_float4(0, 0, 1, 0); 
    rotateMatZ.rows[3] = make_float4(0, 0, 0, 1);
    
    Vec4 newCoord = homoCoord3d;
    newCoord = vectorMultiply(rotateMatX, newCoord);
    newCoord = vectorMultiply(rotateMatY, newCoord);
    newCoord = vectorMultiply(rotateMatZ, newCoord);
    return newCoord;
}

__device__ Vec4 scaleCoord(Mat4 homoCoord3d, float scale) {
    Mat4 mat;
    mat.rows[0] = make_float4(scale, 0, 0, 0);
    mat.rows[1] = make_float4(0, scale, 0, 0);
    mat.rows[2] = make_float4(0, 0, scale, 0);
    mat.rows[3] = make_float4(0, 0, 0, 1);
    return vectorMultiply(mat, homoCoord3d);
}

__device__ void moveCamera(Camera* camera, float3 translate, float3 rotate){
    camera->base = translateCoord(camera->base, translate);
    camera->base = rotateCoord(camera->base, rotate);
    updateAngles(&(camera->angles), rotate)
}

__device__ Vec4 modelToWorld(Vec4 homoCoord3d, float scale, float3 angles, Vec3 base){
    float3 flatBase = make_float3{base.x, base.y, base.z};
    Vec4 worldCoord = homoCoord3d;
    worldCoord = rotateCoord(worldCoord, angles);
    worldCoord = translateCoord(worldCoord, flatBase);
    worldCoord = scaleCoord(worldCoord, scale);
    return worldCoord;
}

__device__ Vec4 worldToCamera(Vec4 homoCoord3d, float3 angles, Vec3 base){
    //translate then rotate
    float3 negBase = make_float3(-base.x, -base.y, -base.z);
    float3 negAngles = {-angles.x, -angles.y, -angles.z};
    Vec4 camCoord = homoCoord3d;
    camCoord = translateCoord(camCoord, negBase);
    camCoord = rotateCoord(camCoord, negAngles);
    return camCoord;
}

__device__ Vec4 orthographicProjectCoord(Vec4 homoCoord3d, utilities::viewVolume viewVolume) {
    float N = viewVolume.N;
    float F = viewVolume.F;
    float L = viewVolume.L;
    float R = viewVolume.R;
    float T = viewVolume.T;
    float B = viewVolume.B;
    Mat4 projectMat;
    projectMat.rows[0] = make_float4(2/(R-L), 0, 0, 0);
    projectMat.rows[1] = make_float4(0, 2/(T-B), 0, 0);
    projectMat.rows[2] = make_float4(0, 0, 1/(F-N), -N/(F-N));
    projectMat.rows[3] = make_float4(0, 0, 0, 1);
    return vectorMultiply(projectMat, homoCoord3d);
}

__device__ Vec4 perspectiveProjectCoord(Vec4 homoCoord3d, utilities::viewVolume viewVolume) {
    float N = viewVolume.N;
    float F = viewVolume.F;
    float L = viewVolume.L;
    float R = viewVolume.R;
    float T = viewVolume.T;
    float B = viewVolume.B;
    Mat4 projectMat;
    projectMat.rows[0] = make_float4((2*N)/(R-L), 0, 0, 0);
    projectMat.rows[1] = make_float4(0, (2*N)/(T-B), 0, 0);
    projectMat.rows[2] = make_float4(0, 0, F/(F-N), -(F*N)/(F-N));
    projectMat.rows[3] = make_float4(0, 0, 1, 0);

    Vec4 projectedCoord = vectorMultiply(projectMat, homoCoord3d);

    float scale = projectedCoord.w;
    Mat4 scaleMat;
    scaleMat.rows[0] = make_float4(1/scale, 0, 0, 0);
    scaleMat.rows[1] = make_float4(0, 1/scale, 0, 0);
    scaleMat.rows[2] = make_float4(0, 0, 1/scale, 0);
    scaleMat.rows[3] = make_float4(0, 0, 0, 1/scale);
    return vectorMultiply(scaleMat, projectedCoord);
}

float2 getBarycentric(Vec4* triangle, float2 point){
    float diff[2] = {point.x - triangle[0].x, point.y - triangle[0].y};
    float mat[2][2] = {
        {triangle[1].x - triangle[0].x, triangle[2].x - triangle[0].x},
        {triangle[1].y - triangle[0].y, triangle[2].y - triangle[0].y}
    };
    // get the inverse of the matrix
    float det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    if (det == 0) {
        return make_float2(-1, -1);
    }
    float inverted[2][2] = {{mat[1][1] / det, -mat[0][1] / det}, {-mat[1][0] / det, mat[0][0] / det}};

    float2 baryVector;
    baryVector.x = inverted[0][0] * diff[0]  + inverted[0][1] * diff[1]; 
    baryVector.y = inverted[1][0] * diff[0]  + inverted[1][1] * diff[1];
    return baryVector;
} 

__device__ bool isValidBarycentric(float2 barycentric){
    float m2 = barycentric.x;
    float m3 = barycentric.y;
    return (m2 >= 0 && m3 >= 0 && m2 + m3 <= 1);
}

__device__ float getDepth(Vec4* triangle, float2 barycentric){
    float m2 = barycentric.x;
    float m3 = barycentric.y;
    float depth = triangle[0].z * (1 - m2 - m3) + triangle[1].z * m2 + triangle[2].z * m3;
    return depth;
}

__device__ Color getColor(float2 barycentric, Color* colors){
    float m2 = barycentric.x;
    float m3 = barycentric.y;
    float m1 = 1 - m2 - m3;
    if ((m1 <1e-2) || (m2 <1e-2) || (m3 <1e-2))
        return make_uchar3(0, 0, 0);
    else {
        uint8_t r = floor(colors[0].x * m1 + colors[1].x * m2 + colors[2].x * m3);
        uint8_t g = floor(colors[0].y * m1 + colors[1].y * m2 + colors[2].y * m3);
        uint8_t b = floor(colors[0].z * m1 + colors[1].z * m2 + colors[2].z * m3);
        return make_uchar3(r, g, b);
    }
}

__device__ utilities::boundingBox getBoundingBox(Vec4* triangle){
    float minX = triangle[0].x;
    float maxX = triangle[0].x;
    float minY = triangle[0].y;
    float maxY = triangle[0].y;
    float minZ = triangle[0].z;
    float maxZ = triangle[0].z;
    for (int i=1; i<3; i++){
        minX = fminf(triangle[i].x, minX);
        maxX = fmaxf(triangle[i].x, maxX);
        minY = fminf(triangle[i].y, minY);
        maxY = fmaxf(triangle[i].y, maxY);
        minZ = fminf(triangle[i].z, minZ);
        maxZ = fmaxf(triangle[i].z, maxZ);
    }
    return {minX, maxX, minY, maxY, minZ, maxZ};
}