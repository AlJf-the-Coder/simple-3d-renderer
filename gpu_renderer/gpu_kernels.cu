#include "gpu_utilities.h"

__host__ __device__ Vec4 vectorMultiply(const Mat4 a, const Vec4 b)
{
    Vec4 resultVector;
    resultVector.x = a.rows[0].x * b.x + a.rows[0].y * b.y + a.rows[0].z * b.z + a.rows[0].w * b.w;
    resultVector.y = a.rows[1].x * b.x + a.rows[1].y * b.y + a.rows[1].z * b.z + a.rows[1].w * b.w;
    resultVector.z = a.rows[2].x * b.x + a.rows[2].y * b.y + a.rows[2].z * b.z + a.rows[2].w * b.w;
    resultVector.w = a.rows[3].x * b.x + a.rows[3].y * b.y + a.rows[3].z * b.z + a.rows[3].w * b.w;
    return resultVector;
}

__host__ __device__ Mat4 matrixMultiply(const Mat4 a, const Mat4 b)
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

__host__ __device__ Mat4 getCanonicalToFullMatrix(utilities::viewVolume viewVolume) {
    const float L = viewVolume.L;
    const float R = viewVolume.R;
    const float T = viewVolume.T;
    const float B = viewVolume.B;

    Mat4 mat;
    mat.rows[0] = make_float4((R-L)/2, 0, 0, (L+R)/2);
    mat.rows[1] = make_float4(0, (T-B)/2, 0, (T+B)/2);
    mat.rows[2] = make_float4(0, 0, 1, 0);
    mat.rows[3] = make_float4(0, 0, 0, 1);

    return mat;
}

__host__ __device__ Mat4 getCartToCanvasMatrix() {
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
    return matrixMultiply(mat2, mat1);
}

// __device__ Mat4 canvasToCartCoords(Vec4 homoCoord3d) {
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

    // return matrixMultiply(mat2, mat1);
// }

__host__ __device__ Mat4 getTranslateMatrix(float3 translate) {
    float dx = translate.x;
    float dy = translate.y;
    float dz = translate.z;

    Mat4 mat;
    mat.rows[0] = make_float4(1, 0, 0, dx);
    mat.rows[1] = make_float4(0, 1, 0, dy);
    mat.rows[2] = make_float4(0, 0, 1, dz);
    mat.rows[3] = make_float4(0, 0, 0, 1);

    return mat;
}

__host__ __device__ Mat4 getRotateMatrix(float3 rotate) {
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
    
    return matrixMultiply(rotateMatZ, matrixMultiply(rotateMatY, rotateMatX));
}

__host__ __device__ Mat4 getScaleMatrix(float scale) {
    Mat4 mat;
    mat.rows[0] = make_float4(scale, 0, 0, 0);
    mat.rows[1] = make_float4(0, scale, 0, 0);
    mat.rows[2] = make_float4(0, 0, scale, 0);
    mat.rows[3] = make_float4(0, 0, 0, 1);
    return mat;
}

__host__ __device__ void moveCamera(utilities::camera* camera, float3 translate, float3 rotate){
    Mat4 mat = matrixMultiply(getTranslateMatrix(translate), getRotateMatrix(rotate));
    Vec4 base = make_float4(camera->base.x, camera->base.y, camera->base.z, 1.0f);
    base = vectorMultiply(mat, base);
    camera->base = make_float3(base.x, base.y, base.z);
    updateAngles(&(camera->angles), rotate);
}

__host__ __device__ Mat4 getModelToWorldMatrix(float scale, float3 angles, Vec3 base){
    return matrixMultiply(getTranslateMatrix(base), matrixMultiply(getRotateMatrix(angles), getScaleMatrix(scale)));
}

__host__ __device__ Mat4 getWorldToCameraMatrix(float3 angles, Vec3 base){
    //translate then rotate
    float3 negBase = make_float3(-base.x, -base.y, -base.z);
    float3 negAngles = make_float3(-angles.x, -angles.y, -angles.z);
    return matrixMultiply(getRotateMatrix(negAngles), getTranslateMatrix(negBase));
}

__host__ __device__ Mat4 getOrthographicProjectMatrix(utilities::viewVolume viewVolume) {
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
    return projectMat;
}

__host__ __device__ Mat4 getPerspectiveProjectMatrix(utilities::viewVolume viewVolume) {
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

    return projectMat;
}

__host__ __device__ Mat4 getModelToCameraMatrix(utilities::object &obj, utilities::camera &cam){
    Mat4 mat = matrixMultiply(getWorldToCameraMatrix(cam.angles, cam.base), getModelToWorldMatrix(obj.scale, obj.angles, obj.base));
    return mat;
}

__host__ __device__ Mat4 getCameraToProjectedMatrix(float3 offset, utilities::viewVolume viewVol, bool perspective){
    Mat4 mat = getTranslateMatrix(offset);
    if (perspective){
        mat = matrixMultiply(getPerspectiveProjectMatrix(viewVol), mat);
    } else {
        mat = matrixMultiply(getOrthographicProjectMatrix(viewVol), mat);
    }
    return mat;
}

__host__ __device__ Mat4 getCanonToScreenMatrix(utilities::viewVolume viewVol){
    return matrixMultiply(getCartToCanvasMatrix(), getCanonicalToFullMatrix(viewVol));
}

__host__ __device__ float2 getBarycentric(Vec4* triangle, float2 point){
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

__host__ __device__ bool isValidBarycentric(float2 barycentric){
    float m2 = barycentric.x;
    float m3 = barycentric.y;
    return (m2 >= 0 && m3 >= 0 && m2 + m3 <= 1);
}

__host__ __device__ float getDepth(Vec4* triangle, float2 barycentric){
    float m2 = barycentric.x;
    float m3 = barycentric.y;
    float depth = triangle[0].z * (1 - m2 - m3) + triangle[1].z * m2 + triangle[2].z * m3;
    return depth;
}

__host__ __device__ Color getColor(float2 barycentric, Color* colors){
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

__host__ __device__ utilities::boundingBox getBoundingBox(Vec4* triangle){
    float minX = triangle[0].x;
    float maxX = triangle[0].x;
    float minY = triangle[0].y;
    float maxY = triangle[0].y;
    for (int i=1; i<3; i++){
        minX = fminf(triangle[i].x, minX);
        maxX = fmaxf(triangle[i].x, maxX);
        minY = fminf(triangle[i].y, minY);
        maxY = fmaxf(triangle[i].y, maxY);
    }
    return {minX, maxX, minY, maxY};
}
__global__ void transformVerticesToCamKernel(
     Vec4* d_transformedVertices, int numVertices, Mat4 modelToCameraMatrix)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) {
        return;
    }
    d_transformedVertices[idx] = vectorMultiply(modelToCameraMatrix, d_transformedVertices[idx]);
}

__global__ void transformCamToCanvasKernel(
    Vec4* d_transformedVertices, int numVertices, Mat4 camToProjectMatrix, Mat4 canonToScreenMatrix)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) {
        return;
    }
    Vec4 t = d_transformedVertices[idx];
    t = vectorMultiply(camToProjectMatrix, t);
    t.x = t.x / t.w;
    t.y = t.y / t.w;
    t.z = t.z / t.w;
    t.w = t.w / t.w;
    d_transformedVertices[idx] = vectorMultiply(canonToScreenMatrix, t);
}

__global__ void rasterizeKernel(
    const Vec4* d_transformedVertices, // canvas coordinates 
    const int* d_faces,               
    const Color* d_face_colors,      
    uint8_t* d_frameBuffer,            
    float* d_depthBuffer,              
    int numFaces,
    Color bgColor
) {
    // Each thread represents a pixel on the screen
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) {
        return; 
    }

    float currentDepth = 1.0f; 
    uchar3 currentColor = bgColor;

    float2 pixelCenter = make_float2(x + 0.5f, y + 0.5f); 
    int index = y * WINDOW_WIDTH + x;
    bool covered = false;

    for (int i = 0; i < numFaces; i++) {
        int base_idx = i * 3;
        Vec4 triangle[3];
        triangle[0] = d_transformedVertices[d_faces[base_idx] - 1];
        triangle[1] = d_transformedVertices[d_faces[base_idx + 1] - 1];
        triangle[2] = d_transformedVertices[d_faces[base_idx + 2] - 1];

        utilities::boundingBox bBox = getBoundingBox(triangle);
        float minX = floor(bBox.minX);
        float minY = floor(bBox.minY);
        float maxX = ceil(bBox.maxX);
        float maxY = ceil(bBox.maxY);

        if (x < minX|| x > maxX|| y < minY|| y > maxY) {
            continue; 
        }

        float2 bary = getBarycentric(triangle, pixelCenter);
        if (isValidBarycentric(bary)) {
            float pointDepth = getDepth(triangle, bary);
            if (pointDepth >= 0.0f && pointDepth <= 1.0f) { 
                // update values if new depth is closer to camera
                if (pointDepth < currentDepth) { 
                    covered = true;
                    Color triColor[3];
                    triColor[0] = d_face_colors[base_idx];
                    triColor[1] = d_face_colors[base_idx + 1];
                    triColor[2] = d_face_colors[base_idx + 2];
                    currentColor = getColor(bary, triColor);
                    currentDepth = pointDepth; 
                }
            }
        }
    }
    int frameBufferBase = index * 3;
    if (covered){
        d_frameBuffer[frameBufferBase] = (uint8_t)floorf((1.0f - currentDepth) * currentColor.x);
        d_frameBuffer[frameBufferBase + 1] = (uint8_t)floorf((1.0f - currentDepth) * currentColor.y);
        d_frameBuffer[frameBufferBase + 2] = (uint8_t)floorf((1.0f - currentDepth) * currentColor.z);
    } else {
        d_frameBuffer[frameBufferBase] = bgColor.x;
        d_frameBuffer[frameBufferBase + 1] = bgColor.y;
        d_frameBuffer[frameBufferBase + 2] = bgColor.z;
    }
    d_depthBuffer[index] = currentDepth;
}

/*
__global__ void rasterizeTriangleKernel(
    const Vec4* d_transformedVertices, // canvas coordinates 
    const int* d_faces,               
    const Color* d_face_colors,      
    uint8_t* d_frameBuffer,            
    float* d_depthBuffer,              
    int numFaces,
    Color bgColor
) {
    // Each thread represents a face of the mesh
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numFaces) {
        return; 
    }

    int base_idx = idx * 3;
    Vec4 triangle[3];
    triangle[0] = d_transformedVertices[d_faces[base_idx] - 1];
    triangle[1] = d_transformedVertices[d_faces[base_idx + 1] - 1];
    triangle[2] = d_transformedVertices[d_faces[base_idx + 2] - 1];

    utilities::boundingBox bBox = getBoundingBox(triangle);
    int minX = max(0, (int) floor(bBox.minX));
    int minY = max(0, (int) floor(bBox.minY));
    int maxX = min(WINDOW_WIDTH - 1, (int) floor(bBox.maxX));
    int maxY = min(WINDOW_HEIGHT- 1, (int) floor(bBox.maxY));

    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            float2 pixelCenter = make_float2(x + 0.5f, y + 0.5f); 
            float2 bary = getBarycentric(triangle, pixelCenter);
            if (isValidBarycentric(bary)) {
                float pointDepth = getDepth(triangle, bary);
                if (pointDepth >= 0.0f && pointDepth <= 1.0f) { 
                    int index = y * WINDOW_WIDTH + x;
                    float currentDepth = atomicExch(d_depthBuffer[index], pointDepth);
                    if (pointDepth < currentDepth) { 
                        Color triColor[3];
                        triColor[0] = d_face_colors[base_idx];
                        triColor[1] = d_face_colors[base_idx + 1];
                        triColor[2] = d_face_colors[base_idx + 2];
                        Color currentColor = getColor(bary, triColor);
                        int frameBufferBase = index * 3;
                        d_frameBuffer[frameBufferBase] = (uint8_t)floorf((
                            1.0f - pointDepth) * currentColor.x);
                        d_frameBuffer[frameBufferBase + 1] = (uint8_t)floorf((
                            1.0f - pointDepth) * currentColor.y);
                        d_frameBuffer[frameBufferBase + 2] = (uint8_t)floorf((
                            1.0f - pointDepth) * currentColor.z);
                        d_depthBuffer[index] = pointDepth;
                    } else {
                        currentDepth = atomicExch(d_depthBuffer[index], currentDepth); 
                    }
                }
            }
        }
    }
}
*/