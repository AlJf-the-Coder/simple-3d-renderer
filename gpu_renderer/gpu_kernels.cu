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

__host__ __device__ Vec4 translateCoord(Vec4 homoCoord3d, float3 translate) {
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

__host__ __device__ Vec4 rotateCoord(Vec4 homoCoord3d, float3 rotate) {
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

__device__ Vec4 scaleCoord(Vec4 homoCoord3d, float scale) {
    Mat4 mat;
    mat.rows[0] = make_float4(scale, 0, 0, 0);
    mat.rows[1] = make_float4(0, scale, 0, 0);
    mat.rows[2] = make_float4(0, 0, scale, 0);
    mat.rows[3] = make_float4(0, 0, 0, 1);
    return vectorMultiply(mat, homoCoord3d);
}

__host__ __device__ void moveCamera(utilities::camera* camera, float3 translate, float3 rotate){
    Vec4 base = make_float4(camera->base.x, camera->base.y, camera->base.z, 1.0f);
    base = translateCoord(base, translate);
    base = rotateCoord(base, rotate);
    camera->base = make_float3(base.x, base.y, base.z);
    updateAngles(&(camera->angles), rotate);
}

__device__ Vec4 modelToWorld(Vec4 homoCoord3d, float scale, float3 angles, Vec3 base){
    Vec4 worldCoord = homoCoord3d;
    worldCoord = rotateCoord(worldCoord, angles);
    worldCoord = translateCoord(worldCoord, base);
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
__global__ void transformVerticesToCamKernel(
     Vec4* d_transformedVertices, int numVertices, float3 d_objAngles, Vec3 d_objBase,
    utilities::camera d_camera, float scale )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) {
        return;
    }
    Vec4 t = d_transformedVertices[idx];
    t = modelToWorld(t, scale, d_objAngles, d_objBase);
    t = worldToCamera(t, d_camera.angles, d_camera.base);
    d_transformedVertices[idx] = t;
}

__global__ void transformCamToCanvasKernel(
    Vec4* d_transformedVertices, int numVertices, float3 offsetVector, bool perspective,
    utilities::viewVolume d_viewVol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) {
        return;
    }
    Vec4 t = d_transformedVertices[idx];
    t = translateCoord(t, offsetVector);
    if (perspective) {
        t = perspectiveProjectCoord(t, d_viewVol);
    }
    else {
        t = orthographicProjectCoord(t, d_viewVol);
    }
    t = canonicalToFullCoords(t, d_viewVol);
    t = cartToCanvasCoords(t);
    d_transformedVertices[idx] = t;
}

__global__ void rasterizeTrianglesKernel(
    Color* frameBuffer, float* depthBuffer,
    Vec4* d_vertices, int numVertices,
    int* d_faces, int numFaces,
    Vec3* d_colors, int numColors)
{
    int triIdx = blockIdx.x;
    int pixelIdx = threadIdx.x;
    if (triIdx >= numFaces || pixelIdx >= WINDOW_WIDTH * WINDOW_HEIGHT) {
        return; // Out of bounds
    }

    // Each block handles one triangle, each thread a pixel in the bounding box
    // TODO: Implement barycentric test and color/depth write
}


/*
// RASTERIZATION KERNEL
__global__ void rasterizeKernel(
    const Vec4* d_canvas_coords,       // Transformed vertices from vertex kernel
    const int* d_faces,                // Flattened face indices
    const Color* d_face_colors,       // Face colors
    uint8_t* d_frameBuffer,             // Output frame buffer
    float* d_depthBuffer,              // Output depth buffer
    int num_faces,
) {
    // Each thread represents a pixel on the screen
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) {
        return; // Thread is outside screen bounds
    }

    float current_depth = 1.0f; // Initialize with max depth (far plane)
    uchar3 current_color = make_uchar3(0, 0, 0); // Initialize with black (background)

    // Pixel center for barycentric calculation
    Vec4 pixel_center = make_float4(x + 0.5f, y + 0.5f, 0.0f, 1.0f); // Z and W are placeholders

    // Iterate through ALL triangles for EACH pixel
    // This is INCREDIBLY INEFFICIENT but gets "something" working.
    // For a real renderer, you'd use a different approach (e.g., triangle setup + grid/tile).
    for (int i = 0; i < num_faces; ++i) {
        // Get vertex indices for the current triangle
        int v0_idx = d_faces[i * 3 + 0];
        int v1_idx = d_faces[i * 3 + 1];
        int v2_idx = d_faces[i * 3 + 2];

        // Get triangle vertices in canvas coordinates
        Vec4 v0 = d_canvas_coords[v0_idx];
        Vec4 v1 = d_canvas_coords[v1_idx];
        Vec4 v2 = d_canvas_coords[v2_idx];

        // Crude bounding box check (optional but helps a tiny bit)
        float minX_tri = fminf(v0.x, fminf(v1.x, v2.x));
        float maxX_tri = fmaxf(v0.x, fmaxf(v1.x, v2.x));
        float minY_tri = fminf(v0.y, fminf(v1.y, v2.y));
        float maxY_tri = fmaxf(v0.y, fmaxf(v1.y, v2.y));

        if (x < minX_tri || x > maxX_tri || y < minY_tri || y > maxY_tri) {
            continue; // Pixel outside triangle's bounding box
        }

        // Calculate barycentric coordinates
        float3 bary = getBarycentric(v0, v1, v2, pixel_center);

        // Check if pixel is inside the triangle
        if (isValidBarycentric(bary)) {
            float pointDepth = getDepth(v0, v1, v2, bary);

            // Z-buffer test: need atomic for this!
            // Initialize depth buffer with a very high value (e.g., 1.0 or FLT_MAX) on host
            // before launching this kernel.
            if (pointDepth >= 0.0f && pointDepth <= 1.0f) { // Ensure depth is in valid range
                // Atomic operation for depth test and update
                // This is CRITICAL for correctness when multiple triangles overlap
                float old_depth = atomicExch(d_depthBuffer + (y * WINDOW_WIDTH + x), pointDepth);
                if (pointDepth < old_depth) { // If new depth is closer
                    // Update the color only if we won the depth test
                    // This is a race condition. If atomics are not used for color,
                    // you might see flickering. Simplest for demo: just set color.
                    // For true correctness, you'd need atomic CAS for both or a more
                    // advanced rendering algorithm.
                    current_color = getFaceColor(d_face_colors, i); // Get solid face color
                    // (Optional: rudimentary depth-based shading, as in your original CPU code)
                    // current_color.x = (uint8_t)floorf((1.0f - pointDepth) * current_color.x);
                    // current_color.y = (uint8_t)floorf((1.0f - pointDepth) * current_color.y);
                    // current_color.z = (uint8_t)floorf((1.0f - pointDepth) * current_color.z);

                    d_frameBuffer[y * WINDOW_WIDTH + x] = current_color;
                } else {
                    // If we lost the depth test, restore the old depth value
                    atomicExch(d_depthBuffer + (y * WINDOW_WIDTH + x), old_depth);
                }
            }
        }
    }
}
*/