
__global__ void transformVerticesToCamKernel(
    Vec4* d_vertices, int numVertices, float3 d_obj_angles, Vec3 d_obj_base,
    utilities::camera d_camera, float scale )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    Vec4 t = d_vertices[idx];
    t = modelToWorld(t, scale, d_obj_angles, d_obj_base);
    t = worldToCamera(t, d_camera.angles, d_camera.base);
    d_vertices[idx] = t;
}

__global__ void transformCamToCanvasKernel(
    Vec4* d_vertices, int numVertices, float offsetZ,
    utilities::viewVolume d_viewVol,)
{
    float3 offsetVector = make_float3(0.0f, 0.0f, offsetZ);
    Vec4 t = d_vertices[idx];
    t = translateCoord(t, offsetVector);
    t = perspectiveFunc(t, d_viewVol);
    t = canonicalToFullCoords(t, d_viewVol);
    t = cartToCanvasCoords(t);
    d_vertices[idx] = t;
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
