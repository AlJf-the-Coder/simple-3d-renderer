#include "gpu_utilities.h"

__global__ void transformVerticesToCamKernel(
     Vec* d_transformedVertices, int numVertices, float3 d_objAngles, Vec3 d_objBase,
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
    Vec4* d_transformedVertices, int numVertices, float3 offsetVector,
    utilities::viewVolume d_viewVol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) {
        return;
    }
    Vec4 t = d_transformedVertices[idx];
    t = translateCoord(t, offsetVector);
    t = perspectiveFunc(t, d_viewVol);
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


// RASTERIZATION KERNEL
__global__ void rasterizeKernel(
    const Vec4* d_canvas_coords,       // Transformed vertices from vertex kernel
    const int* d_faces,                // Flattened face indices
    const Color* d_face_colors,       // Face colors
    uint8_t* d_frameBuffer,             // Output frame buffer
    float* d_depthBuffer,              // Output depth buffer
    int num_faces,
    int WINDOW_WIDTH,
    int WINDOW_HEIGHT
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