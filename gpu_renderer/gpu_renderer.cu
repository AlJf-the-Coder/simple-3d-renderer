#define SDL_MAIN_USE_CALLBACKS 1  /* use the callbacks instead of main() */
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3_image/SDL_image.h>
#include <SDL3/SDL_surface.h>
#include <SDL3/SDL_video.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <iomanip>
#include "gpu_utilities.h"

using namespace std;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        cerr << "CUDA Error: " << file << " " <<  line << " : " << cudaGetErrorString(code) << endl;
        exit(code);
    }
}


static SDL_Window *window = NULL;
static SDL_Renderer *renderer = NULL;
static double totalTime = 0;
// static Uint64 totalPixels = 0;
static Uint64 lastTime = 0;
static Uint64 frameNum = 0;

FrameBuffer frameBuffer;
DepthBuffer depthBuffer;
utilities::camera camera;
utilities::object obj;
const Color bgColor = make_uchar3(16, 110, 190); // Background color

FrameBuffer d_frameBuffer;
DepthBuffer d_depthBuffer;
Vec4* d_vertices;
Vec4* d_transformedVertices;
int* d_faces;
Color* d_colors;


void initBuffers(FrameBuffer &frameBuffer, DepthBuffer &depthBuffer) {
    frameBuffer = new uint8_t[WINDOW_WIDTH * WINDOW_HEIGHT * 3];
    depthBuffer = new float[WINDOW_WIDTH * WINDOW_HEIGHT];
    for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++) {
        int frameBufferBase = i * 3;
        frameBuffer[frameBufferBase] = bgColor.x;
        frameBuffer[frameBufferBase + 1] = bgColor.y;
        frameBuffer[frameBufferBase + 2] = bgColor.z;
        depthBuffer[i] = 1.0f; // Initialize depth buffer to maximum depth
    }
}

void resetBuffers() {
    for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++) {
        int frameBufferBase = i * 3;
        frameBuffer[frameBufferBase] = bgColor.x;
        frameBuffer[frameBufferBase + 1] = bgColor.y;
        frameBuffer[frameBufferBase + 2] = bgColor.z;
        depthBuffer[i] = 1.0f;
    }
}

/*
void drawWireframe(vector<Vec4> canvasCoords, uchar3 strokeColor) {
    auto [r,g,b] = strokeColor;
    SDL_SetRenderDrawColor(renderer, r, g, b, SDL_ALPHA_OPAQUE);
    for (int i=0 i < obj.faces.size(), i+=3){
        // strokeWeight(4);
        for (int j=0; j<polygon.size(); j++){
        vector<float> point1 = flatten_vector(canvasCoords[polygon[j] - 1]);
        vector<float> point2 = flatten_vector(canvasCoords[polygon[(j + 1) % polygon.size()] - 1]);
        SDL_RenderLine(renderer, point1[0], point1[1], point2[0], point2[1]);
        }
        // strokeWeight(10);
    }
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
    for (Matrix coord: canvasCoords){
        vector<float> point = flatten_vector(coord);
        SDL_RenderPoint(renderer, point[0], point[1]);
    }
}
*/

void scanlineRender(Vec4* triangle, Color* triColors) {

    float mat[2][2] = {
        {triangle[1].x - triangle[0].x, triangle[2].x - triangle[0].x},
        {triangle[1].y - triangle[0].y, triangle[2].y - triangle[0].y}
    };
    float det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    // skip degenerate triangles
    if (det == 0) {
        return;
        // return 0;
    }

    utilities::boundingBox bBox = getBoundingBox(triangle);
    float minX = max(0, (int) floor(bBox.minX));
    float minY = max(0, (int) floor(bBox.minY));
    float maxX = min(WINDOW_WIDTH - 1, (int) floor(bBox.maxX));
    float maxY = min(WINDOW_HEIGHT- 1, (int) floor(bBox.maxY));

    // int triPixels = 0;
    for (int y=minY; y <= maxY; y++){
        for (int x=minX; x <= maxX; x++){
            // triPixels += 1;
            float2 point = make_float2(x + 0.5f, y + 0.5f);
            float2 barycentric = getBarycentric(triangle, point);
            if (isValidBarycentric(barycentric)){
                float pointDepth = getDepth(triangle, barycentric);
                int index = (y * WINDOW_WIDTH + x);
                if (pointDepth >= 0 && pointDepth <= 1 && pointDepth <= depthBuffer[index]){
                    Color color = getColor(barycentric, triColors);
                    int frameBufferBase = 3 * index;
                    frameBuffer[frameBufferBase + 0] = floor((1-pointDepth)*color.x);
                    frameBuffer[frameBufferBase + 1] = floor((1-pointDepth)*color.y);
                    frameBuffer[frameBufferBase + 2] = floor((1-pointDepth)*color.z);
                    depthBuffer[index] = pointDepth;
                }
            }
        }
    }
    // return triPixels;
}

void rasterize(vector<Vec4> canvasCoords, utilities::object object) {
    // int modelPixels = 0;
    for (int i = 0; i < object.faces.size(); i+=3){
        Color triColors[3];
        Vec4 mappedTriangle[3];
        for (int j=0; j<3; j++){
            triColors[j] = object.colors[i + j];
            mappedTriangle[j] = canvasCoords[object.faces[i + j] - 1];
        }
        // modelPixels += scanlineRender(mappedTriangle, triColors);
        scanlineRender(mappedTriangle, triColors);
    }
    // return modelPixels;
}

vector<Vec4> transformObjectCoordinates(utilities::object &obj, utilities::camera &cam, bool perspective) {

    vector<Vec4> transformedCoords;
    transformedCoords.reserve(obj.vertices.size());

    Mat4 modelToCamMatrix = getModelToCameraMatrix(obj, cam);
    for (Vec4 coord : obj.vertices){
        transformedCoords.push_back(vectorMultiply(modelToCamMatrix, coord));
    }

    float near = camera.viewVolume.N;
    float minZ = transformedCoords[0].z;
    for (Vec4 t: transformedCoords){
        minZ = fminf(minZ, t.z);
    }
    float3 offsetVector = make_float3(0.0f, 0.0f, near - minZ);

    Mat4 camToProjectMatrix = getCameraToProjectedMatrix(offsetVector, cam.viewVolume, perspective);
    Mat4 canonToScreenMatrix = getCanonToScreenMatrix(cam.viewVolume);
    for (Vec4& t: transformedCoords){
        t = vectorMultiply(camToProjectMatrix, t);
        t.x = t.x / t.w;
        t.y = t.y / t.w;
        t.z = t.z / t.w;
        t.w = 1.0f;
        t = vectorMultiply(canonToScreenMatrix, t);
    }

    return transformedCoords;
}

/* This function runs once at startup. */
SDL_AppResult SDL_AppInit(void **appstate, int argc, char *argv[])
{
    SDL_Log("Initializing\n");

    float near = WINDOW_WIDTH;
    utilities::initCamera(camera, {near, near + WINDOW_WIDTH, -WINDOW_WIDTH/2, WINDOW_WIDTH/2, WINDOW_HEIGHT/2, -WINDOW_HEIGHT/2});
    initBuffers(frameBuffer, depthBuffer);
    try{
        if (argc == 1)
            utilities::loadObject("../../../models/cube.obj", obj);
        else
            utilities::loadObject(argv[1], obj);
    }
    catch (const std::exception &e) {
        cerr << "Error loading object: " << e.what() << endl;
        return SDL_APP_FAILURE;
    }

    cerr << "Loaded object" << "\n";
    cout << "Object has " << obj.vertices.size() << " vertices and " << obj.faces.size() / 3 << " faces" << endl;

    SDL_SetAppMetadata("Simple 3D Renderer", "1.0", "com.example.renderer-points");

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        cerr << "Couldn't initialize SDL: " << SDL_GetError() << endl;
        return SDL_APP_FAILURE;
    }

    if (!SDL_CreateWindowAndRenderer("Test Renderer", WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer)) {
        cerr << "Couldn't create window/renderer: " << SDL_GetError() << endl;
        return SDL_APP_FAILURE;
    }
    CUDA_CHECK(cudaMalloc((void** )&d_frameBuffer, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(Color))); 
    CUDA_CHECK(cudaMalloc((void**) &d_depthBuffer, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &d_vertices, obj.vertices.size() * sizeof(Vec4)));
    CUDA_CHECK(cudaMalloc((void**) &d_transformedVertices, obj.vertices.size() * sizeof(Vec4)));
    CUDA_CHECK(cudaMalloc((void**) &d_faces, obj.faces.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_colors, obj.colors.size() * sizeof(Color)));

    CUDA_CHECK(cudaMemcpy(d_vertices, obj.vertices.data(), obj.vertices.size() * sizeof(Vec4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_faces, obj.faces.data(), obj.faces.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colors, obj.colors.data(), obj.colors.size() * sizeof(Color), cudaMemcpyHostToDevice));

    return SDL_APP_CONTINUE;  /* carry on with the program! */
}

/* This function runs when a new event (mouse input, keypresses, etc) occurs. */
SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event)
{
    if (event->type == SDL_EVENT_QUIT) {
        return SDL_APP_SUCCESS;  /* end the program, reporting success to the OS. */
    }
    return SDL_APP_CONTINUE;  /* carry on with the program! */
}
/* This function runs once per frame, and is the heart of the program. */
SDL_AppResult SDL_AppIterate(void *appstate)
{
    lastTime = SDL_GetPerformanceCounter();
    resetBuffers();  /* clear the frame buffer. */
    
    int numVertices = obj.vertices.size();
    int numFaces = obj.faces.size() / 3; // Each face is a triangle

    CUDA_CHECK(cudaMemcpy(d_transformedVertices, d_vertices, numVertices * sizeof(Vec4), cudaMemcpyDeviceToDevice));
    int numThreads = 256;
    int numBlocks = (numVertices + numThreads - 1) / numThreads;

    Mat4 modelToCameraMatrix = getModelToCameraMatrix(obj, camera);
    transformVerticesToCamKernel<<<numBlocks, numThreads>>>(
        d_transformedVertices, numVertices, modelToCameraMatrix);
    CUDA_CHECK(cudaDeviceSynchronize()); 

    minZUnary unary_op;
    minZBinary binary_op;
    thrust::device_ptr<Vec4> d_verticesPtr(d_transformedVertices);
    float init = unary_op(d_verticesPtr[0]);
    float minZ = thrust::transform_reduce(d_verticesPtr, d_verticesPtr + numVertices, unary_op, init, binary_op);

    float offsetZ = camera.viewVolume.N - minZ;
    float3 offsetVec = make_float3(0.0f, 0.0f, offsetZ);

    Mat4 camToProjectMatrix = getCameraToProjectedMatrix(offsetVec, camera.viewVolume, true);
    Mat4 canonToScreenMatrix = getCanonToScreenMatrix(camera.viewVolume);
    transformCamToCanvasKernel<<<numBlocks, numThreads>>>(
        d_transformedVertices, numVertices, camToProjectMatrix, canonToScreenMatrix);

    float3 objRotate = make_float3(-0.03f, 0.0f, -0.053f);
    updateAngles(&obj.angles, objRotate);

    /*
    vector<Vec4> canvasCoords(numVertices);
    CUDA_CHECK(cudaMemcpy(canvasCoords.data(), d_transformedVertices, numVertices * sizeof(Vec4), cudaMemcpyDeviceToHost));
    rasterize(canvasCoords, obj);
    */

    //rasterize the object
    dim3 threads(16, 16); //256 threads per block
    dim3 blocks(
        (WINDOW_WIDTH + threads.x - 1) / threads.x, 
        (WINDOW_HEIGHT + threads.y - 1) / threads.y
    );
    
    rasterizeKernel<<<blocks, threads>>>(
        d_transformedVertices, d_faces, d_colors, 
        d_frameBuffer, d_depthBuffer, numFaces, 
        bgColor);

    // copy data back to Host
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaMemcpy(frameBuffer, d_frameBuffer, 3 * WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    //save the frame
    SDL_Surface* surface = SDL_CreateSurfaceFrom(
        WINDOW_WIDTH, WINDOW_HEIGHT,
        SDL_PIXELFORMAT_RGB24, frameBuffer, WINDOW_WIDTH * 3 
    );
    if (surface == NULL) {
        cerr << "Couldn't read pixels from renderer: " << SDL_GetError() << endl;
        return SDL_APP_FAILURE;
    }

    // Convert the surface to a texture
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (texture == NULL) {
        cerr << "Couldn't create texture: " << SDL_GetError() << endl;
        SDL_DestroySurface(surface);
        return SDL_APP_FAILURE;
    }

    // Render the texture to the screen
    SDL_RenderTexture(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);

    const string fileString = "../../output/frame_" + to_string(frameNum) + ".png";
    const char *fileName = fileString.c_str();

    if (!IMG_SavePNG(surface, fileName)){
        cerr << "Couldn't save image: " << SDL_GetError() << endl;
    }

    SDL_DestroySurface(surface);
    SDL_DestroyTexture(texture);

    //store metrics in log file
    const Uint64 now = SDL_GetPerformanceCounter();
    const double elapsed = ((double) (now - lastTime) / SDL_GetPerformanceFrequency()) * 1000;  /* seconds since last iteration */
    totalTime += elapsed;
    // totalPixels += pixelCount;


    cout << "frameNum: " << frameNum << " "
     << "time: " << elapsed << endl;
    //  << "pixels: " << pixelCount << endl;
    frameNum++;

    if (frameNum >= 120) {
        double frameTime = totalTime / frameNum;
        cout << "Total time: " << totalTime << endl;
        cout << "Exiting after " << frameNum << " frames" << endl;
        // cout << "Average pixels per frame: " << std::fixed << std::setprecision(4) << (float) totalPixels / frameNum << endl;
        cout << "Average processing time: " << std::fixed << std::setprecision(4) << frameTime << endl;
        cout << "Frame rate: " << std::fixed << std::setprecision(4) << (1.0 / (frameTime / 1000)) << endl;
        CUDA_CHECK(cudaFree(d_frameBuffer))
        CUDA_CHECK(cudaFree(d_depthBuffer))
        CUDA_CHECK(cudaFree(d_vertices))
        CUDA_CHECK(cudaFree(d_transformedVertices))
        CUDA_CHECK(cudaFree(d_faces))
        CUDA_CHECK(cudaFree(d_colors))
        delete[] frameBuffer;
        delete[] depthBuffer;
        return SDL_APP_SUCCESS;  /* end the program, reporting success to the OS. */
    }
    return SDL_APP_CONTINUE;  /* carry on with the program! */
}

/* This function runs once at shutdown. */
void SDL_AppQuit(void *appstate, SDL_AppResult result)
{
    /* SDL will clean up the window/renderer for us. */
}
