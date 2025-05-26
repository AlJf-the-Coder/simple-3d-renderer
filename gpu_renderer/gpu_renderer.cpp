#define SDL_MAIN_USE_CALLBACKS 1  /* use the callbacks instead of main() */
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3_image/SDL_image.h>
#include <SDL3/SDL_surface.h>
#include <SDL3/SDL_video.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "gpu_kernels.cu"
#include "gpu_rasterizer.h"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

using namespace std;

static SDL_Window *window = NULL;
static SDL_Renderer *renderer = NULL;
static double averageTime = 0;
static double totalTime = 0;
static Uint64 totalPixels = 0;
static Uint64 lastTime = 0;
static Uint64 frameNum = 0;

FrameBuffer frameBuffer;
DepthBuffer depthBuffer;
utilities::camera camera;
utilities::object obj;

Vec4* d_vertices;
int* d_faces;
Color* d_colors;

void initBuffers(FrameBuffer* &frameBuffer, DepthBuffer* &depthBuffer, const uchar3 clearColor) {
    frameBuffer = new uint8_t[WINDOW_WIDTH * WINDOW_HEIGHT * 3];
    depthBuffer = new float[WINDOW_WIDTH * WINDOW_HEIGHT];
    for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; ++i) {
        frameBuffer[i * 3] = clearColor.x;
        frameBuffer[i * 3 + 1] = clearColor.y;
        frameBuffer[i * 3 + 2] = clearColor.z;
        depthBuffer[i] = 1.0f; // Initialize depth buffer to maximum depth
    }
}

void resetBuffers(std::array<uint8_t, 3> color) {
    for (int i = 0; i < WINDOW_HEIGHT; i++)
    {
        for (int j = 0; j < WINDOW_WIDTH; j++)
        {
            frameBuffer[i][j] = color;
            depthBuffer[i][j] = 1.0f;
        }
    }
}

void drawWireframe(vector<Matrix> canvasCoords, array<uint8_t, 3> strokeColor= {255, 255, 0}) {
    auto [r,g,b] = strokeColor;
    SDL_SetRenderDrawColor(renderer, r, g, b, SDL_ALPHA_OPAQUE);
    for (array<int, 3> polygon: obj.faces){
        // strokeWeight(4);
        for (int i=0; i<polygon.size(); i++){
        vector<float> point1 = flatten_vector(canvasCoords[polygon[i] - 1]);
        vector<float> point2 = flatten_vector(canvasCoords[polygon[(i + 1) % polygon.size()] - 1]);
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

int scanlineRender(Vec4* triangle, Color* triColors) {

    utilities:boundingBox = getBoundingBox(triangle);
    float minX = triangle[0].x;

    int triPixels = 0;
    for (int y=max(0, (int) floor(minY)); y <= min(WINDOW_HEIGHT- 1, (int) floor(maxY)); y++){
        for (int x=max(0, (int) floor(minX)); x <= min(WINDOW_WIDTH - 1, (int) floor(maxX)); x++){
            triPixels += 1;
            float2 point = make_float2(x + 0.5f, y + 0.5f);
            float2 barycentric = getBarycentric(triangle, point);
            if (isValidBarycentric(barycentric)){
                float pointDepth = getDepth(triangle, barycentric);
                if (pointDepth >= 0 && pointDepth <= 1 && pointDepth <= depthBuffer[y][x]){
                    Color getColor(barycentric, triColors);
                    for (int i=0; i<3; i++){
                        color[i] = floor((1-pointDepth)*color[i]);
                    }
                    frameBuffer[y][x] = color;
                    depthBuffer[y][x] = pointDepth;
                }
            }
        }
    }
    return triPixels;
}

int rasterize(vector<Vec4> canvasCoords, utilities::object object) {
    int modelPixels = 0;
    for (int i = 0; i < object.faces.size(); i+=3){
        Color triColors[3];
        Vec4 mappedTriangle[3];
        for (int j=0; j<3; j++){
            triColors[j] = object.colors[i + j];
            mappedTriangle[j] = canvasCoords[triangle[i + j] - 1];
        }
        modelPixels += scanlineRender(mappedTriangle, triColors);
    }
    return modelPixels;
}

/* This function runs once at startup. */
SDL_AppResult SDL_AppInit(void **appstate, int argc, char *argv[])
{
    SDL_Log("Initializing\n");

    float near = WINDOW_WIDTH;
    initCamera(camera, {near, near + WINDOW_WIDTH, -WINDOW_WIDTH/2, WINDOW_WIDTH/2, WINDOW_HEIGHT/2, -WINDOW_HEIGHT/2});
    try{
        if (argc == 1)
            loadObject("../../../models/cube.obj", obj);
        else
            loadObject(argv[1], obj);
    }
    catch (const std::exception &e) {
        SDL_Log("Error loading object");
        cerr << "Error loading object: " << e.what() << endl;
        return SDL_APP_FAILURE;
    }

    SDL_Log("Loaded object with %d vertices and %d faces\n", obj.vertices.size(), obj.faces.size());
    cout << "Object has " << obj.vertices.size() << " vertices and " << obj.faces.size() << " faces" << endl;

    SDL_SetAppMetadata("Simple 3D Renderer", "1.0", "com.example.renderer-points");

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    if (!SDL_CreateWindowAndRenderer("Test Renderer", WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer)) {
        SDL_Log("Couldn't create window/renderer: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    cudaMalloc(&d_vertices, vertices.size() * sizeof(Vec4));
    cudaMalloc(&d_faces, faces.size() * sizeof(int));
    cudaMalloc(&d_colors, colors.size() * sizeof(Color));

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
    resetBuffers({0, 255, 255});  /* clear the frame buffer. */
    cudaMemcpy(d_vertices, vertices.data(), vertices.size() * sizeof(Vec4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, faces.data(), faces.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors, colors.data(), colors.size() * sizeof(Color), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obj_angles, &obj.angles, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obj_base, &obj.base, sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_camera, &camera, sizeof(utilities::camera), cudaMemcpyHostToDevice);
    cudaMemcpy(d_viewVol, &camera.viewVolume, sizeof(utilities::viewVolume), cudaMemcpyHostToDevice);
    float scale = obj.scale;
    
    int threadsPerBlock = 1024;
    int blocks = (numVertices + threadsPerBlock - 1) / threadsPerBlock;

    transformVerticesToCamKernel<<<blocks, threadsPerBlock>>>(
        d_vertices, numVertices, d_obj_angles, d_obj_base, d_camera, scale)
    cudaDeviceSynchronize(); 

    thrust::device_ptr<Vec4> d_vertices_ptr(d_vertices);
    auto z_accessor_it = thrust::make_transform_iterator(d_vertices_ptr, ExtractZ());
    auto min_z_it = thrust::min_element(z_accessor_it, z_accessor_it + num_vertices);
    float minZ = *min_z_it;

    float offsetZ = camera.viewVolume.N - minZ;

    transformCamToCanvasKernel<<<blocks, threadsPerBlock>>>(
        d_vertices, numVertices, offsetZ, d_viewVol);
    cudaDeviceSynchronize();

    float3 objRotate = make_float3(-0.03f, 0.0f, -0.053f);
    updateAngles(&obj.angles, objRotate);

    
    // Remove after making rasterizer kernel
    vector<Vec4> canvasCoords;
    canvasCoords.reserve(numVertices);
    cudaMemcpy(canvasCoords.data(), d_vertices, vertices.size() * sizeof(Vec4), cudaMemcpyDeviceToHost);
    const int pixelCount = rasterize(canvasCoords, obj);
    // drawWireframe(canvasCoords, {0, 255, 0});

    // copy data back to Host

    //TODO: make it so all pixels are rendered at once
    for (int y=0; y<WINDOW_HEIGHT; y++){
        for (int x=0; x<WINDOW_WIDTH; x++){
            int index = 4 * (y * WINDOW_WIDTH + x);
            auto [r,g,b] = frameBuffer[y][x];
            SDL_SetRenderDrawColor(renderer, r, g, b, SDL_ALPHA_OPAQUE);
            SDL_RenderPoint(renderer, x, y);
            // pixels[index] = frameBuffer[y][x][0];
            // pixels[index + 1] = frameBuffer[y][x][1];
            // pixels[index + 2] = frameBuffer[y][x][2];
        }
    }

    //save the frame
    SDL_Surface *surface = SDL_RenderReadPixels(renderer, NULL);
    if (surface == NULL) {
        SDL_Log("Couldn't read pixels from renderer: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    const string fileString = "../../output/frame_" + to_string(frameNum) + ".png";
    const char *fileName = fileString.c_str();

    if (!IMG_SavePNG(surface, fileName)){
        SDL_DestroySurface(surface);
        SDL_Log("Couldn't save image: %s", SDL_GetError());
    }

    SDL_DestroySurface(surface);
    SDL_RenderPresent(renderer);  /* put it all on the screen! */

    //store metrics in log file
    const Uint64 now = SDL_GetPerformanceCounter();
    const double elapsed = ((double) (now - lastTime) / SDL_GetPerformanceFrequency()) * 1000;  /* seconds since last iteration */
    totalTime += elapsed;
    totalPixels += pixelCount;


    SDL_Log("Frame %llu, Time: %f, Pixels: %d", frameNum, elapsed, pixelCount);
    cout << "frameNum: " << frameNum << " "
     << "time: " << elapsed << " " 
     << "pixels: " << pixelCount << endl;
    frameNum++;

    if (frameNum >= 10) {
        SDL_Log("Total time: %f", totalTime);
        double frameTime = totalTime / frameNum;
        SDL_Log("Exiting after %d frames", frameNum);
        SDL_Log("Average pixels per frame: %.4f", (float) totalPixels / frameNum);
        SDL_Log("Average processing time: %.4f", frameTime);
        SDL_Log("Frame rate: %.4f", 1.0 / (frameTime / 1000));
        return SDL_APP_SUCCESS;  /* end the program, reporting success to the OS. */
    }
    return SDL_APP_CONTINUE;  /* carry on with the program! */
}

/* This function runs once at shutdown. */
void SDL_AppQuit(void *appstate, SDL_AppResult result)
{
    /* SDL will clean up the window/renderer for us. */
}
