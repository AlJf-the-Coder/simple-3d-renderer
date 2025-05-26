#include <iostream>
#include <cstdint>
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
typedef uchar3 Color

struct ExtractZ {
    __host__ __device__
    float operator()(const Vec4& v) const { return v.z; }
}

namespace utilities{
    struct boundingBox {
        float minX, maxX, minY, maxY, minZ, maxZ;
    };

    struct viewVolume
    {
        float N;
        float F;
        float L;
        float R;
        float T;
        float B;
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

    struct GPUObject {
        Vec4* vertices;
        int* faces;
        Color* colors;
        int numVertices;
        int numFaces;
        float scale;
        Vec3 base;
        float3 angles;
    };

    struct light {
        Vec3 base;
        float3 angles;
    };

}

float getScaleFactor(utilities::object obj)
{
    float maxX = obj.vertices[0].x;
    float minX = obj.vertices[0].x;
    float maxY = obj.vertices[0].y;
    float minY = obj.vertices[0].y;
    float maxZ = obj.vertices[0].z;
    float minZ = obj.vertices[0].z;
    for (Matrix vertex: obj.vertices){
        maxX = fmaxf(maxX, vertex.x);
        minX = fminf(minX, vertex.x);
        maxY = fmaxf(maxY, vertex.y);
        minY = fminf(minY, vertex.y);
        maxZ = fmaxf(maxZ, vertex.z);
        minZ = fminf(minZ, vertex.z);
    }
    float dx = maxX - minX;
    float dy = maxY - minY;
    float dz = maxZ - minZ;
    float diagonal = sqrt(dx*dx + dy*dy + dz*dz);
    float screenDiagonal = sqrt(WINDOW_WIDTH * WINDOW_WIDTH + WINDOW_HEIGHT * WINDOW_HEIGHT);
    return screenDiagonal / diagonal * 0.8f; // Add a margin;
}

void loadObject(const char *filename, utilities::object &obj)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if (type == "v")
        {
            Vec4 vertex;
            iss >> vertex.x >> vertex.y >> vertex.z;
            vertex.w = 1;
            obj.vertices.push_back(vertex);
        }
        else if (type == "f")
        {
            int face;
            for (int i=0; i < 3; i++)
            {
                std::string vertex;
                iss >> vertex;
                size_t pos = vertex.find('/');
                if (pos != std::string::npos)
                {
                    face = std::stoi(vertex.substr(0, pos));
                }
                else
                {
                    face = std::stoi(vertex);
                }
                obj.faces.push_back(face)
            }

            //Default color for each face
            // {{{255,255,0}, {255,0,0}, {0,255,0}}};
            obj.colors.push_back(make_uchar3(255, 255, 0));
            obj.colors.push_back(make_uchar3(255, 0, 0));
            obj.colors.push_back(make_uchar3(0, 255, 0)); 
        }
    }
    file.close();

    obj.scale = getScaleFactor(obj);
    obj.base = {
        {0.0f},
        {0.0f},
        {0.0f}
    };
    obj.angles = {0.0f, 0.0f, 0.0f};
}

void initCamera(utilities::camera &cam, const utilities::viewVolume viewVol)
{
    cam.base = {
        {0.0f},
        {0.0f},
        {0.0f}
    };
    cam.angles = {0.0f, 0.0f, 0.0f};
    cam.viewVolume = viewVol;
}

void initLight(utilities::light &light)
{
    light.base = {
        {0.0f},
        {0.0f},
        {0.0f}
    };
    light.angles = {0.0f, 0.0f, 0.0f};
}


void updateAngles(float3* angles, float3 rotate){
    angles->x += rotate.x;
    angles->y += rotate.y;
    angles->z += rotate.z;
    // Normalize angles to [0, 2*PI)
    angles->x = fmod(angles->x, PI * 2);
    angles->y = fmod(angles->y, PI * 2);
    angles->z = fmod(angles->z, PI * 2);

}