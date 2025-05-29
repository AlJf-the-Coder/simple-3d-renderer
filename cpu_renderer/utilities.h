#include <iostream>
#include <array> 
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <functional>

#define PI 3.14159265358979323846

#define WINDOW_WIDTH 500
#define WINDOW_HEIGHT WINDOW_WIDTH

using FrameBuffer = std::array<uint8_t, 3 * WINDOW_WIDTH * WINDOW_HEIGHT>;
using DepthBuffer = std::array<std::array<float, WINDOW_WIDTH>, WINDOW_HEIGHT>;
using Matrix = std::vector<std::vector<float>>;

namespace utilities{
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
        Matrix base;
        std::array<float, 3> angles;
        utilities::viewVolume viewVolume;
    };

    struct object 
    {
        std::vector<Matrix> vertices;
        std::vector<std::array<int, 3>> faces;
        std::vector<std::array<std::array<uint8_t, 3>, 3>> colors;
        float scale;
        Matrix base;
        std::array<float, 3> angles;
    };

    struct light {
        Matrix base;
        std::array<float, 3> angles;
    };
}

using PerspectiveFunc = std::function<Matrix(Matrix, utilities::viewVolume)>;
using TransformFunc= std::function<Matrix(const Matrix, const std::any)>;

void loadObject(const char *filename, utilities::object &obj)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw std::runtime_error(std::string("Error opening file: ") + filename);
    }
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        if (type == "v")
        {
            Matrix vertex(4, std::vector<float>(1, 1.0f));
            iss >> vertex[0][0] >> vertex[1][0] >> vertex[2][0];
            obj.vertices.push_back(vertex);
        }
        else if (type == "f")
        {
            std::array<int, 3> face;
            for (int i=0; i < 3; i++)
            {
                std::string vertex;
                iss >> vertex;
                size_t pos = vertex.find('/');
                if (pos != std::string::npos)
                {
                    face[i] = std::stoi(vertex.substr(0, pos));
                }
                else
                {
                    face[i] = std::stoi(vertex);
                }
            }
            obj.faces.push_back(face);

            //Default color for each face
            // std::array<std::array<uint8_t, 3>, 3> colors = {{{255, 255, 0}, {255, 255, 0}, {255, 255, 0}}};
            std::array<std::array<uint8_t, 3>, 3> colors = {{{255,255,0}, {255,0,0}, {0,255,0}}};
            obj.colors.push_back(colors);
        }
    }
    file.close();

    float maxX = obj.vertices[0][0][0];
    float minX = obj.vertices[0][0][0];
    float maxY = obj.vertices[0][1][0];
    float minY = obj.vertices[0][1][0];
    float maxZ = obj.vertices[0][2][0];
    float minZ = obj.vertices[0][2][0];
    for (Matrix vertex: obj.vertices){
        maxX = fmaxf(maxX, vertex[0][0]);
        minX = fminf(minX, vertex[0][0]);
        maxY = fmaxf(maxY, vertex[1][0]);
        minY = fminf(minY, vertex[1][0]);
        maxZ = fmaxf(maxZ, vertex[2][0]);
        minZ = fminf(minZ, vertex[2][0]);
    }
    float dx = maxX - minX;
    float dy = maxY - minY;
    float dz = maxZ - minZ;
    float diagonal = sqrt(dx*dx + dy*dy + dz*dz);
    float screenDiagonal = sqrt(WINDOW_WIDTH * WINDOW_WIDTH + WINDOW_HEIGHT * WINDOW_HEIGHT);
    obj.scale = screenDiagonal / diagonal * 0.9f;
    std::array <float, 3> center = {
        minX + dx / 2,
        minY + dy / 2,
        minZ + dz / 2
    };
    for (Matrix &vertex : obj.vertices) {
        vertex[0][0] -= center[0];
        vertex[1][0] -= center[1];
        vertex[2][0] -= center[2];
        vertex[3][0] = 1.0f; // Ensure homogeneous coordinate
    }
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

void updateAngles(std::array<float, 3> &angles, std::array<float, 3> changes){
    for (int i = 0; i < 3; i++){
        angles[i] += changes[i];
        angles[i] = fmod(angles[i], PI * 2);
    }
}
