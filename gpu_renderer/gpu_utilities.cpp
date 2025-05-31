#include <iostream>
#include <fstream>
#include <sstream>
#include "gpu_utilities.h"

namespace utilities {

    void initCamera(camera &cam, const viewVolume &viewVol)
    {
        cam.base = make_float3(0.0f, 0.0f, 0.0f);
        cam.angles = make_float3(0.0f, 0.0f, 0.0f);
        cam.viewVolume = viewVol;
    }

    void initLight(light &light)
    {
        light.base = make_float3(0.0f, 0.0f, 0.0f);
        light.angles = make_float3(0.0f, 0.0f, 0.0f);
    }

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
                    obj.faces.push_back(face);
                }

                //Default color for each face
                // {{{255,255,0}, {255,0,0}, {0,255,0}}};
                obj.colors.push_back(make_uchar3(255, 255, 0));
                obj.colors.push_back(make_uchar3(255, 0, 0));
                obj.colors.push_back(make_uchar3(0, 255, 0)); 
            }
        }
        file.close();

        float maxX = obj.vertices[0].x;
        float minX = obj.vertices[0].x;
        float maxY = obj.vertices[0].y;
        float minY = obj.vertices[0].y;
        float maxZ = obj.vertices[0].z;
        float minZ = obj.vertices[0].z;
        for (Vec4 vertex: obj.vertices){
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
        obj.scale = screenDiagonal / diagonal * 0.9f;
        std::array <float, 3> center = {
            minX + dx / 2,
            minY + dy / 2,
            minZ + dz / 2
        };
        //Translate vertices to center
        for (Vec4 &vertex : obj.vertices) {
            vertex.x -= center[0];
            vertex.y -= center[1];
            vertex.z -= center[2];
        }
        obj.base = make_float3(0.0f, 0.0f, 0.0f);
        obj.angles = make_float3(0.0f, 0.0f, 0.0f);
    }
} // namespace utilities

void updateAngles(float3* angles, float3 rotate){
    angles->x += rotate.x;
    angles->y += rotate.y;
    angles->z += rotate.z;
    // Normalize angles to [0, 2*PI)
    angles->x = fmodf(angles->x, PI * 2);
    angles->y = fmodf(angles->y, PI * 2);
    angles->z = fmodf(angles->z, PI * 2);

}