#include <array> 
#include <vector>
#include <any>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <math.h>

#define PI 3.14159265358979323846

#define WINDOW_WIDTH 400
#define WINDOW_HEIGHT WINDOW_WIDTH

using FrameBuffer = std::array<std::array<std::array<uint8_t, 3>, WINDOW_WIDTH>, WINDOW_HEIGHT>;
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

float getScaleFactor(utilities::object obj)
{
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
            Matrix vertex(4, std::vector<float>(1, 1.0f));
            iss >> vertex[0][0] >> vertex[1][0] >> vertex[2][0];
            obj.vertices.push_back(vertex);
        }
        else if (type == "f")
        {
            std::array<int, 3> face;
            iss >> face[0] >> face[1] >> face[2];
            obj.faces.push_back(face);
            // std::array<std::array<uint8_t, 3>, 3> colors = {{255, 255, 0}, {255, 255, 0}, {255, 255, 0}};
            std::array<std::array<uint8_t, 3>, 3> colors = {{{255, 255, 0}, {255, 255, 0}, {255, 255, 0}}};
            obj.colors.push_back(colors);
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

void updateAngles(std::array<float, 3> &angles, std::array<float, 3> changes){
    for (int i = 0; i < 3; i++){
        angles[i] += changes[i];
        angles[i] = fmod(angles[i], PI * 2);
    }
}

Matrix matrixMultiply(const Matrix a, const Matrix b)
{
    if (a[0].size() != b.size())
    {
        std::cerr << "matrix dimensions do not match for multiplication." << std::endl;
        return {};
    }

    Matrix resultMatrix(a.size(), std::vector<float>(b[0].size(), 0.0f));
    for (size_t i = 0; i < a.size(); i++)
    {
        for (size_t j = 0; j < b[0].size(); j++)
        {
            for (size_t k = 0; k < a[0].size(); k++)
            {
                resultMatrix[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return resultMatrix;
}

Matrix canonicalToFullCoords(Matrix homoCoord3d, utilities::viewVolume viewVolume) {
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    auto [N,F,L,R,T,B] = viewVolume;
    const std::vector<std::vector<float>> mat = {
        {(R-L)/2, 0, 0, (L+R)/2},
        {0, (T-B)/2, 0, (T+B)/2},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };

    return matrixMultiply(mat, homoCoord3d);
}

Matrix cartToCanvasCoords(Matrix homoCoord3d)
{
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    Matrix mat1 = {
        {1, 0, 0, 0},
        {0, -1, 0, 0}, 
        {0, 0, 1, 0}, 
        {0, 0, 0, 1}
    };
    Matrix mat2 = {
        {1, 0, 0, WINDOW_WIDTH/2},
        {0, 1, 0, WINDOW_HEIGHT/2}, 
        {0, 0, 1, 0}, 
        {0, 0, 0, 1}
    };
    return matrixMultiply(mat2, matrixMultiply(mat1, homoCoord3d));
}

Matrix canvasToCartCoords(Matrix homoCoord3d) {
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    Matrix mat1 = {
        {1, 0, 0, 0},
        {0, -1, 0, 0}, 
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    Matrix mat2 = {
        {1, 0, 0, -WINDOW_WIDTH/2},
        {0, 1, 0, WINDOW_HEIGHT/2}, 
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    return matrixMultiply(mat2, matrixMultiply(mat1, homoCoord3d));
}

Matrix transformCoord(Matrix homoCoord3d, std::vector<std::pair<TransformFunc, std::any>> argPairs) {
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    Matrix newCoord = homoCoord3d;
    for (auto [func, arg]: argPairs){
        if (arg.type() == typeid(std::array<float, 3>)) {
            newCoord = func(newCoord, std::any_cast<std::array<float, 3>>(arg));
        } else if (arg.type() == typeid(float)) {
            newCoord = func(newCoord, std::any_cast<float>(arg));
        } else {
            std::cerr << "Unsupported argument type in transformCoord" << std::endl;
        }
    }
    return newCoord;
}

Matrix translateCoord(Matrix homoCoord3d, std::any arg) {
    std::array<float, 3> translate = std::any_cast<std::array<float, 3>>(arg);
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    auto [dx,dy,dz] = translate;
    Matrix translateMatrix = {
        {1,0,0,dx}, 
        {0,1,0,dy}, 
        {0,0,1,dz}, 
        {0,0,0,1}
    };
    return matrixMultiply(translateMatrix, homoCoord3d);
}

Matrix rotateCoord(Matrix homoCoord3d, std::any arg) {
    std::array<float, 3> rotate = std::any_cast<std::array<float, 3>>(arg);
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    auto [rx,ry,rz] = rotate;
    Matrix rotateMatrixX= {
        {1,0,0,0}, 
        {0,cos(rx),-sin(rx),0}, 
        {0,sin(rx),cos(rx),0}, 
        {0,0,0,1}
    };
    Matrix rotateMatrixY= {
        {cos(ry),0,sin(ry),0}, 
        {0,1,0,0}, 
        {-sin(ry),0,cos(ry),0}, 
        {0,0,0,1}
    };
    Matrix rotateMatrixZ= {
        {cos(rz),-sin(rz),0,0}, 
        {sin(rz),cos(rz),0,0}, 
        {0,0,1,0}, 
        {0,0,0,1}
    };
    
    Matrix newCoord = homoCoord3d;
    newCoord = matrixMultiply(rotateMatrixX, newCoord);
    newCoord = matrixMultiply(rotateMatrixY, newCoord);
    newCoord = matrixMultiply(rotateMatrixZ, newCoord);
    return newCoord;
}

Matrix scaleCoord(Matrix homoCoord3d, std::any arg) {
    float scale = std::any_cast<float>(arg);
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    Matrix scaleMatrix = {
        {scale,0,0,0}, 
        {0,scale,0,0}, 
        {0,0,scale,0}, 
        {0,0,0,1}
    };
    return matrixMultiply(scaleMatrix, homoCoord3d);
}

void moveCamera(utilities::camera &camera, std::array<float, 3> translate={0,0,0}, std::array<float, 3> rotate={0,0,0}){
    camera.base = transformCoord(camera.base, {{translateCoord,translate}, {rotateCoord,rotate}}); 
    updateAngles(camera.angles, rotate);
}

Matrix modelToWorld(Matrix homoCoord3d, float scale, std::array<float, 3> angles, Matrix base){
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    std::array<float, 3> flatBase = {base[0][0], base[1][0], base[2][0]};
    Matrix worldCoord = transformCoord(homoCoord3d, {{rotateCoord,angles}, {translateCoord,flatBase}, {scaleCoord,scale}});
    return worldCoord;
}

Matrix worldToCamera(Matrix homoCoord3d, std::array<float, 3> angles, Matrix base){
  //translate then rotate
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    std::array<float, 3> negBase = {-base[0][0], -base[1][0], -base[2][0]};
    std::array<float, 3> negAngles = {-angles[0], -angles[1], -angles[2]};
    Matrix camCoord = transformCoord(homoCoord3d, {{translateCoord, negBase}, {rotateCoord, negAngles}});
    return camCoord;
}

Matrix orthographicProjectCoord(Matrix homoCoord3d, utilities::viewVolume viewVolume) {
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    auto [N,F,L,R,T,B] = viewVolume;
    Matrix projectMatrix = {
        {2/(R-L),0,0,0},
        {0,2/(T-B),0,0},
        {0,0,1/(F-N),-N/(F-N)},
        {0,0,0,1}
    };
    return matrixMultiply(projectMatrix, homoCoord3d);
}

Matrix perspectiveProjectCoord(Matrix homoCoord3d, utilities::viewVolume viewVolume) {
    if (homoCoord3d.size() != 4){
        std::cerr << "Vector not a 3d homogeneous coordinate" << std::endl;
        return {};
    }
    auto [N,F,L,R,T,B] = viewVolume;
    Matrix projectMatrix = {
        {(2*N)/(R-L),0,0,0},
        {0,(2*N)/(T-B),0,0},
        {0,0,F/(F-N),-(F*N)/(F-N)},
        {0,0,1,0}
    };
    Matrix projectedCoord = matrixMultiply(projectMatrix, homoCoord3d);
    float scale = projectedCoord[3][0];
    Matrix scaleMatrix = {
        {1/scale,0,0,0},
        {0,1/scale,0,0},
        {0,0,1/scale,0},
        {0,0,0,1/scale},
    };
    return matrixMultiply(scaleMatrix, projectedCoord);
}

std::vector<float> flatten_vector(Matrix vec){
    // return the 1d version of the matrix representation of vector 
    std::vector<float> flatVec;
    for (int i=0; i<vec.size(); i++){
        flatVec.push_back(vec[i][0]);
    }
    return flatVec;
}

Matrix inverseMatrix2d(Matrix mat){
    float det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    if (det == 0) {
        std::cerr << "Matrix is singular and cannot be inverted." << std::endl;
        return {};
    }
    return {{mat[1][1] / det, -mat[0][1] / det}, {-mat[1][0] / det, mat[0][0] / det}};
}

std::array<float, 2> getBarycentric(std::array<Matrix, 3> triangle, Matrix point){
    Matrix diff = {{point[0][0] - triangle[0][0][0]}, {point[1][0] - triangle[0][1][0]}};
    Matrix mat = {
        {triangle[1][0][0] - triangle[0][0][0], triangle[2][0][0] - triangle[0][0][0]},
        {triangle[1][1][0] - triangle[0][1][0], triangle[2][1][0] - triangle[0][1][0]}
    };
    Matrix inverted = inverseMatrix2d(mat);
    if (inverted.size() == 0){
        // work with lines
        return {-1, -1};
    }
    std::vector<float> baryVector = flatten_vector(matrixMultiply(inverseMatrix2d(mat), diff));
    return {baryVector[0], baryVector[1]};
} 

bool isValidBarycentric(std::array<float, 2> barycentric){
    auto [m2, m3] = barycentric;
    return (m2 >= 0 && m3 >= 0 && m2 + m3 <= 1);
}

float getDepth(std::array<Matrix, 3> triangle, std::array<float, 2> barycentric){
    auto [m2, m3] = barycentric;
    float depth = triangle[0][2][0] * (1 - m2 - m3) + triangle[1][2][0] * m2 + triangle[2][2][0] * m3;
    return depth;
}

std::array<uint8_t, 3> getColor(std::array<float, 2> barycentric, std::array<std::array<uint8_t, 3>, 3> colors){
  // std::array<std::array<uint8_t, 3>, 3> colors = {{{255,255,0}, {255,0,0}, {0,255,0}}};
    auto [m2, m3] = barycentric;
    float m1 = 1 - m2 - m3;
    if ((m1 <1e-2) || (m2 <1e-2) || (m3 <1e-2))
        return {0, 0, 0};
    else {
        uint8_t r = floor(colors[0][0] * m1 + colors[1][0] * m2 + colors[2][0] * m3);
        uint8_t g = floor(colors[0][1] * m1 + colors[1][1] * m2 + colors[2][1] * m3);
        uint8_t b = floor(colors[0][2] * m1 + colors[1][2] * m2 + colors[2][2] * m3);
        return {r, g, b};
    }
}

std::array<float, 6> boundingBox(std::array<Matrix, 3> triangle){
    float minX = triangle[0][0][0];
    float maxX = triangle[0][0][0];
    float minY = triangle[0][1][0];
    float maxY = triangle[0][1][0];
    float minZ = triangle[0][2][0];
    float maxZ = triangle[0][2][0];
    for (int i=1; i<3; i++){
        minX = fminf(triangle[i][0][0], minX);
        maxX = fmaxf(triangle[i][0][0], maxX);
        minY = fminf(triangle[i][1][0], minY);
        maxY = fmaxf(triangle[i][1][0], maxY);
        minZ = fminf(triangle[i][2][0], minZ);
        maxZ = fmaxf(triangle[i][2][0], maxZ);
    }
    return {minX, maxX, minY, maxY, minZ, maxZ};
}