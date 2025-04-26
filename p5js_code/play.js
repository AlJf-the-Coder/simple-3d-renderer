//import objects from './objects.json' assert {type: 'json'};

const width = 600;
const height = 400;
camera = {base: [[0],[0],[0]], angles: [-Math.PI/4,0,0], viewVolume: {N:0, F:width, L:-width/2, R:width/2, T:height/2, B:-height/2}}
//obj = objects.obj1;
obj = require("./objects.json").obj1;
//make each vertex homogeneous
camera.base.push([1]);
obj.vertices.forEach((vertex)=>vertex.push([1]));

const depthBuffer = Array(height).fill().map(()=>Array(width).fill(Infinity));
function groundObject(obj, ground){
  let minZ = obj["vertices"].reduce((vertex, acc)=>Math.min(vertex[2], acc), Infinity);
  obj["base"] = [0, 0, ground - minZ]
}

function pointInTriangle(triangle, point){
  //first small triangle
  point, triangle[0], triangle[1]
  point, triangle[1], triangle[2]
  point, triangle[2], triangle[0]
  return //area of three small triangles == area triangle
}

function crossProduct(triangle){
  const [a1, a2, a3] = [[triangle[1][0] -triangle[0][0]], [triangle[1][0] -triangle[0][0]], [triangle[1][2] - triangle[0][2]]];
  const [b1, b2, b3] = [[triangle[2][0] -triangle[0][0]], [triangle[2][0] -triangle[0][0]], [triangle[2][2] - triangle[0][2]]];
  const res = [[a2 * b3 - a3 * b2], [a3 * b1 - a1 * b3], [a1 * b2 - a2 * b1]];
  return res;
}

function getDepth(triangle, point=null){
  if (!point){
    return triangle.reduce((vertex, acc)=>vertex[2] + acc, 0) / 3
  }
  if (!pointInTriangle(triangle, point)){
    return Infinity
  }
  normal = crossProduct(triangle);
  return (point[0] * normal[0], point[1] * normal[1])
  //make plane function
}

function drawPainterly(obj){
  sortedTriangles = obj["triangles"].sort((triangle)=>triangle[2]);
  for (const triangle of sortedTriangles){
    scanline_raster(triangle)
  }
}

function scanline_raster(triangle){
  minX = triangle.reduce((vertex, acc)=>Math.min(vertex[0], acc), Infinity)
  minY = triangle.reduce((vertex, acc)=>Math.min(vertex[1], acc), Infinity)
  maxX = triangle.reduce((vertex, acc)=>Math.max(vertex[1], acc), -Infinity)
  maxY = triangle.reduce((vertex, acc)=>Math.max(vertex[1], acc), -Infinity)
  for (let y=minY; y <= maxY; y++){
    for (let x=minX; x <= maxX; x++){
      if (pointInTriangle(triangle, point))
        if (depthBuffer[y][x] >= getDepth(triangle, point))
          fill_pixel
    }
  }
}
function matMultiply(matA, matB){
  if (matA[0].length != matB.length)
    throw new Error(`inner dimensions do not match: ${matA[0].length} != $${matB.length}`);
  const newMat = Array(matA.length).fill().map(()=>Array(matB[0].length).fill(0));
  for (let i=0; i < matA.length; i++){
    for (let j=0; j < matB[0].length; j++){
      for (let k=0; k < matA[0].length; k++){
        newMat[i][j] += matA[i][k] * matB[k][j];
      }
    }
  }
  return newMat;
}