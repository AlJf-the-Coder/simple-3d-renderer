function preload() {
    // objects = loadJSON('./cube.json', 
    objects = loadJSON('../objects.json', 
    (data) => console.log("successfully loaded objects"), 
    (err)=>console.log("error loading objects", err));
}

function setup() {
  const width = 1000;
  frame_num = 0;
  createCanvas(width, width);
  camera = {base: [[0],[0],[0]], angles: [0,0,0], viewVolume: {N:width, F: 2 * width, L:-width/2, R:width/2, T:height/2, B:-height/2}};
  far = camera.viewVolume.F;
  obj = objects.obj1;
  obj.base = [0, 0, 2]
  depthBuffer = Array(height).fill().map(()=>Array(width).fill(Infinity));
  frameBuffer = Array(height).fill().map(()=>Array(width).fill([255, 255, 255]));

  //make each vertex homogeneous
  camera.base.push([1]);
  obj.vertices.forEach((vertex)=>vertex.push([1]));
}

function draw() {
  //draw a sky blue background
  background(135, 206, 235);
  textAlign(LEFT, TOP);
  text("frame_num: " + frame_num, 0, 0);
  frame_num++;

  depthBuffer = Array(height).fill().map(()=>Array(width).fill(1));
  frameBuffer = Array(height).fill().map(()=>Array(width).fill([255, 255, 255]));

  loadPixels();
  for (let y=0; y<height; y++){
    for (let x=0; x<width; x++){
      const index = 4 * (y * width + x);
      frameBuffer[y][x] = pixels.slice(index, index + 3);
    }
  }

  // let camRotate = [0.3, 0, 0.5];
  // moveCamera(camera, undefined, camRotate);
  let objRotate = [-0.3, 0, -0.5];
  drawRotatedObj(obj, objRotate);
  
  // noLoop()
}

function update_angles(angles, changes){
  const [a1, a2, a3] = angles;
  const [d1, d2, d3] = changes;
  angles = [a1 + d1, a2 + d2, a3 + d3];
  angles = angles.map((angle) => angle % (Math.PI * 2))
  return angles;
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

function canonicalToFullCoords(homoCoord3d, viewVolume) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const {N,F,L,R,T,B} = viewVolume;
  const mat = [
    [(R-L)/2, 0, 0, (L+R)/2],
    [0, (T-B)/2, 0, (T+B)/2],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
  return matMultiply(mat, homoCoord3d);
}
function cartToCanvasCoords(homoCoord3d) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const mat1 = [
	  [1, 0, 0, 0],
	  [0, -1, 0, 0], 
	  [0, 0, 1, 0], 
	  [0, 0, 0, 1]
  ]
  const mat2 = [
	  [1, 0, 0, width/2],
	  [0, 1, 0, height/2], 
	  [0, 0, 1, 0], 
	  [0, 0, 0, 1]
  ]
  return matMultiply(mat2, matMultiply(mat1, homoCoord3d));
}

function canvasToCartCoords(homoCoord3d) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const mat1 = [
	  [1, 0, 0, 0],
	  [0, -1, 0, 0], 
	  [0, 0, 1, 0]
	  [0, 0, 0, 1]
  ]
  const mat2 = [
	  [1, 0, 0, -width/2],
	  [0, 1, 0, height/2], 
	  [0, 0, 1, 0]
	  [0, 0, 0, 1]
  ]
  return matMultiply(mat2, matMultiply(mat1, homoCoord3d));
}

function transformCoord(homoCoord3d, argPairs) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  newCoord = homoCoord3d.slice();
  for (const [func, arg] of argPairs){
    newCoord = func(newCoord, arg);
  }
  return newCoord;
}

function translateCoord(homoCoord3d, translate=[0,0,0]) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const [dx,dy,dz] = translate;
  const translateMatrix = [
	  [1,0,0,dx], 
	  [0,1,0,dy], 
	  [0,0,1,dz], 
	  [0,0,0,1]
  ];
  return matMultiply(translateMatrix, homoCoord3d);
}

function rotateCoord(homoCoord3d, rotate=[0,0,0]) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const [rx,ry,rz] = rotate;
  const rotateMatrixX= [
	  [1,0,0,0], 
	  [0,Math.cos(rx),-Math.sin(rx),0], 
	  [0,Math.sin(rx),Math.cos(rx),0], 
	  [0,0,0,1]
  ];
  const rotateMatrixY= [
	  [Math.cos(ry),0,Math.sin(ry),0], 
	  [0,1,0,0], 
	  [-Math.sin(ry),0,Math.cos(ry),0], 
	  [0,0,0,1]
  ];
  const rotateMatrixZ= [
	  [Math.cos(rz),-Math.sin(rz),0,0], 
	  [Math.sin(rz),Math.cos(rz),0,0], 
	  [0,0,1,0], 
	  [0,0,0,1]
  ];
  
  let newCoord = matMultiply(rotateMatrixX, homoCoord3d);
  newCoord = matMultiply(rotateMatrixY, newCoord);
  newCoord = matMultiply(rotateMatrixZ, newCoord);
  return newCoord
}

function scaleCoord(homoCoord3d, scale=1) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const scaleMatrix = [
	  [scale,0,0,0], 
	  [0,scale,0,0], 
	  [0,0,scale,0], 
	  [0,0,0,1]
  ];
  return matMultiply(scaleMatrix, homoCoord3d);
}

function moveCamera(camera, translate=[0,0,0], rotate=[0,0,0]){
  const [x, y, z] = translate;
  camera.base = transformCoord(camera.base, [[translateCoord,translate], [rotateCoord,rotate]]); 
  camera.angles = update_angles(camera.angles, rotate);
}

function modelToWorld(homoCoord3d, scale, angles, base){
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const worldCoord = transformCoord(homoCoord3d, [[rotateCoord,angles], [translateCoord,base], [scaleCoord,scale]]);
  return worldCoord;
}

function worldToCamera(homoCoord3d, angles, base){
  //translate then rotate
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const camCoord = transformCoord(homoCoord3d, [[translateCoord, base.map((c)=>[-c[0]])], [rotateCoord, angles.map((angle)=>-angle)]]);
  return camCoord;
}

function orthographicProjectCoord(homoCoord3d, viewVolume) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const {N,F,L,R,T,B} = viewVolume;
  const projectMatrix = [
	  [2/(R-L),0,0,0],
	  [0,2/(T-B),0,0],
	  [0,0,1/(F-N),-N/(F-N)],
	  [0,0,0,1]
  ];
  return matMultiply(projectMatrix, homoCoord3d);
}

function perspectiveProjectCoord(homoCoord3d, viewVolume) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const {N,F,L,R,T,B} = viewVolume;
  const projectMatrix = [
	  [(2*N)/(R-L),0,0,0],
	  [0,(2*N)/(T-B),0,0],
	  [0,0,F/(F-N),-(F*N)/(F-N)],
	  [0,0,1,0]
  ];
  let projectedCoord = matMultiply(projectMatrix, homoCoord3d);
  const scale = projectedCoord[3][0];
  const scaleMatrix = [
    [1/scale,0,0,0],
    [0,1/scale,0,0],
    [0,0,1/scale,0],
    [0,0,0,1/scale],
  ];
  return matMultiply(scaleMatrix, projectedCoord);
}


function flatten_vector(vec){
  // return the 1d version of the matrix representation of vector 
  return vec.map(coord=>coord[0]);
}

function inverseMatrix2d(mat){
  const det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
  return [[mat[1][1] / det, -mat[0][1] / det], [-mat[1][0] / det, mat[0][0] / det]];
}

function getBarycentric(triangle, point){
  const diff = [[point[0][0] - triangle[0][0]], [point[1][0] - triangle[0][1]]];
  const mat = [
    [triangle[1][0] - triangle[0][0], triangle[2][0] - triangle[0][0]],
    [triangle[1][1] - triangle[0][1], triangle[2][1] - triangle[0][1]]
  ];
  return flatten_vector(matMultiply(inverseMatrix2d(mat), diff));
} 

function isValidBarycentric(barycentric){
  const [m2, m3] = barycentric;
  return (m2 >= 0 && m3 >= 0 && m2 + m3 <= 1);
}

function getDepth(triangle, barycentric){
  const [m2, m3] = barycentric
  const depth = triangle[0][2][0] * (1 - m2 - m3) + triangle[1][2][0] * m2 + triangle[2][2][0] * m3;
  return depth;
}

function getColor(barycentric){
  // colors = [[255,255,0], [255,0,0], [0,255,0]];
  colors = [[255,255,0], [255,255,0], [255,255,0]];
  const [m2, m3] = barycentric;
  const m1 = 1 - m2 - m3;
  if ((m1 <1e-2) || (m2 <1e-2) || (m3 <1e-2))
    return [0, 0, 0];
  else {
    const r = Math.floor(colors[0][0] * m1 + colors[1][0] * m2 + colors[2][0] * m3);
    const g = Math.floor(colors[0][1] * m1 + colors[1][1] * m2 + colors[2][1] * m3);
    const b = Math.floor(colors[0][2] * m1 + colors[1][2] * m2 + colors[2][2] * m3);
    return [r, g, b];
  }
}

function drawPainterly(camCoords, perspectiveFunc, fillColor=undefined) {
  let projectedCoords = camCoords.map((coord)=>perspectiveFunc(coord, camera.viewVolume));
  const sortedPolygons = obj.faces.sort((poly1, poly2) => {
                                              return Math.max(...poly2.map(ind=>camCoords[ind - 1][2])) - 
                                                     Math.max(...poly1.map(ind=>camCoords[ind - 1][2]));
                                              }
                                        );
  let fullCoords = projectedCoords.map((coord)=>canonicalToFullCoords(coord, camera.viewVolume));
  let canvasCoords = fullCoords.map((coord)=>cartToCanvasCoords(coord));

  drawFace = (indices) => {
      beginShape();
      indices.forEach((ind)=>vertex(...flatten_vector(canvasCoords[ind - 1]).slice(0, 2)));
      endShape(CLOSE);
    }
  
  if (fillColor)
    fill(fillColor);
  else
    noFill();

  for (const polygon of sortedPolygons) {
    stroke("green");
    strokeWeight(1);
    drawFace(polygon)
    stroke("red");
    strokeWeight(0);
    polygon.forEach((ind)=>point(...flatten_vector(canvasCoords[ind - 1]).slice(0, 2)));
  }
}

function drawWireframe(camCoords, perspectiveFunc, strokeColor="Black") {
  let projectedCoords = camCoords.map((coord)=>perspectiveFunc(coord, camera.viewVolume));
  let fullCoords = projectedCoords.map((coord)=>canonicalToFullCoords(coord, camera.viewVolume));
  let canvasCoords = fullCoords.map((coord)=>cartToCanvasCoords(coord));

  drawFace = (indices) => {
    for (let i = 0; i < indices.length; i++) {
      line(...flatten_vector(canvasCoords[indices[i] - 1]).slice(0, 2), ...flatten_vector(canvasCoords[indices[(i+1)%indices.length] - 1]).slice(0, 2));
    }
  }

  obj.faces.forEach((polygon) => {
      stroke(strokeColor);
      strokeWeight(4);
      drawFace(polygon)
      stroke("Red");
      strokeWeight(10);
      polygon.forEach((ind)=>point(...flatten_vector(canvasCoords[ind - 1]).slice(0, 2)));
    }
  )
}

function scanlineRender(triangle, perspectiveFunc){
  let projectedTri = triangle.map((coord)=>perspectiveFunc(coord, camera.viewVolume));
  let fullTri = projectedTri.map((coord)=>canonicalToFullCoords(coord, camera.viewVolume));
  let canvasTri = fullTri.map((coord)=>cartToCanvasCoords(coord));
  const minX = canvasTri.reduce((acc, vertex)=>Math.min(vertex[0], acc), Infinity);
  const minY = canvasTri.reduce((acc, vertex)=>Math.min(vertex[1], acc), Infinity);
  const maxX = canvasTri.reduce((acc, vertex)=>Math.max(vertex[0], acc), -Infinity);
  const maxY = canvasTri.reduce((acc, vertex)=>Math.max(vertex[1], acc), -Infinity);

  for (let y=Math.max(0, Math.floor(minY)); y <= Math.min(height - 1, Math.floor(maxY)); y++){
    for (let x=Math.max(0, Math.floor(minX)); x <= Math.min(width - 1, Math.floor(maxX)); x++){
      const point = [[x + 0.5], [y + 0.5]];
      const barycentric = getBarycentric(canvasTri, point);
      if (isValidBarycentric(barycentric)){
        const pointDepth = getDepth(canvasTri, barycentric);
        if (pointDepth >= 0 && pointDepth <= 1 && pointDepth <= depthBuffer[y][x]){
          frameBuffer[y][x] = getColor(barycentric).map(color=>Math.floor((1-pointDepth)*color));
          depthBuffer[y][x] = pointDepth;
        }
      }
    }
  }
}

function drawRaster(camCoords, perspectiveFunc) {
  for (const triangle of obj.faces){
     scanlineRender(triangle.map(ind=>camCoords[ind - 1]), perspectiveFunc);
  }
}

function drawRotatedObj(obj, angles) {
  let scale, base;
  ({ scale, base } = obj);
  obj.angles = update_angles(obj.angles, angles);
  let worldCoords = obj.vertices.map((coord)=>modelToWorld(coord, scale, obj.angles, base));
  ({ angles, base } = camera);
  let camCoords = worldCoords.map((coord)=>worldToCamera(coord, angles, base));

  const perspectiveFunc = orthographicProjectCoord;
  // const perspectiveFunc = perspectiveProjectCoord;

  drawRaster(camCoords, perspectiveFunc);

  for (let y=0; y<height; y++){
    for (let x=0; x<width; x++){
      const index = 4 * (y * width + x);
      pixels[index] = frameBuffer[y][x][0];
      pixels[index + 1] = frameBuffer[y][x][1];
      pixels[index + 2] = frameBuffer[y][x][2];
    }
  }

  updatePixels();
  drawWireframe(camCoords, perspectiveFunc, "Green");
}

/*
function getAngle(x,y){
  let theta1,theta2,theta3;
  let [x,y] = canvasToCartCoords(mouseX, mouseY)
  theta1 =  getAngle(x,y);
  theta2 = mouseX/(width/2) * Math.PI;
  theta3 = mouseY/(height/2) * Math.PI;
  let angle = Math.abs(Math.atan(y/x));
  if (Number.isNaN(angle))
    angle = Math.PI/2;
  if (x<0 && y>=0)
    angle = Math.PI - angle;
  else if (x<0 && y<0)
    angle = angle + Math.PI;
  else if (x>=0 && y<0)
    angle = 2*Math.PI - angle;
  return angle
}
*/
