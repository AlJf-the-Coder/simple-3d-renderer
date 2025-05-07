function preload() {
    objects = loadJSON('./cube.json', 
    (data) => console.log("successfully loaded objects"), 
    (err)=>console.log("error loading objects", err));
}

function setup() {
  const width = 400;
  const height = 400;
  frame_num = 0;
  createCanvas(width, height);
  camera = {base: [[0],[0],[0]], angles: [-Math.PI/4,0,0], viewVolume: {N:0, F:width, L:-width/2, R:width/2, T:height/2, B:-height/2}}
  obj = objects.obj1;
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

  depthBuffer = Array(height).fill().map(()=>Array(width).fill(Infinity));
  frameBuffer = Array(height).fill().map(()=>Array(width).fill([255, 255, 255]));

  loadPixels();
  for (let y=0; y<height; y++){
    for (let x=0; x<width; x++){
      const index = 4 * (y * width + x);
      frameBuffer[y][x] = pixels.slice(index, index + 3);
    }
  }

  updatePixels();
  
  let camRotate = [1, -1, 1];
  moveCamera(camera, undefined, camRotate);
  let objRotate = [0, 0, 0];
  drawRotatedObj(obj, objRotate);
  
  for (let y=0; y<height; y++){
    for (let x=0; x<width; x++){
      const index = 4 * (y * width + x);
      pixels[index] = frameBuffer[y][x][0];
      pixels[index + 1] = frameBuffer[y][x][1];
      pixels[index + 2] = frameBuffer[y][x][2];
    }
  }

  updatePixels();
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

function cartToCanvasCoords(homoCoord2d) {
  if (homoCoord2d.length != 3)
    throw new Error("coordinate is not a homogeneous 2d coordinate");
  const mat1 = [
	  [1, 0, 0],
	  [0, -1, 0], 
	  [0, 0, 1]
  ]
  const mat2 = [
	  [1, 0, width/2],
	  [0, 1, height/2], 
	  [0, 0, 1]
  ]
  return matMultiply(mat2, matMultiply(mat1, homoCoord2d));
}

function canvasToCartCoords(homoCoord2d) {
  if (homoCoord2d.length != 3)
    throw new Error("coordinate is not a homogeneous 2d coordinate");
  const mat1 = [
	  [1, 0, 0],
	  [0, -1, 0], 
	  [0, 0, 1]
  ]
  const mat2 = [
	  [1, 0, -width/2],
	  [0, 1, height/2], 
	  [0, 0, 1]
  ]
  return matMultiply(mat2, matMultiply(mat1, homoCoord2d));
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
  const worldCoord = transformCoord(homoCoord3d, [[scaleCoord,scale], [rotateCoord,angles], [translateCoord,base]]);
  return worldCoord;
}

function worldToCamera(homoCoord3d, angles, base){
  //translate then rotate
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const camCoord = transformCoord(homoCoord3d, [[translateCoord, base.map((c)=>[-c[0]])], [rotateCoord, angles.map((angle)=>-angle)]]);
  return camCoord;
}

function orthogonalProjectCoord(homoCoord3d, viewVolume) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const {N,F,L,R,T,B} = viewVolume;
  const projectMatrix = [
	  [1,0,0,0],
	  [0,1,0,0],
	  [0,0,0,1],
	  [0,0,0,1]
  ];
  return matMultiply(projectMatrix, homoCoord3d);
}

function perspectiveProjectCoord(homoCoord3d, viewVolume) {
  if (homoCoord3d.length != 4)
    throw new Error("coordinate is not a homogeneous 3d coordinate");
  const {N,F,L,R,T,B} = viewVolume;
  const projectMatrix = [
	  [1,0,0,0],
	  [0,1,0,0],
	  [0,0,1,0],
	  [0,0,1,0]
  ];
  let projectedCoord = matMultiply(projectMatrix, homoCoord3d);
  projectedCoord = projectedCoord.map((i)=>[i[0]/projectedCoord[3][0]]);
  return projectedCoord;
}

function drawPainterly(camCoords, perspectiveFunc, fillColor=undefined) {
  let projectedCoords = camCoords.map((coord)=>perspectiveFunc(coord, camera.viewVolume));
  const sortedPolygons = obj.faces.sort((poly1, poly2) => {
                                              return Math.max(...poly2.map(ind=>camCoords[ind - 1][2])) - 
                                                     Math.max(...poly1.map(ind=>camCoords[ind - 1][2]));
                                              //return (poly2.reduce((acc, ind)=>camCoords[ind - 1][2] + acc, 0) / poly2.length) - 
                                              //       (poly1.reduce((acc, ind)=>camCoords[ind - 1][2] + acc, 0) / poly1.length);
                                              }
                                        );
  let canvasCoords = projectedCoords.map((coord)=>cartToCanvasCoords(coord.slice(0, 3)).slice(0, 2))

  drawFace = (indices) => {
    for (let i = 0; i < indices.length; i++) {
      line(...canvasCoords[indices[i] - 1].map((c)=>c[0]), ...canvasCoords[indices[(i+1)%indices.length] - 1].map((c)=>c[0]));
    }
  }
  /*
  drawFace = (indices) => {
      beginShape();
      indices.forEach((ind)=>vertex(...canvasCoords[ind - 1].map((c)=>c[0])));
      endShape(CLOSE);
    }
  */
  
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
    polygon.forEach((ind)=>point(...canvasCoords[ind - 1].map((c)=>c[0])));
  }
}

function flatten_vector(vec){
  return vec.map(coord=>coord[0]);
}

function getDepth(triangle, barycentric){
  const [m2, m3] = barycentric
  const depth = triangle[0][2][0] * (1 - m2 - m3) + triangle[1][2][0] * m2 + triangle[2][2][0] * m3;
  return depth;
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

function getColor(barycentric){
  colors = [[255,255,0], [255,0,0], [0,255,0]];
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

function scanlineRender(triangle, perspectiveFunc){
  let projectedTri = triangle.map((coord)=>perspectiveFunc(coord, camera.viewVolume));
  let canvasTri = projectedTri.map((coord)=>cartToCanvasCoords(coord.slice(0, 3)));
  const minX = canvasTri.reduce((acc, vertex)=>Math.min(vertex[0], acc), Infinity);
  const minY = canvasTri.reduce((acc, vertex)=>Math.min(vertex[1], acc), Infinity);
  const maxX = canvasTri.reduce((acc, vertex)=>Math.max(vertex[0], acc), -Infinity);
  const maxY = canvasTri.reduce((acc, vertex)=>Math.max(vertex[1], acc), -Infinity);

  for (let y=Math.max(0, Math.floor(minY)); y <= Math.min(height - 1, Math.floor(maxY)); y++){
    for (let x=Math.max(0, Math.floor(minX)); x <= Math.min(width - 1, Math.floor(maxX)); x++){
      //add depth
      for (let i = 0; i < 3; i++)
        canvasTri[i][2][0] = triangle[i][2][0]
      const point = [[x + 0.5], [y + 0.5]];
      const barycentric = getBarycentric(canvasTri, point);
      if (isValidBarycentric(barycentric)){
        // const pointDepth = getDepth(canvasTri, point);
        const pointDepth = getDepth(canvasTri, barycentric);
        if (depthBuffer[y][x] >= pointDepth){
          frameBuffer[y][x] = getColor(barycentric);
          depthBuffer[y][x] = pointDepth;
        }
      }
    }
  }
}

function drawRaster(camCoords, perspectiveFunc, fillColor="Black") {
  stroke(fillColor)
  for (const triangle of obj.faces){
     scanlineRender(triangle.map(ind=>camCoords[ind - 1]), perspectiveFunc);
  }
}

function drawRotatedObj(obj, angles) {
  let scale, base;
  ({ scale, base } = obj);
  let worldCoords = obj.vertices.map((coord)=>modelToWorld(coord, scale, angles, base));
  obj.angles = update_angles(obj.angles, angles);
  ({ angles, base } = camera);
  let camCoords = worldCoords.map((coord)=>worldToCamera(coord, angles, base));
  drawRaster(camCoords, orthogonalProjectCoord, "orange");
  // drawPainterly(camCoords, orthogonalProjectCoord, "orange");

  //color vertices
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
