function preload() {
  objects = loadJSON('./cube.json', 
  // objects = loadJSON('../muscles.json', 
    (data) => console.log("successfully loaded objects"), 
    (err)=>console.log("error loading objects", err));
}

function setup() {
  const width = 600;
  const height = 600;
  createCanvas(width, height);
  camera = {base: [[0],[0],[0]], angles: [-Math.PI/4,0,0], viewVolume: {N:0, F:width, L:-width/2, R:width/2, T:height/2, B:-height/2}}
  obj = objects.obj1;
  //make each vertex homogeneous
  camera.base.push([1]);
  obj.vertices.forEach((vertex)=>vertex.push([1]));
}

function draw() {
  //draw a sky blue background
  background(135, 206, 235);

  let camRotate = [0.01, 0.01, 0.01];
  moveCamera(camera, undefined, camRotate);
  let objRotate = [0.01, 0.01, 0.01];
  drawRotatedObj(obj, objRotate);
  //noLoop()
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

  /*
  drawFace = (indices) => {
    for (let i = 0; i < indices.length; i++) {
      line(...canvasCoords[indices[i] - 1].map((c)=>c[0]), ...canvasCoords[indices[(i+1)%indices.length] - 1].map((c)=>c[0]));
    }
  }
  */
  drawFace = (indices) => {
      beginShape();
      indices.forEach((ind)=>vertex(...canvasCoords[ind - 1].map((c)=>c[0])));
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
    polygon.forEach((ind)=>point(...canvasCoords[ind - 1].map((c)=>c[0])));
  }

}

function drawRaster(camCoords, perspectiveFunc, fillColor=undefined) {
  //hahah
  const depthBuffer = Array(height).fill().map(()=>Array(width).fill(Infinity));

}

function drawRotatedObj(obj, angles) {
  let scale, base;
  ({ scale, base } = obj);
  let worldCoords = obj.vertices.map((coord)=>modelToWorld(coord, scale, angles, base));
  obj.angles = update_angles(obj.angles, angles);
  ({ angles, base } = camera);
  let camCoords = worldCoords.map((coord)=>worldToCamera(coord, angles, base));
  drawPainterly(camCoords, orthogonalProjectCoord, "orange");

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
