import sys
import json



'''
filename = sys.argv[1]
width = int(sys.argv[2])

with open(filename) as f:
    lines = f.readlines()

vertices = []
faces = []

for line in lines:
    line = line.split()
    if not line:
        continue
    if line[0] == "v":
        vertex = [[float(line[i])] for i in range(1, 4)]
        vertices.append(vertex)
    elif line[0] == "f":
        indices = []
        for i in range(1, len(line)):
            ind = int(line[i].split('/')[0])
            indices.append(ind)
        faces.append(indices)

def get_length(vertices):
    length = 0
    for i in range(3):
        min_val = max_val = vertices[0][i][0]
        for vertex in vertices:
            min_val = min(min_val, vertex[i][0])
            max_val = max(max_val, vertex[i][0])
        length = max(length, max_val - min_val)
    return length



object = {"obj1": {
              "scale": 0.9 * width / get_length(vertices),
              "angles": [0,0,0],
              "base": [0,0,0],
              "vertices": vertices, 
              "faces": faces
            }
          }

assert all(map(lambda vert: len(vert) == 3, vertices)), "some vertices not 3d"
assert all(map(lambda face: len(face) >= 3, faces)), "some faces only have 2 points"

with open("objects.json", "w") as f:
    json.dump(object, f)
'''

with open("p5js_code/cube.json", "r") as f:
    cube = json.load(f)
    for i in range(len(cube["obj1"]["faces"])):
        cube["obj1"]["faces"][i] = list(map(lambda ind: ind + 1, cube["obj1"]["faces"][i]))

with open("p5js_code/cube.json", "w") as f:
    json.dump(cube, f)

