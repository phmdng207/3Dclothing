import numpy as np
import cv2
import matplotlib.pyplot as plt 
import glob
import os
import sys


# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
    
def inside_object(mask, pt1, pt2, pt3):

    pt = (int((pt1[0] + pt2[0] + pt3[0])/3), int((pt1[1] + pt2[1]+ pt3[1])/3)) # centroid
    
    if mask[pt[1], pt[0]] > 0:
        return True
    else:
        return False
        
def filter_inner_triangles(r, subdiv, object_mask) :

    triangleList = subdiv.getTriangleList();
   
    inside_triangleList = []
    for t in triangleList :
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            if inside_object(object_mask, pt1, pt2, pt3) :    
                inside_triangleList.append([pt1, pt2, pt3])
                
    return  inside_triangleList           
                
def draw_delaunay(img, triangleList, delaunay_color) :
    triangleList = np.array(triangleList, dtype = np.int64)
    for t in triangleList :
        cv2.line(img, t[0], t[1], delaunay_color, 1) #, cv2.CV_AA, 0)
        cv2.line(img, t[1], t[2], delaunay_color, 1) #, cv2.CV_AA, 0)
        cv2.line(img, t[2], t[0], delaunay_color, 1) #, cv2.CV_AA, 0)


def uv2mesh(img, mask):

    
    # 1. contour from mask 
    contours,hierarchy = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE  )
    cnt = contours[0]
    print(f"contours.shape:{contours[0].shape}")
    print(f"hierarchy:{hierarchy}")
    print(f"shape:{contours[0].shape[0]}")
    cv2.drawContours(img,contours,0,(0,0,255),2)
    plt.imshow(img[:,:,::-1])
    plt.show()

    
    # 2. Delaunoi points 
    
    # the exterior points 
    contour  =  contours[0].reshape([-1,2])
    #print(contour)
    
    contour_points = []
    #= contour.tolist()
    # add contour points 
    contour_step = 5
    grid_ratio= 20
    for i in range(0, len(contour), contour_step): 
        #print(contours[0][i])
        contour_points.append((int(contour[i,0]),int(contour[i,1])) )
    
    for i in range(0, len(contour_points), contour_step):
        cv2.drawMarker(img, (contour_points[i][0],contour_points[i][1]), color=(0,255,0), markerType=cv2.MARKER_CROSS, markerSize = 4, thickness=1)
        
    # add inside points 
    min_x, min_y = np.min(contour, 0)
    max_x, max_y = np.max(contour, 0)
    print( min_x, min_y)
    print(max_x, max_y)
    grid_step = (max_y - min_y)//grid_ratio
    #print(f"step:{grid_step}")
    interior_points = []
    for x in range(min_x + grid_step, max_x, grid_step): 
        for y in range(min_y + grid_step, max_y, grid_step):
            #print( x, y)
            if mask[y,x] > 0:
                contour_points.append((x, y))
                interior_points.append((x, y))
    
    for i in range(0, len(interior_points)):
        cv2.drawMarker(img, (interior_points[i][0],interior_points[i][1]), color=(0,255,0), markerType=cv2.MARKER_CROSS, markerSize = 4, thickness=1)
       
    plt.imshow(img)
    plt.show()
    
    # 3. triangle mesh 
    rect = (0, 0, mask.shape[1], mask.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    
    for p in contour_points:
        subdiv.insert(p)
    for p in interior_points:
        subdiv.insert(p)
 
    r = (0, 0, img.shape[1], img.shape[0])
    triangleList = filter_inner_triangles(r, subdiv, mask)
    draw_delaunay(img, triangleList, (255, 0, 0))
   
    plt.imshow(img)
    plt.show()
    return triangleList

def makeMask(cloth, outputPath):
    _, mask = cv2.threshold(cloth, 0, 255,  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) 
    #mophology
    kernel = np.ones((11,11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(outputPath, mask)
    return mask

    
def make3DMesh(triangleListF, fImg, bImg):
    
    '''
      using only front size vertices: from triangle 
    
    '''
    #
    triangleArr = np.array(triangleListF)
    triangleVerticesArr = triangleArr.reshape((-1,2))
    
    unique = np.unique(triangleVerticesArr, axis=0)  # @TODO triagnles share vertices (duplicated) => using traingle...
    
    vertices = np.zeros_like(unique)
    verticesLen = vertices.shape[0]
    faceF = np.zeros(triangleVerticesArr.shape[:1], dtype=np.uint32) #이후 3개씩 끊어서 사용
    faceB = np.zeros(triangleVerticesArr.shape[:1], dtype=np.uint32)
    for j, tv in enumerate(triangleVerticesArr):     # tv = (x, y) in pixel domain 
        for i, val in enumerate(unique):             # searching the vertices with same (x, y) in triangle vertices 
            if(tv[0] == val[0])and(tv[1]==val[1]):   # is same in x and y coordinate 
                faceF[j] = i+1              # in list or numpy 0-index but obj file use 1-index 
                faceB[j] = i+1+verticesLen  # verticesLen : backside faces are after front 

    faceF = faceF.reshape((-1,3))
    faceB = faceB.reshape((-1,3))

    (h,w) = clothF.shape[:2]
    ratio = w/h
    
    uv_vertices = unique.copy()
    # nomalize (0,1) for uvmap cooridniate 
    uv_vertices[:,0] = uv_vertices[:,0]/w  # x 
    uv_vertices[:,1] = -uv_vertices[:,1]/h # y 

    #shift -0.5 to 0.5
    vertices[:,0] = (uv_vertices[:,0]-0.5) * ratio
    vertices[:,1] = uv_vertices[:,1]+0.5

    return vertices, uv_vertices, faceF, faceB



def save_mesh_as_obj(vertices, uv_vertices, faceF, faceB, mtl_filename, obj_path ):

    with open(obj_path, 'w') as f:

        f.write( 'mtllib %s\n' % (mtl_filename))   # Linking info 
    
        #front vertices
        for x,y in vertices:
            f.write('v %.4f %.4f %.4f\n'%(x, y, 0.5))
         
        #back vertices
        for x,y in vertices:
            f.write('v %.4f %.4f %.4f\n'%(x, y, -0.5))
          
        #front normal vector
        f.write("vn 0.0000 -1.0000 0.0000\n")
        #back normal vector
        f.write("vn 0.0000 1.0000 0.0000\n")

        #  v = 1.0
        #  *--------------
        #   FRONT |  BACK
        #   UVMAP |  UVMAP
        #  ---------------> u =1.0
        # (0,0) 
        #front UVmap
        for x, y in uv_vertices:
            f.write('vt %.4f %.4f\n'%(x/2, y))
        
        #back UVmap
        for x, y in uv_vertices:
            f.write('vt %.4f %.4f\n'%(0.5 + x/2, y))
       
        #front faces
        for a, b, c in faceF:
            f.write('f %d/%d/1 %d/%d/1 %d/%d/1\n'%(c,c,b,b,a,a))                   
            #f.write('f %d/%d/1 %d/%d/1 %d/%d/1\n'%(a,a,b,b,c,c))
            
        #back faces
        for a, b, c in faceB:
            f.write('f %d/%d/2 %d/%d/2 %d/%d/2\n'%(a,a,b,b,c,c))
      
  
def align_front_back(clothF_mask, clothB_mask, clothB):

    '''
       align the front and backside mask as much as possible (using PCA) @TODO: better Align method e.g. PROCUSTES etc  
       take only the intersection for valid area @TODO 
       @TODO left right flipped
       
    '''
    #1. calcuate PCA for all pts 
    ptsF = np.array(np.where(clothF_mask>127)).T  # the clothing area pixels coorinates
    ptsB = np.array(np.where(clothB_mask>127)).T
    originF = (np.mean(ptsF[:,1]), np.mean(ptsF[:,0]))
    originB = (np.mean(ptsB[:,1]), np.mean(ptsB[:,0]))
    covF = np.cov(ptsF[:,1], ptsF[:,0])
    covB = np.cov(ptsB[:,1], ptsB[:,0])
    evalF, evecF = np.linalg.eig(covF)
    evalB, evecB = np.linalg.eig(covB)

    if evalF[1]>evalF[0]:
        evalF = evalF[::-1]
        evecF = evecF[:,::-1]
    
    if evalB[1]>evalB[0]:
        evalB = evalB[::-1]
        evecB = evecB[:,::-1]

    evalF = np.sqrt(evalF)
    evalB = np.sqrt(evalB)

    affinePtsF = np.float32([originF, [originF[0]+int(evalF[0]*evecF[0][0]), originF[1]+int(evalF[0]*evecF[0][1])], [originF[0]+int(evalF[1]*evecF[1][0]), originF[1]+int(evalF[1]*evecF[1][1])]])
    affinePtsB = np.float32([originB, [originB[0]+int(evalB[0]*evecB[0][0]), originB[1]+int(evalB[0]*evecB[0][1])], [originB[0]+int(evalB[1]*evecB[1][0]), originB[1]+int(evalB[1]*evecB[1][1])]])
    
    # get Affine transfrorm between two mask 

    M = cv2.getAffineTransform(affinePtsB, affinePtsF)
    clothB = cv2.warpAffine(clothB, M, (w,h))
    clothB_mask = cv2.warpAffine(clothB_mask, M, (w,h))
    # cv2.line(clothF, affinePtsF.astype(np.uint16)[0], affinePtsF.astype(np.uint16)[1], (0,0,255), 4)
    # cv2.line(clothF, affinePtsF.astype(np.uint16)[0], affinePtsF.astype(np.uint16)[2], (255,0,0), 4)
    # cv2.line(clothF, affinePtsB.astype(np.uint16)[0], affinePtsB.astype(np.uint16)[1], (0,255,255), 4)
    # cv2.line(clothF, affinePtsB.astype(np.uint16)[0], affinePtsB.astype(np.uint16)[2], (255,255,0), 4)
    # cv2.imshow('a', cv2.resize(clothF, (w//2, h//2)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # cv2.imshow('a', affineCloth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 3. Intersection area for common mask area 
    mask = np.bitwise_and(clothF_mask, clothB_mask)
    
    return mask, clothB
 
if __name__ == "__main__":
    
    
    # 1. check input images 
    imgDir = "cloth"  # base  dir
    frontnameList = sorted(glob.glob(imgDir + "/*_F.jpg"))  # all front images 
    backnameList = sorted(glob.glob(imgDir + "/*_B.jpg"))   # all backside images 
    #check pairs of front and back 
    filenames = []
    for i in range(len(frontnameList)):
        a = frontnameList[i][len(imgDir)+1:-6]
        b = backnameList[i][len(imgDir)+1:-6]
        if(a != b):
            print("E: Uncorresponding pair exists.")
            sys.exit(1)
        else: filenames.append(a)
    

    # convert all
    for filename in filenames:
        # 2. load color and mask images 
        clothF_path = os.path.join(imgDir, filename + "_F.jpg")
        clothB_path = os.path.join(imgDir, filename + "_B.jpg")
        clothF_mask_path = os.path.join("cloth-mask", filename + "_mask_F.png")
        clothB_mask_path = os.path.join("cloth-mask", filename + "_mask_B.png")
        obj_path = "obj" # os.path.join("obj", filename)

        clothF = cv2.imread(clothF_path)
        clothB = cv2.imread(clothB_path)
        clothF_mask = makeMask(cv2.cvtColor(clothF, cv2.COLOR_BGR2GRAY), clothF_mask_path)
        clothB_mask = makeMask(cv2.cvtColor(clothB, cv2.COLOR_BGR2GRAY), clothB_mask_path)
        
        (h,w) = clothF.shape[:2]

        # 2. align front and back 
        mask, clothB = align_front_back(clothF_mask, clothB_mask, clothB)
        
        
        # 3. make mesh 
        # 3.1 2D mesh traingle 
        triangleListF = uv2mesh(clothF.copy(), mask)
        # 3.2 combine front and back and into 3D mesh 
        vertices, uv_vertices, faceF, faceB = make3DMesh(triangleListF, clothF, clothB)
        
        # 4.1 Geometry (main Obj file)
        objfile_path = os.path.join(obj_path, filename + ".obj")
        mtl_filename = filename + ".mtl"
        save_mesh_as_obj(vertices, uv_vertices, faceF, faceB, mtl_filename, objfile_path)
        
        # 4.2 materal file 
        with open(objfile_path.replace('.obj', '.mtl'), "w") as f:         # linking file    
            f.write("newmtl material_0\n")
            f.write("# shader_type beckmann\n")
            f.write("map_Kd %s"%(filename + "_texture.png"))
            print('..Output mesh saved to: ', objfile_path.replace('.obj', '.mtl')) 
        
        # 4.2. combine front and back into one texture image 
        texture = cv2.hconcat([clothF, clothB])  # horizontal direction 
        cv2.imwrite(os.path.join(obj_path, filename + "_texture.png"), texture)

        print("Done!!!")
    

        
        