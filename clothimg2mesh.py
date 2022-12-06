import numpy as np
import cv2
import matplotlib.pyplot as plt 
import os

'''

  convert clothing color image + mask to 2D triangle mesh 
  for maniplation as mesh 
    

 2022.11.10. (0.3.1) bug fix:  check the landmarks is already in the contour points list 
 2022.11.11  (0.4.0) triangulation algorithm change using traingle package 
 
'''

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
    
def inside_object_simple(mask, pt0, pt1, pt2):
    
    # @ToDO check all pixel positions 
    # now simply check centroid of triangle 

    pt = (int((pt0[0] + pt1[0] + pt2[0])/3), int((pt0[1] + pt1[1]+ pt2[1])/3)) # centroid
    
    if mask[pt[1], pt[0]] > 0:
        return True
    else:
        return False

def inside_object(mask, pt0, pt1, pt2):
    
    # @ToDO check all pixel positions 
    # now simply check centroid of triangle 

    v1 = (pt1[0] - pt0[0], pt1[1] - pt0[1])
    v2 = (pt2[0] - pt0[0], pt2[1] - pt0[1])
    
    for t1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for t2 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            if t1 + t2 < 1.0:
                p = (int(pt0[0] + t1 * v1[0] + t2*v2[0]), int(pt0[1] + t1 * v1[1] + t2*v2[1])) 
                if mask[p[1], p[0]] <= 0:
                    return False
                    
    return True
        
    
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

    for t in triangleList :
        cv2.line(img, t[0], t[1], delaunay_color, 1) #, cv2.CV_AA, 0)
        cv2.line(img, t[1], t[2], delaunay_color, 1) #, cv2.CV_AA, 0)
        cv2.line(img, t[2], t[0], delaunay_color, 1) #, cv2.CV_AA, 0)


def getCurvature(contour, stride=1):
    
    curvature=[]
    
    assert stride < len(contour),"stride must be shorther than length of contour"
 
    for i in range(len(contour)):
    
        before=i-stride+len(contour) if i-stride<0 else i-stride
        after=i+stride-len(contour) if i+stride>=len(contour) else i+stride
        
        if False:
            f1x,f1y=(contour[after]-contour[before])/stride
            f2x,f2y=(contour[after]-2*contour[i]+contour[before])/stride**2
            denominator=(f1x**2+f1y**2)**3+1e-11
            
            curvature_at_i  =  np.sqrt(4*(f2y*f1x-f2x*f1y)**2/denominator) if denominator > 1e-12 else -1
        else:
            v1  =   contour[before] - contour[i]
            v2  =   contour[after]-contour[i]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            cross = np.cross(v1, v2)
            curvature_at_i  = np.arcsin(cross/(v1_norm*v2_norm))*180.0/np.pi
            
        curvature.append(curvature_at_i)
    
    curvature = np.array(curvature)
    
    return curvature
    
def find_nearest_point(pt, contour):
    
    ''' KD tree is not for this, since only one time used '''
    
    distances = (contour[:,0]-pt[0])**2 + (contour[:,1]-pt[1])**2 
    return np.argmin(distances)
    
   

def nearest_dist(nearest_idx, landmarks, contour):


    xt, yt = contour[nearest_idx,0], contour[nearest_idx,1]
    distances = (contour[landmarks,0]-xt)**2 + (contour[landmarks,1]-yt)**2 
    return np.min(distances)
  
    # @TODO efficency 
    '''
  
    xt, yt = contour[nearest_idx,0], contour[nearest_idx,1]
    
    x, y  = contour[landmarks[0],0], contour[landmarks[0],1]
    min_dist =  (x-xt)*(x-xt) + (y-yt)*(y-yt)
    
    for i in range(1, len(landmarks)):
        x, y = contour[landmarks[i],0], contour[landmarks[i],1]
        dist = (x-xt)*(x-xt) + (y-yt)*(y-yt)
        if dist < min_dist:
            min_dist = dist
        
    return min_dist
    '''
    
    
def detect_sewing_edges(contour, landmarks, img, vis = True):

    # ask to user for sewing edge 
    # return list of (start index, end index) of contours for sewing edges
    # @TODO make it automatically (DL?)

    # 1. visualize the landmarks 
    plt.ion()
    plt.imshow(img)
    plt.title("please select the sewing edges")
    plt.show()
    
    # 2. get the edge from users
    print("please select the sewing edges")
    sewing_edges = []
    for i in range(len(landmarks)):    
        y = input(f"{i} to {(i+1)%len(landmarks)} (y/n):")
        if y == 'y' or y == 'Y':
            s, e = landmarks[i], landmarks[(i+1)%len(landmarks)]
            sewing_edges.append( (s,e))            
            if vis:
                if s < e:
                    idx_list = list(range(s,(e+1)))
                else:
                    idx_list = list(range(s,len(contour))) + list(range(0,e+1))
                segment = contour[idx_list,:].reshape([-1,1,2])
                print(f"segment:{s},{e}") # ,{segment}")
                #cv2.drawContours(img, [segment], -1, (255,255,0)) 
                cv2.polylines(img, [segment], isClosed = False, color = (255,255,0), thickness = 5)            
                plt.imshow(img), plt.title('sewing edges')
                plt.show()
                
                
    #plt.ioff()    
    
    return sewing_edges

def detect_landmarks(contour, contour_points, img,  epsilon = 0.001, min_dist = 25, debug = False):

    # contour:  the exterior contour points in numpy 
    # contour_points : sampled points list in contours
    # epsilon:  torelance for cv2.approxPolyDP
    # min_dist: radius of non-multiple points samples
    # return : return the index list for the corners 
   

    #  Detect Corners 
    # Approximate the contour,  @TODO not use appxomiation for sampling 
    epsilon = epsilon*cv2.arcLength(contour, True)    
    approx = cv2.approxPolyDP(contour, epsilon, True)
    contour_approx  =  approx.reshape([-1,2])
    # 2.2 get angles 
    curvature = getCurvature(contour_approx) 
    
    #print(f"curvature:{curvature}")    # 4, 5 
    landmarks = []
    # 2.3 detect and marking the corners 
    curvature_thres  = 30 # degree
    if debug:
        print(f"len(contour_approx)={len(contour_approx)}")
        print(f"curvature={curvature}")
        
    count_landmarks = 0
    for i in range(len(contour_approx)):
        if curvature[i] > curvature_thres or  curvature[i] < -curvature_thres:  # some meaningfull angles 
            # find closest contour points 
            nearest_idx = find_nearest_point((contour_approx[i,0],contour_approx[i,1]), contour)
            #print(f"approx={contour_approx[i,:]}, contour={contour[nearest_idx]} at {nearest_idx}")
            if not(nearest_idx in contour_points):  # BUG FIX: 2022.11.10
                if (len(landmarks) == 0) or (not(nearest_idx in landmarks) and nearest_dist(nearest_idx,landmarks, contour) > min_dist**2) : 
                    landmarks.append(nearest_idx)  
                    contour_points.append(nearest_idx)
                    # viualize 
                    cv2.drawMarker(img, (contour[nearest_idx,0],contour[nearest_idx,1]), color=(255,0,0), markerType=cv2.MARKER_STAR, markerSize = 4, thickness=1)
                    cv2.putText(cloth, str(count_landmarks), org = (contour[nearest_idx,0],contour[nearest_idx,1]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,0,0), thickness = 1)
                    count_landmarks += 1

    return landmarks
    

def uv2mesh_old(img, mask, cloth_type):

    # find contours 
    # sample contour points 
    # get the landmarks from the contours (though approximation and cuvatures) and non-maxum supression 
    # ask the sewing boundary 
    # trainglurization 
    
    # 1. contour from mask 
    if cv2.__version__[0] == '3':  
        __,contours,hierarchy = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE  )
    else:
        contours,hierarchy = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE  )
    
    print(f"len(contours):{len(contours)}")
    print(f"contours[0].shape:{contours[0].shape}")
    print(f"hierarchy:{hierarchy}")
    #cv2.drawContours(img,[box],0,(0,0,255),2)
    
    # the exterior points 
    contour  =  contours[0].reshape([-1,2])
    #print(contour)
    min_x, min_y = np.min(contour, 0)
    max_x, max_y = np.max(contour, 0)
    #print( min_x, min_y)
    #print(max_x, max_y)
    grid_step = (max_y - min_y)//20
    #print(f"step:{grid_step}")
        
    contour_points = []
    #= contour.tolist()
    # add contour points
    i = 0
    contour_points.append(i)
    cv2.drawMarker(img, (contour[i,0],contour[i,1]), color=(0,255,0), markerType=cv2.MARKER_CROSS, markerSize = 4, thickness=1)
    for i in range(1, len(contour)): 
        dx, dy = contour[i-1,:] - contour[i,:]
        if dx*dx + dy*dy > grid_step*grid_step:
            contour_points.append(i)
            cv2.drawMarker(img, (contour[i,0],contour[i,1]), color=(0,255,0), markerType=cv2.MARKER_CROSS, markerSize = 4, thickness=1)

    # 2. detect landmark corners 
    landmarks = detect_landmarks(contour, contour_points, img)    
    print(landmarks)

    # 3. detect sewing edge 
    sewing_eddges =  detect_sewing_edges(contour, landmarks, img)
    print(f"sewing_edge: {sewing_eddges}")
    '''
    for r in sewing_eddges:
        s, e = r[0], r[1]
        if s < e:
            idx_list = list(range(s,(e+1)))
        else:
            idx_list = list(range(s,len(contour))) + list(range(0,e+1))
        segment = contour[idx_list,:].reshape([-1,1,2])
        print(f"segment:{s},{e}") # ,{segment}")
        #cv2.drawContours(img, [segment], -1, (255,255,0)) 
        cv2.polylines(img, [segment], isClosed = False, color = (255,255,0), thickness = 5)
        
    plt.imshow(img), plt.title('sewing edges')
    plt.show()
    _ = input(f"next?")
    '''
    
    # 4. create 2D triangle mesh 
    # 4.1. ceate subdiv 
    rect = (0, 0, mask.shape[1], mask.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    # 4.2. add boundary points 
    for i in contour_points:
        p = (contour[i, 0], contour[i,1])
        subdiv.insert(p)
    
    # 4.3 add internal points 
    
    interior_points = []
    for x in range(min_x + grid_step, max_x, grid_step): 
        for y in range(min_y + grid_step, max_y, grid_step):
            if mask[y,x] > 0:            
                subdiv.insert( (x,y))
                cv2.drawMarker(img, (x,y), color=(0,255,0), markerType=cv2.MARKER_CROSS, markerSize = 4, thickness=1)
    
    # 4.3 remove traingles outside of the contour             
    r = (0, 0, img.shape[1], img.shape[0])
    triangleList = filter_inner_triangles(r, subdiv, cloth_mask)
    draw_delaunay(img, triangleList, (255, 0, 0))
   
    plt.imshow(img), plt.title('traingle mesh')
    plt.show()
    _ = input(f"next?")
    
    
def save_tri_obj(tri, file_path, texture_path = None, texture_size = None):

    ''' save as Waterfront OBJ file (@TODO Does Traingle package have an API for it?)
        
        none texture version and texture version 
        
    '''
    with open(file_path, "w") as fp:

        # 1. header part if texture used
        if texture_path is not None:
             mtl_path = file_path + ".mtl"
             fp.write( 'mtllib %s\n' % (mtl_path))   # Linking info 
             with open(mtl_path, "w") as fp2:         # linking file    
                fp2.write("newmtl material_0\n")
                fp2.write("# shader_type beckmann\n")
                fp2.write("map_Kd %s"%(texture_path))
                print('..Output mesh saved to: ', mtl_path) 
          
        # 2. vertices     
        v = tri['vertices']
        v[:,0] = v[:,0]/texture_size[0]  # x width normalization   
        v[:,1] = v[:,1]/texture_size[1]  # y height normalization 
        for i in range(len(v)): 
            fp.write( 'v %.2f %.2f %.2f\n' % ( v[i,0], 1.0 - v[i,1], 0) )  # x, y, z = 0, up side down
           
        # 3. texture vertices if used    
        if texture_path is not None:    
            vt = v
            for i in range(len(vt)): 
                fp.write( 'vt %.2f %.2f %.2f\n' % (vt[i,0], 1.0 - vt[i,1], 0) )  # x, y, z = 0  The UV coordinate u-right, v up   
        
        # 4. faces 
        if texture_path is not None:                   
            f = tri['triangles']   
            for i in range(len(f)): # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d/%d %d/%d %d/%d\n' %  (f[i,0]+1, f[i,0]+1, f[i,1]+1, f[i,1]+1, f[i,2]+1, f[i,2]+1) )  # index from 1 in OBJ file 
        else:
            f = tri['triangles']   
            for i in range(len(f)): # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[i,0]+1, f[i,1]+1, f[i,2]+1) )   # index from 1 in OBJ file 
    ## Print message
    print('..Output mesh saved to: ', file_path)       
    

def uv2mesh_using_trangle(img, mask, cloth_type, num_grid = 10, debug = False):

    import triangle as tr
    # pip install triangle  
    

    # find contours 
    # sample contour points 
    # @TODO: recover !!! get the landmarks from the contours (though approximation and cuvatures) and non-maxum supression 
    # trainglurization 
    
    # 1. contour from mask 
    if cv2.__version__[0] == '3':  
        __,contours,hierarchy = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE  )
    else:
        contours,hierarchy = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE  )
    
    print(f"len(contours):{len(contours)}")
    print(f"contours[0].shape:{contours[0].shape}")
    print(f"hierarchy:{hierarchy}")
    #cv2.drawContours(img,[box],0,(0,0,255),2)
    
    # the exterior points 
    contour  =  contours[0].reshape([-1,2])
    #print(contour)
    min_x, min_y = np.min(contour, 0)
    max_x, max_y = np.max(contour, 0)
    #print( min_x, min_y)
    #print(max_x, max_y)
    grid_step = 5 #(max_y - min_y)//num_grid
    print(f"step:{grid_step}")
        
    contour_points = []
    #= contour.tolist()
    # add contour points
    i = 0
    contour_points.append(i)
    cv2.drawMarker(img, (contour[i,0],contour[i,1]), color=(0,255,0), markerType=cv2.MARKER_CROSS, markerSize = 4, thickness=1)
    for i in range(1, len(contour)): 
        dx, dy = contour[i-1,:] - contour[i,:]
        if dx*dx + dy*dy > grid_step*grid_step:
            contour_points.append(i)
            cv2.drawMarker(img, (contour[i,0],contour[i,1]), color=(0,255,0), markerType=cv2.MARKER_CROSS, markerSize = 4, thickness=1)

    # 2. detect landmark corners 
    landmarks = detect_landmarks(contour, contour_points, img)    
    if debug:
        print(f"landmarks:{landmarks}")
        plt.imshow(img)
        plt.show()
        
    # 3. detect sewing edge 
    if True:
        sewing_eddges =  detect_sewing_edges(contour, landmarks, img)
        print(f"sewing_edge: {sewing_eddges}")

    #print(f"cp-before:{contour_points}")
    contour_points.sort()
    #print(f"cp:{contour_points}")
    pts = contour[contour_points,:]
    N = len(pts)
    i = np.arange(N)
    seg = np.stack([i, i + 1], axis=1) % N
    if debug:
        print(f"N={N}")
        print(f"pts={pts}")
        print(f"seg={seg}")
        
    meshinfo = dict(vertices=pts.astype(np.float32), segments=seg)
    mesh = tr.triangulate(meshinfo, 'qpa700')
     
    if debug:
        tr.compare(plt, meshinfo, mesh)
        plt.show()
        
    return mesh    
 
if __name__ == "__main__":

    cloth_list = ["dTest_F", "dTest_B", "uTest_F", "uTest_B"]
    mask_list = ["dTest_mask_F", "dTest_mask_B", "uTest_mask_F", "uTest_mask_B"]
    cloth_type_list = ["top_short", "top_short", "pants_long", "pants_long"]
        
    for cloth_id, mask_id, cloth_type in zip(cloth_list, mask_list, cloth_type_list):

        cloth_path = os.path.join("cloth", cloth_id + ".jpg")
        cloth_mask_path = os.path.join("cloth-mask", mask_id + ".png")
        
        print(f"{cloth_path}, {cloth_mask_path}")
        
        cloth = cv2.imread(cloth_path)
        cloth_mask = cv2.imread(cloth_mask_path, cv2.IMREAD_UNCHANGED)
        
        if cloth is None or cloth_mask is None:
            print(f"No such a file {cloth_path} ")
            exit()
        
        if cloth_mask is None or cloth_mask is None:
            print(f"No such a file {cloth_mask_path} ")
            exit()    
        '''    
        plt.subplot(1,2,1)
        plt.imshow(cloth)
        plt.subplot(1,2,2)
        plt.imshow(cloth_mask)
        plt.show()
        '''
        #uv2mesh_old(cloth, cloth_mask, cloth_type)
        tri_mesh = uv2mesh_using_trangle(cloth, cloth_mask, cloth_type, num_grid = 20, debug = True)
        save_tri_obj(tri_mesh, file_path = cloth_id +".obj", texture_path = cloth_path, texture_size =(cloth.shape[1], cloth.shape[0]))