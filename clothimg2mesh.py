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
    
def find_nearest_point_index(pt, contour):

    ''' KD tree is not for this, since only one time used '''
    distances = (contour[:,0]-pt[0])**2 + (contour[:,1]-pt[1])**2 
    return np.argmin(distances)


def nearest_dist(nearest_idx, landmarks, contour):

    xt, yt = contour[nearest_idx,0], contour[nearest_idx,1]
    distances = (contour[landmarks,0]-xt)**2 + (contour[landmarks,1]-yt)**2 
    return np.min(distances)
     
    
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
    # return : return the index list for the corners and curvatures  
   

    #  Detect Corners 
    # Approximate the contour,  @TODO not use appxomiation for sampling 
    epsilon = epsilon*cv2.arcLength(contour, True)    
    approx = cv2.approxPolyDP(contour, epsilon, True)
    contour_approx  =  approx.reshape([-1,2])
    # 2.2 get angles 
    curvature = getCurvature(contour_approx, 1) 
    
    #print(f"curvature:{curvature}")    # 4, 5 
    landmarks = []
    curvature_landmark = []
    # 2.3 detect and marking the corners 
    curvature_thres  = 30 #30 # degree
    if debug:
        print(f"len(contour_approx)={len(contour_approx)}")
        print(f"curvature={curvature}")
        
    count_landmarks = 0
    for i in range(len(contour_approx)):
        if curvature[i] > curvature_thres or  curvature[i] < -curvature_thres:  # some meaningfull angles 
            # find closest contour points 
            nearest_idx = find_nearest_point_index((contour_approx[i,0],contour_approx[i,1]), contour)
            #print(f"approx={contour_approx[i,:]}, contour={contour[nearest_idx]} at {nearest_idx}")
            if not(nearest_idx in contour_points):  # BUG FIX: 2022.11.10
                if (len(landmarks) == 0) or (not(nearest_idx in landmarks) and nearest_dist(nearest_idx,landmarks, contour) > min_dist**2) : 
                    landmarks.append(nearest_idx)  
                    curvature_landmark.append(curvature[i])
                    contour_points.append(nearest_idx)
                    # viualize 
                    if debug:
                        cv2.drawMarker(img, (contour[nearest_idx,0],contour[nearest_idx,1]), color=(255,0,0), markerType=cv2.MARKER_STAR, markerSize = 4, thickness=1)
                        cv2.putText(img, f"{count_landmarks}({curvature[i]:.0f})", org = (contour[nearest_idx,0],contour[nearest_idx,1]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,0,0), thickness = 1)
                    count_landmarks += 1

    return landmarks, curvature_landmark
 
 
def save_tri_obj(tri, base_dir, file_name, texture_size = None):

    ''' 
        tri : front 2-d mesh 
        base_dir:  base_directory 
        file_name:  used for obj, mtl, texture files 
        
        save as Waterfront OBJ file (@TODO Does Traingle package have an API for it?)      
    '''
    with open(os.path.join(file_path, file_name + '.obj'), "w") as fp:

        # 1. header part if texture used
        if texture_size is not None:
             mtl_path = os.path.join(file_path, file_name + ".mtl")
             fp.write( 'mtllib %s\n' % (file_name + ".mtl"))   # Linking info 
             with open(mtl_path, "w") as fp2:         # linking file    
                fp2.write("newmtl material_0\n")
                fp2.write("# shader_type beckmann\n")
                fp2.write("map_Kd %s"%(file_name + ".png"))
                print('..Output mesh saved to: ', mtl_path) 
          
        # 2. vertices     
        v = tri['vertices']
        v[:,0] = v[:,0]/texture_size[0]  # x width normalization   
        v[:,1] = v[:,1]/texture_size[1]  # y height normalization 
        for i in range(len(v)): 
            fp.write( 'v %.2f %.2f %.2f\n' % ( v[i,0], 1.0 - v[i,1], 0) )  # x, y, z = 0, up side down
           
        # 3. texture vertices if used    
        if texture_size is not None:    
            vt = v
            for i in range(len(vt)): 
                fp.write( 'vt %.2f %.2f %.2f\n' % (vt[i,0], 1.0 - vt[i,1], 0) )  # x, y, z = 0  The UV coordinate u-right, v up   
        
        # 4. faces 
        if texture_size is not None:                   
            f = tri['triangles']   
            for i in range(len(f)): # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d/%d %d/%d %d/%d\n' %  (f[i,0]+1, f[i,0]+1, f[i,1]+1, f[i,1]+1, f[i,2]+1, f[i,2]+1) )  # index from 1 in OBJ file 
        else:
            f = tri['triangles']   
            for i in range(len(f)): # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[i,0]+1, f[i,1]+1, f[i,2]+1) )   # index from 1 in OBJ file 
    ## Print message
    print('..Output mesh saved to: ', os.path.join(file_path, file_name + '.obj'))       
    
       
def save_both_tri_obj(tri, base_dir, file_name, texture_size = None):

    ''' 
        tri : front 2-d mesh 
        base_dir:  base_directory 
        file_name:  used for obj, mtl, texture files 
        
        save two side as Waterfront OBJ file 
        none texture version and texture version 
        
    '''
    with open(os.path.join(base_dir, file_name + '.obj'), "w") as fp:

        # 1. header part if texture used
        if texture_size is not None:
             mtl_path = os.path.join(base_dir, file_name + ".mtl")
             fp.write( 'mtllib %s\n' % (file_name + ".mtl"))   # Linking info 
             with open(mtl_path, "w") as fp2:         # linking file    
                fp2.write("newmtl material_0\n")
                fp2.write("# shader_type beckmann\n")
                fp2.write("map_Kd %s"%(file_name + ".png"))
                print('..Output mesh saved to: ', mtl_path) 
          
        # 2. vertices     
        v = tri['vertices'].copy()
        v[:,0] = v[:,0]/texture_size[0]  # x width normalization   
        v[:,1] = v[:,1]/texture_size[1]  # y height normalization 
        zgap = 0.25
        # front 
        for i in range(len(v)): 
            fp.write( 'v %.2f %.2f %.2f\n' % ( v[i,0], 1.0 - v[i,1],-zgap) )  # x, y, z = 0, up side down
          
        # back 
        for i in range(len(v)): 
            fp.write( 'v %.2f %.2f %.2f\n' % ( v[i,0], 1.0 - v[i,1], zgap) )  # x, y, z = 0, up side down
      
      
        #front normal vector
        fp.write("vn 0.000 0.000 -1.000\n")
        #back normal vector
        fp.write("vn 0.000 0.000 1.000\n")
      
        # 3. texture vertices if used    
        if texture_size is not None:    
            vt = tri['vertices'].copy()
            vt[:,0] = vt[:,0]/texture_size[0]  # [0,1/2]  u   
            vt[:,1] = vt[:,1]/texture_size[1]    # [0,1] v  
            for i in range(len(vt)): 
                fp.write( 'vt %.2f %.2f\n' % (vt[i,0], 1.0 - vt[i,1]) )  # x, y, The UV coordinate u-right, v up   
            vt[:,0] = vt[:,0] + 0.5  # x width normalization   
            for i in range(len(vt)): 
                fp.write( 'vt %.2f %.2f\n' % (vt[i,0], 1.0 - vt[i,1]) )  # x, y,  The UV coordinate u-right, v up   
        
        
        # 4. faces 
        num_v = len(v)  # number of one side vertices, offset for the back side 
        if texture_size is not None:                   
            f = tri['triangles'] 
            # front 
            for i in range(len(f)): # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d/%d/1 %d/%d/1 %d/%d/1\n' %  
                    (f[i,0]+1, f[i,0]+1, f[i,1]+1, f[i,1]+1, f[i,2]+1, f[i,2]+1) )  # cw,  1-index 
       
            # back 
            f[:,:] = f[:,:] + num_v
            for i in range(len(f)): # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d/%d/2 %d/%d/2 %d/%d/2\n' % 
                            (f[i,2]+1, f[i,2]+1, f[i,1]+1, f[i,1]+1, f[i,0]+1, f[i,0]+1) )  # ccw, 1-index
      
        else:
            f = tri['triangles']   
            # front 
            for i in range(len(f)): # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d/1 %d/1 %d/1\n' %  (f[i,0]+1, f[i,1]+1, f[i,2]+1) )   # cw,  1-index  
            
            # back 
            f[:,:] = f[:,:] + num_v
            for i in range(len(f)): # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d/2 %d/2 %d/2\n' %  (f[i,0]+1, f[i,1]+1, f[i,2]+1) )   # ccw, 1-index            
                
    ## Print message
    print('..Output mesh saved to: ', os.path.join(base_dir, file_name + '.obj'))       
    
    
def mask2mesh_using_triangle(img, mask, num_grid = 10,  debug = False):

    ''' make well shaped triangle mesh 
        for front side of clothing 
        back side will be copied from front side
        
        return mesh
               mesh['vertices']  = the vertices given + added 
               N  (number contours points)
    '''
    
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
    
    # 1. samling the exterior points with min_dist
    contour  =  contours[0].reshape([-1,2])
    #print(contour)
    min_x, min_y = np.min(contour, 0)
    max_x, max_y = np.max(contour, 0)
    #print( min_x, min_y)
    #print(max_x, max_y)
    min_dist = (max_y - min_y)//num_grid   # 5
    #print(f"min_dist:{min_dist}")
        
    contour_points = []
    i = 0 # 1st contour point 
    contour_points.append(i) 
    latest_contour_pts = contour[i,:]
    if debug:
        cv2.drawMarker(img, (contour[i,0],contour[i,1]), color=(0,255,0), markerType=cv2.MARKER_CROSS, markerSize = 4, thickness=1)
    # sampling the minimum distance of min_dist
    for i in range(1, len(contour)): 
        dx, dy = latest_contour_pts - contour[i,:]
        if dx*dx + dy*dy > min_dist*min_dist:
            contour_points.append(i)
            latest_contour_pts = contour[i,:]
            if debug: 
                cv2.drawMarker(img, (contour[i,0],contour[i,1]), color=(0,255,0), markerType=cv2.MARKER_CROSS, markerSize = 4, thickness=1)

    # 2. detect landmark corners 
    landmarks, _  = detect_landmarks(contour, contour_points, img)    
    if debug:
        print(f"landmarks:{landmarks}")
        plt.imshow(img)
        plt.show()
        
    # 3. detect sewing edge 
    if False:
        sewing_eddges =  detect_sewing_edges(contour, landmarks, img)
        print(f"sewing_edge: {sewing_eddges}")

    #print(f"cp-before:{contour_points}")
    contour_points.sort()
    #print(f"cp:{contour_points}")
    pts = contour[contour_points,:]  # make the points unique here 
    N = len(pts)
    i = np.arange(N)
    seg = np.stack([i, i + 1], axis=1) % N
    if False:
        print(f"N={N}")
        print(f"pts={pts}")
        print(f"seg={seg}")
        
    meshinfo = dict(vertices=pts.astype(np.float32), segments=seg)
    mesh = tr.triangulate(meshinfo, 'qpa700')
     
    if debug:
        #print(f"input vertices:{pts}")
        #print(f"first out:{mesh['vertices'][:N]}")
        tr.compare(plt, meshinfo, mesh)
        plt.show()
        
    return mesh, N   
 
def match_back2front_pca(clothF, clothF_mask, clothB, clothB_mask):

    ''' 
        matching/warping back to front using PCA method 
    
    '''
  
    (h,w) = clothF.shape[:2]
  
    # 1. covariance 
    ptsF = np.array(np.where(clothF_mask>127)).T
    ptsB = np.array(np.where(clothB_mask>127)).T
    centerF = (np.mean(ptsF[:,1]), np.mean(ptsF[:,0]))  # center 
    centerB = (np.mean(ptsB[:,1]), np.mean(ptsB[:,0]))  # center
    covF = np.cov(ptsF[:,1], ptsF[:,0])
    covB = np.cov(ptsB[:,1], ptsB[:,0])
    
    # 2. eigen vec and values 
    evalF, evecF = np.linalg.eig(covF)
    evalB, evecB = np.linalg.eig(covB)

    # eigen vectors 
    if evalF[1]>evalF[0]:
        evalF = evalF[::-1]
        evecF = evecF[:,::-1]
    
    if evalB[1]>evalB[0]:
        evalB = evalB[::-1]
        evecB = evecB[:,::-1]

    # eignen values 
    evalF = np.sqrt(evalF)
    evalB = np.sqrt(evalB)

    # 3. elipse approximation and 3 points (center and long and short end points)
    affinePtsF = np.float32([centerF, 
                            [centerF[0]+int(evalF[0]*evecF[0][0]), centerF[1]+int(evalF[0]*evecF[0][1])],
                            [centerF[0]+int(evalF[1]*evecF[1][0]), centerF[1]+int(evalF[1]*evecF[1][1])]])
    affinePtsB = np.float32([centerB, 
                            [centerB[0]+int(evalB[0]*evecB[0][0]), centerB[1]+int(evalB[0]*evecB[0][1])],
                            [centerB[0]+int(evalB[1]*evecB[1][0]), centerB[1]+int(evalB[1]*evecB[1][1])]])
    
    # 4. estimate Affine matrix from 3 points 
    M = cv2.getAffineTransform(affinePtsB, affinePtsF)
    
    # 5. warping back to match front as much as possible 
    clothB_warped = cv2.warpAffine(clothB, M, (w,h))
    clothB_mask_warped = cv2.warpAffine(clothB_mask, M, (w,h))
    
    return clothB_warped, clothB_mask_warped
  

def match_back2front_landmarks(clothF, clothF_mask, clothB, clothB_mask, debug = True):

    ''' 
        matching/warping back to front using landmarks  
        
        unfinished code
    '''
  
    (h,w) = clothF.shape[:2]
    # FRONT 
    if cv2.__version__[0] == '3':  
        __,contours,hierarchy = cv2.findContours(clothF_mask, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE  )
    else:
        contours,hierarchy = cv2.findContours(clothF_mask, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE  )
    
    #print(f"len(contours):{len(contours)}")
    #print(f"contours[0].shape:{contours[0].shape}")
    #print(f"hierarchy:{hierarchy}")
    #cv2.drawContours(img,[box],0,(0,0,255),2)
    
    # the exterior points 
    contourF  =  contours[0].reshape([-1,2])
  
    # 2. detect landmark corners 
    contour_pointsF = []
    landmarksF, curvatureF = detect_landmarks(contourF, contour_pointsF, clothF, debug=True)   

    ## BACK 
    if cv2.__version__[0] == '3':  
        __,contours,hierarchy = cv2.findContours(clothB_mask, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE  )
    else:
        contours,hierarchy = cv2.findContours(clothB_mask, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE  )
    
    #print(f"len(contours):{len(contours)}")
    #print(f"contours[0].shape:{contours[0].shape}")
    #print(f"hierarchy:{hierarchy}")
    #cv2.drawContours(img,[box],0,(0,0,255),2)
    
    # the exterior points 
    contourB  =  contours[0].reshape([-1,2])
  
    # 2. detect landmark corners 
    contour_pointsB = []
    landmarksB, curvatureB = detect_landmarks(contourB, contour_pointsB, clothB,  debug=True)   

    
    if debug:
        print(f"landmarksF:{landmarksF}")
        print(f"landmarksB:{landmarksB}")
        plt.subplot(1,3,1)
        plt.imshow(clothF)
        plt.subplot(1,3,2)
        plt.imshow(clothB)
        plt.subplot(1,3,3)
        # draw the contours and lanmaks
        img = np.zeros_like(clothB)
        count = 0
        for i in landmarksF:
            cv2.drawMarker(img, (contourF[i,0],contourF[i,1]), color=(255,0,0), markerType=cv2.MARKER_STAR, markerSize = 4, thickness=1)
            cv2.putText(img, f"{count}({curvatureF[count]:.0f})", org = (contourF[i,0],contourF[i,1]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,0,0), thickness = 1)
            count +=1
        count = 0
        for i in landmarksB:
            cv2.drawMarker(img, (contourB[i,0],contourB[i,1]), color=(0,0,255), markerType=cv2. MARKER_DIAMOND , markerSize = 4, thickness=1)
            cv2.putText(img, f"{count}({curvatureB[count]:.0f})", org = (contourB[i,0],contourB[i,1]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,255), thickness = 1)
            count +=1
        plt.imshow(img[:,:,::-1])        
        
        plt.show()
   
   
    # 2. find the matching withinin distance 
    dist_thres = 10  # 0.5 between the minum corner distance 
   
    if False:
   
        # 3. elipse approximation and 3 points (center and long and short end points)
        affinePtsF = np.float32()
        affinePtsB = np.float32()
        
        # 4. estimate Affine matrix from 3 points 
        M = cv2.getAffineTransform(affinePtsB, affinePtsF)
        
        # 5. warping back to match front as much as possible 
        clothB_warped = cv2.warpAffine(clothB, M, (w,h))
        clothB_mask_warped = cv2.warpAffine(clothB_mask, M, (w,h))
        
        return clothB_warped, clothB_mask_warped  
  
  

def test_make_mesh_single_image():

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


def test_make_mesh_both_image():

    # test sample for top and bottom
    cloth_id_list = ["dTest", "uTest"]
    cloth_front_list = ["dTest_F", "uTest_F"]
    cloth_back_list = ["dTest_B", "uTest_B"]
    mask_front_list = ["dTest_mask_F", "uTest_mask_F"]
    mask_back_list = [ "dTest_mask_B", "uTest_mask_B"]
    cloth_type_list = ["top_short", "pants_long"]
        
    for cloth_id, cloth_front, cloth_back, mask_front, mask_back \
            in zip(cloth_id_list, cloth_front_list, cloth_back_list, mask_front_list, mask_back_list):

        # 1. loading clothings and masks
        
        cloth_front_path = os.path.join("cloth", cloth_front + ".jpg")
        cloth_back_path = os.path.join("cloth", cloth_back + ".jpg")
        mask_front_path = os.path.join("cloth-mask", mask_front + ".png")
        mask_back_path = os.path.join("cloth-mask", mask_back + ".png")
        
        #print(f"{cloth_path}, {cloth_mask_path}")
        
        cloth_front = cv2.imread(cloth_front_path)
        cloth_back  = cv2.imread(cloth_back_path)
        mask_front = cv2.imread(mask_front_path, cv2.IMREAD_UNCHANGED)
        mask_back = cv2.imread(mask_back_path, cv2.IMREAD_UNCHANGED)
       
       
        if cloth_front is None or mask_front is None:
            print(f"No such a file {cloth_path} ")
            exit()
        
        if cloth_back is None or mask_back is None:
            print(f"No such a file {cloth_mask_path} ")
            exit()    
            
            
        if False:    
            plt.subplot(2,2,1)
            plt.imshow(cloth_front[:,:,::-1])
            plt.subplot(2,2,2)
            plt.imshow(cloth_back[:,:,::-1])
            plt.subplot(2,2,3)
            plt.imshow(mask_front, cmap='gray')
            plt.subplot(2,2,4)
            plt.imshow(mask_back, cmap='gray')
            
            plt.show()
            
            
        # 2. flipping the back clothing
        cloth_back_flipped = cloth_back[:,::-1, :]
        mask_back_flipped  = mask_back[:,::-1]
        
        if False:
            test = np.zeros_like(cloth_front)
            test[:,:,0] = mask_back
            test[:,:,1] = mask_back_flipped
            test[:,:,2] = 0
            plt.subplot(2,1,1)
            plt.title('back vs back flipped')
        
            plt.imshow(test[:,:,::-1])
            
            test[:,:,0] = mask_front
            test[:,:,1] = mask_back_flipped
            test[:,:,2] = 0
            plt.subplot(2,1,2)
            plt.imshow(test[:,:,::-1])
            plt.title('front vs back flipped')
            plt.show()


        # 3. matching two parts 
        
        # using PCA method         
        cloth_back_flipped, mask_back_flipped = match_back2front_pca(cloth_front, mask_front, cloth_back_flipped, mask_back_flipped)

        # refine using landmarks 
        if False:
            match_back2front_landmarks(cloth_front, mask_front, cloth_back_flipped, mask_back_flipped)
            #cloth_back_flipped, mask_back_flipped = match_back2front_landmarks(cloth_front, mask_front, cloth_back_flipped, mask_back_flipped)

        '''
          # @TODO DUNG  (SCM)
        # 3.1 find key points of clothings
        # I recommend to use contours ...

        # 3.2 find matching
      
        match_index_pairs = context_shape_match( .....)
        # also need the x, y coordinate for the indexed points too
        
        # 3.3 find affine transform 
        affine_mat = estimate_affine(match_index_pairs, front_pts, back_pts2)
        
        # 3.4 warp the back part to match front part 
        cloth_back_warped = cv.warpAffine(cloth_back_flipped, affine_mat, dsize[, dst[, flags[, borderMode[, borderValue]]]]	) ->	dst
        mask_back_warped = cv.warpAffine(mask_back_flipped, affine_mat, dsize[, dst[, flags[, borderMode[, borderValue]]]]	) ->	dst
        '''
        
        # 3.5 get intersection or front and back   
        mask_intersection = cv2.bitwise_and(mask_back_flipped, mask_front)
      
        # after warping, the intersection should be almost union !
        if True:
            test = np.zeros_like(cloth_front)
            test[:,:,0] = mask_back
            test[:,:,1] = mask_back_flipped
            test[:,:,2] = mask_intersection
            plt.subplot(2,3,1)
            plt.title('back vs back flipped and intersect')  
            plt.imshow(test[:,:,::-1])
            
            test[:,:,0] = mask_front
            test[:,:,1] = mask_back_flipped
            test[:,:,2] = mask_intersection
            plt.subplot(2,3,2)
            plt.imshow(test[:,:,::-1])
            plt.title('front vs back flipped and intersect')
            
            plt.subplot(2,3,3)
            plt.imshow( mask_intersection  )
            
            plt.subplot(2,3,4)
            cloth_front[mask_intersection < 127, : ] = 0
            plt.imshow(  cloth_front[:,:,::-1] )
            
            plt.subplot(2,3,5)
            cloth_back_flipped[mask_intersection < 127, : ] = 0
            plt.imshow(cloth_back_flipped[:,:,::-1] )
            
            plt.show()
        
        # do we need to modify the color clothing based on the mask ? maybe not
        
        # 4. make mesh
        # Professor will make this part first for you.
        # 4.1 get contours for the intersection 
        
        print(f"type of mask_intersection:{mask_intersection.dtype}")
        tri_mesh_front, n_contours = mask2mesh_using_triangle(cloth_front, mask_intersection, num_grid = 20, debug = True)
      
        # 4.2 take key points for mesh 
        # 4.3 make a 2D triangle mesh (using trimesh package)
        # 5. save retult
        # 5.1 make a 3D mesh for front and back (with z gap)
        # 5.2 save the mesh into OBJ file 
        
       
        if True:
            texture  = cv2.hconcat([cloth_front, cloth_back_flipped])
            cloth_texture_path = os.path.join('obj',  cloth_id +".png")
            cv2.imwrite(cloth_texture_path, texture)
            save_both_tri_obj(tri_mesh_front, base_dir = 'obj', file_name = cloth_id, 
                        texture_size =(texture.shape[1], texture.shape[0]))   
        else:
            cloth_both = cloth_front
            cloth_texture_path = os.path.join('obj',  cloth_id +".png")
            cv2.imwrite(cloth_texture_path, cloth_both)
            save_tri_obj(tri_mesh_front, base_dir = 'obj', file_name = cloth_id, 
                        texture_size =(cloth_both.shape[1], cloth_both.shape[0]))        
 
        # @TODO develope a algorithm to decide which vertices are used for sewing ***
        #       This is quite important contribution to the paper
        # 5.3 save sewing vertices indices pairs between front and back into npy file or pickle file 
        # now saving the all contours vertices index 
        
        import pickle
        edge_info = {'front': range(1, n_contours + 1),
                     'back' : range(len(tri_mesh_front['vertices']), len(tri_mesh_front['vertices']) + n_contours +1)}
        print(f"edge_info:{edge_info}")
        with open(os.path.join('obj', cloth_id +'.pickle'), 'wb') as handle:
            pickle.dump(edge_info, handle)
     
if __name__ == "__main__":

    test_make_mesh_both_image()