import numpy as np

def show_dim():
    mtx=np.ones((3,2))
    print(mtx.shape)
    print(mtx)
    print(mtx.flatten().shape)
    print(mtx.flatten())
    print(mtx.reshape(2,-1))
    print(mtx.reshape(2,-1).shape)
    ## Expand dimensions
    
    print(np.expand_dims(mtx,0).shape) ## at dim_post 0 
    print(np.expand_dims(mtx,0))

    print(np.expand_dims(mtx,1))
    print(np.expand_dims(mtx,1).shape) ## at dim_pos 1
    
    print(np.expand_dims(mtx,2))
    print(np.expand_dims(mtx,2).shape) ## at dim_pos 2

    print(mtx.reshape(2,1,-1).shape)
    
    print(mtx[0].shape)#2
    print(mtx[1].shape)#2
    print(mtx[2].shape)#2
    try:
        print(mtx[3].shape)#2
    except Exception as e:
        print(e)

show_dim()

print(len([]))