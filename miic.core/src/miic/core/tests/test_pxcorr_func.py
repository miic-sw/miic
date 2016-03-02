import miic.core.pxcorr_func as px
import numpy as np

def test_spectral_whitening():
    A = [[1,2,3,5,4,5,6,6,7,4],
         [2,65,7,3,7,3,1,6,8,3],
         [7,3,7,9,5,1,7,1,6,2],
         [7,4,7,9,32,1,5,8,9,54],
         [4,6,32,7,98,9,3,5,8,32],
         [5,3,1,5,6,3,1,5,7,98]]
    A = np.array(A,dtype=float).T
    # pseudo random phi
    phi = np.fliplr(A/13)
    B = A*(np.cos(phi) + 1j*np.sin(phi))
    # individual normalization
    Bt = B.copy()
    Bw = px.spectralWhitening(Bt,{},{})
    assert np.allclose(np.angle(B),np.angle(Bw),atol=1e-12), 'phase does not match'
    assert np.allclose(1.,np.absolute(Bw),atol=1e-12), 'amplitude does not match'
    # joint normalization
    Bt = B.copy()    
    Bw  = px.spectralWhitening(Bt,{'joint_norm':True},{})
    assert np.allclose(np.angle(B),np.angle(Bw),atol=1e-12), 'phase does not match'
    Bn = np.hstack((np.tile(np.atleast_2d(np.mean(A[:,:3],axis=1)).T,[1,3]),
                    np.tile(np.atleast_2d(np.mean(A[:,3:],axis=1)).T,[1,3])))
    amp = A/Bn
    #print amp
    #print np.absolute(Bw)
    assert np.allclose(amp,np.absolute(Bw),atol=1e-12), 'amplitude does not match'

	
