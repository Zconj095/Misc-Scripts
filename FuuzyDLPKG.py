import numpy as np
from PIL import Image

class TrilinearVLF:
    def __init__(self, texture):
        self.texture = texture
        
    def filter(self, uv):
        # Get surrounding texels
        u, v = uv
        u0 = int(np.floor(u))
        u1 = u0 + 1
        v0 = int(np.floor(v))
        v1 = v0 + 1
        
        # Clamp texture bounds
        u0 = min(max(u0, 0), self.texture.width - 1)  
        u1 = min(max(u1, 0), self.texture.width - 1)
        v0 = min(max(v0, 0), self.texture.height - 1)
        v1 = min(max(v1, 0), self.texture.height - 1)
        
        c00 = self.texture[v0, u0] 
        c01 = self.texture[v0, u1]
        c10 = self.texture[v1, u0]
        c11 = self.texture[v1, u1]
        
        vec00 = self.vector_freq(uv, (u0, v0))
        vec01 = self.vector_freq(uv, (u1, v0)) 
        vec10 = self.vector_freq(uv, (u0, v1))
        vec11 = self.vector_freq(uv, (u1, v1))
        
        freq = (vec00 + vec01 + vec10 + vec11) / 4
        
        return self.interpolate(c00, c01, c10, c11, freq)
    
    def vector_freq(self, uv, texel):
        du = uv[0] - texel[0]
        dv = uv[1] - texel[1]
        return np.sqrt(du**2 + dv**2)
        
    def interpolate(self, c00, c01, c10, c11, freq):
        fu = freq*(c01[0] - c00[0]) + c00[0]
        fv = freq*(c10[1] - c00[1]) + c00[1]
        fc = freq*(c11[2] - c00[2]) + c00[2]
        return (int(fu), int(fv), int(fc))
